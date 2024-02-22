import numpy as np
import re
from tqdm import tqdm
from collections import Counter
from image import Image, elements_to_order

class OUTCAR(object):
    def __init__(self, outcar_file) -> None:
        self.image_list:list[Image] = []
        self.number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
        self.load_outcar_file(outcar_file)

    def get(self):
        return self.image_list

    def load_outcar_file(self, outcar_file):
        # seperate content to image contents
        with open(outcar_file, 'r') as rf:
            outcar_contents = rf.readlines()

        atom_names = []
        atom_type_num = None
        nelm = None
        for idx, ii in enumerate(outcar_contents):
            if "POTCAR" in ii:
                # get atom names from POTCAR info, tested only for PAW_PBE ...
                _ii = ii.split()[2]
                if '_' in _ii:
                    atom_names.append(_ii.split('_')[0])
                else:
                    atom_name = _ii
                if atom_name not in atom_names:
                    atom_names.append(atom_name)
            elif 'ions per type' in ii:
                atom_type_num_ = [int(s) for s in ii.split()[4:]]
                if atom_type_num is None:
                    atom_type_num = atom_type_num_
                else:
                    assert (atom_type_num == atom_type_num_), "inconsistent number of atoms in OUTCAR"
            elif 'NELM   =' in ii:
                nelm = int(ii.split()[2][:-1])
                break
        assert (nelm is not None), "cannot find maximum steps for each SC iteration"
        assert (atom_type_num is not None), "cannot find ion type info in OUTCAR"
        atom_names = atom_names[:len(atom_type_num)]
        atom_types_image = []
        for idx, ii in enumerate(atom_type_num) :
            for _ in range(ii) :
                atom_types_image.append(idx+1)
        atom_nums = sum(atom_type_num)
        atom_types_image = elements_to_order(atom_names, atom_types_image, atom_nums)

        max_scf_idx = 0
        prev_idx = 0
        # temp_images = {}
        converged_images = []
        max_insw = -1
        for idx, ii in tqdm(enumerate(outcar_contents), total=len(outcar_contents), desc="Processing data"):
            if "Ionic step" in ii:
                if prev_idx == 0:
                    prev_idx = idx
                    # continue
                else:
                    # temp_images[idx] = outcar_contents[prev_idx:idx]
                    if max_insw < nelm:
                        converged_images.append(outcar_contents[max_scf_idx:idx])
                max_insw = 0
            if "Iteration" in ii:
                scf_index = int(ii.split()[3][:-1])
                if scf_index > max_insw:
                    max_insw = scf_index
                    max_scf_idx = idx
            if "Elapsed time (sec):" in ii:
                if max_insw < nelm:
                    converged_images.append(outcar_contents[max_scf_idx:idx])

        for converged_image in tqdm(converged_images, total=len(converged_images), desc="Loading converged data"):
            for idx, line in enumerate(converged_image):
                if "Iteration" in line:
                    image = Image()
                    self.image_list.append(image)
                    image.scf = int(line.split()[3][:-1])
                elif "in kB" in line:
                    virial_info = self.parse_virial_info(converged_image[idx - 1])
                    image.stress = virial_info["virial"]
                elif "VOLUME and BASIS" in line:
                    lattice_info = self.parse_lattice(converged_image[idx+5:idx+8])
                    image.lattice = lattice_info["lattice"]
                elif "TOTAL-FORCE" in line:
                    force_info = self.parse_force(converged_image[idx+2:idx+2+atom_nums])
                    image.force = force_info["force"] 
                    image.position = force_info["position"]   
                    image.cartesian = True            
                elif "free  energy   TOTEN" in line:
                    energy_info = self.parse_energy_info(line)
                    image.Ep = energy_info["Etot"]
            image.atom_nums = atom_nums
            image.atom_types_image = atom_types_image
            image.atom_type = list(Counter(atom_types_image).keys())
            image.atom_type_num = atom_type_num
            # If Atomic-Energy is not in the file, calculate it from the Ep
            if image is not None and len(image.atomic_energy) == 0 and image.atom_type_num:
                atomic_energy, _, _, _ = np.linalg.lstsq([image.atom_type_num], np.array([image.Ep]), rcond=1e-3)
                atomic_energy = np.repeat(atomic_energy, image.atom_type_num)
                image.atomic_energy = atomic_energy.tolist()
        # atom_type_num = list(counter.values())
        image.image_nums = len(self.image_list)
        print("Load data %s successfully! \t\t\t\t Image nums: %d" % (outcar_file, image.image_nums))
        
    def parse_virial_info(self, virial_content):
        numbers = self.number_pattern.findall(virial_content)
        tmp_virial = [float(_) for _ in numbers]
        virial = np.zeros(9)
        virial[0] = tmp_virial[0]     # xx
        virial[4] = tmp_virial[1]     # yy
        virial[8] = tmp_virial[2]     # zz
        virial[1] = tmp_virial[3]     # xy
        virial[3] = tmp_virial[3]     # yx
        virial[5] = tmp_virial[4]     # yz
        virial[7] = tmp_virial[4]     # zy
        virial[2] = tmp_virial[5]     # xz
        virial[6] = tmp_virial[5]     # zx
        return {"virial": virial}
    
    def parse_lattice(self, lattice_content):
        lattice1 = [float(_) for _ in self.number_pattern.findall(lattice_content[0])]
        lattice2 = [float(_) for _ in self.number_pattern.findall(lattice_content[1])]
        lattice3 = [float(_) for _ in self.number_pattern.findall(lattice_content[2])]
        lattice = [lattice1[:3], lattice2[:3], lattice3[:3]]
        return {"lattice": lattice}
    
    def parse_force(self, force_content):
        force = []
        position = []
        for i in range(0, len(force_content)):
            numbers = self.number_pattern.findall(force_content[i])
            position.append([float(_) for _ in numbers[:3]])
            force.append([float(_) for _ in numbers[3:6]])
        return {"position": position, "force": force}
    
    def parse_energy_info(self, energy_content):
        numbers = self.number_pattern.findall(energy_content)
        Etot = float(numbers[0])
        return {"Etot": Etot}