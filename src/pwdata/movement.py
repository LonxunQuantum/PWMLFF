import numpy as np
import re
from tqdm import tqdm
from collections import Counter
from image import Image
from calculators.const import deltaE

class MOVEMENT(object):
    def __init__(self, movement_file):
        self.image_list:list[Image] = []
        self.number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
        self.load_movement_file(movement_file)

    def get(self):
        return self.image_list

    def load_movement_file(self, movement_file):
        # seperate content to image contents
        with open(movement_file, 'r') as rf:
            mvm_contents = rf.readlines()
        
        for idx, ii in tqdm(enumerate(mvm_contents), total=len(mvm_contents), desc="Loading data"):
            if "Iteration" in ii:
                energy_info = self.parse_energy_info(ii)
                image = Image(**energy_info)
                self.image_list.append(image)
            elif "Lattice" in ii:
                lattice_stress = self.parse_lattice_stress(mvm_contents[idx+1:idx+4])
                image.lattice = lattice_stress["lattice"]
                image.stress = lattice_stress["stress"]
            elif " Position" in ii:
                position = self.parse_position(mvm_contents[idx+1:idx+image.atom_nums+1])
                image.position = position["position"]
                image.atom_type = position["atom_type"]
                image.atom_type_num = position["atom_type_num"]
                image.atom_types_image = position["atom_types_image"]
            elif "Force" in ii:
                force = self.parse_force(mvm_contents[idx+1: idx+ image.atom_nums+1])
                image.force = force["force"]
            elif "Atomic-Energy" in ii:
                atomic_energy = self.parse_atomic_energy(mvm_contents[idx+1: idx+ image.atom_nums+1])
                for i, atom_type in enumerate(image.atom_types_image):
                    atomic_energy["atomic_energy"][i] += deltaE[atom_type]
                image.atomic_energy = atomic_energy["atomic_energy"]
            elif "-------------" in ii:
                image.content = mvm_contents[idx:]
                continue
            else:
                # If Atomic-Energy is not in the file, calculate it from the Ep
                if image is not None and len(image.atomic_energy) == 0 and image.atom_type_num:
                    atomic_energy, _, _, _ = np.linalg.lstsq([image.atom_type_num], np.array([image.Ep]), rcond=1e-3)
                    atomic_energy = np.repeat(atomic_energy, image.atom_type_num)
                    image.atomic_energy = atomic_energy.tolist()
        image.image_nums = len(self.image_list)
        print("Load data %s successfully! \t\t\t\t Image nums: %d" % (movement_file, image.image_nums))
    
    def parse_energy_info(self, energy_content):
        # "76 atoms,Iteration (fs) =   -0.1000000000E+01, Etot,Ep,Ek (eV) =   -0.1335474369E+05  -0.1335474369E+05   0.0000000000E+00, SCF =    14"
        # after re extract, numbers = ['76', '-0.1000000000E+01', '-0.1335474369E+05', '-0.1335474369E+05', '0.0000000000E+00', '14']
        numbers = self.number_pattern.findall(energy_content)
        atom_nums = int(numbers[0])
        iteration = format(float(numbers[1]), '.2f')
        Etot, Ep, Ek = float(numbers[2]), float(numbers[3]), float(numbers[4])
        if len(numbers) >= 5:
            scf = numbers[5]
        else:
            scf = None
        return {"atom_nums": atom_nums, "iteration": iteration, "Etot": Etot, "Ep": Ep, "Ek": Ek, "scf": scf}
    
    def parse_lattice_stress(self, lattice_content):
        lattice1 = [float(_) for _ in self.number_pattern.findall(lattice_content[0])]
        lattice2 = [float(_) for _ in self.number_pattern.findall(lattice_content[1])]
        lattice3 = [float(_) for _ in self.number_pattern.findall(lattice_content[2])]
        if "stress" in lattice_content[0]:
            stress = [lattice1[3:], lattice2[3:], lattice3[3:]]
        else:
            stress = []
        lattice = [lattice1[:3], lattice2[:3], lattice3[:3]]
        return {"lattice": lattice, "stress": stress}
    
    def parse_position(self, position_content):
        atom_types_image = []
        position = []
        for i in range(0, len(position_content)):
            numbers = self.number_pattern.findall(position_content[i])
            atom_types_image.append(int(numbers[0]))
            position.append([float(_) for _ in numbers[1:4]])
        counter = Counter(atom_types_image)
        atom_type = list(counter.keys())
        atom_type_num = list(counter.values())
        assert sum(atom_type_num) == len(position)
        return {"atom_type": atom_type, "atom_type_num": atom_type_num, "atom_types_image": atom_types_image, "position": position}
    
    def parse_force(self, force_content):
        force = []
        for i in range(0, len(force_content)):
            numbers = self.number_pattern.findall(force_content[i])
            force.append([- float(_) for _ in numbers[1:4]])    # PWmat have -force
        return {"force": force}
    
    def parse_atomic_energy(self, atomic_energy_content):
        atomic_energy = []
        for i in range(0, len(atomic_energy_content)):
            numbers = self.number_pattern.findall(atomic_energy_content[i])
            atomic_energy.append(float(numbers[1]))
        return {"atomic_energy": atomic_energy}