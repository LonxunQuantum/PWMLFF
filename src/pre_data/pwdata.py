import numpy as np
import os, re, sys, glob
import numpy.linalg as LA
from math import ceil
from tqdm import tqdm
from collections import Counter
# import time
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../lib")

elements = ["0", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", 
                "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", 
                "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
                "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", 
                "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", 
                "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", 
                "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", 
                "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", 
                "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", 
                "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", 
                "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg"]
class Image(object):
    def __init__(self, 
                 atom_type = None, atom_type_num = None, atom_nums = None, atom_types_image = None, 
                 iteration=None, Etot = None, Ep = None, Ek = None, scf = None, lattice = None, 
                 stress = None, position = None, force = None, atomic_energy = None,
                 content = None, image_nums = None):
        self.atom_nums = atom_nums
        self.iteration = iteration
        self.atom_type = atom_type
        self.atom_type_num = atom_type_num
        self.atom_types_image = atom_types_image
        self.Etot = Etot
        self.Ep = Ep
        self.Ek = Ek
        self.scf = scf
        self.image_nums = image_nums
        self.lattice = lattice if lattice is not None else []
        self.stress = stress if stress is not None else []
        self.position = position if position is not None else []
        self.force = force if force is not None else []
        self.atomic_energy = atomic_energy if atomic_energy is not None else []
        self.content = content if content is not None else []

class MOVEMENT(object):
    def __init__(self, movement_file) -> None:
        self.image_list:list[Image] = []
        self.number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
        self.deltaE ={
        1: -45.140551665, 3: -210.0485218888889, 4: -321.1987119, 5: -146.63024691666666, 6: -399.0110205833333, 
        7: -502.070125, 8: -879.0771215, 9: -1091.0652775, 11: -1275.295054, 12: -2131.9724644444445, 13: -2412.581311, 
        14: -787.3439924999999, 15: -1215.4995769047619, 16: -1705.5754946875, 17: -557.9141695, 19: -1544.3553605, 
        20: -1105.0024515, 21: -1420.574128, 22: -1970.9374273333333, 23: -2274.598644, 24: -2331.976294, 
        25: -2762.3960913793107, 26: -3298.6401545, 27: -3637.624857, 28: -4140.3502, 29: -5133.970898611111, 
        30: -5498.13054, 31: -2073.70436625, 32: -2013.83114375, 33: -463.783827, 34: -658.83885375, 35: -495.05260075, 
        37: -782.22601375, 38: -1136.1897344444444, 39: -1567.6510633333335, 40: -2136.8407, 41: -2568.946113, 
        42: -2845.9228975, 43: -3149.6645705, 44: -3640.458547, 45: -4080.81555, 46: -4952.347355, 
        47: -5073.703895555555, 48: -4879.3604305, 49: -2082.8865266666667, 50: -2051.94076125, 51: -2380.010715, 
        52: -2983.2449, 53: -3478.003375, 55: -1096.984396724138, 56: -969.538106, 72: -2433.925215, 73: -2419.015324, 
        74: -2872.458516, 75: -4684.01374, 76: -5170.37679, 77: -4678.720765, 78: -5133.04942, 79: -5055.7201, 
        80: -5791.21431, 81: -1412.194369, 82: -2018.85905225, 83: -2440.8732966666666
        }
        self.load_movement_file(movement_file)

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
                    atomic_energy["atomic_energy"][i] += self.deltaE[atom_type]
                image.atomic_energy = atomic_energy["atomic_energy"]
            elif "-------------" in ii:
                image.content = mvm_contents[idx:]
                continue
            else:
                # If Atomic-Energy is not in the file, calculate it from the Ep
                if image and len(image.atomic_energy) == 0 and image.atom_type_num:
                    atomic_energy, _, _, _ = np.linalg.lstsq([image.atom_type_num], np.array([image.Ep]), rcond=1e-3)
                    atomic_energy = np.repeat(atomic_energy, image.atom_type_num)
                    image.atomic_energy = atomic_energy.tolist()
        print("Load data %s successfully! \t\t\t\t Image nums: %d" % (movement_file, image.image_nums))
        image.image_nums = len(self.image_list)
    
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
    
class CONFIG(object):
    def __init__(self, config_file) -> None:
        self.image_list:list[Image] = []
        self.number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
        self.load_config_file(config_file)

    def load_config_file(self, config_file):
        # seperate content to image contents
        with open(config_file, 'r') as rf:
            config_contents = rf.readlines()
        
        for idx, ii in tqdm(enumerate(config_contents), total=len(config_contents), desc="Loading data"):
            if "lattice" in ii.lower():
                atom_nums = int(config_contents[idx-1].split()[0])
                lattice_info = self.parse_lattice(config_contents[idx+1:idx+4])
                image = Image(**lattice_info)
                self.image_list.append(image)
                image.atom_nums = atom_nums
            elif "position" in ii.lower():
                position = self.parse_position(config_contents[idx+1:idx+image.atom_nums+1])
                image.position = position["position"]
                image.atom_type = position["atom_type"]
                image.atom_type_num = position["atom_type_num"]
                image.atom_types_image = position["atom_types_image"]
        image.image_nums = len(self.image_list)
        print("Load data %s successfully!" % config_file)
    
    def parse_lattice(self, lattice_content):
        lattice1 = [float(_) for _ in self.number_pattern.findall(lattice_content[0])]
        lattice2 = [float(_) for _ in self.number_pattern.findall(lattice_content[1])]
        lattice3 = [float(_) for _ in self.number_pattern.findall(lattice_content[2])]
        lattice = [lattice1[:3], lattice2[:3], lattice3[:3]]
        return {"lattice": lattice}
    
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
    
class OUTCAR(object):
    def __init__(self, outcar_file) -> None:
        self.image_list:list[Image] = []
        self.number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
        self.load_outcar_file(outcar_file)

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
                elif "free  energy   TOTEN" in line:
                    energy_info = self.parse_energy_info(line)
                    image.Etot = energy_info["Etot"]
            image.atom_nums = atom_nums
            image.atom_types_image = atom_types_image
            image.atom_type = list(Counter(atom_types_image).keys())
            image.atom_type_num = atom_type_num
            # If Atomic-Energy is not in the file, calculate it from the Ep
            if image and len(image.atomic_energy) == 0 and image.atom_type_num:
                atomic_energy, _, _, _ = np.linalg.lstsq([image.atom_type_num], np.array([image.Etot]), rcond=1e-3)
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
    
class POSCAR(object):
    def __init__(self, poscar_file) -> None:
        self.image_list:list[Image] = []
        self.number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
        self.load_poscar_file(poscar_file)

    def load_poscar_file(self, poscar_file):
        # seperate content to image contents
        with open(poscar_file, 'r') as rf:
            poscar_contents = rf.readlines()
        
        for idx, ii in tqdm(enumerate(poscar_contents), total=len(poscar_contents), desc="Loading data"):
            if idx == 0:
                image = Image()
                self.image_list.append(image)
                lattice_info = self.parse_lattice(poscar_contents[idx+2:idx+5])
                image.lattice = lattice_info["lattice"]
                atom_names = poscar_contents[idx+5].split()
                image.atom_type_num = [int(_) for _ in poscar_contents[idx+6].split()]
                image.atom_nums = sum(image.atom_type_num)
            elif "direct" in ii.lower():
                position = self.parse_position(poscar_contents[idx+1:idx+image.atom_nums+1], atom_names, image.atom_type_num)
                image.position = position["position"]
                atom_types_image = position["atom_types_image"]
                image.atom_type = [elements.index(atom_names[_]) for _ in range(len(image.atom_type_num))]
                image.atom_types_image = [elements.index(atom_types_image[_]) for _ in range(image.atom_nums)]
        image.image_nums = len(self.image_list)
        print("Load data %s successfully!" % poscar_file)
    
    def parse_lattice(self, lattice_content):
        lattice1 = [float(_) for _ in self.number_pattern.findall(lattice_content[0])]
        lattice2 = [float(_) for _ in self.number_pattern.findall(lattice_content[1])]
        lattice3 = [float(_) for _ in self.number_pattern.findall(lattice_content[2])]
        lattice = [lattice1[:3], lattice2[:3], lattice3[:3]]
        return {"lattice": lattice}
    
    def parse_position(self, position_content, atom_names, atom_type_num):
        position = []
        atom_types_image = []
        for i in range(0, len(position_content)):
            numbers = self.number_pattern.findall(position_content[i])
            position.append([float(_) for _ in numbers[:3]])
            if position_content[i].split()[3] in elements:
                atom_types_image.append(elements.index(position_content[i].split()[3]))
        
        if len(atom_types_image) == 0:
            for atom_name, num in zip(atom_names, atom_type_num):
                atom_types_image.extend([atom_name] * num)   
        return {"position": position, "atom_types_image": atom_types_image}

class Save_Data(object):
    def __init__(self, data_path, datasets_path = "./PWdata", train_data_path = "train", valid_data_path = "valid", 
                 train_ratio = None, random = False, seed = 2024, format = None, retain_raw = False) -> None:
        if format == "config":
            self.image_data = CONFIG(data_path)
        elif format == "poscar":
            self.image_data = POSCAR(data_path)
        elif format == "dump":
            pass
        else:
            self.data_name = os.path.basename(data_path)
            self.labels_path = os.path.join(datasets_path, self.data_name)
            if os.path.exists(datasets_path) is False:
                os.makedirs(datasets_path, exist_ok=True)
            if not os.path.exists(self.labels_path):
                os.makedirs(self.labels_path, exist_ok=True)
            if len(glob.glob(os.path.join(self.labels_path, train_data_path, "*.npy"))) > 0:
                print("Data %s has been processed!" % self.data_name)
                return
            if format == "movement":
                self.image_data = MOVEMENT(data_path)
            elif format == "outcar":
                self.image_data = OUTCAR(data_path)
            elif format == "xml":
                pass
        self.lattice, self.position, self.energies, self.ei, self.forces, self.virials, self.atom_type, self.atom_types_image, self.image_nums = get_all(self.image_data)

        if format != "config" and train_ratio is not None:  # inference 时不存数据
            self.train_ratio = train_ratio        
            self.split_and_save_data(seed, random, self.labels_path, train_data_path, valid_data_path, retain_raw)
    
    def split_and_save_data(self, seed, random, labels_path, train_path, val_path, retain_raw):
        if seed:
            np.random.seed(seed)
        indices = np.arange(self.image_nums)    # 0, 1, 2, ..., image_nums-1
        if random:
            np.random.shuffle(indices)              # shuffle the indices
        train_size = ceil(self.image_nums * self.train_ratio)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        # image_nums = [self.image_nums]
        atom_types_image = self.atom_types_image.reshape(1, -1)

        train_data = [self.lattice[train_indices], self.position[train_indices], self.energies[train_indices], 
                      self.forces[train_indices], atom_types_image, self.atom_type,
                      self.ei[train_indices]]
        val_data = [self.lattice[val_indices], self.position[val_indices], self.energies[val_indices], 
                    self.forces[val_indices], atom_types_image, self.atom_type,
                    self.ei[val_indices]]

        if len(self.virials) != 0:
            train_data.append(self.virials[train_indices])
            val_data.append(self.virials[val_indices])
        else:
            train_data.append([])
            val_data.append([])

        if self.train_ratio == 1.0 or len(val_indices) == 0:
            labels_path = os.path.join(labels_path, train_path)
            if not os.path.exists(labels_path):
                os.makedirs(labels_path)
            if retain_raw:
                self.save_to_raw(train_data, train_path)
            self.save_to_npy(train_data, labels_path)
        else:
            train_path = os.path.join(labels_path, train_path) 
            val_path = os.path.join(labels_path, val_path)
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            if not os.path.exists(val_path):
                os.makedirs(val_path)
            if retain_raw:
                self.save_to_raw(train_data, train_path)
                self.save_to_raw(val_data, val_path)
            self.save_to_npy(train_data, train_path)
            self.save_to_npy(val_data, val_path)
                
    def save_to_raw(self, data, directory):
        filenames = ["lattice.dat", "position.dat", "energies.dat", "forces.dat", "image_type.dat", "atom_type.dat", "ei.dat", "virials.dat"]
        formats = ["%.8f", "%.16f", "%.8f", "%.16f", "%d", "%d", "%.8f", "%.8f"]
        # for i in tqdm(range(len(data)), desc="Saving to raw files"):
        for i in range(len(data)):
            if i != 7 or (i == 7 and len(data[7]) != 0):
                np.savetxt(os.path.join(directory, filenames[i]), data[i], fmt=formats[i])

    def save_to_npy(self, data, directory):
        filenames = ["lattice.npy", "position.npy", "energies.npy", "forces.npy", "image_type.npy", "atom_type.npy", "ei.npy", "virials.npy"]
        # for i in tqdm(range(len(data)), desc="Saving to npy files"):
        for i in range(len(data)):
            if i != 7 or (i == 7 and len(data[7]) != 0):
                np.save(os.path.join(directory, filenames[i]), data[i])

class OUTCAR2MOVEMENT(object):
    def __init__(self, outcar_file, output_path, output_file) -> None:
        self.image_data = OUTCAR(outcar_file)
        self.lattice, self.position, self.energies, self.ei, self.forces, self.virials, self.atom_type, self.atom_types_image, self.image_nums = get_all(self.image_data)
        self.output_path = os.path.abspath(output_path)
        self.output_file = output_file
        self.save_to_movement()

    def save_to_movement(self):
        output_file = open(os.path.join(self.output_path, self.output_file), 'w')
        for i in range(self.image_nums):
            image_data = self.image_data.image_list[i]
            image_data.position = np.dot(image_data.position, LA.inv(image_data.lattice))
            # with open(os.path.join(self.output_path, self.output_file), 'a') as wf:
            output_file.write(" %d atoms,Iteration (fs) = %16.10E, Etot,Ep,Ek (eV) = %16.10E  %16.10E   %16.10E, SCF = %d\n"\
                              % (image_data.atom_nums, 0.0, image_data.Etot, image_data.Etot, 0.0, self.image_data.image_list[i].scf))
            output_file.write(" MD_INFO: METHOD(1-VV,2-NH,3-LV,4-LVPR,5-NHRP) TIME(fs) TEMP(K) DESIRED_TEMP(K) AVE_TEMP(K) TIME_INTERVAL(fs) TOT_TEMP(K) \n")
            output_file.write("          *    ************   ********   ********   ********    ********    ********\n")
            output_file.write("     TOTAL MOMENTUM\n")
            output_file.write("     ********    ********    ********\n")
            output_file.write(" MD_VV_INFO: Basic Velocity Verlet Dynamics (NVE), Initialized total energy(Hartree)\n")
            output_file.write("          *******              \n")
            output_file.write("Lattice vector (Angstrom)\n")
            for j in range(3):
                if image_data.stress != []:
                    output_file.write("  %16.10E    %16.10E    %16.10E     stress (eV): %16.10E    %16.10E    %16.10E\n" % (image_data.lattice[j][0], image_data.lattice[j][1], image_data.lattice[j][2], image_data.virials[j][0], image_data.virials[j][1], image_data.virials[j][2]))
                else:
                    output_file.write("  %16.10E    %16.10E    %16.10E\n" % (image_data.lattice[j][0], image_data.lattice[j][1], image_data.lattice[j][2]))
            output_file.write("  Position (normalized), move_x, move_y, move_z\n")
            for j in range(image_data.atom_nums):
                output_file.write(" %4d    %20.15F    %20.15F    %20.15F    1 1 1\n"\
                                  % (image_data.atom_types_image[j], image_data.position[j][0], image_data.position[j][1], image_data.position[j][2]))
            output_file.write("  Force (-force, eV/Angstrom)\n")
            for j in range(image_data.atom_nums):
                output_file.write(" %4d    %20.15F    %20.15F    %20.15F\n"\
                                  % (image_data.atom_types_image[j], -image_data.force[j][0], -image_data.force[j][1], -image_data.force[j][2]))
            output_file.write(" -------------------------------------\n")
        output_file.close()

def elements_to_order(atom_names, atom_types_image, atom_nums):
    """
    Replaces the atom types's order (from 1) to the order of the elements in the atom_names list.

    Args:
        atom_names (list): List of atom names.
        atom_types_image (list): List of atom types.
        atom_nums (int): Number of atoms.

    Example:
        >>> atom_names = ['C', 'N']
        >>> atom_types_image = [1, 1, 1, 1, 1, ... , 2, 2, 2, 2, 2, ... , 2]
        >>> atom_nums = 56
        >>> elements_to_order(atom_names, atom_types_image, atom_nums)
        [6, 6, 6, 6, 6, ... , 7, 7, 7, 7, 7, ... , 7]
        
    Returns:
        list: Updated list of atom types per atom.
    """
    for idx, name in enumerate(atom_names):
        for ii in range(atom_nums):
            if name in elements and atom_types_image[ii] == idx+1:
                atom_types_image[ii] = elements.index(name)
    return atom_types_image

def get_all(image_data):
    # Initialize variables to store data
    all_lattices = []
    all_postions = []
    all_energies = []
    all_ei = []
    all_forces = []
    all_virials = []
    for image in image_data.image_list:
        all_lattices.append(image.lattice)
        all_postions.append(image.position)
        all_energies.append(image.Etot)
        all_forces.append(image.force)
        all_ei.append(image.atomic_energy)
        if len(image.stress) != 0:
            all_virials.append(image.stress)  
    image_nums = image.image_nums
    atom_type = np.array(image.atom_type).reshape(1, -1)
    atom_types_image = np.array(image.atom_types_image)
    all_lattices = np.array(all_lattices).reshape(image_nums, 9)
    all_postions = np.array(all_postions).reshape(image_nums, -1)
    all_energies = np.array(all_energies).reshape(image_nums, 1)
    all_forces = np.array(all_forces).reshape(image_nums, -1)
    all_ei = np.array(all_ei).reshape(image_nums, -1)
    if len(all_virials) != 0:
        all_virials = np.array(all_virials).reshape(image_nums, -1)
    return all_lattices, all_postions, all_energies, all_ei, all_forces, all_virials, atom_type, atom_types_image, image_nums

if __name__ == "__main__":

    # MOVEMENT("/data/home/hfhuang/2_MLFF/2-DP/19-json-version/4-CH4-dbg/MOVEMENT")
    # start = time.time()
    Save_Data("/data/home/hfhuang/2_MLFF/2-DP/19-json-version/4-CH4-dbg/pwdata/MOVEMENT", 0.8)
    # end = time.time()
    # print(end-start)
    import argparse

    parser = argparse.ArgumentParser(description='Convert OUTCAR to MOVEMENT')
    parser.add_argument('--outcar_file', type=str, required=True, help='Path to the OUTCAR file')
    parser.add_argument('--output_path', type=str, required=False, help='Path to the output directory', default="./")
    parser.add_argument('--output_file', type=str, required=False, help='Name of the output file', default="MOVEMENT")

    args = parser.parse_args()
    OUTCAR2MOVEMENT(args.outcar_file, args.output_path, args.output_file)