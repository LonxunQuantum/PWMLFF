import numpy as np
import os, re, sys, glob
from math import ceil
from tqdm import tqdm
from collections import Counter
# import time
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../lib")

class Image(object):
    def __init__(self, 
                 atom_type = None, atom_type_num = None, atom_num = None, atom_types_image = None, 
                 iteration=None, Etot = None, Ep = None, Ek = None, scf = None, lattice = None, 
                 stress = None, position = None, force = None, atomic_energy = None,
                 content = None, image_nums = None):
        self.atom_num = atom_num
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
                position = self.parse_position(mvm_contents[idx+1:idx+image.atom_num+1])
                image.position = position["position"]
                image.atom_type = position["atom_type"]
                image.atom_type_num = position["atom_type_num"]
                image.atom_types_image = position["atom_types_image"]
            elif "Force" in ii:
                force = self.parse_force(mvm_contents[idx+1: idx+ image.atom_num+1])
                image.force = force["force"]
            elif "Atomic-Energy" in ii:
                atomic_energy = self.parse_atomic_energy(mvm_contents[idx+1: idx+ image.atom_num+1])
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
        print("Load data %s successfully!" % movement_file)
        image.image_nums = len(self.image_list)
    
    def parse_energy_info(self, energy_content):
        # "76 atoms,Iteration (fs) =   -0.1000000000E+01, Etot,Ep,Ek (eV) =   -0.1335474369E+05  -0.1335474369E+05   0.0000000000E+00, SCF =    14"
        # after re extract, numbers = ['76', '-0.1000000000E+01', '-0.1335474369E+05', '-0.1335474369E+05', '0.0000000000E+00', '14']
        numbers = self.number_pattern.findall(energy_content)
        atom_num = int(numbers[0])
        iteration = format(float(numbers[1]), '.2f')
        Etot, Ep, Ek = float(numbers[2]), float(numbers[3]), float(numbers[4])
        if len(numbers) >= 5:
            scf = numbers[5]
        else:
            scf = None
        return {"atom_num": atom_num, "iteration": iteration, "Etot": Etot, "Ep": Ep, "Ek": Ek, "scf": scf}
    
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
                atom_num = int(config_contents[idx-1].split()[0])
                lattice_info = self.parse_lattice(config_contents[idx+1:idx+4])
                image = Image(**lattice_info)
                self.image_list.append(image)
                image.atom_num = atom_num
            elif "position" in ii.lower():
                position = self.parse_position(config_contents[idx+1:idx+image.atom_num+1])
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
    
class Save_Data(object):
    def __init__(self, data_path, datasets_path = "./PWdata", train_data_path = "train", valid_data_path = "valid", 
                 train_ratio = None, random = False, seed = 2024, retain_raw = False, format = "movement") -> None:
        if format == "config":
            self.image_data = CONFIG(data_path)
        elif format == "movement":
            self.data_name = os.path.basename(data_path)
            self.labels_path = os.path.join(datasets_path, self.data_name)
            if os.path.exists(datasets_path) is False:
                os.makedirs(datasets_path, exist_ok=True)
            if not os.path.exists(self.labels_path):
                os.makedirs(self.labels_path, exist_ok=True)
            if len(glob.glob(os.path.join(self.labels_path, train_data_path, "*.npy"))) > 0:
                print("Data %s has been processed!" % self.data_name)
                return
            self.image_data = MOVEMENT(data_path)
        self.lattice, self.position, self.energies, self.ei, self.forces, self.virials, self.atom_type, self.atom_types_image, self.image_nums = self.get_all(self.image_data)

        if format != "config" and train_ratio is not None:  # inference 时不存数据
            self.train_ratio = train_ratio        
            self.split_and_save_data(seed, random, self.labels_path, train_data_path, valid_data_path, retain_raw)
        
    def get_all(self, image_data):
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

if __name__ == "__main__":

    # MOVEMENT("/data/home/hfhuang/2_MLFF/2-DP/19-json-version/4-CH4-dbg/MOVEMENT")
    # start = time.time()
    Save_Data("/data/home/hfhuang/2_MLFF/2-DP/19-json-version/4-CH4-dbg/pwdata/MOVEMENT", 0.8)
    # end = time.time()
    # print(end-start)
