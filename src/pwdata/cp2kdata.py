import re, os, glob
import numpy as np
from tqdm import tqdm
from collections import Counter
from image import Image
from calculators.const import ELEMENTTABLE

class CP2KMD(object):
    def __init__(self, stdout_file):
        self.image_list:list[Image] = []
        stdout_path = os.path.dirname(stdout_file)
        traj_file = sorted(glob.glob(f"{stdout_path}/*-pos*"))[0]
        label_f_file = sorted(glob.glob(f"{stdout_path}/*-frc*"))[0]
        _, self.format = os.path.splitext(traj_file)
        self.format = self.format.lstrip('.')
        self.read_stdout(stdout_file)

        if self.format == "pdb":
            self.read_pdb(traj_file)
        elif self.format == "xyz":
            self.read_xyz(traj_file)
        elif self.format == "dcd":
            raise NotImplementedError("DCD format is not supported yet!")
        
        self.read_label_f(label_f_file)

        assert len(self.image_list) > 0, "No system loaded!"
    
    def get(self):
        return self.image_list

    def read_stdout(self, stdout_file):
        with open(stdout_file, "r") as f:
            stdout_contents = f.readlines()
        read_volume = True
        pbc_dict = {
                    "NONE": [False, False, False],
                    "X": [True, False, False],
                    "XY": [True, True, False],
                    "XYZ": [True, True, True],
                    "XZ": [True, False, True],
                    "Y": [False, True, False],
                    "YZ": [False, True, True],
                    "Z": [False, False, True]}
        for idx, ii in tqdm(enumerate(stdout_contents), total=len(stdout_contents), desc="Reading lattice from STDOUT"):
            if "CELL_TOP| Volume" in ii and read_volume:
                volume = float(stdout_contents[idx].split()[-1])    # angstrom^3
                lattice_angle = parse_lattice_angle(stdout_contents[idx+1:idx+7])
                pbc = pbc_dict.get(stdout_contents[idx+8].split()[-1], [False, False, False])
                read_volume = False
            elif "SCF run converged in" in ii:
                image = Image(lattice=lattice_angle["lattice"], pbc=pbc)
                self.image_list.append(image)
            elif "STRESS| Analytical stress tensor" in ii:
                stress_info = parse_stress(stdout_contents[idx+2:idx+5])
                stress = gpa2ev(stress_info["stress"], volume)
                image.stress = stress
              
    def read_pdb(self, traj_file):
        with open(traj_file, "r") as f:
            pdb_contents = f.readlines()
        step_pattern = re.compile(r"Step (\d+), time = (.+), E = (.+)")
        for idx, ii in tqdm(enumerate(pdb_contents), total=len(pdb_contents), desc="Reading PDB"):
            if ii.startswith("REMARK"):
                iteration = int(step_pattern.findall(ii)[0][0])
                Ep = float(step_pattern.findall(ii)[0][-1])       # Pot.[a.u.]
                image = self.image_list[iteration]
                image.iteration = iteration
                image.Ep = Ep * 27.2113838565563    # convert to eV
                # atom_names = []
                # atom_types_image = []
                atom_positions = []
                # position = []
                # atom_type_num = {}
            elif ii.startswith("ATOM"):
                atom = ii.split()
                atom_name = atom[2]
                atom_positions.append((atom_name, [float(_) for _ in atom[3:6]]))  # Store atom name and position together
                # position.append([float(_) for _ in atom[3:6]])
                # if atom_name not in atom_names:
                #     atom_names.append(atom_name)
                #     atom_type_num[atom_name] = 1
                # else:
                #     atom_type_num[atom_name] += 1
                # atom_types_image.append(ELEMENTTABLE[atom_name])
                atom_nums = int(atom[1])
            elif "END" in ii:
                atom_positions.sort(key=lambda x: x[0])  # Sort atom_positions by atom name
                position = [pos for _, pos in atom_positions]  # Extract sorted positions
                atom_names = [elem for elem, _ in atom_positions]  # Extract sorted atom names
                sc = Counter(atom_names)
                atom_type = list(sc.keys())
                atom_type_num = list(sc.values())
                image.cartesian = True
                image.atom_nums = atom_nums
                image.atom_types_image = [ELEMENTTABLE[atom] for atom in atom_names]
                image.atom_type = [ELEMENTTABLE[atom] for atom in atom_type]
                image.atom_type_num = atom_type_num
                image.position = position
                # If Atomic-Energy is not in the file, calculate it from the Ep
                atomic_energy, _, _, _ = np.linalg.lstsq([image.atom_type_num], np.array([image.Ep]), rcond=1e-3)
                atomic_energy = np.repeat(atomic_energy, image.atom_type_num)
                image.atomic_energy = atomic_energy.tolist()
    
    def read_xyz(self, traj_file):
        with open(traj_file, "r") as f:
            xyz_contents = f.readlines()
        step_pattern = re.compile(r"i =\s+(\d+), time =\s+([\d\.]+), E =\s+([-?\d\.]+)")
        for idx, ii in tqdm(enumerate(xyz_contents), total=len(xyz_contents), desc="Reading XYZ"):
            if "i = " in ii:
                iteration = int(step_pattern.findall(ii)[0][0])
                Ep = float(step_pattern.findall(ii)[0][-1])       # Pot.[a.u.]
                image = self.image_list[iteration]
                image.iteration = iteration
                image.atom_nums = int(xyz_contents[idx-1])
                atom_name = [atom.split()[0] for atom in xyz_contents[idx+1:idx+image.atom_nums+1]]
                atom_positions = [[float(_) for _ in atom.split()[1:]] for atom in xyz_contents[idx+1:idx+image.atom_nums+1]]
                atom_positions = list(zip(atom_name, atom_positions))
                atom_positions.sort(key=lambda x: x[0])
                position = [pos for _, pos in atom_positions]
                atom_names = [elem for elem, _ in atom_positions]
                sc = Counter(atom_names)
                atom_type = list(sc.keys())
                atom_type_num = list(sc.values())
                image.Ep = Ep * 27.2113838565563    # convert to eV
                image.position = position
                image.atom_type = [ELEMENTTABLE[atom] for atom in atom_type]
                image.atom_type_num = atom_type_num
                image.atom_types_image = [ELEMENTTABLE[atom] for atom in atom_names]
                image.cartesian = True
                # If Atomic-Energy is not in the file, calculate it from the Ep
                atomic_energy, _, _, _ = np.linalg.lstsq([image.atom_type_num], np.array([image.Ep]), rcond=1e-3)
                atomic_energy = np.repeat(atomic_energy, image.atom_type_num)
                image.atomic_energy = atomic_energy.tolist()

    def read_label_f(self, label_f_file):
        with open(label_f_file, "r") as f:
            label_f_contents = f.readlines()
        step_pattern = re.compile(r"i =\s+(\d+), time =\s+([\d\.]+), E =\s+([-?\d\.]+)")
        for idx, ii in tqdm(enumerate(label_f_contents), total=len(label_f_contents), desc="Reading label_force file"):
            if "i = " in ii:
                iteration = int(step_pattern.findall(ii)[0][0])
                image = self.image_list[iteration]
                assert image.iteration == iteration, "Iteration number not match!"
                assert image.atom_nums == int(label_f_contents[idx-1]), "Atom number not match!"
                atom_name = [atom.split()[0] for atom in label_f_contents[idx+1:idx+image.atom_nums+1]]
                atom_forces = [[float(_) for _ in atom.split()[1:]] for atom in label_f_contents[idx+1:idx+image.atom_nums+1]]
                atom_forces = list(zip(atom_name, atom_forces))
                atom_forces.sort(key=lambda x: x[0])  # Sort atom_forces by atom name
                image.force = [[tmp * 51.42206318571696 for tmp in force] for _, force in atom_forces]  # convert to eV/angstrom
            

class CP2KSCF(object):
    def __init__(self, stdout_file):
        self.image_list:list[Image] = []
        self.read_stdout(stdout_file)
        assert len(self.image_list) > 0, "No system loaded!"

    def get(self):
        return self.image_list

    def read_stdout(self, stdout_file):
        with open(stdout_file, "r") as f:
            stdout_contents = f.readlines()
        read_volume = True
        pbc_dict = {
                    "NONE": [False, False, False],
                    "X": [True, False, False],
                    "XY": [True, True, False],
                    "XYZ": [True, True, True],
                    "XZ": [True, False, True],
                    "Y": [False, True, False],
                    "YZ": [False, True, True],
                    "Z": [False, False, True]}
        for idx, ii in tqdm(enumerate(stdout_contents), total=len(stdout_contents), desc="Reading lattice from STDOUT"):
            if "SCF run NOT converged" in ii:
                print("SCF run did not converge. Stopping.")
                break
            elif "CELL_TOP| Volume" in ii and read_volume:
                volume = float(stdout_contents[idx].split()[-1])    # angstrom^3
                lattice_angle = parse_lattice_angle(stdout_contents[idx+1:idx+7])
                pbc = pbc_dict.get(stdout_contents[idx+8].split()[-1], [False, False, False])
                read_volume = False
                image = Image(lattice=lattice_angle["lattice"], pbc=pbc)
                self.image_list.append(image)
            elif "TOTAL NUMBERS AND MAXIMUM NUMBERS" in ii:
                atom_nums = int(stdout_contents[idx+3].split()[-1])
                image.atom_nums = atom_nums
            elif "ATOMIC COORDINATES IN ANGSTROM" in ii:
                position_info = self.parse_position(stdout_contents[idx+3:idx+3+atom_nums])
                image.position = position_info["position"]
                image.atom_type = position_info["atom_type"]
                image.atom_type_num = position_info["atom_type_num"]
                image.atom_types_image = position_info["atom_types_image"]
                image.cartesian = True
            # elif "SCF run converged in" in ii:
            elif "ENERGY| Total FORCE_EVAL" in ii:
                # energy = float(stdout_contents[idx-2].split()[-2])  # Pot.[a.u.]
                energy = float(ii.split()[-1])  # Pot.[a.u.]
                image.Ep = energy * 27.2113838565563   # convert to eV
                atomic_energy, _, _, _ = np.linalg.lstsq([image.atom_type_num], np.array([image.Ep]), rcond=1e-3)
                atomic_energy = np.repeat(atomic_energy, image.atom_type_num)
                image.atomic_energy = atomic_energy.tolist()
                # atomic_energy_dict = dict(zip(image.atom_types_image, atomic_energy))
                # image.atomic_energy = [atomic_energy_dict[atom_type] for atom_type in image.atom_types_image]
            elif "ATOMIC FORCES in [a.u.]" in ii:
                force_info = self.parse_force(stdout_contents[idx+3:idx+3+atom_nums])
                image.force = force_info["force"]
            elif "STRESS| Analytical stress tensor" in ii:
                stress_info = parse_stress(stdout_contents[idx+2:idx+5])
                stress = gpa2ev(stress_info["stress"], volume)
                image.stress = stress
    
    def parse_position(self, position_content):
        position = [[float(_) for _ in atom.split()[4:7]] for atom in position_content]
        type = [atom.split()[1] for atom in position_content]
        atom_positions = list(zip(type, position))
        atom_types_image = list(zip(type, [atom.split()[3] for atom in position_content]))
        atom_positions.sort(key=lambda x: x[0])
        atom_types_image.sort(key=lambda x: x[0])
        
        position = [pos for _, pos in atom_positions]
        atom_types_image = [int(type) for _, type in atom_types_image]
        # atom_types_image = [int(atom.split()[3]) for atom in position_content]
        sc = Counter(atom_types_image)
        atom_type = list(sc.keys())
        atom_type_num = list(sc.values())
        return {"position": position, "atom_type": atom_type, "atom_type_num": atom_type_num, "atom_types_image": atom_types_image}
    
    def parse_force(self, force_content):
        force = [[float(_) * 51.42206318571696 for _ in atom.split()[3:6]] for atom in force_content] # convert to eV/angstrom
        atom_forces = list(zip([atom.split()[1] for atom in force_content], force))
        atom_forces.sort(key=lambda x: x[0])
        force = [force for _, force in atom_forces]    
        return {"force": force}


def parse_lattice_angle(lattice_content):
    vector_pattern = r"Vector ([abc]) \[angstrom\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
    angle_pattern = r"Angle \([abc],[abc]\), ([\w]+)\s+\[degree\]:\s+(\d+\.\d+)"
    vector_pattern = re.compile(vector_pattern)
    angle_pattern = re.compile(angle_pattern)
    lattice_a = [float(_) for _ in vector_pattern.findall(lattice_content[0])[0][1:]]
    lattice_b = [float(_) for _ in vector_pattern.findall(lattice_content[1])[0][1:]]
    lattice_c = [float(_) for _ in vector_pattern.findall(lattice_content[2])[0][1:]]
    alpha = float(angle_pattern.findall(lattice_content[3])[0][1])
    beta = float(angle_pattern.findall(lattice_content[4])[0][1])
    gamma = float(angle_pattern.findall(lattice_content[5])[0][1])
    lattice = [lattice_a[:3], lattice_b[:3], lattice_c[:3]]
    angle = [alpha, beta, gamma]
    return {"lattice": lattice, "angle": angle} 
    
def parse_stress(stress_content):
    stress_pattern = re.compile(r"(-?\d+\.\d+)")
    stress_x = [float(_) for _ in stress_pattern.findall(stress_content[0])]
    stress_y = [float(_) for _ in stress_pattern.findall(stress_content[1])]
    stress_z = [float(_) for _ in stress_pattern.findall(stress_content[2])]
    stress = [stress_x, stress_y, stress_z]
    return {"stress": stress}

def gpa2ev(stress, volume):
    """
    default unit in cp2k is GPa
    virial = stress * volume
    """
    return np.array(stress) * volume / 160.21766208