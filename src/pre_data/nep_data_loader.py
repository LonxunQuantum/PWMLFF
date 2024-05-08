import numpy as np
import os
import glob
from torch.utils.data import Dataset
import torch
import yaml
from src.lib.NeighConst import neighconst
from src.user.input_param import InputParam
from pwdata import Save_Data
import time
from src.pre_data.find_neigh.findneigh import FindNeigh

class MovementDataset(Dataset):
    def __init__(self, data_paths, config:dict, input_param, energy_shift, max_atom_nums):
        super(MovementDataset, self).__init__()
        self.dirs = data_paths  # include all movement data path

        self.atom_types = np.array([(_['type']) for _ in config["atomType"]])   # input atom type order
        self.m_neigh = config['maxNeighborNum']
        self.Rc_type = np.array([(_['Rc']) for _ in config["atomType"]])
        self.Rm_type = np.array([(_['Rm']) for _ in config["atomType"]])
        self.Rc_M = config['Rc_M']
        self.img_max_types = len(self.atom_types)
        self.Egroup = config['train_egroup']
        self.input_param = input_param
        self.img_max_atom_num = max_atom_nums # for multi batch size training 
        self.ener_shift = np.array(energy_shift)
        self.all_movement_data, self.total_images, self.images_per_dir, self.atoms_per_dir = self.__concatenate_data()

        self.calc = FindNeigh()
        # test

        # data = self.__load_data(0)
        # print()

    def __load_data(self, index):
        type_index = np.searchsorted(np.cumsum(self.images_per_dir), index + 1)
        data = {}
        data["Force"] = self.all_movement_data["forces.npy"][index]
        data["Ei"] = self.all_movement_data["ei.npy"][index]
        data["Etot"] = self.all_movement_data["energies.npy"][index]
        # data["ListNeighbor"] = self.all_movement_data["ListNeighbor"][index]
        # data["ImageDR"] = self.all_movement_data["ImageDR"][index]
        data["Position"] = self.all_movement_data["position.npy"][index]
        data["Lattice"] = self.all_movement_data["lattice.npy"][index]
        data["AtomType"] = self.all_movement_data["atom_type.npy"][type_index]
        data["AtomTypeMap"] = self.all_movement_data["type_maps"][type_index]
        atom_nums = self.atoms_per_dir[type_index]
        data["ImageAtomNum"] = self.atoms_per_dir[type_index]

        # #radial and angular is set from nep.txt file
        list_neigh, dR_neigh, Egroup_weight, Divider, Egroup  = self.find_neigh_nep(atom_nums, 
                            list(data["AtomTypeMap"]),
                            list(data["Lattice"].transpose(1, 0).reshape(-1)),
                            list(data["Position"].transpose(1, 0).reshape(-1)))

        data["ListNeighbor"] = list_neigh
        data["ImageDR"] = dR_neigh
        data["max_ri"] = 0
        Egroup_weight = None
        Divider = None
        Egroup = None

        if self.Egroup:
            if Egroup.shape[0] < self.img_max_atom_num:
                Egroup_weight = np.pad(Egroup_weight, ((0, self.img_max_atom_num - Egroup.shape[0]), (0, self.img_max_atom_num - Egroup.shape[0])))
                Divider = np.pad(Divider, ((0, self.img_max_atom_num - Egroup.shape[0])))
                Egroup = np.pad(Egroup, ((0, self.img_max_atom_num - Egroup.shape[0])))
            data["Egroup_weight"] = Egroup_weight
            data["Divider"] = Divider
            data["Egroup"] = Egroup
        if "virials.npy" in self.all_movement_data.keys():
            data["Virial"] = self.all_movement_data["virials.npy"][index]

        return data

    def find_neigh_nep(self, atom_nums, AtomTypeMap, Lattice, Position):
        # tall 18.700355052947998 t2 0.32842254638671875 t3 0.07180023193359375 t4 18.628554821014404 t5 0.022504329681396484 t6 0.007369279861450195
        # t1 = time.time()
        # from src.pre_data.nep_neigh.calculate import NEP
        # self.calc2 = NEP("/data/home/wuxingxing/datas/pwmat_mlff_workdir/hfo2/nep_ff_1image/model_record/nep.txt")
        # t2 = time.time()
        # self.calc2.calc.setAtoms(atom_nums, 
        #                         AtomTypeMap,
        #                         Lattice,
        #                         Position)
        # d12, NL_radial, NL_angular, NN_radial, NN_angular = self.calc2.calc.getNeigh()
        # t3 = time.time()
        # # self.calc.setAtoms(atom_nums, \
        # #                    self.input_param.descriptor.cutoff[0],self.input_param.descriptor.cutoff[1],
        # #                     AtomTypeMap, Lattice, Position)
        # neigh_radial_list  = np.zeros([atom_nums, self.m_neigh*self.img_max_types])
        # neigh_angular_list = np.zeros([atom_nums, self.m_neigh*self.img_max_types])

        # neigh_radial_rij   = np.zeros([atom_nums, self.m_neigh*self.img_max_types, 4])
        # neigh_angular_rij  = np.zeros([atom_nums, self.m_neigh*self.img_max_types, 4])        

        # split = int(len(d12)/6)
        # for i, num_neigh in enumerate(NN_radial):
        #     num_neigh_a = NN_angular[i]
        #     neigh_radial_list[i, :num_neigh]        = [NL_radial[_*atom_nums+i] + 1 for _ in range(0, num_neigh)]
        #     neigh_radial_rij[i, :num_neigh, 1]      = [d12[       :  split][_*atom_nums+i] for _ in range(0, num_neigh)]
        #     neigh_radial_rij[i, :num_neigh, 2]      = [d12[split  :2*split][_*atom_nums+i] for _ in range(0, num_neigh)]
        #     neigh_radial_rij[i, :num_neigh, 3]      = [d12[2*split:3*split][_*atom_nums+i] for _ in range(0, num_neigh)]
            
        #     neigh_angular_list[i, :num_neigh_a]     = [NL_angular[_*atom_nums+i] + 1 for _ in range(0, num_neigh_a)]
        #     neigh_angular_rij[i, :num_neigh_a, 1]   = [d12[3*split:4*split][_*atom_nums+i] for _ in range(0, num_neigh_a)]
        #     neigh_angular_rij[i, :num_neigh_a, 2]   = [d12[4*split:5*split][_*atom_nums+i] for _ in range(0, num_neigh_a)]
        #     neigh_angular_rij[i, :num_neigh_a, 3]   = [d12[5*split:       ][_*atom_nums+i] for _ in range(0, num_neigh_a)]
        # neigh_radial_rij[:, :, 0]  = np.sqrt(np.sum(neigh_radial_rij[:, :, 1:]**2, axis=-1))
        # neigh_angular_rij[:, :, 0] = np.sqrt(np.sum(neigh_angular_rij[:, :, 1:]**2, axis=-1))

        # if neigh_radial_list.shape[0] < self.img_max_atom_num:
        #     neigh_radial_list = np.pad(neigh_radial_list, ((0, self.img_max_atom_num - neigh_radial_list.shape[0]), (0, 0)))
        #     neigh_radial_rij = np.pad(neigh_radial_rij, ((0, self.img_max_atom_num - neigh_radial_rij.shape[0]), (0, 0), (0, 0)))

        # if neigh_angular_list.shape[0] < self.img_max_atom_num:
        #     neigh_angular_list = np.pad(neigh_angular_list, ((0, self.img_max_atom_num - neigh_angular_list.shape[0]), (0, 0)))
        #     neigh_angular_rij = np.pad(neigh_angular_rij, ((0, self.img_max_atom_num - neigh_angular_rij.shape[0]), (0, 0), (0, 0)))

        # list_neigh = np.concatenate((neigh_radial_list, neigh_angular_list), axis=0)
        # dR_neigh = np.concatenate((neigh_radial_rij, neigh_angular_rij), axis=0)

        Egroup_weight, Divider, Egroup = None, None, None
        # 34622.19498329725 d12_radial
        d12_radial, d12_agular, NL_radial2, NL_angular2, NN_radial2, NN_angular2 = self.calc.getNeigh(
                           self.input_param.descriptor.cutoff[0],self.input_param.descriptor.cutoff[1], 
                            atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num, Lattice, Position)
        neigh_radial_rij   = np.array(d12_radial).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num, 4)
        neigh_angular_rij  = np.array(d12_agular).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num, 4)
        neigh_radial_list  = np.array(NL_radial2).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num)
        neigh_angular_list = np.array(NL_angular2).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num)
        # print("tall {} t2 {} t3 {} t4 {} t5 {} t6 {}".format(t4-t2, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5))
        return neigh_radial_list, neigh_radial_rij, Egroup_weight, Divider, Egroup

    def __getitem__(self, index):

        data = self.__load_data(index)
        # if self.train_hybrid is True:
        #     data = self.__completing_tensor_rows(data)
        return data
    
    def __len__(self): 
        return self.total_images

    def get_stat(self):
        return self.ener_shift, self.atom_types
    
    def __concatenate_data(self):
        """
        Concatenates the data from multiple directories into a single dictionary.

        Returns:
            data (dict): A dictionary containing the concatenated data.
            total_images (int): The total number of images in the concatenated data.
            images_per_dir (list): A list containing the number of images in each directory.
            atoms_per_dir (list): A list containing the number of atoms in each directory.
        """
        data = {}
        images_per_dir = []
        atoms_per_dir = []
        all_has_virial = True # maybe some dirs do not have virials.npy, do not load virials.npy file
        for dir in self.dirs:
            vir_list = glob.glob(os.path.join(dir, "virials.npy"))
            if len(vir_list) == 0:
                all_has_virial = False
                break
        for dir in self.dirs:
            npy_files = [f for f in os.listdir(dir) if f.endswith(".npy")]
            file_data_dict = {}
            for npy_file in npy_files:
                if all_has_virial is False and "virials.npy" == os.path.basename(npy_file):
                    continue
                file_path = os.path.join(dir, npy_file)
                file_data = np.load(file_path)

                if npy_file == "forces.npy":
                    images = file_data.shape[0]
                    file_data = file_data.reshape(images, -1, 3)
                    images_per_dir.append(images)
                    atoms_per_dir.append(file_data.shape[1])
                elif npy_file == "lattice.npy":
                    file_data = file_data.reshape(-1, 3, 3)
                elif npy_file == "position.npy":
                    images = file_data.shape[0]
                    file_data = file_data.reshape(images, -1, 3)

                file_data_dict[npy_file] = file_data
            type_maps = type_map(file_data_dict["image_type.npy"][0], self.atom_types)
            file_data_dict["type_maps"] = np.array(type_maps).reshape(1, -1)

            for vars_file, file_data in file_data_dict.items():
                if vars_file == "image_type.npy":
                    continue
                elif vars_file == "atom_type.npy" and file_data.shape[1] < self.img_max_types:
                    file_data = np.pad(file_data, ((0, 0), (0, self.img_max_types - file_data.shape[1])))
                elif vars_file == "ei.npy" and file_data.shape[1] < self.img_max_atom_num:
                    file_data = np.pad(file_data, ((0, 0), (0, self.img_max_atom_num - file_data.shape[1])))
                elif vars_file == "forces.npy" and file_data.shape[1] < self.img_max_atom_num:
                    file_data = np.pad(file_data, ((0, 0), (0, self.img_max_atom_num - file_data.shape[1]), (0, 0)))
                elif vars_file == "position.npy":
                    file_data = np.pad(file_data, ((0, 0), (0, self.img_max_atom_num - file_data.shape[1]), (0, 0)))
                elif vars_file == "type_maps" and file_data.shape[1] < self.img_max_atom_num:
                    file_data = np.pad(file_data, ((0, 0), (0, self.img_max_atom_num - file_data.shape[1])), mode='constant', constant_values=-1)
                # elif vars_file == "ListNeighbor" and file_data.shape[1] < self.img_max_atom_num:
                #     file_data = np.pad(file_data, ((0, 0), (0, self.img_max_atom_num - file_data.shape[1]), (0, 0)))
                # elif vars_file == "ImageDR" and file_data.shape[1] < self.img_max_atom_num:
                #     file_data = np.pad(file_data, ((0, 0), (0, self.img_max_atom_num - file_data.shape[1]), (0, 0), (0, 0)))

                if vars_file not in data:
                    data[vars_file] = file_data
                else:
                    data[vars_file] = np.concatenate((data[vars_file], file_data), axis=0)
        total_images = data["energies.npy"].shape[0]    
        return data, total_images, images_per_dir, atoms_per_dir
    
def type_map(atom_types_image, atom_type):
    """
    Maps the atom types to their corresponding indices in the atom_type array.

    Args:
    atom_types_image (numpy.ndarray): Array of atom types to be mapped.
    atom_type (numpy.ndarray): Array of integers representing the atom type of each atom in the system.

    Returns:
    list: List of indices corresponding to the atom types in the atom_type array.

    Raises:
    AssertionError: If no atom types in atom_types_image are found in atom_type.

    Examples: CH4 molecule
    >>> atom_types_image = array([6, 1, 1, 1, 1])
    >>> atom_type = array([6, 1])
    >>> type_map(atom_types_image, atom_type)
    [0, 1, 1, 1, 1]
    """
    atom_type_map = []
    for elem in atom_types_image:
        if elem in atom_type:
            atom_type_map.append(np.where(atom_type == elem)[0][0])
    assert len(atom_type_map) != 0, "this atom type didn't found"
    return atom_type_map

def find_neighbore(AtomTypeMap, Position, Lattice, ImageAtomNum, Ei, 
                   img_max_types, Rc_type, Rm_type, m_neigh, Rc_M, train_egroup):
    """
    Call the Fortran subroutine that finds the neighbors for each atom in the system.

    Args:
        AtomTypeMap (numpy.ndarray): List of atom types to index.
        Position (numpy.ndarray): List of atomic positions.
        Lattice (numpy.ndarray): List of lattice vectors.
        ImageAtomNum (int): The number of atoms in the system.
        Ei (numpy.ndarray): List of atomic energies.
        img_max_types (int): The maximum number of atom types in the system.
        Rc_type (numpy.ndarray): List of cutoff radii for each atom type.
        Rm_type (numpy.ndarray): List of minimum cutoff radii for each atom type.
        m_neigh (int): The maximum number of neighbors for each atom.
        Rc_M (float): The maximum cutoff radius for the system.
        train_egroup (bool): Whether to train the energy group.

    Returns:
        tuple: A tuple containing list_neigh, ImageDR, and max_ri.
            - list_neigh (numpy.ndarray): The list of neighbors.
            - ImageDR (numpy.ndarray): The displacement vectors for each neighbor.
            - max_ri (float): The maximum value of Ri.
    """
    images = 1
    ntypes = img_max_types
    natoms = ImageAtomNum
    Rc_type = np.asfortranarray(Rc_type)
    type_maps = np.asfortranarray(AtomTypeMap[:natoms] + 1)
    lattice = np.asfortranarray(Lattice.reshape(images, 3, 3))
    position = np.asfortranarray(Position[:natoms].reshape(images, natoms, 3))

    neighconst.find_neighbore(images, lattice, position, ntypes, natoms, 
                                m_neigh, Rc_M, Rc_type, type_maps)
    _list_neigh = neighconst.list_neigh
    _dR_neigh = neighconst.dr_neigh
    list_neigh = np.transpose(_list_neigh.copy(), (3, 2, 1, 0)).reshape(images, natoms, ntypes*m_neigh)
    dR_neigh = np.transpose(_dR_neigh.copy(), (4, 3, 2, 1, 0)).reshape(images, natoms, ntypes*m_neigh, 3)
    if train_egroup:
        Ei = np.asfortranarray(Ei[:natoms].reshape(images, natoms))
        neighconst.calc_egroup(images, lattice, position, natoms, Rc_M, type_maps, Ei)
        _Egroup_weight = neighconst.fact
        _Divider = neighconst.divider
        _Egroup = neighconst.energy_group
        Egroup_weight = np.transpose(_Egroup_weight.copy(), (2, 1, 0)).squeeze(0)
        Divider = np.transpose(_Divider.copy(), (1, 0)).squeeze(0)
        Egroup = np.transpose(_Egroup.copy(), (1, 0)).squeeze(0)
    else:
        Egroup_weight = None
        Divider = None
        Egroup = None
    neighconst.dealloc()
    
    max_ri, Rij = compute_Ri(list_neigh, dR_neigh, Rc_type, Rm_type)
    ImageDR = np.concatenate((Rij, dR_neigh), axis=-1)
    return list_neigh.squeeze(0), ImageDR.squeeze(0), max_ri, Egroup_weight, Divider, Egroup

def compute_Ri(list_neigh, dR_neigh, Rc_type, Rm_type):
    """
    Compute the Ri values for a given list of neighbors and their displacement vectors.

    Args:
        list_neigh (list): List of neighbor indices.
        dR_neigh (list): List of displacement vectors for each neighbor.

    Returns:
        tuple: A tuple containing max_ri, and Rij.
            - max_ri (torch.Tensor): The maximum value of Ri.
            - Rij (numpy.ndarray): The squared root of the sum of the squared displacement vectors.
    """
    device = torch.device("cpu")
    image_dR = torch.tensor(dR_neigh, device=device, dtype=torch.float64)
    list_neigh = torch.tensor(list_neigh, device=device, dtype=torch.int64)

    mask = list_neigh > 0
    dR2 = torch.zeros_like(list_neigh, dtype=torch.float64)
    Rij = torch.zeros_like(list_neigh, dtype=torch.float64)
    dR2[mask] = torch.sum(image_dR[mask] * image_dR[mask], -1)
    Rij[mask] = torch.sqrt(dR2[mask])

    nr = torch.zeros_like(dR2)
    inr = torch.zeros_like(dR2)

    dR2_copy = dR2.unsqueeze(-1).repeat(1, 1, 1, 3)
    Ri_xyz = torch.zeros_like(dR2_copy)

    nr[mask] = dR2[mask] / Rij[mask]
    Ri_xyz[mask] = image_dR[mask] / dR2_copy[mask]
    inr[mask] = 1 / Rij[mask]

    uu = torch.zeros_like(nr)
    vv = torch.zeros_like(nr)
    res = torch.zeros_like(nr)

    # x < rcut_min vv = 1;
    mask_min = nr < Rm_type[0] # why just use the first element of Rm_type? so why? emmm..
    mask_1 = mask & mask_min
    vv[mask_1] = 1

    # rcut_min< x < rcut_max;
    mask_max = nr < Rc_type[0]
    mask_2 = ~mask_min & mask_max & mask
    # uu = (xx - rmin) / (rmax - rmin);
    uu[mask_2] = (nr[mask_2] - Rm_type[0]) / (Rc_type[0] - Rm_type[0])
    vv[mask_2] = (
        uu[mask_2]
        * uu[mask_2]
        * uu[mask_2]
        * (-6 * uu[mask_2] * uu[mask_2] + 15 * uu[mask_2] - 10)
        + 1
    )
    mask_3 = ~mask_max & mask
    vv[mask_3] = 0

    res[mask] = 1.0 / nr[mask] 
    Ri = torch.cat((res.unsqueeze(-1), Ri_xyz), dim=-1)

    vv_copy = vv.unsqueeze(-1).repeat(1, 1, 1, 4)
    Ri[mask] *= vv_copy[mask]
    max_ri = torch.max(Ri[:,:,:,0])
    Rij = Rij.unsqueeze(-1).numpy()
    return max_ri, Rij

'''
description: 

param {InputParam} config
param {*} stat_add
param {*} datasets_path
param {*} work_dir
param {*} chunk_size
return {*}
    energy_shift , this is for model created
    max atom numbers of the image, this is for multibatch training
    the image path
author: wuxingxing
'''
def get_stat(config:InputParam, stat_add=None, datasets_path=None, work_dir=None, chunk_size=10):
    train_data_path = config.file_paths.trainDataPath
    ntypes = len(config.atom_type)
    input_atom_type = config.atom_type   # input atom type order
    if stat_add is not None:
        # load from prescribed path
        print("input_atom_type and energy_shift are from model checkpoint")
        input_atom_type, energy_shift = stat_add
    else:
        energy_shift = None
    
    max_atom_nums = 0
    valid_chunk = False
    for dataset_path in datasets_path:
        atom_types_image = np.load(os.path.join(dataset_path, train_data_path, "image_type.npy"))
        max_atom_nums = max(max_atom_nums, atom_types_image.shape[1])
        if energy_shift is None:
            _atom_types = np.load(os.path.join(dataset_path, train_data_path, "atom_type.npy"))
            if _atom_types.shape[1] != ntypes:
                continue
            lattice = np.load(os.path.join(dataset_path, train_data_path, "lattice.npy"))
            img_per_mvmt = lattice.shape[0]
            if img_per_mvmt < chunk_size:
                continue
            valid_chunk = True
            # position = np.load(os.path.join(dataset_path, train_data_path, "position.npy"))
            _Ei = np.load(os.path.join(dataset_path, train_data_path, "ei.npy"))
            type_maps = np.array(type_map(atom_types_image[0], input_atom_type))
            types, type_incides, atom_types_nums = np.unique(type_maps, return_index=True, return_counts=True)
            atom_types_nums = atom_types_nums[np.argsort(type_incides)]
            energy_shift = calculate_energy_shift(chunk_size, _Ei, atom_types_nums)
            energy_shift = adjust_order_same_as_user_input(energy_shift, _atom_types[0].tolist(), input_atom_type)
            # set feature scaler
            
    if not valid_chunk and energy_shift is None:
        raise ValueError("Invalid 'chunk_size', the number of images (include all atom types) in the movement is too big, \nPlease set a smaller chunk_size (default: 10) or add more images in the movement")

    return energy_shift, max_atom_nums, os.path.join(dataset_path, train_data_path)

'''
description: 
param {list} energy_shift
param {list} atom_type_order: the atom order in the current images
param {list} atom_type_list: user input atom type order 
return {*}
author: wuxingxing
'''
def adjust_order_same_as_user_input(energy_shift:list, atom_type_order:list, atom_type_list:list):
    energy_shift_res = []
    for i, atom in enumerate(atom_type_list):
        energy_shift_res.append(energy_shift[atom_type_order.index(atom)])
    return energy_shift_res

def calculate_energy_shift(chunk_size, _Ei, atom_types_nums):
    Ei = _Ei[:chunk_size]
    res = []
    current_type = 0
    for atom_type_num in atom_types_nums:
        current_type_indices = current_type + atom_type_num
        avg_Ei = np.mean(Ei[:, current_type:current_type_indices])
        res.append(avg_Ei)
        current_type = current_type_indices
    return res

def gen_train_data(train_ratio, raw_data_path, datasets_path,
                   train_data_path, valid_data_path, 
                   data_shuffle=True, seed=2024, format="movement"):
    """
    Generate training data for MLFF model.

    Args:
        train_ratio (float): Ratio of training data to total data.
        raw_data_path (list): List of paths to raw data. MOVEMENT, OUTCAR, etc.
        datasets_path (str): Path to the directory containing the temp *.npy files.
        train_data_path (str): Path to the directory containing the training data.
        valid_data_path (str): Path to the directory containing the validation data.
        data_shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        seed (int, optional): Random seed for shuffling the data. Defaults to 2024.
        format (str, optional): Format of the raw data. Defaults to "movement".

    Returns:
        list: List of paths to the labels.
    """
    labels_path = []
    for data_path in raw_data_path:
        data_name = os.path.basename(data_path)
        labels_path.append(os.path.join(datasets_path, data_name))
        Save_Data(data_path, datasets_path, train_data_path, valid_data_path, 
                    train_ratio, data_shuffle, seed, format)
    return labels_path

def main():
    input_json = "/data/home/wuxingxing/codespace/PWMLFF_nep/pwmat_mlff_workdir/hfo2/nep_train/train.json"
    nep_param = InputParam(input_json, "train")
    #     print("Read Config successful")
    # import ipdb; ipdb.set_trace()
    # load_type, nep_param:InputParam, energy_shift, max_atom_nums
    energy_shift = [-6.052877426125001, -12.10575485225]
    max_atom_nums = 96
    dataset = MovementDataset("train", nep_param, energy_shift, max_atom_nums)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    for i, sample_batches in enumerate(dataloader):
        # import ipdb;ipdb.set_trace()

        print(sample_batches["Force"].shape)
        print(sample_batches["Virial"].shape)
        print(sample_batches["Ei"].shape)
        print(sample_batches["Etot"].shape)
        #print(sample_batches["Egroup"].shape)
        #print(sample_batches["Divider"].shape)
        #print(sample_batches["Egroup_weight"].shape)

        print(sample_batches["ListNeighbor"].shape)
        print(sample_batches["Position"].shape)
        # print(sample_batches["NeighborType"].shape)
        # print(sample_batches["Ri"].shape)
        # print(sample_batches["ImageDR"].shape)
        # print(sample_batches["Ri_d"].shape)
        print(sample_batches["ImageAtomNum"].shape)


if __name__ == "__main__":
    main()
