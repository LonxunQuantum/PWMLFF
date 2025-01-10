import numpy as np
import os
import glob
from torch.utils.data import Dataset
import torch
import yaml
from src.user.input_param import InputParam
from pwdata import Save_Data
from pwdata import Config
from src.feature.nep_find_neigh.findneigh import FindNeigh

class NepTestData():
    def __init__(self, input_param:InputParam):
        self.image_list = []
        self.input_param = input_param
        data_paths = []
        for data_path in self.input_param.file_paths.datasets_path:
            if os.path.exists(os.path.join(os.path.join(data_path, "train", "position.npy"))):
                data_paths.append(os.path.join(os.path.join(data_path, "train"))) #train dir
            if os.path.exists(os.path.join(os.path.join(data_path, "valid", "position.npy"))):
                data_paths.append(os.path.join(os.path.join(data_path, "valid"))) #valid dir
            if os.path.exists(os.path.join(data_path, "position.npy")) > 0: # add train or valid data
                data_paths.append(data_path)

        if len(data_paths) > 0:
            for config in data_paths:
                image_read = Config(data_path=config, format="pwmlff/npy").images
                if isinstance(image_read, list):
                    self.image_list.extend(image_read)
                else:
                    self.image_list.append(image_read)
        for image in self.image_list:
            # if image.cartesian is True:
            #     image._set_fractional()
            if image.cartesian is False:
                image._set_cartesian()

        # return image_list
        
# from numpy.ctypeslib import ndpointer
# import ctypes
# lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# libfindneigh = ctypes.CDLL(os.path.join(lib_path, 'feature/nep/build/lib/libfind_neigh.so')) # multi-descriptor
# libfindneigh.CreateFindNeigh.argtypes = [ctypes.c_double, 
#                                          ctypes.c_double, 
#                                          ctypes.c_int, 
#                                          ctypes.c_int]
# libfindneigh.CreateFindNeigh.restype = ctypes.POINTER(ctypes.c_void_p)

# libfindneigh.DestroyFindNeigh.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
# libfindneigh.DestroyFindNeigh.restype = None

# libfindneigh.get_neighs.argtypes = [ctypes.POINTER(ctypes.c_void_p), 
#                                     ctypes.c_int, 
#                                     ctypes.POINTER(ctypes.c_int), 
#                                     ctypes.POINTER(ctypes.c_double), 
#                                     ctypes.POINTER(ctypes.c_double)]
# libfindneigh.get_neighs.restype = ctypes.POINTER(ctypes.c_void_p)

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
        # self.cal_neigh = libfindneigh.CreateFindNeigh(input_param.descriptor.cutoff[0],\
        #                                               input_param.descriptor.cutoff[1],\
        #                                             self.img_max_atom_num, \
        #                                             self.m_neigh*self.img_max_types)

        # test

        # data = self.__load_data(0)
        # print()

    def __load_data(self, index):
        type_index = np.searchsorted(np.cumsum(self.images_per_dir), index + 1)
        data = {}
        data["Force"] = self.all_movement_data["forces.npy"][index]
        data["Ei"] = self.all_movement_data["ei.npy"][index]
        data["Etot"] = self.all_movement_data["energies.npy"][index]
        data["Position"] = self.all_movement_data["position.npy"][index]
        data["Lattice"] = self.all_movement_data["lattice.npy"][index]
        data["AtomType"] = self.all_movement_data["atom_type.npy"][type_index]
        data["AtomTypeMap"] = self.all_movement_data["type_maps"][type_index]
        atom_nums = self.atoms_per_dir[type_index]
        data["ImageAtomNum"] = self.atoms_per_dir[type_index]

        # #radial and angular is set from nep.txt file 
        list_neigh_radial, dR_neigh_radial, list_neigh_type_radial, \
            list_neigh_angular, dR_neigh_angular, list_neigh_type_angular, Egroup_weight, Divider, Egroup  = self.find_neigh_nep(
                            list(data["AtomTypeMap"]),
                            list(data["Lattice"].transpose(1, 0).reshape(-1)),
                            list(data["Position"].transpose(1, 0).reshape(-1)))

        # for batch > 1, the demision of each type data should be the same when use the dataloader
        if list_neigh_radial.shape[0] < self.img_max_atom_num:
            list_neigh_radial = np.pad(list_neigh_radial, ((0, self.img_max_atom_num - list_neigh_radial.shape[0]), (0, 0)))
            dR_neigh_radial = np.pad(dR_neigh_radial, ((0, self.img_max_atom_num - dR_neigh_radial.shape[0]), (0, 0), (0, 0)))
            list_neigh_type_radial = np.pad(list_neigh_type_radial, ((0, self.img_max_atom_num - list_neigh_type_radial.shape[0]), (0, 0)))
            
            list_neigh_angular = np.pad(list_neigh_angular, ((0, self.img_max_atom_num - list_neigh_angular.shape[0]), (0, 0)))
            dR_neigh_angular = np.pad(dR_neigh_angular, ((0, self.img_max_atom_num - dR_neigh_angular.shape[0]), (0, 0), (0, 0)))
            list_neigh_type_angular = np.pad(list_neigh_type_angular, ((0, self.img_max_atom_num - list_neigh_type_angular.shape[0]), (0, 0)))

        data["ListNeighbor"] = list_neigh_radial
        data["ListNeighborType"]=list_neigh_type_radial
        data["ImageDR"] = dR_neigh_radial
        data["ListNeighborAngular"] = list_neigh_angular
        data["ListNeighborTypeAngular"]=list_neigh_type_angular
        data["ImageDRAngular"] = dR_neigh_angular
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

    '''
    description: 
        AtomTypeMap: len is same as max_image_nums, is less, full with -1
    param {*} self
    param {*} AtomTypeMap
    param {*} Lattice
    param {*} Position
    return {*}
    author: wuxingxing
    '''    
    def find_neigh_nep(self, AtomTypeMap, Lattice, Position):
        Egroup_weight, Divider, Egroup = None, None, None
        # 34622.19498329725 d12_radial
        atom_nums = len(AtomTypeMap)

        d12_radial, d12_agular, NL_radial, NL_angular, NLT_radial, NLT_angular = self.calc.getNeigh(
                           self.input_param.descriptor.cutoff[0],self.input_param.descriptor.cutoff[1], 
                            len(self.input_param.atom_type)*self.input_param.max_neigh_num, AtomTypeMap, Lattice, Position)
        
        neigh_radial_rij   = np.array(d12_radial).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num, 4)
        neigh_radial_list  = np.array(NL_radial).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num)
        neigh_radial_type_list  =  np.array(NLT_radial).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num)
        neigh_angular_rij  = np.array(d12_agular).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num, 4)
        neigh_angular_list = np.array(NL_angular).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num)
        neigh_angular_type_list = np.array(NLT_angular).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num)            
        # np.sum(neigh_radial_rij)
        # 16070.331055526085
        # print("tall {} t2 {} t3 {} t4 {} t5 {} t6 {}".format(t4-t2, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5))
        
        #neigh_radial_list[0,:10]
        # array([ 2,  2,  3,  3,  5,  5,  9, 11, 13, 14])
        # neigh_radial_rij.shape
        # (96, 200, 4)
        # neigh_radial_rij[0,:2,:]
        # array([[ 5.20969973, -0.86372257,  0.        ,  5.13760202],
        #     [ 5.20970035,  0.86372267,  0.        , -5.13760264]])
        return neigh_radial_list, neigh_radial_rij, neigh_radial_type_list, neigh_angular_list, neigh_angular_rij, neigh_angular_type_list, Egroup_weight, Divider, Egroup

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
        # all_has_virial = True # maybe some dirs do not have virials.npy, do not load virials.npy file
        # for dir in self.dirs:
        #     vir_list = glob.glob(os.path.join(dir, "virials.npy"))
        #     if len(vir_list) == 0:
        #         all_has_virial = False
        #         break
        for dir in self.dirs:
            npy_files = [f for f in os.listdir(dir) if f.endswith(".npy")]
            has_virial = False
            file_data_dict = {}
            for npy_file in npy_files:
                # if all_has_virial is False and "virials.npy" == os.path.basename(npy_file):
                #     continue
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
                elif npy_file == "virials.npy":
                    ones_column = np.ones((file_data.shape[0], 1))
                    file_data = np.hstack((file_data, ones_column))
                    has_virial = True
                file_data_dict[npy_file] = file_data
            if has_virial is False:
                file_data_dict["virials.npy"] = np.zeros((images, 10)) # cur-dir has no virial, use 0 to occupy space to 10
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
        if len(data.keys()) == 0:
            total_images = 0
        else:
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
    if isinstance(atom_types_image.tolist(), int):
        atom_types_image = [atom_types_image.tolist()]
    for elem in atom_types_image:
        if elem in atom_type:
            atom_type_map.append(np.where(atom_type == elem)[0][0])
    assert len(atom_type_map) != 0, "this atom type didn't found"
    return atom_type_map

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
    input_atom_type = config.atom_type   # input atom type order
    energy_dict = {}
    for  _atom in input_atom_type:
        energy_dict[_atom] = []
    energy_dict['E'] = []

    if stat_add is not None:
        # load from prescribed path
        print("input_atom_type and energy_shift are from model checkpoint")
        input_atom_type, energy_shift = stat_add
    else:
        energy_shift = None
    
    max_atom_nums = 0
    for dataset_path in datasets_path:
        if os.path.exists(os.path.join(dataset_path, train_data_path, "image_type.npy")):
            data_load_path = os.path.join(dataset_path, train_data_path)
        elif os.path.exists(os.path.join(dataset_path, "image_type.npy")):
            data_load_path = dataset_path
        atom_types_image = np.load(os.path.join(data_load_path, "image_type.npy"))
        max_atom_nums = max(max_atom_nums, atom_types_image.shape[1])

        if energy_shift is None:
            cout_type, cout_num = np.unique(atom_types_image, return_counts=True)
            atom_types_image_dict = dict(zip(cout_type, cout_num))
            for element in input_atom_type:
                if element in atom_types_image_dict.keys():
                    energy_dict[element].append(atom_types_image_dict[element])
                else:
                    energy_dict[element].append(0)
            _energy = np.load(os.path.join(data_load_path, "energies.npy"))
            energy_dict['E'].append(np.mean(_energy[:min(_energy.shape[0], 10), :]))
            
    if energy_shift is None:
        _num_matrix = []
        for key in energy_dict.keys():
            if key != 'E':
                _num_matrix.append(energy_dict[key])
        x, residuals, rank, s = np.linalg.lstsq(np.array(_num_matrix).T, energy_dict['E'], rcond=None)
        energy_shift = x.tolist()
    return energy_shift, max_atom_nums, os.path.join(data_load_path)

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

'''
description: 
 nouse
param {*} chunk_size
param {*} _Ei
param {*} atom_types_nums
return {*}
author: wuxingxing
'''
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
                   data_shuffle=True, seed=2024, format="pwmat/movement", atom_names=None):
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
    for idx, data_path in enumerate(raw_data_path):
        data_name = "{}_{}".format(os.path.basename(data_path), idx)
        # labels_path.append(os.path.join(datasets_path, data_name))
        save_dir = os.path.join(datasets_path, data_name)
        labels_path.append(os.path.join(save_dir, os.path.basename(data_path)))
        Save_Data(data_path, save_dir, train_data_path, valid_data_path, 
                    train_ratio, data_shuffle, seed, format, atom_names = atom_names)
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
