import numpy as np
import os
import glob
from torch.utils.data import Dataset
import torch
import yaml
from NeighConst import neighconst
from numpy.ctypeslib import ndpointer
import ctypes
lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib = ctypes.CDLL(os.path.join(lib_path, 'feature/chebyshev/build/lib/libneighborList.so')) # multi-neigh-list
lib.CreateNeighbor.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                               ndpointer(ctypes.c_int, ndim=1, flags="C_CONTIGUOUS"), 
                               ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"), 
                               ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")]
                                  
lib.CreateNeighbor.restype = ctypes.c_void_p

# lib.ShowNeighbor.argtypes = [ctypes.c_void_p]
lib.DestroyNeighbor.argtypes = [ctypes.c_void_p]

lib.GetNumNeighAll.argtypes = [ctypes.c_void_p]
lib.GetNumNeighAll.restype = ctypes.POINTER(ctypes.c_int)
lib.GetNeighborsListAll.argtypes = [ctypes.c_void_p]
lib.GetNeighborsListAll.restype = ctypes.POINTER(ctypes.c_int)
lib.GetDRNeighAll.argtypes = [ctypes.c_void_p]
lib.GetDRNeighAll.restype = ctypes.POINTER(ctypes.c_double)


class MovementDataset(Dataset):
    def __init__(self, load_type, input_param, max_atom_nums):
        super(MovementDataset, self).__init__()
        if load_type == "train":
            data_paths = [os.path.join(_, input_param.file_paths.trainDataPath) for _ in input_param.file_paths.datasets_path]
                         
        elif load_type == "valid":
            data_paths = [os.path.join(_, input_param.file_paths.validDataPath) 
                          for _ in input_param.file_paths.datasets_path
                          if os.path.exists(os.path.join(_, input_param.file_paths.validDataPath))]

        self.dirs = data_paths  # include all movement data path
        self.atom_types = np.array(input_param.atom_type)   # input atom type order
        self.m_neigh = input_param.max_neigh_num
        self.img_max_types = len(self.atom_types)
        self.Rc_M = input_param.descriptor.Rmax
        self.rcut_smooth = input_param.descriptor.Rmin
        self.Rm_type = np.array([input_param.descriptor.Rmin for _ in self.atom_types])
        self.Egroup = input_param.optimizer_param.train_egroup
        self.img_max_atom_num = max_atom_nums
        # self.ener_shift = np.array(energy_shift)
        self.all_movement_data, self.total_images, self.images_per_dir, self.atoms_per_dir = self.__concatenate_data()
            
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
        data["ImageAtomNum"] = self.atoms_per_dir[type_index]

        num_neigh, list_neigh, dR_neigh, Egroup_weight, Divider, Egroup, max_ri = find_neighbore(data["AtomTypeMap"], data["Position"], data["Lattice"], data["ImageAtomNum"], data["Ei"], 
                                                                                      self.img_max_types, self.m_neigh, self.Rc_M, self.rcut_smooth, self.Egroup)
        if list_neigh.shape[0] < self.img_max_atom_num:
            num_neigh = np.pad(num_neigh, ((0, self.img_max_atom_num - num_neigh.shape[0]), (0, 0)))
            list_neigh = np.pad(list_neigh, ((0, self.img_max_atom_num - list_neigh.shape[0]), (0, 0), (0, 0)))
            dR_neigh = np.pad(dR_neigh, ((0, self.img_max_atom_num - dR_neigh.shape[0]), (0, 0), (0, 0), (0, 0)))
        data["NumNeighbor"] = num_neigh
        data["ListNeighbor"] = list_neigh
        data["ImageDR"] = dR_neigh
        data["max_ri"] = max_ri
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

        
    def __getitem__(self, index):

        data = self.__load_data(index)
        # if self.train_hybrid is True:
        #     data = self.__completing_tensor_rows(data)
        return data
    
    def __len__(self): 
        return self.total_images

    def get_stat(self):
        return self.atom_types
    
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

def find_neighbore(AtomTypeMap, Position, Lattice, natoms, Ei, 
                   ntypes, m_neigh, Rc_M, rcut_smooth, train_egroup):
    """
    Call the Fortran subroutine that finds the neighbors for each atom in the system.

    Args:
        AtomTypeMap (numpy.ndarray): List of atom types to index.
        Position (numpy.ndarray): List of atomic positions.
        Lattice (numpy.ndarray): List of lattice vectors.
        natoms (int): The number of atoms in the system.
        Ei (numpy.ndarray): List of atomic energies.
        ntypes (int): The maximum number of atom types in the system.
        m_neigh (int): The maximum number of neighbors for each atom.
        Rc_M (float): The maximum cutoff radius for the system.
        rcut_smooth (float): minimum cutoff radii for the system
        train_egroup (bool): Whether to train the energy group.

    Returns:
        num_neigh_all (numpy.ndarray): The number of neighbors for each atom in the system.
        list_neigh_all (numpy.ndarray): The list of neighbors for each atom in the system.
        dr_neigh_all (numpy.ndarray): The distance vector between each atom and its neighbors.
        max_ri (float): The maximum distance between an atom and its neighbors.
    """
    images = 1
    type_maps = AtomTypeMap[:natoms].astype(np.int32)

    # Create neighbor list
    mnl = lib.CreateNeighbor(images, Rc_M, m_neigh, ntypes, natoms, type_maps, Position.flatten(), Lattice.flatten())
    # lib.ShowNeighbor(mnl)
    num_neigh_all = lib.GetNumNeighAll(mnl)
    list_neigh_all = lib.GetNeighborsListAll(mnl)
    dr_neigh_all = lib.GetDRNeighAll(mnl)
    num_neigh_all = np.ctypeslib.as_array(num_neigh_all, (images, natoms, ntypes)).squeeze(0)
    list_neigh_all = np.ctypeslib.as_array(list_neigh_all, (images, natoms, ntypes, m_neigh)).squeeze(0)
    dr_neigh_all = np.ctypeslib.as_array(dr_neigh_all, (images, natoms, ntypes, m_neigh, 4)).squeeze(0)   # rij, delx, dely, delz
    max_ri = np.max(dr_neigh_all[:, :, :, 0])

    if train_egroup:
        type_maps = np.asfortranarray(AtomTypeMap[:natoms] + 1)
        lattice = np.asfortranarray(Lattice.reshape(images, 3, 3))
        position = np.asfortranarray(Position[:natoms].reshape(images, natoms, 3))
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

    # Delete neighbor list
    num_neigh = num_neigh_all.copy()
    list_neigh = list_neigh_all.copy()
    dr_neigh = dr_neigh_all.copy()
    lib.DestroyNeighbor(mnl)
    return num_neigh, list_neigh, dr_neigh, Egroup_weight, Divider, Egroup, max_ri


def main():

    with open("config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    #     print("Read Config successful")
    # import ipdb; ipdb.set_trace()
    dataset = MovementDataset("./train")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
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
        # print(sample_batches["NeighborType"].shape)
        # print(sample_batches["Ri"].shape)
        print(sample_batches["ImageDR"].shape)
        # print(sample_batches["Ri_d"].shape)
        print(sample_batches["ImageAtomNum"].shape)


if __name__ == "__main__":
    main()
