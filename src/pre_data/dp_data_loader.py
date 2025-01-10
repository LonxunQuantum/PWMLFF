import numpy as np
import os
import glob
from torch.utils.data import Dataset
import torch
import yaml
from NeighConst import neighconst

class MovementDataset(Dataset):
    def __init__(self, data_paths, config, davg, dstd, energy_shift, max_atom_nums):
        super(MovementDataset, self).__init__()

        self.dirs = data_paths  # include all movement data path
        self.m_neigh = config['maxNeighborNum']
        self.img_max_types = len(config["atomType"])
        self.atom_types = np.array([(_['type']) for _ in config["atomType"]])   # input atom type order
        self.Rc_M = config['Rc_M']
        self.Rc_type = np.array([(_['Rc']) for _ in config["atomType"]])
        self.Rm_type = np.array([(_['Rm']) for _ in config["atomType"]])
        self.Egroup = config['train_egroup']
        self.img_max_atom_num = max_atom_nums
        self.davg = np.array(davg)
        self.dstd = np.array(dstd)
        self.ener_shift = np.array(energy_shift)
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

        list_neigh, dR_neigh, max_ri, Egroup_weight, Divider, Egroup = find_neighbore(data["AtomTypeMap"], data["Position"], data["Lattice"], data["ImageAtomNum"], data["Ei"], 
                                                                                      self.img_max_types, self.Rc_type, self.Rm_type, self.m_neigh, self.Rc_M, self.Egroup)
        if list_neigh.shape[0] < self.img_max_atom_num:
            list_neigh = np.pad(list_neigh, ((0, self.img_max_atom_num - list_neigh.shape[0]), (0, 0)))
            dR_neigh = np.pad(dR_neigh, ((0, self.img_max_atom_num - dR_neigh.shape[0]), (0, 0), (0, 0)))
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
        return self.davg, self.dstd, self.ener_shift, self.atom_types
    
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
        # for dir in self.dirs:
            # vir_list = glob.glob(os.path.join(dir, "virials.npy"))
            # if len(vir_list) == 0:
            #     all_has_virial = False       
            #     break
        for dir in self.dirs:
            npy_files = [f for f in os.listdir(dir) if f.endswith(".npy")]
            has_virial = False
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
                elif npy_file == "virials.npy":
                    ones_column = np.ones((file_data.shape[0], 1))
                    file_data = np.hstack((file_data, ones_column))
                    has_virial = True
                file_data_dict[npy_file] = file_data
            if has_virial is False:
                ones_column = np.zeros((file_data.shape[0], 1))
                none_virial = -1e6 * np.ones((images, 9))
                file_data_dict["virials.npy"] = np.hstack((none_virial, ones_column)) # cur-dir has no virial, use 0 to occupy space to 10

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
        if len(data.keys()) > 0:
            total_images = data["energies.npy"].shape[0]
        else:
            total_images = 0
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
    mask_min = nr < Rm_type[0] # why just use the first element of Rm_type?
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
