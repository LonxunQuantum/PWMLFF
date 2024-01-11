import numpy as np
import os
from torch.utils.data import Dataset
import torch
import yaml
from NeighConst import neighconst

class MovementDataset(Dataset):
    def __init__(self, data_paths, config):
        super(MovementDataset, self).__init__()

        self.dirs = data_paths  # include all movement data path
        self.normalized_data_path = os.path.dirname(os.path.dirname(self.dirs[0]))
        self.m_neigh = config['maxNeighborNum']
        self.img_max_types = len(config["atomType"])
        self.atom_map = np.array([(_['type']) for _ in config["atomType"]])   # input atom type order
        self.Rc_M = config['Rc_M']
        self.Rc_type = np.array([(_['Rc']) for _ in config["atomType"]])
        self.Rm_type = np.array([(_['Rm']) for _ in config["atomType"]])
        self.Egroup = config['train_egroup']
        self.img_max_atom_num = np.load(os.path.join(self.normalized_data_path, "max_atom_nums.npy")).tolist()
        self.davg = np.load(os.path.join(self.normalized_data_path, "davg.npy"))
        self.dstd = np.load(os.path.join(self.normalized_data_path, "dstd.npy"))
        self.ener_shift = np.load(os.path.join(self.normalized_data_path, "energy_shift.npy"))
        if self.ener_shift.size == 1:
            self.ener_shift = [self.ener_shift.tolist()]
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
        data["AtomTypeMap"] = self.all_movement_data["type_maps.npy"][type_index]
        data["ImageAtomNum"] = self.atoms_per_dir[type_index]

        list_neigh, dR_neigh, max_ri, Egroup_weight, Divider, Egroup = self.find_neighbore(data["AtomTypeMap"], data["Position"], data["Lattice"], data["ImageAtomNum"], data["Ei"])
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
        return self.davg, self.dstd, self.ener_shift, self.atom_map
    
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
        for dir in self.dirs:
            npy_files = [f for f in os.listdir(dir) if f.endswith(".npy")]
            file_data_dict = {}
            for npy_file in npy_files:
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
            # list_neigh, dR_neigh, max_ri, natoms = self.find_neighbore(file_data_dict)
            # file_data_dict["ListNeighbor"] = list_neigh
            # file_data_dict["ImageDR"] = dR_neigh

            for vars_file, file_data in file_data_dict.items():
                if vars_file == "atom_type.npy" and file_data.shape[1] < self.img_max_types:
                    file_data = np.pad(file_data, ((0, 0), (0, self.img_max_types - file_data.shape[1])))
                elif vars_file == "ei.npy" and file_data.shape[1] < self.img_max_atom_num:
                    file_data = np.pad(file_data, ((0, 0), (0, self.img_max_atom_num - file_data.shape[1])))
                elif vars_file == "forces.npy" and file_data.shape[1] < self.img_max_atom_num:
                    file_data = np.pad(file_data, ((0, 0), (0, self.img_max_atom_num - file_data.shape[1]), (0, 0)))
                elif vars_file == "type_maps.npy" and file_data.shape[1] < self.img_max_atom_num:
                    file_data = np.pad(file_data, ((0, 0), (0, self.img_max_atom_num - file_data.shape[1])), mode='constant', constant_values=-1)
                elif vars_file == "position.npy":
                    file_data = np.pad(file_data, ((0, 0), (0, self.img_max_atom_num - file_data.shape[1]), (0, 0)))
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
    
    def find_neighbore(self, AtomTypeMap, Position, Lattice, ImageAtomNum, Ei):
        """
        Call the Fortran subroutine that finds the neighbors for each atom in the system.

        Args:
            AtomTypeMap (numpy.ndarray): List of atom types to index.
            Position (numpy.ndarray): List of atomic positions.
            Lattice (numpy.ndarray): List of lattice vectors.
            ImageAtomNum (int): The number of atoms in the system.
            Ei (numpy.ndarray): List of atomic energies.

        Returns:
            tuple: A tuple containing list_neigh, ImageDR, and max_ri.
                - list_neigh (numpy.ndarray): The list of neighbors.
                - ImageDR (numpy.ndarray): The displacement vectors for each neighbor.
                - max_ri (float): The maximum value of Ri.
        """
        images = 1
        ntypes = self.img_max_types
        natoms = ImageAtomNum
        Rc_type = np.asfortranarray(self.Rc_type)
        type_maps = np.asfortranarray(AtomTypeMap[:natoms] + 1)
        lattice = np.asfortranarray(Lattice.reshape(images, 3, 3))
        position = np.asfortranarray(Position[:natoms].reshape(images, natoms, 3))

        neighconst.find_neighbore(images, lattice, position, ntypes, natoms, 
                                    self.m_neigh, self.Rc_M, Rc_type, type_maps)
        _list_neigh = neighconst.list_neigh
        _dR_neigh = neighconst.dr_neigh
        list_neigh = np.transpose(_list_neigh.copy(), (3, 2, 1, 0)).reshape(images, natoms, ntypes*self.m_neigh)
        dR_neigh = np.transpose(_dR_neigh.copy(), (4, 3, 2, 1, 0)).reshape(images, natoms, ntypes*self.m_neigh, 3)
        if self.Egroup:
            Ei = np.asfortranarray(Ei[:natoms].reshape(images, natoms))
            neighconst.calc_egroup(images, lattice, position, natoms, self.Rc_M, type_maps, Ei)
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
        
        max_ri, Rij = self.compute_Ri(list_neigh, dR_neigh)
        ImageDR = np.concatenate((Rij, dR_neigh), axis=-1)
        return list_neigh.squeeze(0), ImageDR.squeeze(0), max_ri, Egroup_weight, Divider, Egroup

    def compute_Ri(self, list_neigh, dR_neigh):
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
        mask_min = nr < self.Rm_type[0] # why just use the first element of Rm_type?
        mask_1 = mask & mask_min
        vv[mask_1] = 1

        # rcut_min< x < rcut_max;
        mask_max = nr < self.Rc_type[0]
        mask_2 = ~mask_min & mask_max & mask
        # uu = (xx - rmin) / (rmax - rmin);
        uu[mask_2] = (nr[mask_2] - self.Rm_type[0]) / (self.Rc_type[0] - self.Rm_type[0])
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
    In mixed training, during the automatic loading process of DataLoader objects, \
    it is necessary for each data object to have the same number of rows, otherwise the following error will occur.
        return torch.stack(batch, 0, out=out)
        RuntimeError: stack expects each tensor to be equal size, but got [64, 3] at entry 0 and [76, 3] at entry 1
    
    This function is used to fill in the Tensor row count.
    param {*} self
    param {*} data
    return {*}
    author: wuxingxing
    '''    
    def __completing_tensor_rows(self, data):
        dest = {}
        if data["ImageAtomNum"][0] == self.img_max_atom_num:
            return data
        else:
            data["Force"] = __complet_tensor(data["Force"], self.img_max_atom_num)
        np.resize()
    
        def __complet_tensor(souce, img_max_atom_num):
            
            pass
        # data["Force"] = -1 * np.load(os.path.join(path, "Force.npy"))

        # if os.path.exists(os.path.join(path, "Virial.npy")):
        #     data["Virial"] = np.load(os.path.join(path, "Virial.npy"))

        # data["Ei"] = np.load(os.path.join(path, "Ei.npy"))
        # data["Etot"] = np.load(os.path.join(path, "Etot.npy"))

        # if os.path.exists(os.path.join(path, "Egroup.npy")):
        #     data["Egroup"] = np.load(os.path.join(path, "Egroup.npy"))
        #     data["Divider"] = np.load(os.path.join(path, "Divider.npy"))
        #     data["Egroup_weight"] = np.load(os.path.join(path, "Egroup_weight.npy"))

        # data["ListNeighbor"] = np.load(os.path.join(path, "ListNeighbor.npy"))
        # data["ImageDR"] = np.load(os.path.join(path, "ImageDR.npy"))
        # data["Ri"] = np.load(os.path.join(path, "Ri.npy"))
        # data["Ri_d"] = np.load(os.path.join(path, "Ri_d.npy"))
        # data["ImageAtomNum"] = np.load(os.path.join(path, "ImageAtomNum.npy")).reshape(-1)
        # atom_type_list = list(np.load(os.path.join(path, "AtomType.npy")))
        # data["AtomType"] = np.array(sorted(set(atom_type_list), key=atom_type_list.index))

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
