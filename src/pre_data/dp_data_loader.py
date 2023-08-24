import numpy as np
import os
from torch.utils.data import Dataset
import torch
import yaml

class MovementDataset(Dataset):
    '''
    description: 
    param {*} self
    param {*} data_path
    param {*} train_hybrid False is for single system training, True is for multi different systems training.
    return {*}
    author: wuxingxing
    '''    
    def __init__(self, data_paths):
        super(MovementDataset, self).__init__()

        self.dirs = []

        for data_path in data_paths:
            for current_dir, child_dir, child_file in os.walk(data_path):
                if len(child_dir) == 0 and "Ri.npy" in child_file:
                    self.dirs.append(current_dir)

        self.dirs = sorted(self.dirs, key=lambda x: int(x.split('_')[-1]))
        self.img_max_atom_num, self.img_max_types, file_index = self.__set_max_atoms()
        # self.__compute_stat_output(10, 1e-3)
        if file_index is not None:
            data_path = os.path.dirname(file_index)
            self.davg = np.load(os.path.join(data_path, "davg.npy"))
            self.dstd = np.load(os.path.join(data_path, "dstd.npy"))
            self.ener_shift = np.loadtxt(os.path.join(data_path, "energy_shift.raw"))
            if self.ener_shift.size == 1:
                self.ener_shift = [self.ener_shift.tolist()]
            self.atom_map = np.loadtxt(os.path.join(data_path, "atom_map.raw"), dtype=int)
        else:
            self.davg, self.dstd, self.ener_shift, self.atom_map = None, None, None, None
            
    def __load_data(self, path):

        data = {}

        data["Force"] = -1 * np.load(os.path.join(path, "Force.npy"))

        if os.path.exists(os.path.join(path, "Virial.npy")):
            data["Virial"] = np.load(os.path.join(path, "Virial.npy"))

        data["Ei"] = np.load(os.path.join(path, "Ei.npy"))
        data["Etot"] = np.load(os.path.join(path, "Etot.npy"))

        if os.path.exists(os.path.join(path, "Egroup.npy")):
            data["Egroup"] = np.load(os.path.join(path, "Egroup.npy"))
            data["Divider"] = np.load(os.path.join(path, "Divider.npy"))
            data["Egroup_weight"] = np.load(os.path.join(path, "Egroup_weight.npy"))

        data["ListNeighbor"] = np.load(os.path.join(path, "ListNeighbor.npy"))
        data["ImageDR"] = np.load(os.path.join(path, "ImageDR.npy"))
        data["Ri"] = np.load(os.path.join(path, "Ri.npy"))
        data["Ri_d"] = np.load(os.path.join(path, "Ri_d.npy"))
        data["ImageAtomNum"] = np.load(os.path.join(path, "ImageAtomNum.npy")).reshape(-1)
        atom_type_list = list(np.load(os.path.join(path, "AtomType.npy")))
        data["AtomType"] = np.array(sorted(set(atom_type_list), key=atom_type_list.index))
        
        # this block is used for hybrid training
        if data["ImageAtomNum"][0] < self.img_max_atom_num:
            # pad_num = self.img_max_atom_num - data["Force"].shape[0]
            data["Force"].resize((self.img_max_atom_num, data["Force"].shape[1]), refcheck=False)
            data["Ei"].resize(self.img_max_atom_num, refcheck=False)
            if "Egroup" in data.keys():
                # doing Egroup things 
                data["Egroup"].resize(self.img_max_atom_num, refcheck=False)
                data["Divider"].resize(self.img_max_atom_num, refcheck=False)
                data["Egroup_weight"].resize((self.img_max_atom_num, self.img_max_atom_num), refcheck=False)
            
            data["ListNeighbor"].resize((self.img_max_atom_num, data["ListNeighbor"].shape[1]), refcheck=False)
            data["ImageDR"].resize((self.img_max_atom_num, data["ImageDR"].shape[1], data["ImageDR"].shape[2]), refcheck=False)
            data["Ri"].resize((self.img_max_atom_num, data["Ri"].shape[1], data["Ri"].shape[2]), refcheck=False)
            data["Ri_d"].resize((self.img_max_atom_num, data["Ri_d"].shape[1], data["Ri_d"].shape[2], data["Ri_d"].shape[3]), refcheck=False)
        if len(data["AtomType"]) < self.img_max_types:
            data["AtomType"] = np.append(data["AtomType"], [0 for _ in range(0,self.img_max_types-len(data["AtomType"]))])
        
        return data
        
    def __getitem__(self, index):

        file_path = self.dirs[index]
        data = self.__load_data(file_path)
        # if self.train_hybrid is True:
        #     data = self.__completing_tensor_rows(data)
        return data
    
    def __len__(self): 
        return len(self.dirs)

    def get_stat(self):
        return self.davg, self.dstd, self.ener_shift, self.atom_map
    
    
    '''
    description: 
        get the max atom num of images in training set
        file_index is the image index which has max atom num
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def __set_max_atoms(self):
        ImageAtomNum = []
        AtomType = []
        AtomType_size_list = []
        for path in self.dirs:
            ImageAtomNum.append(np.load(os.path.join(path, "ImageAtomNum.npy")).reshape(-1)[0])
            atom_type_list = list(np.load(os.path.join(path, "AtomType.npy")))
            AtomType.append(np.array(sorted(set(atom_type_list), key=atom_type_list.index)))
            AtomType_size_list.append(len(AtomType[-1]))
        
        if len(self.dirs) == 0:
            return 0, 0, None
        else:
            file_index = self.dirs[AtomType_size_list.index(max(AtomType_size_list))]
            return max(ImageAtomNum), max(AtomType_size_list), file_index
    
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
        print(sample_batches["Ri"].shape)
        print(sample_batches["ImageDR"].shape)
        print(sample_batches["Ri_d"].shape)
        print(sample_batches["ImageAtomNum"].shape)


if __name__ == "__main__":
    main()
