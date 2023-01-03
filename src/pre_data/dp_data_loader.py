import numpy as np
import os
from torch.utils.data import Dataset
import torch
import yaml


class MovementDataset(Dataset):
    def __init__(self, config, data_path):
        super(MovementDataset, self).__init__()
        self.config = config
        self.davg = np.load(os.path.join(data_path, "davg.npy"))
        self.dstd = np.load(os.path.join(data_path, "dstd.npy"))

        self.dirs = []
        self.range_map = {}
        self.cache_set_num = 10
        self.work_space = [None] * self.cache_set_num
        self.index = 0

        for current_dir, child_dir, child_file in os.walk(data_path):
            if len(child_dir) == 0 and "Ri.npy" in child_file:
                self.dirs.append(current_dir)
                start_index = int(current_dir.split("/")[-1].split("_")[-2])
                end_index = int(current_dir.split("/")[-1].split("_")[-1])
                self.range_map[current_dir] = range(start_index, end_index)

        for i in range(self.cache_set_num):
            if i < len(self.dirs):
                self.__load_data(self.dirs[i])

        self.__compute_stat_output(10, 1e-3)

    def __get_img_num(self, path):
        start_index = int(path.split("/")[-1].split("_")[-2])
        end_index = int(path.split("/")[-1].split("_")[-1])
        return end_index - start_index

    def __load_data(self, path):

        data = {}

        data["start_index"] = int(path.split("/")[-1].split("_")[-2])
        data["end_index"] = int(path.split("/")[-1].split("_")[-1])
        data["Force"] = -1* np.load(os.path.join(path, "Force.npy"))
        data["Ei"] = np.load(os.path.join(path, "Ei.npy"))
        data["ListNeighbor"] = np.load(os.path.join(path, "ListNeighbor.npy"))
        data["Ri"] = np.load(os.path.join(path, "Ri.npy"))
        data["Ri_d"] = np.load(os.path.join(path, "Ri_d.npy"))
        data["ImageAtomNum"] = np.load(os.path.join(path, "ImageAtomNum.npy"))
        data["ImageIndex"] = np.insert(data["ImageAtomNum"][:, 0], 0, 0).cumsum()

        self.work_space[self.index] = data
        self.index = (self.index + 1) % self.cache_set_num

    def __getitem__(self, index):
        for data in self.work_space:
            if index < data["start_index"] or index >= data["end_index"]:
                continue

            real_index = index - data["start_index"]

            dic = {
                "Force": data["Force"][
                    data["ImageIndex"][real_index] : data["ImageIndex"][real_index + 1]
                ],
                "Ei": data["Ei"][
                    data["ImageIndex"][real_index] : data["ImageIndex"][real_index + 1]
                ],
                "ListNeighbor": data["ListNeighbor"][
                    data["ImageIndex"][real_index] : data["ImageIndex"][real_index + 1]
                ],
                "Ri": data["Ri"][
                    data["ImageIndex"][real_index] : data["ImageIndex"][real_index + 1]
                ],
                "Ri_d": data["Ri_d"][
                    data["ImageIndex"][real_index] : data["ImageIndex"][real_index + 1]
                ],
                "ImageAtomNum": data["ImageAtomNum"][real_index : real_index + 1]
            }
            return dic

        for dir, dir_range in self.range_map.items():
            if index in dir_range:
                self.__load_data(dir)

        for data in self.work_space:
            if index < data["start_index"] or index >= data["end_index"]:
                continue

            real_index = index - data["start_index"]
            dic = {
                "Force": data["Force"][
                    data["ImageIndex"][real_index] : data["ImageIndex"][real_index + 1]
                ],
                "Ei": data["Ei"][
                    data["ImageIndex"][real_index] : data["ImageIndex"][real_index + 1]
                ],
                "ListNeighbor": data["ListNeighbor"][
                    data["ImageIndex"][real_index] : data["ImageIndex"][real_index + 1]
                ],
                "Ri": data["Ri"][
                    data["ImageIndex"][real_index] : data["ImageIndex"][real_index + 1]
                ],
                "Ri_d": data["Ri_d"][
                    data["ImageIndex"][real_index] : data["ImageIndex"][real_index + 1]
                ],
                "ImageAtomNum": data["ImageAtomNum"][real_index : real_index + 1]
            }
            return dic

    def __len__(self):
        num = 0
        for dir in self.dirs:
            num += self.__get_img_num(dir)
        return num

    def __compute_stat_output(self, image_num=10, rcond=1e-3):

        data = self.work_space[0]

        self.ener_shift = []
        natoms_sum = data["ImageAtomNum"][0, 0]
        natoms_per_type = data["ImageAtomNum"][0, 1:]
        # only for one atom type
        if image_num > data["ImageAtomNum"].shape[0]:
            image_num = data["ImageAtomNum"].shape[0]

        energy = data["Ei"][data["ImageIndex"][0] : data["ImageIndex"][image_num]]
        energy = np.reshape(energy, (-1, natoms_sum, 1))
        # natoms_sum = 0
        # for ntype in range(self.ntypes):
        #     energy_ntype = energy[:, natoms_sum:natoms_sum+natoms_per_type[ntype]]
        #     natoms_sum += natoms_per_type[ntype]
        #     energy_sum = energy_ntype.sum(axis=1)
        #     energy_one = np.ones_like(energy_sum) * natoms_per_type[ntype]
        #     ener_shift, _, _, _ = np.linalg.lstsq(energy_one, energy_sum, rcond=rcond)
        #     self.ener_shift.append(ener_shift[0, 0])
        # energy_ntype = energy[:, natoms_sum:natoms_sum+natoms_per_type[ntype]]
        # natoms_sum += natoms_per_type[ntype]
        energy_sum = energy.sum(axis=1)
        energy_avg = np.average(energy_sum)
        # energy_one = np.ones_like(energy_sum) * natoms_per_type[ntype]
        ener_shift, _, _, _ = np.linalg.lstsq(
            [natoms_per_type], [energy_avg], rcond=rcond
        )
        self.ener_shift = ener_shift.tolist()

    def get_stat(self):
        return self.davg, self.dstd, self.ener_shift


def main():

    with open("config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    #     print("Read Config successful")
    import ipdb; ipdb.set_trace()
    dataset = MovementDataset(config, "./train")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    for i, sample_batches in enumerate(dataloader):
        
        print(sample_batches["Force"].shape)
        print(sample_batches["Ei"].shape)
        print(sample_batches["ListNeighbor"].shape)
        print(sample_batches["Ri"].shape)
        print(sample_batches["Ri_d"].shape)
        print(sample_batches["ImageAtomNum"].shape)
        # print(sample_batches["ImageIndex"].shape)
        import ipdb;ipdb.set_trace()
        # print(sample_batches["ListNeighbor"].shape)


if __name__ == "__main__":
    main()
