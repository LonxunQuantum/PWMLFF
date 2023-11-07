from src.user.input_param import InputParam
from utils.extract_movement import MOVEMENT

class nep_network:
    def __init__(self, nep_param:InputParam):
        self.nep_param = nep_param

    def construct_data_dir(self):
        # movement do classification
        mvm_list = self.nep_param.file_paths.train_movement_path
        mvm_sorted = []
        mvm_dict = {}
        for i, mvm_file in enumerate(mvm_list):
            mvm = MOVEMENT(mvm_file)
            atom_type = mvm.image_list[0].atom_type
            atom_type_num_list = mvm.image_list[0].atom_type_num
            key1 = "_".join(atom_type_num_list)
            key2 = "_".join(atom_type)
            mvm_dict[i] = "{}_{}".format(key1, key2)
        tmp = sorted(mvm_dict.items(), key = lambda x: len(x[1]['types']), reverse=True)
        for t in tmp:
            mvm_sorted.append(mvm_list[t[0]])
        return mvm_sorted
        # saperated movements to training and valid by random or last 20%

        # 