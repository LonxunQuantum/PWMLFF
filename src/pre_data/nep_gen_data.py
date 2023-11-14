from src.user.input_param import InputParam
from utils.extract_movement import MOVEMENT
from utils.random_utils import random_index
from utils.mvm2xyz import Structure
import os, shutil

def convert_mvmfiles_to_xyz(mvm_file_list:list, train_save_path:str, valid_save_path:str, valid_shuffle:bool=False, ratio:float=0.2):
    mvm_classed_list = classify_mvm(mvm_file_list)
    # saperated movements to training and valid by random or last 20%
    write_train_valid_movement(train_save_path, valid_save_path, mvm_classed_list, ratio, valid_shuffle)
        
def classify_mvm(mvm_file_list: list[str]):
    mvm_sorted = {}
    mvm_dict = {}
    mvm_obj = []
    for i, mvm_file in enumerate(mvm_file_list):
        mvm = MOVEMENT(mvm_file)
        mvm_obj.append(mvm)
        atom_type = mvm.image_list[0].atom_type
        atom_type_num_list = mvm.image_list[0].atom_type_num
        key1 = "_".join(str(item) for item in atom_type_num_list)
        key2 = '_'.join(str(item) for item in atom_type)
        mvm_dict[i] = "{}_{}".format(key1, key2)
    tmp = sorted(mvm_dict.items(), key = lambda x: len(x[1]), reverse=True)
    for t in tmp:
        if t[1] not in mvm_sorted.keys():
            mvm_sorted[t[1]] = [{"file": mvm_file_list[t[0]], "obj":mvm_obj[t[0]]}]
        else:
            mvm_sorted[t[1]].append({"file": mvm_file_list[t[0]], "obj":mvm_obj[t[0]]})
    return mvm_sorted

def write_train_valid_movement(train_save_path, valid_save_path, mvm_sorted:dict, ratio:float, valid_shuffle:bool):
    # separate mvm files to train_movement and valid_movement
    train_file_list = []
    valid_file_list = []
    # delete tmp files, train.xyz, and test.xyz if exist before
    os.system("rm train_mvm_* valid_mvm_* train.xyz test.xyz -r")

    for i, mvm_type_key in enumerate(mvm_sorted.keys()):
        mvm_list = mvm_sorted[mvm_type_key]
        tmp_train = "train_mvm_{}_{}".format(mvm_type_key, i)
        tmp_valid = "valid_mvm_{}_{}".format(mvm_type_key, i)

        for mvm in mvm_list:
            train_indexs, valid_indexs = random_index(mvm["obj"].image_nums, ratio, valid_shuffle)
    
            with open(tmp_train, 'a') as af:
                for j in train_indexs:
                    for line in mvm["obj"].image_list[j].content:
                        af.write(line)

            with open(tmp_valid, 'a') as af:
                for j in valid_indexs:
                    for line in mvm["obj"].image_list[j].content:
                        af.write(line)

            print("{} separted to train and valid momvement done (valid shuffle {})!".format(mvm['file'], valid_shuffle))
        
        train_file_list.append(tmp_train)
        valid_file_list.append(tmp_valid)
        
    # convert movement to xyz 
    mvm2xyz(train_file_list, train_save_path)
    mvm2xyz(valid_file_list, valid_save_path)
    # delete tmp files
    for mvm in train_file_list:
        os.remove(mvm)
    for mvm in valid_file_list:
        os.remove(mvm)

def mvm2xyz(mvm_list:list[str], file_name:str):
    # if exist file_name, delete them
    if os.path.exists(file_name):
        os.remove(file_name)
        
    for mvm in mvm_list:
        a=Structure(path=mvm, type="MOVEMENT")
        a.coordinate2cartesian()
        a.out_extxyz(file_name,None) # Write to the file in append mode
        print("{} convert to xyz format done!".format(mvm))
    