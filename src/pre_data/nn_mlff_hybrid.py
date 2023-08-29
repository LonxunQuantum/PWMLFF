import os
import shutil
import subprocess
import default_para as pm

'''
description: do cluster according to movement [atom type, atom type order, atom type nums] 
param {list} movement_list
return {*}
author: wuxingxing
'''
def get_cluster_dirs(movement_list:list):
    # read movement dir from root_dir
    cluster_dir_list = {}
    for mvm_path in movement_list:
        types, nums, key, atom_num = get_atom_info_from_movement(mvm_path)
        if key not in cluster_dir_list.keys():
            cluster_dir_list[key] = {"mvm_path":[os.path.abspath(mvm_path)], "types":types, "type_nums":nums}
        else:
            cluster_dir_list[key]["mvm_path"].append(mvm_path)
    # sorted by atom_type
    tmp_cluster = sorted(cluster_dir_list.items(), key = lambda x: len(x[1]['types']), reverse=True)
    for tmp in tmp_cluster:
        cluster_dir_list[tmp[0]] = tmp[1]
    return cluster_dir_list, atom_num

'''
description: 
param {*} feature_dir
param {list} cluster_dir_list
return {*}
author: wuxingxing
'''
def make_work_dir(feature_dir:str, pwdata_name:str, movement_name:str, cluster_dir_list: list):
    sub_work_dir = []
    if os.path.exists(feature_dir):
        shutil.rmtree(feature_dir)
    os.makedirs(feature_dir)
    for cluster in cluster_dir_list.items():
        tmp_dir = os.path.join(feature_dir, "{}".format(cluster[0]), pwdata_name)
        # copy movement
        for index, mvm_path in enumerate(cluster[1]['mvm_path']):
            mvm_dir = os.path.join(tmp_dir, "{}_{}".format(os.path.basename(mvm_path), index))
            os.makedirs(mvm_dir)
            shutil.copy(mvm_path, os.path.join(mvm_dir, movement_name))
        sub_work_dir.append(os.path.dirname(tmp_dir))
    return sub_work_dir

'''
description: 
    read atom infos from first image of movement
    example: for a Li-Si system with 76 atoms (60 Li atoms, 16 Si atoms)
        return [3, 14], [60, 16], '3_14_60_16'
param {*} file_path: the movement file path
return {*} types, nums, key
'''
def get_atom_info_from_movement(file_path):
    # read first line to get atom nums info, then read the Position block to count atom info.
    position_block = []
    from itertools import islice
    with open(file_path, 'r') as rf:
        first_line = rf.readline()
        atom_num = int(first_line.strip().split()[0])
        for line in islice(rf, atom_num+ 20):
            position_block.append(line)

    start_index = 0
    while "Position " not in position_block[start_index]:
        start_index += 1
    atom_list = []
    for atom_line in position_block[start_index+1:start_index+1+atom_num]:
        atom_list.append(int(atom_line.strip().split()[0]))
    atom_type = {}
    for atom in atom_list:
        if atom not in atom_type.keys():
            atom_type[atom] = [atom]
        else:
            atom_type[atom].append(atom)
    key = ""
    types = []
    nums = []
    for t in list(atom_type.keys()):
        key += "{}_".format(t)
        types.append(t)
    for v in atom_type.values():
        key += "{}_".format(len(v))
        nums.append(len(v))
        
    return types, nums, key[:-1], atom_num

def mv_featrues(source_dir, dest_dir, index):
    if os.path.exists(dest_dir) is False:
        os.makedirs(dest_dir)
    npy_files = os.listdir(source_dir)
    for file in npy_files:
        if '.npy' in file:
            dest_path = os.path.join(dest_dir, "{}_{}".format(index, file))
            shutil.move(os.path.join(source_dir, file), dest_path)

def copy_file(source_file, dest_file):
    if os.path.exists(dest_file):
        if os.path.isdir(dest_file):
            shutil.rmtree(dest_file)
        else:
            os.remove(dest_file)
    shutil.copy(source_file, dest_file)

def line_file(source_file, dest_filse):
    if os.path.exists(dest_filse):
        os.remove(dest_filse)
        
        


