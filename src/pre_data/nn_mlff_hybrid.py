import os
import shutil
import subprocess
import default_para as pm

'''
    get sub_dir of root_dir, for each sub_dir, it should contains at least 1 MOVEMENT file
description: 
param {*} root_dir
param {*} dest_dir
param {*} file_name
return {*}
author: wuxingxing
'''
def get_cluster_dirs(root_dir, file_name="MOVEMENT"):
    # read movement dir from root_dir
    cluster_dir_list = {}
    mvm_dirs = []
    for path,dirList,fileList in os.walk(os.path.join(root_dir, "PWdata")):
        if file_name in fileList:
            mvm_dirs.append(path)
    for mvm_dir in mvm_dirs:
        mvm_path = os.path.join(mvm_dir, file_name)
        types, nums, key = get_atom_info_from_movement(mvm_path)
        if key not in cluster_dir_list.keys():
            cluster_dir_list[key] = {"mvm_path":[mvm_path], "types":types, "type_nums":nums}
        else:
            cluster_dir_list[key]["mvm_path"].append(mvm_path)
    
    # sorted by atom_type
    tmp_cluster = sorted(cluster_dir_list.items(), key = lambda x: len(x[1]['types']), reverse=True)
    for tmp in tmp_cluster:
        cluster_dir_list[tmp[0]] = tmp[1]

    return cluster_dir_list

def make_work_dir(work_dir, cluster_dir_list: list):
    sub_work_dir = []
    for cluster in cluster_dir_list.items():
        tmp_dir = os.path.join(work_dir, "{}/PWdata".format(cluster[0]))
        if os.path.exists(tmp_dir) is False:
            os.makedirs(tmp_dir)
        # copy movement
        for mvm_path in cluster[1]['mvm_path']:
            mvm_dir = os.path.dirname(mvm_path)
            if os.path.exists(os.path.join(tmp_dir, os.path.basename(mvm_dir))) is False:
                shutil.copytree(mvm_dir, os.path.join(tmp_dir, os.path.basename(mvm_dir)))
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
        
    return types, nums, key[:-1]

def mv_featrues(source_dir, dest_dir, index):
    if os.path.exists(dest_dir) is False:
        os.makedirs(dest_dir)
    npy_files = os.listdir(source_dir)
    for file in npy_files:
        if '.npy' in file:
            dest_path = os.path.join(dest_dir, "{}_{}".format(index, file))
            shutil.move(os.path.join(source_dir, file), dest_path)

def mv_file(source_file, dest_file):
    if os.path.exists(dest_file):
        if os.path.isdir(dest_file):
            shutil.rmtree(dest_file)
        else:
            os.remove(dest_file)
    shutil.move(source_file, dest_file)

def line_file(source_file, dest_filse):
    if os.path.exists(dest_filse):
        os.remove(dest_filse)
        
        


