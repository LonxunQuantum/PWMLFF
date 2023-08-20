import os
import shutil

import numpy as np
def write_line_to_file(file_path, line, write_patthen="w"):
    with open(file_path, write_patthen) as wf:
        wf.write(line)
        wf.write('\n')
    
def write_arrays_to_file(file_path, arrays, write_patthen="w"):
    with open(file_path, write_patthen) as wf:
        for data in arrays:
            if isinstance(data, list):
                line = ' '.join(np.array(data).astype('str'))
            else:
                line = "{}".format(data)
            wf.write(line)
            wf.write('\n') 

'''
description: copy file from souce to target dir:
    if target dir does not exist, create it
    if target_file is exist, replace it by source file
param {*} source_file_path
param {*} target_file_path
param {*} target_dir
return {*}
author: wuxingxing
'''
def copy_file(source_file_path, target_file_path, target_dir):
    if os.path.exists(target_dir) is False:
        os.makedirs(target_dir)
    if os.path.exists(target_file_path):
        os.remove(target_file_path)
    shutil.copy(source_file_path, target_file_path)

'''
description: 
    copy source_movement_list to target_dir, each movement file at target_dir has a sinlge dir. for example:
        souce_movement_paths=[".../MOVMENET1", ".../MOVEMENT2"], after copy, at the target_dir:
        target_dir/MOVEMENT1_0/MOVEMENT1, target_dir/MOVEMENT2_1/MOVEMENT2
param {*} source_movement_paths
param {*} target_dir
param {*} trainSetDir
param {*} movement_name
return pwdata_work_dir
author: wuxingxing
'''
def copy_movements_to_work_dir(source_movement_paths, target_dir, trainSetDir, movement_name):
    if os.path.exists(target_dir) is True:
        shutil.rmtree(target_dir)
    pwdata_work_dir = os.path.join(target_dir, trainSetDir)
    for i, mvm in enumerate(source_movement_paths):
        mvm_dir = os.path.join(pwdata_work_dir, "{}_{}".format(os.path.basename(mvm), i))
        target_mvm = os.path.join(mvm_dir, movement_name)
        copy_file(mvm, target_mvm, mvm_dir)
    return pwdata_work_dir
