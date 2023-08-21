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


'''
description: 
    copy model dir under workdir to target_dir
    copy forcefild dir under workdir to target_dir
    delete feature files under workdir
param {*} target_dir
param {*} model_store_dir
param {*} forcefield_dir
param {*} train_dir
return {*}
author: wuxingxing
'''
def post_process_train(target_dir, model_store_dir, forcefield_dir, train_dir):
    # copy model
    target_model_path = os.path.join(target_dir, os.path.basename(model_store_dir))
    if os.path.exists(target_model_path):
        shutil.rmtree(target_model_path)
    shutil.copytree(model_store_dir, target_model_path)
    # os.symlink(os.path.realpath(target_model_path), os.path.realpath(model_store_dir))

    # copy forcefild
    target_forcefield_dir = os.path.join(target_dir, os.path.basename(forcefield_dir))
    if os.path.exists(target_forcefield_dir):
        shutil.rmtree(target_forcefield_dir)
    shutil.copytree(forcefield_dir, target_forcefield_dir)
    # os.symlink(os.path.realpath(target_forcefield_dir), os.path.realpath(source_forcefield_dir))

    # delete feature data
    shutil.rmtree(train_dir)

    # for NN model, copy fread_dfeat and input dir to model dir
    # source_fread_dfeat = os.path.join(train_dir, "fread_dfeat")
    # if os.path.exists(source_fread_dfeat):
    #     target_fread_dfeat = os.path.join(model_store_dir, os.path.basename(source_fread_dfeat))
    #     if os.path.exists(target_fread_dfeat):
    #         shutil.rmtree(target_fread_dfeat)
    #     shutil.copytree(source_fread_dfeat, target_fread_dfeat)

    # source_input = os.path.join(train_dir, "input")
    # if os.path.exists(source_input):
    #     target_input = os.path.join(model_store_dir, os.path.basename(source_input))
    #     if os.path.exists(target_input):
    #         shutil.rmtree(target_input)
    #     shutil.copytree(source_input, target_input)

    # this commit code for: copy the feature path under work dir to json file dir
    # target_data_path = os.path.join(self.dp_params.file_paths.target_dir, "gen_feature")
    # index = 0
    # #If a feature path already exists under the JSON dir path, then the feature_ Path+1
    # while os.path.exists(target_data_path): 
    #     target_data_path = os.path.join(self.dp_params.file_paths.target_dir, "{}_{}".format(os.path.basename(train_dir), index))
    #     index +=1
    # os.symlink(os.path.realpath(target_data_path), os.path.realpath(train_dir))
    # copy fread/input

#self.dp_params.file_paths.test_dir
'''
description: 
    copy inference dir under work dir to target_dir
param {*} test_dir
param {*} json_dir
return {*}
author: wuxingxing
'''
def post_process_test(target_dir, test_dir):
    # copy inference result
    target_test_dir = os.path.join(target_dir, os.path.basename(test_dir))
    if os.path.exists(target_test_dir):
        shutil.rmtree(target_test_dir)
    shutil.copytree(test_dir, target_test_dir)
    # os.symlink(os.path.realpath(target_test_dir), os.path.realpath(source_test_dir))

'''
description: delete diractory
param {*} dir
return {*}
author: wuxingxing
'''
def delete_dir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)