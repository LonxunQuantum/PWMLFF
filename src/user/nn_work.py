import os
import json
import torch
from src.user.input_param import InputParam
from src.PWMLFF.nn_param_extract import extract_force_field
from src.PWMLFF.nn_network import nn_network
from utils.file_operation import delete_tree, copy_tree, copy_file
from utils.json_operation import get_parameter, get_required_parameter
'''
description: do nn training
    step1. generate feature from MOVEMENTs
    step2. load features and do training
    step3. extract forcefield files
    step4. copy features, trained model files to the same level directory of jsonfile
param {json} input_json
return {*}
author: wuxingxing
'''
def nn_train(input_json: json, cmd:str):
    nn_param = InputParam(input_json, cmd) 
    nn_param.print_input_params(json_file_save_name="std_input.json")
    nn_trainer = nn_network(nn_param)
    if len(nn_param.file_paths.train_movement_path) > 0:
        feature_path = nn_trainer.generate_data()
        # feature_path = "/data/home/wuxingxing/datas/pwmat_mlff_workdir/lisi/ref_nn/work_train_dir/feature"
        nn_param.file_paths.set_train_feature_path([feature_path])
    nn_trainer.load_and_train()
    # if the input epochs to the end, model will not be trained and will not be saved at work_dir
    if os.path.exists(nn_param.file_paths.model_save_path) is False:
        if os.path.exists(nn_param.file_paths.model_load_path):
            nn_param.file_paths.model_save_path = nn_param.file_paths.model_load_path
    extract_force_field(nn_param)
    
    if os.path.realpath(nn_param.file_paths.json_dir) != os.path.realpath(nn_param.file_paths.work_dir) :
        copy_train_result(nn_param.file_paths.json_dir, \
                    nn_param.file_paths.model_store_dir, nn_param.file_paths.forcefield_dir)
        
        if nn_param.file_paths.reserve_feature is False:
            delete_tree(nn_param.file_paths.train_dir)

        if nn_param.file_paths.reserve_work_dir is False:
            delete_tree(nn_param.file_paths.work_dir)

def gen_nn_feature(input_json: json, cmd:str):
    nn_param = InputParam(input_json, cmd) 
    nn_param.print_input_params(json_file_save_name="std_input.json")
    nn_trainer = nn_network(nn_param)
    if len(nn_param.file_paths.train_movement_path) > 0:
        feature_path = nn_trainer.generate_data()
    print("feature generated done, the dir path is: \n{}".format(feature_path))
    return feature_path

'''
description: 
    do nn inference:
    step1. generate feature, the movement from json file 'test_movement_path'
    step2. load model and do inference
    step3. copy inference result files to the same level directory of jsonfile
param {json} input_json
param {str} cmd
return {*}
author: wuxingxing
'''
def nn_test(input_json: json, cmd:str):
    model_load_path = get_required_parameter("model_load_file", input_json)
    model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"))
    json_dict_train = model_checkpoint["json_file"]

    json_dict_train["work_dir"] = get_parameter("work_dir", input_json, "work_test_dir")
    
    nn_param = InputParam(json_dict_train, "test".upper())
    # set inference param
    nn_param.set_test_relative_params(input_json)
    nn_param.print_input_params(json_file_save_name="std_input.json")

    nn_trainer = nn_network(nn_param)
    if len(nn_param.file_paths.test_movement_path) > 0:
        gen_feat_dir = nn_trainer.generate_data()
        nn_param.file_paths.set_test_feature_path([gen_feat_dir])

    nn_trainer.load_and_train()

    if os.path.realpath(nn_param.file_paths.json_dir) != os.path.realpath(nn_param.file_paths.work_dir) :
        copy_test_result(nn_param.file_paths.json_dir, nn_param.file_paths.test_dir)
        if nn_param.file_paths.reserve_work_dir is False:
            delete_tree(nn_param.file_paths.work_dir)

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
def copy_train_result(target_dir, model_store_dir, forcefield_dir):
    # copy model
    target_model_path = os.path.join(target_dir, os.path.basename(model_store_dir))
    copy_tree(model_store_dir, target_model_path)
    # copy forcefild
    target_forcefield_dir = os.path.join(target_dir, os.path.basename(forcefield_dir))
    copy_tree(forcefield_dir, target_forcefield_dir)

'''
description: 
    copy inference dir under work dir to target_dir
param {*} test_dir
param {*} json_dir
return {*}
author: wuxingxing
'''
def copy_test_result(target_dir, test_dir):
    # copy inference result
    target_test_dir = os.path.join(target_dir, os.path.basename(test_dir))
    for file in os.listdir(test_dir):
        # if file.endswith(".txt") or file.endswith('.csv'):
        copy_file(os.path.join(test_dir, file), os.path.join(target_test_dir, os.path.basename(file)))