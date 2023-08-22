'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-08-21 22:43:22
LastEditors: your name
LastEditTime: 2023-08-22 10:07:50
FilePath: /PWMLFF/src/user/nn_work.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import json

from src.user.model_param import DpParam
from src.PWMLFF.nn_param_extract import extract_force_field
from src.PWMLFF.nn_network import nn_network
from utils.file_operation import post_process_train, post_process_test, delete_dir

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
    nn_param = DpParam(input_json, cmd) 
    nn_param.print_input_params()
    nn_trainer = nn_network(nn_param)
    if len(nn_param.file_paths.train_movement_path) > 0:
        feature_path = nn_trainer.generate_data()
    nn_param.file_paths.set_train_feature_path([feature_path])
    nn_trainer.load_and_train()
    extract_force_field(nn_param)
    
    if os.path.realpath(nn_param.file_paths.json_dir) != os.path.realpath(nn_param.file_paths.work_dir) :
        post_process_train(nn_param.file_paths.json_dir, \
                    nn_param.file_paths.model_store_dir, nn_param.file_paths.forcefield_dir, nn_param.file_paths.train_dir)
        if nn_param.file_paths.reserve_work_dir is False:
            delete_dir(nn_param.file_paths.work_dir)

def gen_nn_feature(input_json: json, cmd:str):
    nn_param = DpParam(input_json, cmd) 
    nn_param.print_input_params()
    nn_trainer = nn_network(nn_param)
    feature_path = nn_trainer.generate_data()
    print("feature generated done, the dir path is: \n{}".format(feature_path))

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
    nn_param = DpParam(input_json, cmd)
    nn_param.print_input_params()
    nn_trainer = nn_network(nn_param)
    if len(nn_param.file_paths.test_movement_path) > 0:
        gen_feat_dir = nn_trainer.generate_data()
    nn_param.file_paths.set_test_feature_path([gen_feat_dir])
    nn_trainer.load_and_train()

    if os.path.realpath(nn_param.file_paths.json_dir) != os.path.realpath(nn_param.file_paths.work_dir) :
        post_process_test(nn_param.file_paths.json_dir, nn_param.file_paths.test_dir)
        if nn_param.file_paths.reserve_work_dir is False:
            delete_dir(nn_param.file_paths.work_dir)