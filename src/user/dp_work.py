import os
import json

from src.user.model_param import DpParam
from src.PWMLFF.dp_param_extract import extract_force_field
from src.PWMLFF.dp_network import dp_network

'''
description: do dp training
    step1. generate feature from MOVEMENTs
    step2. load features and do training
param {json} input_json
return {*}
author: wuxingxing
'''
def dp_train(input_json: json, cmd:str):
    dp_param = DpParam(input_json, cmd) 
    dp_param.print_input_params()
    dp_trainer = dp_network(dp_param)
    if len(dp_param.file_paths.train_movement_path) > 0:
        feature_path = dp_trainer.generate_data()
    dp_param.file_paths.set_train_feature_path([feature_path])
    dp_trainer.load_and_train()
    extract_force_field(dp_param)
    dp_trainer.post_process_train()
    
def gen_dp_feature(input_json: json, cmd:str):
    dp_param = DpParam(input_json, cmd) 
    dp_param.print_input_params()
    dp_trainer = dp_network(dp_param)
    feature_path = dp_trainer.generate_data()
    print("feature generated done, the dir path is: \n{}".format(feature_path))

def dp_test(input_json: json, cmd:str):
    dp_param = DpParam(input_json, cmd)
    dp_param.print_input_params()
    dp_trainer = dp_network(dp_param)
    gen_feat_dir = dp_trainer.generate_data()
    dp_param.file_paths.set_test_feature_path([gen_feat_dir])
    dp_trainer.load_and_train()
    dp_trainer.post_process_test()
