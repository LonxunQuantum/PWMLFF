import os
import json

from src.user.model_param import DpParam
from src.PWMLFF.dp_param_extract import extract_force_field
from src.PWMLFF.dp_network import dp_network
from utils.file_operation import delete_tree, copy_tree, copy_file
'''
description: do dp training
    step1. generate feature from MOVEMENTs
    step2. load features and do training
    step3. extract forcefield files
    step4. copy features, trained model files to the same level directory of jsonfile
param {json} input_json
return {*}
author: wuxingxing
'''
def dp_train(input_json: json, cmd:str):
    dp_param = DpParam(input_json, cmd) 
    dp_param.print_input_params(json_file_save_name="dp_train_final.json")
    dp_trainer = dp_network(dp_param)
    # if len(dp_param.file_paths.train_movement_path) > 0:
        # feature_path = dp_trainer.generate_data()
        # dp_param.file_paths.set_train_feature_path([feature_path])
    feature_path = dp_param.file_paths.train_dir
    if os.path.exists(feature_path):
        dp_param.file_paths.set_train_feature_path([feature_path])
        dp_trainer.load_and_train()
    else:
        if len(dp_param.file_paths.train_movement_path) > 0:
            feature_path = dp_trainer.generate_data()
            dp_param.file_paths.set_train_feature_path([feature_path])
        dp_trainer.load_and_train()
    extract_force_field(dp_param)

    if os.path.realpath(dp_param.file_paths.json_dir) != os.path.realpath(dp_param.file_paths.work_dir) :
        post_process_train(dp_param.file_paths.json_dir, \
                       dp_param.file_paths.model_store_dir, dp_param.file_paths.forcefield_dir, dp_param.file_paths.train_dir)
        if dp_param.file_paths.reserve_work_dir is False:
            if dp_param.model_num == 1:
                delete_tree(dp_param.file_paths.work_dir)

def gen_dp_feature(input_json: json, cmd:str):
    dp_param = DpParam(input_json, cmd) 
    dp_param.print_input_params(json_file_save_name="dp_train_final.json")
    dp_trainer = dp_network(dp_param)
    if len(dp_param.file_paths.train_movement_path) > 0:
        feature_path = dp_trainer.generate_data()
    dp_param.file_paths.set_train_feature_path([feature_path])
    print("feature generated done, the dir path is: \n{}".format(feature_path))

'''
description: 
    do dp inference:
    step1. generate feature, the movement from json file 'test_movement_path'
    step2. load model and do inference
    step3. copy inference result files to the same level directory of jsonfile
param {json} input_json
param {str} cmd
return {*}
author: wuxingxing
'''
def dp_test(input_json: json, cmd:str):
    dp_param = DpParam(input_json, cmd)
    dp_param.print_input_params(json_file_save_name="dp_test_final.json")
    dp_trainer = dp_network(dp_param)
    gen_feat_dir = dp_trainer.generate_data()
    dp_param.file_paths.set_test_feature_path([gen_feat_dir])
    dp_trainer.load_and_train()
    if os.path.realpath(dp_param.file_paths.json_dir) != os.path.realpath(dp_param.file_paths.work_dir) :
        post_process_test(dp_param.file_paths.json_dir, dp_param.file_paths.test_dir)
        if dp_param.file_paths.reserve_work_dir is False:
            delete_tree(dp_param.file_paths.work_dir)


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
    copy_tree(model_store_dir, target_model_path)
    # copy forcefild
    target_forcefield_dir = os.path.join(target_dir, os.path.basename(forcefield_dir))
    copy_tree(forcefield_dir, target_forcefield_dir)
    # delete feature data
    delete_tree(train_dir)

'''
description: 
    copy inference result under work dir to target_dir

param {*} test_dir
param {*} json_dir
return {*}
author: wuxingxing
'''
def post_process_test(target_dir, test_dir):
    # copy inference result
    target_test_dir = os.path.join(target_dir, os.path.basename(test_dir))
    for file in os.listdir(test_dir):
        if file.endswith(".txt") or file.endswith('.csv'):
            copy_file(os.path.join(test_dir, file), os.path.join(target_test_dir, os.path.basename(file)))
    
