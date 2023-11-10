import os
import json
from src.user.input_param import InputParam
from src.PWMLFF.nep_network import NepNetwork
from utils.file_operation import delete_tree, copy_tree, copy_file
'''
description: do nep training
    step1. generate feature from MOVEMENTs
        movement -> .xyz (train & valid)
    step2. load features and do training
    step3. extract forcefield files
    step4. copy features, trained model files to the same level directory of jsonfile
param {json} input_json
return {*}
author: wuxingxing
'''
def nep_train(input_json: json, cmd:str):
    nep_param = InputParam(input_json, cmd)
    nep_param.print_input_params(json_file_save_name="std_input.json")
    nep_trainer = NepNetwork(nep_param)
    if len(nep_param.file_paths.train_movement_path) > 0:
        nep_trainer.generate_data()
    nep_trainer.train()

    if os.path.realpath(nep_param.file_paths.json_dir) != os.path.realpath(nep_param.file_paths.work_dir) :
        if nep_param.file_paths.reserve_feature is False:
            if os.path.exists(nep_param.file_paths.nep_train_xyz_path):
                os.remove(nep_param.file_paths.nep_train_xyz_path)
            if os.path.exists(nep_param.file_paths.nep_test_xyz_path):
                os.remove(nep_param.file_paths.nep_test_xyz_path)

        # copy the whole work dir to nep_training_dir or model_record?
        copy_tree(nep_param.file_paths.work_dir, os.path.join(nep_param.file_paths.json_dir, os.path.basename(nep_param.file_paths.model_store_dir)))
        if nep_param.file_paths.reserve_work_dir is False:
            delete_tree(nep_param.file_paths.work_dir)

def gen_nep_feature(input_json: json, cmd:str):
    nep_param = InputParam(input_json, cmd)
    nep_param.print_input_params(json_file_save_name="std_input.json")
    nep_trainer = NepNetwork(nep_param)
    if len(nep_param.file_paths.train_movement_path) > 0:
        nep_trainer.generate_data()
    # copy the train.xyz and test.xyz to json file dir
    target_train = os.path.join(nep_param.file_paths.json_dir, os.path.basename(nep_param.file_paths.nep_train_xyz_path))
    target_valid = os.path.join(nep_param.file_paths.json_dir, os.path.basename(nep_param.file_paths.nep_test_xyz_path))
    copy_file(nep_param.file_paths.nep_train_xyz_path, target_train)
    copy_file(nep_param.file_paths.nep_test_xyz_path, target_valid)

'''
description: 
    do dp inference:
    setp0. read params from mode.cpkt file, and set model related params to test
    step1. generate feature, the movement from json file 'test_movement_path'
    step2. load model and do inference
    step3. copy inference result files to the same level directory of jsonfile
param {json} input_json
param {str} cmd
return {*}
author: wuxingxing
'''
def nep_test(input_json: json, cmd:str):
    # model_load_path = get_required_parameter("model_load_file", input_json)
    # model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"))
    # json_dict_train = model_checkpoint["json_file"]

    # json_dict_train["work_dir"] = get_parameter("work_dir", input_json, "work_test_dir")
    
    # dp_param = InputParam(json_dict_train, "test".upper())
    # # set inference param
    # dp_param.set_test_relative_params(input_json)
    # dp_param.print_input_params(json_file_save_name="std_input.json")

    # nep_trainer = dp_network(dp_param)
    # if len(dp_param.file_paths.test_movement_path) > 0:
    #     gen_feat_dir = nep_trainer.generate_data()
    #     dp_param.file_paths.set_test_feature_path([gen_feat_dir])
    # nep_trainer.inference()
    # if os.path.realpath(dp_param.file_paths.json_dir) != os.path.realpath(dp_param.file_paths.work_dir) :
    #     copy_test_result(dp_param.file_paths.json_dir, dp_param.file_paths.test_dir)
        
    #     if dp_param.file_paths.reserve_feature is False:
    #         delete_tree(dp_param.file_paths.train_dir)
            
    #     if dp_param.file_paths.reserve_work_dir is False:
    #         delete_tree(dp_param.file_paths.work_dir)
    pass
