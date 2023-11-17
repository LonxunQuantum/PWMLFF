import os
import json
from src.user.input_param import InputParam
from src.PWMLFF.nep_network import NepNetwork
from utils.file_operation import delete_tree, copy_tree, copy_file
from utils.atom_type_emb_dict import element_table
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
        the params need to be set by nep.txt or nep.in file:

    step1. generate feature, the movement from json file 'test_movement_path'
    step2. load model and do inference
    step3. copy inference result files to the same level directory of jsonfile
param {json} input_json
param {str} cmd
return {*}
author: wuxingxing
'''
def nep_test(input_json: json, cmd:str):
    nep_txt_file = input_json["model_load_file"]
    nep_info = read_nep_info(nep_txt_file)
    input_json["atom_type"] = nep_info["atom_type"]
    if "model" in input_json.keys() and \
        "prediction" in input_json["model"].keys():
            input_json["model"]["prediction"] = 1
    nep_param = InputParam(input_json, "test".upper())
    nep_param.set_test_relative_params(input_json)
    nep_param.print_input_params(json_file_save_name="std_input.json")
    nep_trainer = NepNetwork(nep_param)
    if len(nep_param.file_paths.test_movement_path) > 0:
        nep_trainer.generate_data()
    nep_trainer.inference()

    if os.path.realpath(nep_param.file_paths.json_dir) != os.path.realpath(nep_param.file_paths.work_dir) :
        if nep_param.file_paths.reserve_feature is False:
            if os.path.exists(nep_param.file_paths.nep_train_xyz_path):
                os.remove(nep_param.file_paths.nep_train_xyz_path)

    # copy the whole work dir to nep_training_dir or model_record?
    copy_tree(nep_param.file_paths.work_dir, os.path.join(nep_param.file_paths.json_dir, os.path.basename(nep_param.file_paths.test_dir)))
    if nep_param.file_paths.reserve_work_dir is False:
        delete_tree(nep_param.file_paths.work_dir)

def read_nep_info(nep_txt_file: str):
    if not os.path.exists(nep_txt_file):
        raise Exception("ERROR! The nep.txt file does not exist at {}!".format(nep_txt_file))
    with open(nep_txt_file, 'r') as rf:
        lines = rf.readlines()
    res = {}
    # read version and atom type with order
    first = lines[0].strip().split()
    res["version"]=int(first[0].lower().replace("nep", ""))
    res["atom_type"] = []
    for atom_name in first[2:]:
        res["atom_type"].append(element_table.index(atom_name))
    return res
