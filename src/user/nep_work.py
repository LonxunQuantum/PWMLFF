import os
import torch
import json
from src.user.input_param import InputParam
from src.PWMLFF.nep_network import nep_network
from utils.file_operation import delete_tree, copy_tree, copy_file
from utils.atom_type_emb_dict import element_table
from utils.file_operation import delete_tree, copy_tree, copy_file
from utils.json_operation import get_parameter, get_required_parameter

'''
description: do nep training
    step1. generate feature to xyz format files
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
    nep_trainer = nep_network(nep_param)
    if len(nep_param.file_paths.raw_path) > 0:
        data_paths = nep_trainer.generate_data()
        print(data_paths)
        nep_param.file_paths.set_datasets_path(data_paths)
    nep_trainer.train()

    # if os.path.realpath(nep_param.file_paths.json_dir) != os.path.realpath(nep_param.file_paths.work_dir) :
    #     if nep_param.file_paths.reserve_feature is False:
    #         if os.path.exists(nep_param.file_paths.nep_train_xyz_path):
    #             os.remove(nep_param.file_paths.nep_train_xyz_path)
    #         if os.path.exists(nep_param.file_paths.nep_test_xyz_path):
    #             os.remove(nep_param.file_paths.nep_test_xyz_path)

    #     # copy the whole work dir to nep_training_dir or model_record?
    #     copy_tree(nep_param.file_paths.work_dir, os.path.join(nep_param.file_paths.json_dir, os.path.basename(nep_param.file_paths.model_store_dir)))
    #     if nep_param.file_paths.reserve_work_dir is False:
    #         delete_tree(nep_param.file_paths.work_dir)

# '''
# description: 
#     This method has been deprecated
# param {json} input_json
# param {str} cmd
# return {*}
# author: wuxingxing
# '''
# def gen_nep_feature(input_json: json, cmd:str):
#     nep_param = InputParam(input_json, cmd)
#     nep_param.print_input_params(json_file_save_name="std_input.json")
#     nep_trainer = NepNetwork(nep_param)
#     nep_trainer.generate_data()
#     # copy the train.xyz and test.xyz to json file dir
#     target_train = os.path.join(nep_param.file_paths.json_dir, os.path.basename(nep_param.file_paths.nep_train_xyz_path))
#     target_valid = os.path.join(nep_param.file_paths.json_dir, os.path.basename(nep_param.file_paths.nep_test_xyz_path))
#     copy_file(nep_param.file_paths.nep_train_xyz_path, target_train)
#     copy_file(nep_param.file_paths.nep_test_xyz_path, target_valid)

# '''
# description: 
#     do dp inference:
#     setp0. read params from mode.cpkt file, and set model related params to test
#         the params need to be set by nep.txt or nep.in file:

#     step1. generate feature, the movement from json file 'test_movement_path'
#     step2. load model and do inference
#     step3. copy inference result files to the same level directory of jsonfile
# param {json} input_json
# param {str} cmd
# return {*}
# author: wuxingxing
# '''
def nep_test(input_json: json, cmd:str):
    model_load_path = get_required_parameter("model_load_file", input_json)
    model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"))
    json_dict_train = model_checkpoint["json_file"]
    model_checkpoint["json_file"]["datasets_path"] = []
    json_dict_train["optimizer"] = {}
    json_dict_train["optimizer"]["optimizer"] = "LKF"
    nep_param = InputParam(json_dict_train, "test".upper())
    # set inference param
    nep_param.set_test_relative_params(input_json)
    # nep_param.print_input_params(json_file_save_name="std_input.json")
    nep_trainer = nep_network(nep_param)
    if len(nep_param.file_paths.raw_path) > 0:
        data_paths = nep_trainer.generate_data()
        nep_param.file_paths.set_datasets_path(data_paths)
    nep_trainer.inference()

def togpumd(cmd_list:list[str]):
    ckpt_file = cmd_list[0]
    model_checkpoint = torch.load(ckpt_file, map_location=torch.device("cpu"))
    json_dict_train = model_checkpoint["json_file"]
    if json_dict_train["model_type"] != "NEP":
        raise Exception("Error! The input model is not a nep model!")
    model_checkpoint["json_file"]["datasets_path"] = []
    json_dict_train["optimizer"] = {}
    json_dict_train["optimizer"]["optimizer"] = "LKF"
    json_dict_train["model_load_file"] = ckpt_file
    nep_param = InputParam(json_dict_train, "test".upper())
    nep_param.set_test_relative_params(json_dict_train)
    nep_trainer = nep_network(nep_param)
    energy_shift, max_atom_nums, image_path = nep_trainer._get_stat()
    # energy_shift, atom_map, train_loader, val_loader = nep_trainer.load_data(energy_shift, max_atom_nums)
    model, optimizer = nep_trainer.load_model_optimizer(energy_shift)
    nep_trainer.convert_to_gpumd(model, save_dir=os.path.dirname(os.path.abspath(ckpt_file)))
    
# def read_nep_info(nep_txt_file: str):
#     if not os.path.exists(nep_txt_file):
#         raise Exception("ERROR! The nep.txt file does not exist at {}!".format(nep_txt_file))
#     with open(nep_txt_file, 'r') as rf:
#         lines = rf.readlines()
#     res = {}
#     # read version and atom type with order
#     first = lines[0].strip().split()
#     res["version"]=int(first[0].lower().replace("nep", ""))
#     res["atom_type"] = []
#     for atom_name in first[2:]:
#         res["atom_type"].append(element_table.index(atom_name))
#     return res
