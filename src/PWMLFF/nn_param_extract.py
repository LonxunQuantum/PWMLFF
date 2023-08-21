import os
import shutil
from src.user.model_param import DpParam
import torch
import numpy as np
import src.aux.extract_ff as extract_ff
from src.user.model_param import DpParam
def extract_force_field():
    pass

'''
description: 
    extracting network parameters and scaler values for fortran MD routine
param {DpParam} dp_params
param {*} model_name
return {*}
author: wuxingxing
'''
def extract_model_para(dp_params:DpParam):
    from src.aux.extract_nn import read_wij, read_scaler 

    load_model_path = dp_params.file_paths.model_save_path
    print ("extracting parameters from:", load_model_path)
    read_wij(load_model_path)

    load_scaler_path = dp_params.file_paths.model_store_dir + "scaler.pkl"
    print ("extracting scaler values from:", load_scaler_path) 
    read_scaler(load_model_path)

def extract_force_field(dp_params:DpParam):
    forcefield_dir = dp_params.file_paths.forcefield_dir
    if os.path.exists(forcefield_dir):
        shutil.rmtree(forcefield_dir)
    os.makedirs(forcefield_dir)
    # copy fread_deat dir to forcefield dir
    shutil.copytree(os.path.join(dp_params.file_paths.train_feature_path[0], "fread_dfeat"),
                    os.path.join(forcefield_dir, "fread_dfeat"))
    shutil.copytree(os.path.join(dp_params.file_paths.train_feature_path[0], "input"),
                    os.path.join(forcefield_dir, "input"))
    
    cwd = os.getcwd()
    os.chdir(forcefield_dir)
    from src.aux.extract_ff import extract_ff
    extract_model_para(dp_params)
    extract_ff(name = dp_params.file_paths.forcefield_name, model_type = 3)
    os.chdir(cwd)

def load_scaler_from_checkpoint(model_path):
    model_checkpoint = torch.load(model_path,map_location=torch.device("cpu"))
    scaler = model_checkpoint['scaler']
    return scaler

# scaler_path = self.dp_params.file_paths.model_store_dir + 'scaler.pkl'
#             print("loading scaler from file",scaler_path)
#             self.scaler = load(scaler_path)
#             print("transforming feat with loaded scaler")