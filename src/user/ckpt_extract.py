import os
import shutil
from src.user.model_param import DpParam
import torch
import numpy as np
import src.aux.extract_ff as extract_ff
import json
from src.PWMLFF.dp_param_extract import extract_force_field as dp_extract_force_field
from src.PWMLFF.nn_param_extract import extract_force_field as nn_extract_force_field

def extract_force_field(ckpt_file, cmd_type):
    #json_file
    model_checkpoint = torch.load(ckpt_file,map_location=torch.device("cpu"))
    json_dict = model_checkpoint["json_file"]
    dp_param = DpParam(json_dict, "train".upper())
    # set forcefiled save path
    dp_param.file_paths.forcefield_dir = os.path.join(os.getcwd(), os.path.basename(dp_param.file_paths.forcefield_dir))
    dp_param.file_paths.model_save_path = os.path.join(os.getcwd(), ckpt_file) if os.path.isabs(ckpt_file) is False else ckpt_file
    if dp_param.model_type.upper() == "DP".upper():
        dp_extract_force_field(dp_param)
    elif dp_param.model_type.upper() == "NN".upper():
        nn_extract_force_field(dp_param)
    else:
        raise Exception("Error! The extract command {} not realized. ".format(cmd_type))
    