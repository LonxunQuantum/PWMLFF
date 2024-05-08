import os
from src.user.input_param import InputParam
import torch
from src.PWMLFF.dp_network import dp_network
from src.PWMLFF.nep_network import nep_network
from src.PWMLFF.dp_param_extract import extract_force_field as dp_extract_force_field
from src.PWMLFF.nn_param_extract import extract_force_field as nn_extract_force_field

def extract_force_field(ckpt_file, cmd_type):
    #json_file
    model_checkpoint = torch.load(ckpt_file,map_location=torch.device("cpu"))
    json_dict = model_checkpoint["json_file"]
    dp_param = InputParam(json_dict, "train".upper())
    # set forcefiled save path
    dp_param.file_paths.forcefield_dir = os.path.join(os.getcwd(), os.path.basename(dp_param.file_paths.forcefield_dir))
    dp_param.file_paths.model_save_path = os.path.join(os.getcwd(), ckpt_file) if os.path.isabs(ckpt_file) is False else ckpt_file
    if dp_param.model_type.upper() == "DP".upper():
        dp_extract_force_field(dp_param)
    elif dp_param.model_type.upper() == "NN".upper():
        nn_extract_force_field(dp_param)
    else:
        raise Exception("Error! The extract command {} not realized. ".format(cmd_type))

def script_model(ckpt_file):
    # Step 1.
    model_checkpoint = torch.load(ckpt_file,map_location=torch.device("cpu"))
    model_type = model_checkpoint["json_file"]["model_type"].upper()
    if model_type == "DP":
        script_dp_model(model_checkpoint, ckpt_file)
    elif model_type == "NEP":
        script_nep_model(model_checkpoint, ckpt_file)

def script_dp_model(model_checkpoint, ckpt_file):
    stat = [model_checkpoint["davg"], model_checkpoint["dstd"], model_checkpoint["energy_shift"]]
    # Step 2.
    model_checkpoint["json_file"]["model_load_file"] = ckpt_file #the model will reload from this path
    model_checkpoint["json_file"]["datasets_path"] = []
    dp_param = InputParam(model_checkpoint["json_file"], "train".upper())
    dp_param.inference = True # set the model_load_file and inference, then the model will load from model_load_file path
    # dp_model_param = dp_param.get_dp_net_dict()
    dp_trainer = dp_network(dp_param)
    # Step 3. 初始化 DP model
    model = dp_trainer.load_model_with_ckpt(davg=stat[0], dstd=stat[1], energy_shift=stat[2])
    # model = DP(config=dp_model_param, davg=stat[0], dstd=stat[1], energy_shift=stat[2])
    model.load_state_dict(model_checkpoint["state_dict"])
    if dp_param.dp_params.descriptor.type_embedding:
        dp_log = "Type embedding DP model" 
    else:
        dp_log = "DP model"
    if "compress" in model_checkpoint.keys():
        model.set_comp_tab(model_checkpoint["compress"])
        dp_log += " with compress dx = {}".format(model_checkpoint["compress"]["dx"])
    # Step 4. 
    torch_script_module = torch.jit.script(model)
    torch_script_path = os.path.dirname(os.path.abspath(ckpt_file))
    torch_script_module.save(torch_script_path + "/torch_script_module.pt")# need to change as dp_module
    # the full out will be 'Type Eembeding Dp model with compress dx = 0.001'
    print("Tracing {} successfully! The torch script module is saved in {}".format(dp_log, torch_script_path + "/torch_script_module.pt"))

def script_nep_model(model_checkpoint, ckpt_file):
    energy_shift = model_checkpoint["energy_shift"]
    # Step 2.
    model_checkpoint["json_file"]["model_load_file"] = ckpt_file #the model will reload from this path
    model_checkpoint["json_file"]["datasets_path"] = []
    model_checkpoint["json_file"]["optimizer"] = {}
    nep_param = InputParam(model_checkpoint["json_file"], "train".upper())
    nep_param.inference = True
    nep_trainer = nep_network(nep_param)
    # Step 3. 初始化 NEP model
    model, optimizer = nep_trainer.load_model_optimizer(energy_shift)
    # Step 4. 
    torch_script_module = torch.jit.script(model)
    torch_script_path = os.path.dirname(os.path.abspath(ckpt_file))
    torch_script_module.save(torch_script_path + "/jit_nep_module.pt")
    print("Tracing NEP model successfully! The torch script module is saved in {}".format(torch_script_path + "/jit_nep_module.pt"))    