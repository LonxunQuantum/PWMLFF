#!/usr/bin/env python
import json
import os, sys
from src.user.dp_work import dp_train, gen_dp_feature, dp_test
from src.user.nn_work import nn_train, gen_nn_feature, nn_test
from src.user.linear_work import linear_train, linear_test
from src.user.input_param import help_info
from src.user.active_work import ff2lmps_explore
from utils.json_operation import get_parameter, get_required_parameter
from utils.gen_multi_train import multi_train
from src.user.ckpt_extract import extract_force_field
from src.user.ckpt_compress import compress_force_field
if __name__ == "__main__":
    cmd_type = sys.argv[1].upper()
    # cmd_type = "test".upper()
    # cmd_type = "train".upper()
    # cmd_type = "gen_feat".upper()
    # cmd_type = "multi_train".upper()
    # cmd_type = "explore".upper()
    if cmd_type == "help".upper():
        help_info()
    elif cmd_type == "extract_ff".upper():
        ckpt_file = sys.argv[2]
        extract_force_field(ckpt_file, cmd_type)
    elif cmd_type == "compress".upper():
        ckpt_file = sys.argv[2]
        compress_force_field(ckpt_file, cmd_type)
    else:
        json_path = sys.argv[2]
        # cmd_type = "test".upper()
        
        # json_path = "/data/home/hfhuang/2_MLFF/2-DP/19-json-version/4-CH4-dbg/dp_train_final.json"
        os.chdir(os.path.dirname(os.path.abspath(json_path)))
        json_file = json.load(open(json_path))
        model_type = get_required_parameter("model_type", json_file).upper()  # model type : dp or nn or linear
        model_num = get_parameter("model_num", json_file, 1)
        if model_num > 1 and cmd_type == "train".upper():
            # for multi train, need to input slurm file
            slurm_file = sys.argv[3]
            multi_train(json_path, cmd_type, slurm_file)

        if cmd_type == "train".upper():
            if model_type == "DP".upper():
                dp_train(json_file, cmd_type)
            elif model_type == "NN".upper():
                nn_train(json_file, cmd_type)
            elif model_type == "Linear".upper():
                linear_train(json_file, cmd_type)
            else:
                raise Exception("Error! the model_type param in json file does not existent, you could use DP or NN or Linear")
            
        elif cmd_type == "test".upper():
            if model_type == "DP".upper():
                dp_test(json_file, cmd_type)
            elif model_type == "NN".upper():
                nn_test(json_file, cmd_type)
            elif model_type == "Linear".upper():
                linear_test(json_file, cmd_type)
            else:
                raise Exception("Error! the model_type param in json file does not existent, you could use DP or NN or Linear")
          
        elif cmd_type == "gen_feat".upper():
            if model_type == "DP".upper():
                gen_dp_feature(json_file, cmd_type)
            elif model_type == "NN".upper():
                gen_nn_feature(json_file, cmd_type)
            else:
                raise Exception("Error! the model_type param in json file does not existent, you could use DP or NN or Linear")
        
        # elif cmd_type == "multi_train".upper():
        #     # for multi train, need to input slurm file
        #     slurm_file = sys.argv[3]
        #     multi_train(json_path,slurm_file)

        elif cmd_type == "explore".upper():
            # for now, only support explore for DP model
            ff2lmps_explore(json_file)

        else:
            raise Exception("Error! the cmd type {} does not existent, you could use train or test!".format(cmd_type))
        
