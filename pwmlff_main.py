import json
import os, sys
from src.user.dp_work import dp_train, gen_dp_feature, dp_test
from src.user.model_param import help_info
from utils.json_operation import get_parameter, get_required_parameter

if __name__ == "__main__":
    cmd_type = sys.argv[1].upper()
    if cmd_type == "help".upper():
        help_info()
    else:
        json_path = sys.argv[2]
        cmd_type = "test".upper()
        
        os.chdir("/data/home/wuxingxing/datas/pwmat_mlff_workdir/lisi/ref_dp_ef/")
        json_path = "lisi_train.json"
        json_file = json.load(open(json_path))
        model_type = get_parameter("model_type", json_file, "DP").upper()  # model type : dp or nn or linear
        
        if cmd_type == "train".upper():
            if model_type == "DP".upper():
                dp_train(json_file, cmd_type)
            elif model_type == "NN".upper():
                pass
            elif model_type == "Linear".upper():
                pass
            else:
                raise Exception("Error! the model_type param in json file does not existent, you could use DP or NN or Linear")
            
        elif cmd_type == "test".upper():
            if model_type == "DP".upper():
                dp_test(json_file, cmd_type)
            elif model_type == "NN".upper():
                pass
            elif model_type == "Linear".upper():
                pass
            else:
                raise Exception("Error! the model_type param in json file does not existent, you could use DP or NN or Linear")

        else:
            raise Exception("Error! the cmd type {} does not existent, you could use train or test!".format(cmd_type))
        