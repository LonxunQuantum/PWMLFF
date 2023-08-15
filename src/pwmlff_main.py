import json
import os, sys
from src.user.dp_work import dp_train, gen_dp_feature
if __name__ == "__main__":
    args = sys.argv()
    cmd_type = args[1]
    json_file = json.load(open(args[2]))
    work_dir = json_file["work_dir"]
    cwd = os.getcwd()
    os.chdir(work_dir)
    
    if cmd_type == "gen_dp_feature":
        pass

    if cmd_type == "gen_nn_feature":
        pass

    if cmd_type == "gen_linear_feature":
        pass

    if cmd_type == "dp_train":
        pass   # do dp training
        
    if cmd_type == "nn_train":
        pass
    
    if cmd_type == "linear_train":
        pass
    
    if cmd_type == "dp_test":
        pass   # do dp testing
        
    if cmd_type == "nn_test":
        pass
    
    if cmd_type == "linear_test":
        pass

    os.chdir(cwd)