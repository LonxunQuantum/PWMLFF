import json
import os, sys
from src.user.dp_work import dp_train, gen_dp_feature, dp_test
from src.user.model_param import help_info
if __name__ == "__main__":
    cmd_type = sys.argv[1].upper()
    if cmd_type == "help".upper():
        help_info()
    else:
        json_path = sys.argv[2]
        # cmd_type = "dp_train".upper()
        # json_path = "/data/home/wuxingxing/datas/pwmat_mlff_workdir/lisi/ref_dp_ef/lisi_train.json"
        json_file = json.load(open(json_path))
        work_dir = json_file["work_dir"]
        cwd = os.getcwd()
        os.chdir(work_dir)
        
        # dp interfaces
        if cmd_type == "dp_gen_feat".upper():
            gen_dp_feature(json_file, cmd_type)
        
        if cmd_type == "dp_train".upper():
            dp_train(json_file, cmd_type)
        
        if cmd_type == "dp_test".upper():
            dp_test(json_file, cmd_type)

        if cmd_type == "gen_nn_feature".upper():
            pass

        if cmd_type == "gen_linear_feature".upper():
            pass

    
        if cmd_type == "nn_train".upper():
            pass
        
        if cmd_type == "linear_train".upper():
            pass
        
        if cmd_type == "nn_test".upper():
            pass
        
        if cmd_type == "linear_test".upper():
            pass

        os.chdir(cwd)