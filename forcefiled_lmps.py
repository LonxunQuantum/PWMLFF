#!/usr/bin/env python
import json
import os, sys
from src.mods.adaptive_trainer import adaptive_trainer
from src.user.lmps_param import LmpParam
if __name__ == "__main__":
    # cmd_type = sys.argv[1].upper()
    json_file = sys.argv[1]
    os.chdir(os.path.dirname(os.path.abspath(json_file)))
    json_file = json.load(open(json_file))
    lmps_param = LmpParam(json_file, os.getcwd())
    adpt_trainer = adaptive_trainer(lmps_param)
    #adpt_trainer.initialize() 
    adpt_trainer.explore()
    #adpt_trainer.dbg_explore()  
    #adpt_trainer.run_scf()
    #adpt_trainer.train() 
    #adpt_trainer.lmp_get_err()



    