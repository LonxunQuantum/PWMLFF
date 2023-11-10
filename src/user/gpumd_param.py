import json
import os 

from utils.json_operation import get_parameter, get_required_parameter

class GPUmdParam(object):
    '''
    description: 
        check files 
    param {*} self
    param {json} lmp_json
    param {str} working_dir
    return {*}
    author: wuxingxing
    '''    
    def __init__(self, lmp_json:json) -> None:
        # work dir, init_config, run.in file, potential.txt
        self.user_dir = os.getcwd()
        self.working_dir = get_parameter("working_dir", lmp_json,  "gpumd_work_dir")
        if os.path.realpath(self.user_dir) == os.path.realpath(self.working_dir):
            self.working_dir = os.path.join(self.user_dir, "work_dir")
            print("The work dir in input json file is same as json file dir, change is as {}".format(self.working_dir))

        self.run_in_file = get_required_parameter("run_in_file", lmp_json)
        if not os.path.exists(self.run_in_file):
            raise Exception("run.in file not exist, please check the dir {}".format(os.path.abspath(self.run_in_file)))
        
        self.potential_file = get_required_parameter("potential_file", lmp_json)
        if not os.path.exists(self.run_in_file):
            raise Exception("potential file not exist, please check the dir {}".format(os.path.abspath(self.potential_file)))
        
        self.md_init_config_file = get_required_parameter("md_init_config", lmp_json)
        if not os.path.exists(self.run_in_file):
            raise Exception("md init position file not exist, please check the dir {}".format(os.path.abspath(self.md_init_config_file)))
        

class RunIn(object):
    def __init__(self) -> None:
        pass

    
