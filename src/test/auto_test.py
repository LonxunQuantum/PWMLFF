import os
import sys
import json

from test_pwmlff.model_test import do_model_test
from test_pwmlff.lammps_test import do_lammps_test
from test_util.json_operation import convert_keys_to_lowercase, get_parameter, get_required_parameter

def cmd_infos():
    cmd_info = ""
    cmd_info += "test_pwmlff lmps_params.json\\nn"
    cmd_info += "test_pwmlff train_params.json"
    print(cmd_info)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    json_file = sys.argv[1]
    json_dict = json.load(open(json_file))
    json_dict = convert_keys_to_lowercase(json_dict)
    json_dict["path_prefix"] = os.path.abspath(json_dict["path_prefix"])
    work_dir = get_parameter("work_dir", json_dict, "./test_workdir")
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    os.chdir(work_dir)
    print("change the workdir to {}".format(os.getcwd()))
    work_type = get_required_parameter("work_list", json_dict)
    if isinstance(work_type, dict):
        work_type = [work_type]
    if "lmps_inputs" in work_type[0].keys():
        # do lammps test
        do_lammps_test(json_dict)
    elif "train_inputs" in work_type[0].keys():
        # do training test
        do_model_test(json_dict)
    else:
        print("ERROR! The input json {} can not be recognized, please check.".format(sys.argv[1]))

if __name__ == "__main__":
    main()
