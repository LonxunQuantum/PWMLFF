import os
import glob
import sys
import json

from test_util.file_operation import copy_dir, copy_file, link_file
from slurm.slurm import SlurmJob, Mission
from test_util.json_operation import get_required_parameter

def check_train_slurm_jobs(slurm_files:list[str]):
    # check epochs
    # get moddel type
    # read from std_json
    success_test = {}
    failed_test = {}
    for slurm_file in slurm_files:
        failed_check  = {}
        work_dir = os.path.dirname(slurm_file)
        work_dir_name = os.path.basename(work_dir)
        if not os.path.exists(os.path.join(work_dir, "std_input.json")):
            failed_check[work_dir_name] = "no std_input.json"
            continue
        std_json = json.load(open(os.path.join(work_dir, "std_input.json")))
        model_type = get_required_parameter("model_type", std_json)
        if model_type.lower() == "LINEAR".lower():
            if os.path.exists(os.path.join(work_dir, "test_result/evaluation_plots.png")):
                success_test[work_dir_name] = True
            else:
                failed_test[work_dir_name] = "Linear Failed"
            continue
        epochs = get_required_parameter("epochs", std_json["optimizer"])
        res1, check1 = check_ckpt_model(model_record=os.path.join(work_dir, "model_record"), model_type=model_type)
        res2, check2 = check_epoch(epoch_valid=os.path.join(work_dir, "model_record/epoch_valid.dat"), epochs=epochs)
        res3, check3 = check_test_result(test_result=os.path.join(work_dir, "test_result"), model_type=model_type)
        if check1 and check2 and check3:
            success_test[work_dir_name] = True
        else:
            if not check1:
                failed_check["model"] = res1
            if not check2:
                failed_check["epoch"] = res2
            if not check3:
                failed_check["test"] = res3
            if len(failed_check.keys()) > 0:
                failed_test[work_dir_name] = failed_check
        
    return success_test, failed_test

def check_epoch(epoch_valid:str, epochs:int):
    epoch_check = {}
    if not os.path.exists(epoch_valid):
        epoch_check["has_file"]= False
    else:
        try:
            with open(epoch_valid, 'r') as rf:
                line = rf.readlines()[-1]
                epoch = int(line.split()[0])
                if epoch < epochs:
                    epoch_check["less_epoch"]= "trained {} need {}".format(epoch, epochs)
        except:
            epoch_check["has_epoch"]= False
    if len(epoch_check.keys()) > 0:
        return epoch_check, False
    else:
        return epoch_check, True

def check_ckpt_model(model_record:str, model_type):
    model_check = {}
    if len(glob.glob(os.path.join(model_record, "*.ckpt"))) == 0:
        model_check["model_exists"]=False
    else:
        if model_type == "DP":
            if len(glob.glob(os.path.join(model_record, "jit_dp_cpu.pt"))) == 0:
                model_check["jit_model_exists"]=False
            # if not os.path.exists(work_dir, "")
        if model_type == "NEP":
            if not os.path.exists(os.path.join(model_record, "nep_to_lmps.txt")):
                model_check["nep_to_lammps_exists"]=False
        if model_type == "NN" or model_type == "DP":
            if not os.path.exists(os.path.join(os.path.dirname(model_record), "forcefield/forcefield.ff")):
                model_check["forcefield.ff_exists"]=False

    return model_check, False if len(model_check.keys()) > 0 else True

def check_test_result(test_result, model_type):
    test_check = {}
    if not os.path.exists(os.path.join(test_result, "dft_atomic_energy.txt")):
        test_check["dft_atomic_energy_exists"]=False
    if not os.path.exists(os.path.join(test_result, "inference_force.txt")):
        test_check["inference_force_exists"]=False
    if not os.path.exists(os.path.join(test_result, "dft_force.txt")):
        test_check["txt_exists"]=False
    if not os.path.exists(os.path.join(test_result, "inference_loss.csv")):
        test_check["inference_loss_exists"]=False
    if not os.path.exists(os.path.join(test_result, "dft_total_energy.txt")):
        test_check["dft_total_energy_exists"]=False
    if not os.path.exists(os.path.join(test_result, "inference_summary.txt")):
        test_check["inference_summary_exists"]=False
    if not os.path.exists(os.path.join(test_result, "image_atom_nums.txt")):
        test_check["image_atom_nums_exists"]=False
    if not os.path.exists(os.path.join(test_result, "inference_total_energy.txt")):
        test_check["inference_total_energy_exists"]=False
    if not os.path.exists(os.path.join(test_result, "inference_atomic_energy.txt")):
        test_check["inference_atomic_energy_exists"]=False

    return test_check, False if len(test_check.keys()) > 0 else True



