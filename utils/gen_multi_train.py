#!/usr/bin/env python
import json
import os, random, shutil, sys
import subprocess
from utils.file_operation import smlink_file
from utils.json_operation import get_parameter

"""
This script generates multiple training jobs for PWmatMLFF. It takes a json file and a slurm file as input, and generates multiple training jobs based on the number of models specified in the json file. The script first generates features using the PWMLFF gen_feat command, and then trains each model using the PWMLFF train command. The script also modifies the json file for each model to specify the working directory, model store directory, and forcefield directory. The script uses subprocess to submit jobs to the slurm scheduler.
"""

def multi_train(json_path,slurm_file):
    # os.chdir("/data/home/hfhuang/2_MLFF/2-DP/19-json-version/3-Si/01.Iter1/00.train")
    # json_path = "dp.json"
    # slurm_file = "slurm.sh"
    # json_path = sys.argv[1]
    # slurm_file = sys.argv[2]
    json_dir = os.getcwd()
    command_gen_feat = "PWMLFF gen_feat " + json_path
    command_submit_gen_feat = "sbatch gen_slurm.sh"
    command_submit_train = "sbatch {}/train_slurm.sh".format(json_dir)

    json_file = json.load(open(json_path))
    model_num = json_file['model_num']

    shutil.copy(slurm_file, "gen_slurm.sh")

    with open("gen_slurm.sh", 'r+') as f:
        lines = f.readlines()
        if command_gen_feat + '\n' not in lines:
            f.write(command_gen_feat + '\n')


    # 提交gen_feat任务
    proc = subprocess.Popen(command_submit_gen_feat, shell=True, stdout=subprocess.PIPE)
    out, err = proc.communicate()
    # 获取gen_feat的jobid
    gen_feat_jobid = out.split()[-1].decode('utf-8')

    for i in range(model_num):
        json_file = json.load(open(json_path))
        tmp_work_dir = "{:03d}".format(i)
        tmp_work_dir = os.path.join(os.path.join(json_dir, json_file['work_dir']), tmp_work_dir)
        tmp_json_path = json_path + "_{}".format(i)
        tmp_json_path = os.path.join(json_dir, os.path.basename(tmp_json_path))

        shutil.copy(slurm_file, "train_slurm.sh")
        command_train = "PWMLFF train " + tmp_json_path
        with open("train_slurm.sh", 'r+') as f:
            lines = f.readlines()
            f.write("#SBATCH --dependency=afterok:{}\n".format(gen_feat_jobid))
            if command_train + '\n' not in lines:
                f.write(command_train + '\n')

        os.makedirs(tmp_work_dir, exist_ok=True)
        
        shutil.copy(json_path, tmp_json_path)
        seed = random.randint(1,1000000)
        command_seed = "sed -i 's/\"seed\":[0-9]*/\"seed\":{}/g' {}".format(seed, tmp_json_path)
        os.system(command_seed)

        feature_path = os.path.join(os.path.join(json_dir, json_file['work_dir']), "feature")
        tmp_feature_path = os.path.join(tmp_work_dir, "feature")
        smlink_file(feature_path,tmp_feature_path)

        json_file = json.load(open(tmp_json_path))
        json_file['work_dir'] = tmp_work_dir
        model_store_dir = get_parameter("model_store_dir", json_file, "model_record")
        model_store_dir = os.path.join(tmp_work_dir, model_store_dir)
        forcefield_dir = get_parameter("forcefield_dir", json_file, "forcefield")
        forcefield_dir = os.path.join(tmp_work_dir, forcefield_dir)
        json_file['model_store_dir'] = model_store_dir + "_{}".format(i)
        json_file['forcefield_dir'] = forcefield_dir + "_{}".format(i)
        # Save the modified JSON data back to a file within tmp_work_dir
        with open(tmp_json_path, 'w') as f:
            json.dump(json_file, f, indent=4)

        # train model
        # os.chdir(tmp_work_dir)
        proc = subprocess.Popen(command_submit_train, shell=True, stdout=subprocess.PIPE)
        out, err = proc.communicate()
        train_jobid = out.split()[-1].decode('utf-8')
        
        # 在循环的下一次迭代之前，等待当前的train任务完成
        # f.write("#SBATCH --dependency=afterok:{}\n".format(gen_feat_jobid))