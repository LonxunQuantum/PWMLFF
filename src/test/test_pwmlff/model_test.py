import os
import sys
import json
import glob
from test_util.file_operation import del_dir, link_file, copy_file, write_to_file
from test_pwmlff.param_extract import TrainParam
from test_pwmlff.params import Resource, TrainInput
from slurm.slurm_script import set_slurm_script_content
from test_pwmlff.check_model import check_train_slurm_jobs
from slurm.slurm import Mission, SlurmJob

def do_model_test(input_dict:dict):
    input_param = TrainParam(input_dict)
    # for work in input_param.work_list:
    slurm_list = make_train_works(input_param)
    do_slurm_jobs(slurm_list)
    # set slurm jobs
    success_test, failed_test = check_train_slurm_jobs(slurm_list)
    print("Successful works:")
    print(success_test)
    print("Failed works:")
    print(failed_test)
    json.dump(success_test, open(os.path.join(input_param.work_dir, "success_train_test.json"), "w"), indent=4)
    json.dump(failed_test,  open(os.path.join(input_param.work_dir, "failed_train_test.json"), "w"), indent=4)

def make_train_works(input_parm:TrainParam):
    slurm_list = []
    for idx, work in enumerate(input_parm.work_list):
        train_input, resource , epoch = work
        #set work dir
        env_type = resource.env_type
        model_type = train_input.model_type
        optimizer_type = train_input.optimizer

        train_dir = "{}_{}_{}_{}".format(idx, env_type, model_type, optimizer_type)
        work_dir = os.path.join(input_parm.work_dir, train_dir)

        # copy file to work_dir
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        # copy pwdata
        # link_file(train_input.pwdata_dir, os.path.join(input_parm.work_dir, os.path.basename(train_input.pwdata_dir)))
        # set train.json: read json and set epoch, then save to target_dir
        set_json(train_input, epoch, os.path.join(work_dir, "train.json"))
        # set test.json: read json and write to test.json
        make_test_json(os.path.join(work_dir, "train.json"), os.path.join(work_dir, "test.json"))
        # make slurm file with envs
        run_cmd = set_train_cmd(train_input, resource)
        jobname = os.path.basename(train_dir)
        group_slurm_script = set_slurm_script_content(gpu_per_node=resource.gpu_per_node, 
                number_node = resource.number_node, 
                cpu_per_node = resource.cpu_per_node,
                queue_name = resource.queue_name,
                custom_flags = resource.custom_flags,
                env_script = resource.env_script,
                job_name = jobname,
                run_cmd_template = run_cmd,
                work_dir = work_dir,
                task_tag = "train.success", 
                task_tag_faild = "tarin.failed",
                parallel_num=1,
                check_type=None
                )
        slurm_script_name = "train.job"
        slurm_job_file = os.path.join(work_dir, slurm_script_name)
        write_to_file(slurm_job_file, group_slurm_script, "w")
        slurm_list.append(slurm_job_file)
    return slurm_list

def make_test_json(train_file:str, save_path:str):
    train_json = json.load(open(train_file))
    test_dict = {}
    test_dict["model_type"] = train_json["model_type"]
    test_dict["format"] = train_json["format"]
    test_dict["test_data"] = train_json["train_data"]
    
    if test_dict["model_type"].upper() == "LINEAR":
        test_dict["atom_type"] = train_json["atom_type"]
        mvm_file = train_json["train_data"][0]
        md_dir = os.path.join(os.path.dirname(save_path), "MD")
        if not os.path.exists(md_dir):
            os.makedirs(md_dir)
        copy_file(mvm_file, os.path.join(md_dir, "MOVEMENT"))
        test_dict["test_data"] = ["./MD/MOVEMENT"]

    if test_dict["model_type"] == "DP":
        test_dict["model_load_file"] = "./model_record/dp_model.ckpt"
    elif test_dict["model_type"] == "NEP":
        test_dict["model_load_file"] = "./model_record/nep_model.ckpt"
    elif test_dict["model_type"] == "NN":
        test_dict["model_load_file"] = "./model_record/nn_model.ckpt"
    json.dump(test_dict, open(save_path, "w"), indent=4)
    return save_path

def set_json(train_input:TrainInput, epoch:int, save_path:str):
    train_json = json.load(open(train_input.json_file))

    train_json["train_data"] = []

    if epoch is not None:
        if "optimizer" in train_json.keys():
                train_json["optimizer"]["epochs"] = epoch
        else:
            train_json["optimizer"] = {}
            train_json["optimizer"]["epochs"] = epoch
    if len(train_input.train_data) > 0:
        train_json["train_data"] = train_input.train_data
        train_json["format"] = train_input.format
        train_json["valid_data"] = train_input.train_data
    json.dump(train_json, open(save_path, "w"), indent=4)

def set_train_cmd(train:TrainInput, resource:Resource):
    script = ""
    pwmlff = resource.command
    script += "    {} {} {} >> {}\n\n".format(pwmlff, "train", "train.json", "train.out")
    script += "    {} {} {} >> {}\n\n".format(pwmlff, "test", "test.json", "test.out")

    # do nothing for nep model
    if train.model_type == "DP":
        # do script
        if resource.gpu_per_node is not None and resource.gpu_per_node > 0:
            script += "    cd model_record\n"
            script += "    {} compress dp_model.ckpt -d 0.01 -o 3 -s cmp_dp_model\n".format(pwmlff)
            script += "    {} script dp_model.ckpt\n".format(pwmlff)
            script += "    {} script cmp_dp_model.ckpt\n".format(pwmlff)
            script += "    export CUDA_VISIBLE_DEVICES=''\n"
            script += "    {} script dp_model.ckpt\n".format(pwmlff)
            script += "    {} script cmp_dp_model.ckpt\n".format(pwmlff)
            script += "    cd .."
        else:
            script += "    cd model_record\n"
            script += "    export CUDA_VISIBLE_DEVICES=''\n"
            script += "    {} compress dp_model.ckpt -d 0.01 -o 3 -s cmp_dp_model_cpu\n".format(pwmlff)
            script += "    {} script dp_model.ckpt\n".format(pwmlff)
            script += "    {} script cmp_dp_model_cpu.ckpt\n".format(pwmlff)
            script += "    cd .."
        # if input_param.strategy.compress:
        #     script += "    {} {} {} -d {} -o {} -s {}/{} >> {}\n\n".format(pwmlff, MODEL_CMD.compress, model_path, \
        #         input_param.strategy.compress_dx, input_param.strategy.compress_order, TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.compree_dp_name, SLURM_OUT.train_out)
        #     cmp_model_path = "{}/{}".format(TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.compree_dp_name)
        
        # if input_param.strategy.md_type == FORCEFILED.libtorch_lmps:
        #     if resource.explore_resource.gpu_per_node is None or resource.explore_resource.gpu_per_node == 0:
        #         script += "    export CUDA_VISIBLE_DEVICES=''\n"
        #     if cmp_model_path is None:
        #         # script model_record/dp_model.ckpt the torch_script_module.pt will in model_record dir
        #         script += "    {} {} {} {}/{} >> {}\n".format(pwmlff, MODEL_CMD.script, model_path, TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.script_dp_name, SLURM_OUT.train_out)
        #     else:
        #         script += "    {} {} {} {}/{} >> {}\n\n".format(pwmlff, MODEL_CMD.script, cmp_model_path, TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.script_dp_name, SLURM_OUT.train_out)

    return script

def do_slurm_jobs(slurm_files:list[str]):
    # 过滤已经测试完毕的目录
    mission = Mission()
    for i, script_path in enumerate(slurm_files):
        slurm_job = SlurmJob()
        tag_name = "train.success"
        tag = os.path.abspath(os.path.join(os.path.dirname(script_path),tag_name))
        slurm_job.set_tag(tag)
        slurm_job.set_cmd(script_path)
        mission.add_job(slurm_job)
    if len(mission.job_list) > 0:
        mission.commit_jobs()
        mission.check_running_job()
        mission.all_job_finished(error_type=None)
