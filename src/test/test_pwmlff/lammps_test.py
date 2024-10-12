import os
import json
from test_util.file_operation import copy_file, write_to_file
from test_pwmlff.param_extract import LmpsParam, Resource, LmpsInput
from slurm.slurm_script import set_slurm_script_content
from slurm.slurm import Mission, SlurmJob

def do_lammps_test(input_dict:dict):
    input_param = LmpsParam(input_dict)
    slurm_list = make_lmps_works(input_param)
    do_slurm_jobs(slurm_list)
    success_test, failed_test = check_lmps_slurm_jobs(slurm_list)
    print("Successful works:")
    print(success_test)
    print("Failed works:")
    print(failed_test)
    json.dump(success_test, open(os.path.join(input_param.work_dir, "success_lmps_test.json"), "w"), indent=4)
    json.dump(failed_test,  open(os.path.join(input_param.work_dir, "failed_lmps_test.json"), "w"), indent=4)

def make_lmps_works(input_parm:LmpsParam):
    slurm_list = []
    for idx, work in enumerate(input_parm.work_list):
        lmps_input, resource = work
        #set work dir
        env_type = resource.env_type
        model_type = lmps_input.model_type
        lmps_dir = "{}_{}_{}".format(idx, env_type, model_type)
        work_dir = os.path.join(input_parm.work_dir, lmps_dir)
        # copy file to work_dir
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        device_type = "gpu" if resource.gpu_per_node > 0 else "cpu"
        if model_type == "DP":
            # if env is cpu
            dp_slurms = make_dp_lmps_dirs(work_dir, lmps_input, resource=resource, device_type=device_type)
            slurm_list.extend(dp_slurms)
        elif model_type == "NEP":
            # copy files
            # make traj dir
            if not os.path.exists(os.path.join(work_dir, "traj")):
                os.makedirs(os.path.join(work_dir, "traj"))
            for file in lmps_input.files:
                target_file = os.path.join(work_dir, os.path.basename(file))
                copy_file(file, target_file)
            cmd = make_lmps_cmd(resource.gpu_per_node, resource.cpu_per_node)
            # make slurm 
            jobname = lmps_dir
            group_slurm_script = set_slurm_script_content(gpu_per_node=resource.gpu_per_node, 
                    number_node = resource.number_node, 
                    cpu_per_node = resource.cpu_per_node,
                    queue_name = resource.queue_name,
                    custom_flags = resource.custom_flags,
                    env_script = resource.env_script,
                    job_name = jobname,
                    run_cmd_template = cmd,
                    work_dir = lmps_dir,
                    task_tag = "lmps.success", 
                    task_tag_faild = "lmps.failed",
                    parallel_num=1,
                    check_type=None
                    )
            slurm_script_name = "lmps.job"
            slurm_job_file = os.path.join(work_dir, slurm_script_name)
            write_to_file(slurm_job_file, group_slurm_script, "w")
            slurm_list.append(slurm_job_file)
    return slurm_list

def make_lmps_cmd(gpu_per_node:int, cpu_per_node:int):
    if gpu_per_node is not None and gpu_per_node > 0:
        cmd = "mpirun -np {} lmp_mpi_gpu -in lmp.in".format(gpu_per_node)
    else:
        cmd = "mpirun -np {} lmp_mpi -in lmp.in".format(cpu_per_node)
    return cmd

def make_dp_lmps_dirs(work_dir:str, input_parm:LmpsInput, resource:Resource, device_type:str):
    pts = []
    lmps_file = []
    slurm_list = []
    for file in input_parm.files:
        if "lmp.in" in file:
            lmp_in = file
        if '.pt' not in file:
            lmps_file.append(file)
        if device_type == "cpu":
            if ".pt" in file and 'cpu' in file:
                pts.append(file)
        else:
            if ".pt" in file and 'gpu' in file:
                pts.append(file)
    for pt in pts:
        tmp_work_dir = os.path.join(work_dir, os.path.basename(pt).split(".")[0])
        if not os.path.exists(tmp_work_dir):
            os.makedirs(tmp_work_dir)
        for file in lmps_file:
            target_file = os.path.join(tmp_work_dir, os.path.basename(file))
            copy_file(file, target_file)
        # copy pt file
        copy_file(pt, os.path.join(tmp_work_dir, os.path.basename(pt)))
        # make traj dir
        if not os.path.exists(os.path.join(tmp_work_dir, "traj")):
            os.makedirs(os.path.join(tmp_work_dir, "traj"))
        # set lmp.in
        set_lmp_in(lmp_in, os.path.basename(pt), os.path.join(tmp_work_dir, "lmp.in"))
        # make slurm 
        jobname = os.path.basename(tmp_work_dir)
        cmd = make_lmps_cmd(resource.gpu_per_node, resource.cpu_per_node)
        group_slurm_script = set_slurm_script_content(gpu_per_node=resource.gpu_per_node, 
                number_node = resource.number_node, 
                cpu_per_node = resource.cpu_per_node,
                queue_name = resource.queue_name,
                custom_flags = resource.custom_flags,
                env_script = resource.env_script,
                job_name = jobname,
                run_cmd_template = cmd,
                work_dir = tmp_work_dir,
                task_tag = "lmps.success", 
                task_tag_faild = "lmps.failed",
                parallel_num=1,
                check_type=None
                )
        slurm_script_name = "lmps.job"
        slurm_job_file = os.path.join(tmp_work_dir, slurm_script_name)
        write_to_file(slurm_job_file, group_slurm_script, "w")
        slurm_list.append(os.path.join(tmp_work_dir, slurm_script_name))
    return slurm_list

def set_lmp_in(lmp_in:str, pt_name:str, save_path:str):
    # read lmp.in and set pairstyle
    with open(lmp_in, 'r') as rf:
        lines = rf.readlines()
    start_idx = 0
    while(start_idx < len(lines)):
        if "pair_style" in lines[start_idx].lower() \
            and "pwmlff"  in lines[start_idx].lower():
            break
        start_idx += 1
    new_line = lines[:start_idx]
    new_line.append("pair_style   pwmlff   1 {}\n".format(pt_name))
    new_line.extend(lines[start_idx+1:])
    with open(save_path, 'w') as wf:
        wf.writelines(new_line)
        
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

def check_lmps_slurm_jobs(slurm_files):
    success_list = {}
    failed_list = {}
    for slurm_file in slurm_files:
        error_dict = {}
        log_lmps = os.path.join(os.path.dirname(slurm_file), "log.lammps")
        if not os.path.exists(log_lmps):
            error_dict["log.lammps"] = "not exists"
        else:
            with open(log_lmps, 'r') as rf:
                line = rf.readlines()[-1]
            if "Total wall time".lower() not in line.lower():
                error_dict["log.lammps"] = "running error, last log: {}".format(line)
        if len(error_dict.keys()) > 0:
            failed_list[slurm_file] = error_dict
        else:
            success_list[slurm_file] = "success"
    return success_list, failed_list