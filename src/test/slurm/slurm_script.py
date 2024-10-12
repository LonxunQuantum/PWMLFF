import glob
import os
from math import ceil
GPU_SCRIPT_HEAD = \
"#!/bin/sh\n\
#SBATCH --job-name={}\n\
#SBATCH --nodes={}\n\
#SBATCH --ntasks-per-node={}\n\
#SBATCH --gres=gpu:{}\n\
#SBATCH --gpus-per-task={}\n\
#SBATCH --partition={}\n\
\
"

CPU_SCRIPT_HEAD = \
"#!/bin/sh\n\
#SBATCH --job-name={}\n\
#SBATCH --nodes={}\n\
#SBATCH --ntasks-per-node={}\n\
#SBATCH --partition={}\n\
\
"

def set_slurm_script_content(
                            number_node, 
                            gpu_per_node, #None
                            cpu_per_node,
                            queue_name,
                            custom_flags,
                            env_script,
                            job_name,
                            run_cmd_template,
                            work_dir,
                            task_tag:str,
                            task_tag_faild:str,
                            parallel_num:int=1,
                            check_type:str=None
                            ):
        # set head
        script = ""
        if gpu_per_node is None or gpu_per_node == 0:
            script += CPU_SCRIPT_HEAD.format(job_name, number_node, cpu_per_node, queue_name)
            script += "export CUDA_VISIBLE_DEVICES=''\n"
        else:
            script += GPU_SCRIPT_HEAD.format(job_name, number_node, cpu_per_node, gpu_per_node, 1, queue_name)
        
        for custom_flag in custom_flags:
            script += custom_flag + "\n"
        
        # set conda env
        script += "\n"
        # script += CONDA_ENV
        # script += "\n"

        script += "echo \"SLURM_SUBMIT_DIR is $SLURM_SUBMIT_DIR\"\n\n"
        script += "echo \"Starting job $SLURM_JOB_ID at \" `date`\n\n"
        script += "echo \"Running on nodes: $SLURM_NODELIST\"\n\n"

        script += "start=$(date +%s)\n"

        # set source_list
        script += env_script
        script += "\n"
        
        job_cmd = ""
        
        # check_info = common_check_success(task_tag, task_tag_faild)
            
        job_cmd += "{\n"
        # job_cmd += "cd {}\n".format(work_dir)
        job_cmd += "{}\n".format(run_cmd_template)
        job_cmd += "    touch {}\n".format(task_tag)
        job_cmd += "} &\n\n"
        job_cmd += "wait\n\n"
        
        script += job_cmd
        script += "echo \"Job $SLURM_JOB_ID done at \" `date`\n\n"

        right_script = ""
        right_script += "end=$(date +%s)\n"
        right_script += "take=$(( end - start ))\n"
        right_script += "echo Time taken to execute commands is ${take} seconds\n"
        right_script += "exit 0\n" 
        # error_script  = "    exit 1\n"
        # script += get_job_tag_check_string(job_tag_list, right_script, error_script)
        script += right_script
        return script

def common_check_success(task_tag:str, task_tag_failed:str):
    script = ""
    script += "    if test $? == 0; then\n"
    script += "        touch {}\n".format(task_tag)
    script += "    else\n"
    script += "        touch {}\n".format(task_tag_failed)
    script += "    fi\n"
    return script

def get_job_tag_check_string(job_tags:list[str], true_script:str="", error_script:str=""):
    script = "if "
    for index, job_tag in enumerate(job_tags):
        script += "[ -f {} ]".format(job_tag)
        if index < len(job_tags)-1:
            script += " && "
    script += "; then\n"
    script += true_script
    script += "else\n"
    script += error_script
    script += "fi\n"
    return script