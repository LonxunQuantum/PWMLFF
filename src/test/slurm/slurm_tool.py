import subprocess
import os

def get_job_ids():
    # 使用 squeue 获取当前用户的所有作业ID列表
    user = os.getlogin()  # 获取当前登录的用户名
    result = subprocess.run(['squeue', '-u', user, '-o', '%i', '-h'], capture_output=True, text=True)
    job_ids = result.stdout.strip().split()
    return job_ids

def get_slurm_script_path(job_id):
    # 使用 scontrol 显示作业详细信息，并从中提取 slurm 脚本路径
    result = subprocess.run(['scontrol', 'show', 'job', str(job_id)], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "Command=" in line:
            script_path = line.split('=')[1]
            return script_path
    return None

def get_jobs(work_dir:str):
    job_ids = get_job_ids()
    jobs = []
    for job_id in job_ids:
        script_path = get_slurm_script_path(job_id)
        # if script_path:
        #     print(f"Job ID: {job_id}, Slurm Script Path: {script_path}")
        # else:
        #     print(f"Job ID: {job_id}, Slurm Script Path: Not Found")
        if work_dir in script_path:
            jobs.append(job_id)
    return jobs
        

# if __name__ == "__main__":
#     get_jobs("run_iter/iter.0000/temp_run_iter_work/label/scf")