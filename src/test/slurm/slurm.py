from enum import Enum
from subprocess import Popen, PIPE
import os
import sys
import time
import shutil
from slurm.slurm_tool import get_jobs
class JobStatus (Enum) :
    unsubmitted = 1 #
    waiting = 2 # PD
    running = 3 # R
    terminated = 4
    finished = 5
    unknown = 100
    resubmit_failed = 6
    submit_limit:int = 1

def get_slurm_sbatch_cmd(job_dir:str, job_name:str):
    cmd = "cd {} && sbatch {}".format(job_dir, job_name)
    return cmd

class SlurmJob(object):
    def __init__(self, job_id=None, status=JobStatus.unsubmitted, user=None, name=None, nodes=None, nodelist=None, partition=None) -> None:
        self.job_id = job_id
        self.status = status
        self.user = user
        self.name = name
        self.partition=partition
        self.nodes = nodes
        self.nodelist = nodelist
        self.submit_num = 0
        
    def set_cmd(self, script_path:str):
        #such as "sbatch main_MD_test.sh"
        self.slurm_job_run_dir = os.path.dirname(script_path)
        self.slurm_job_name = os.path.basename(script_path)
        slurm_cmd = get_slurm_sbatch_cmd(self.slurm_job_run_dir, self.slurm_job_name)
        self.submit_cmd = slurm_cmd
    
    '''
    description: 
        the job_type could be:
            cp2k/relax, cp2k/scf, cp2k/aimd, pwmat/relax, pwmat/scf, pwmat/aimd, vasp/relax, vasp/scf, vasp/aimd, lammps
    param {*} self
    param {*} tag
    param {str} job_type
    return {*}
    author: wuxingxing
    '''    
    def set_tag(self, tag, job_type:str=None):
        self.job_finish_tag = tag
        if job_type is not None: # use to determine if the lammps md task has terminated due to "ERROR: there are two atoms too close" reason
            self.job_type = job_type.lower()
        else:
            self.job_type = None

    def submit(self):
        # ret = Popen([self.submit_cmd + " " + self.job_script], stdout=PIPE, stderr=PIPE, shell = True)
        ret = Popen([self.submit_cmd], stdout=PIPE, stderr=PIPE, shell = True)
        stdout, stderr = ret.communicate()
        if str(stderr, encoding='ascii') != "":
            raise RuntimeError (stderr)
        job_id = str(stdout, encoding='ascii').replace('\n','').split()[-1]
        self.job_id = job_id
        self.submit_num += 1
        status = self.update_status()
        print ("# job {} submitted!".format(self.job_id))

    def scancel_job(self):
        ret = Popen (["scancel " + self.job_id], shell=True, stdout=PIPE, stderr=PIPE)
        time.sleep(1)
        stdout, stderr = ret.communicate()
        print("scancel job {}".format(self.job_id))
        # print(str(stderr, encoding='ascii'))

    def update_status(self):
        self.status = self.check_status()
        return self.status

    def check_status_no_tag(self):
        ret = Popen (["squeue --job " + self.job_id], shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = ret.communicate()
        if (ret.returncode != 0) :
            if str("Invalid job id specified") in str(stderr, encoding='ascii') :
                return JobStatus.finished
            else :
                print("status command " + "squeue" + " fails to execute")
                print("erro info: " + str(stderr, encoding='ascii'))
                print("return code: " + str(ret.returncode))
                sys.exit ()
        status_line = str(stdout, encoding='ascii').split ('\n')[-2]
        status_word = status_line.split ()[4]
        if   status_word in ["PD","CF","S"] :
            return JobStatus.waiting
        elif status_word in ["R","CG"] :
            return JobStatus.running
        elif status_word in ["C","E","K","BF","CA","CD","F","NF","PR","SE","ST","TO"] :
            return JobStatus.finished
        elif status_word in ["RH"] : #for job in 'RH' status, scancel the job and return terminated
            self.scancel_job()
            return JobStatus.finished
        else:
            return JobStatus.unknown

    def check_status(self):
        ret = Popen (["squeue --job " + self.job_id], shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = ret.communicate()
        if (ret.returncode != 0) :
            if str("Invalid job id specified") in str(stderr, encoding='ascii') :
                if os.path.exists (self.job_finish_tag) :
                    print("job {} finished: the cmd is {}.".format(self.job_id, self.submit_cmd))
                    return JobStatus.finished
                else :
                    return JobStatus.terminated
            else :
                print("status command " + "squeue" + " fails to execute")
                print("erro info: " + str(stderr, encoding='ascii'))
                print("return code: " + str(ret.returncode))
                sys.exit ()
        status_line = str(stdout, encoding='ascii').split ('\n')[-2]
        status_word = status_line.split ()[4]
        if   status_word in ["PD","CF","S"] :
            return JobStatus.waiting
        elif status_word in ["R","CG"] :
            return JobStatus.running
        elif status_word in ["C","E","K","BF","CA","CD","F","NF","PR","SE","ST","TO"] :
            if os.path.exists (self.job_finish_tag) :
                print("job {} finished: the cmd is {}.".format(self.job_id, self.submit_cmd))
                return JobStatus.finished
            else :
                # for lammps md job, if the job stops because of 'ERROR: there are two atoms too close', set the job.status to finished
                if self.job_type is not None and self.job_type == "lammps":
                    end_normal = self.check_lammps_out_file()
                    if end_normal:
                        with open(self.job_finish_tag, 'w') as wf:
                            wf.writelines("Job done!")
                        print("job {} finished: the cmd is {}.".format(self.job_id, self.submit_cmd))
                        return JobStatus.finished
                return JobStatus.terminated
        elif status_word in ["RH"] : #for job in 'RH' status, scancel the job and return terminated
                self.scancel_job()
                if os.path.exists (self.job_finish_tag) :
                    print("job {} finished: the cmd is {}.".format(self.job_id, self.submit_cmd))
                    return JobStatus.finished
                else:
                    return JobStatus.terminated
        else :
            return JobStatus.unknown

    def running_work(self):
        self.submit()
        while True:
            status = self.check_status()
            if (status == JobStatus.waiting) or \
                (status == JobStatus.running):
                time.sleep(10)
            else:
                break
        
        assert(status == JobStatus.finished)
        return status

    def get_slurm_works_dir(self):
        with open(os.path.join(self.slurm_job_run_dir, self.slurm_job_name), 'r') as rf:
            lines = rf.readlines()
        work_dir_list = []
        for line in lines:
            if 'cd ' in line:
                work_dir = line.split()[-1].strip()
                work_dir_list.append(work_dir)
        return work_dir_list

    '''
    description: 
        if the job is md task, and stoped because of 'ERROR: there are two atoms too close' let the task end normally
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def check_lammps_out_file(self):
        # read last line of md.log file
        md_dirs = self.get_slurm_works_dir()
        try:
            for md_dir in md_dirs:
                tag_md_file = os.path.join(md_dir, "tag.md.success")
                md_log = os.path.join(md_dir, "md.log")
                if os.path.exists(tag_md_file):
                    continue
                if not os.path.exists(md_log):
                    return False

                with open(md_log, "rb") as file:
                    file.seek(-2, 2)  # 定位到文件末尾前两个字节
                    while file.read(1) != b'\n':  # 逐字节向前查找换行符
                        file.seek(-2, 1)  # 向前移动两个字节
                    last_line = file.readline().decode().strip()  # 读取最后一行并去除换行符和空白字符
                if "ERROR: there are two atoms" in last_line:
                    with open(tag_md_file, 'w') as wf:
                        wf.writelines("ERROR: there are two atoms too close")
                    return True
                elif "Total wall time" in last_line:
                    with open(tag_md_file, 'w') as wf:
                        wf.writelines("Job Done!")
                    return True
                else:
                    return False
            return True
        except Exception as e:
            return False


class Mission(object):
    def __init__(self, mission_id=None) -> None:
        self.mission_id = mission_id
        self.job_list: list[SlurmJob]= []
    
    def add_job(self, job:SlurmJob):
        self.job_list.append(job)

    def pop_job(self, job_id):
        del_job, index = self.get_job(job_id)
        self.job_list.remove(del_job)

    def get_job(self, job_id):
        for i, job in enumerate(self.job_list):
            if job.job_id == job_id:
                return job, i

    def update_job_state(self, job_id, state):
        up_job, index = self.get_job(job_id)
        up_job.status = state
        self.job_list[index] = up_job
    
    def get_running_jobs(self):
        job_list: list[SlurmJob] = []
        for job in self.job_list:
            if (job.status == JobStatus.waiting) or (job.status == JobStatus.running):
                job_list.append(job)
        return job_list

    def move_slurm_log_to_slurm_work_dir(self, slurm_log_dir_source:str):
        for job in self.job_list:
            slurm_log_source = os.path.join(slurm_log_dir_source, "slurm-{}.out".format(job.job_id))
            slurm_job_log_target = os.path.join(os.path.dirname(job.slurm_job_path), os.path.basename(slurm_log_source))
            if os.path.exists(slurm_log_source):
                shutil.move(slurm_log_source, slurm_job_log_target)
                
    '''
    Description: 
    job_finish_tag does not exist means this job running in error 
    param {*} self
    Returns: 
    Author: WU Xingxing
    '''
    def get_error_jobs(self):
        job_list: list[SlurmJob] = []
        for job in self.job_list:
            if os.path.exists(job.job_finish_tag) is False:
                job_list.append(job)
        return job_list
    
    def all_job_finished(self, error_type:str=None):
        error_jobs = self.get_error_jobs()
        if len(error_jobs) >= 1:
            error_log_content = ""
            for error_job in error_jobs:
                error_log_path = os.path.join(error_job.slurm_job_run_dir, "slurm-{}.out".format(error_job.job_id))
                error_log_content += "JOB ERRIR! The cmd '{}' failed!\nFor more details on errors, please refer to the following documents:\n"\
                    .format(error_job.submit_cmd)

                slurm_content = "    Slurm script file is {}\n    The slurm log is {}\n"\
                    .format(os.path.join(error_job.slurm_job_run_dir, error_job.slurm_job_name), error_log_path)

                tmp_error = None
                if error_type is not None:
                    work_dirs = error_job.get_slurm_works_dir()
                    if len(work_dirs) > 0:
                        tmp_error = "    Task logs under this slurm job:\n"
                        for _ in work_dirs:
                            job_error_log  = "{}/{}".format(_, error_type)
                            job_finish_tag = "{}/{}".format(_, error_job.job_finish_tag)
                            if os.path.exists(job_error_log) and not os.path.exists(job_finish_tag):
                                tmp_error += "        {}\n".format(job_error_log)

                error_log_content += slurm_content
                if tmp_error is not None:
                    error_log_content += tmp_error
                error_log_content += "\n\n"
            print(error_log_content)
            # raise Exception(error_log_content)
        return True
    
    def commit_jobs(self):
        for job in self.job_list:
            if job.status == JobStatus.unsubmitted:
                job.submit()
    
    '''
    description: 
        return all job ids, the job id is the slurm job id
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def get_all_job_ids(self):
        job_id_list = []
        for job in self.job_list:
            job_id_list.append(job.job_id)
        return job_id_list
    
    def check_running_job(self):
        while True:
            for idx, job in enumerate(self.job_list):
                # print(job.status, job.job_id)
                if job.status == JobStatus.resubmit_failed or job.status == JobStatus.finished: # For job resubmitted more than 3 times, do not check again
                    # print("the job success, job_id={}, job_cmd={}".format(job.job_id, job.submit_cmd))
                    continue
                status = job.check_status()
                self.update_job_state(job.job_id, status)
            # if the job failed, resubmit it until the resubmit time more than 3 times
            # self.resubmit_jobs()
            if len(self.get_running_jobs()) == 0:
                break
            time.sleep(10)
        # error_jobs = self.get_error_jobs()
        # if len(error_jobs) > 0:
        #     error_info = "job error: {}".format([_.job_id for _ in error_jobs])
        #     raise Exception(error_info)
        return True
    
    def resubmit_jobs(self):
        for job in self.job_list:
            if job.status == JobStatus.terminated:
                if job.submit_num <= JobStatus.submit_limit.value:
                    print("resubmit job {}: {}, the time is {}\n".format(job.job_id, job.submit_cmd, job.submit_num))
                    job.submit()
                else:
                    job.status = JobStatus.resubmit_failed                    
                
    '''
    Description: 
    after some jobs finished with some jobs terminated, we should try to recover these terminated jobs.
    param {*} self
    Returns: 
    Author: WU Xingxing
    '''
    def re_submmit_terminated_jobs(self):
        error_jobs = self.get_error_jobs()
        if len(error_jobs) == 0:
            return
        self.job_list.clear()
        self.job_list.extend(error_jobs)
        self.reset_job_state()
        self.commit_jobs()
        self.check_running_job()

    def reset_job_state(self):
        for job in self.job_list:
            job.status == JobStatus.unsubmitted

def scancle_job(work_dir:str):
    job_id_list = get_jobs(work_dir)
    print("the job to be scancelled is:")
    print(job_id_list)
    for job_id in job_id_list:
        job = SlurmJob(job_id=job_id)
        status = job.check_status_no_tag()#get status
        if status == JobStatus.waiting or status == JobStatus.running: # is running 
            job.scancel_job()
            # time.sleep(2)
            # status = job.check_status_no_tag()
            # if JobStatus.finished == status:
            #     print("scancel job {} successfully\n\n".format(job_id))
            # else:
            #     print("Scancel job {} failed, Please manually check and cancel this task!\n\n".format(job_id))
    time.sleep(5)
    for job_id in job_id_list:
        job = SlurmJob(job_id=job_id)
        status = job.check_status_no_tag()#get status
        if JobStatus.finished == status:
            print("scancel job {} successfully".format(job_id))
        else:
            print("Scancel job {} failed, Please manually check and cancel this task!\n".format(job_id))

