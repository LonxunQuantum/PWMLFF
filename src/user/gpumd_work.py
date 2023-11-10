import os, shutil
from utils.file_operation import copy_file
from src.user.gpumd_param import GPUmdParam
from utils.mvm2xyz import atomconfig2xyz, POSCAR_OUTCAR2xyz
class GPUMD(object):
    def __init__(self, input_param: GPUmdParam) -> None:
        self.input_param = input_param

    def run_md(self):
        # set work dir 
        self.set_work_files()
        # run work
        cwd = os.getcwd()
        os.chdir(self.input_param.working_dir)
        result = os.system("gpumd")
        if result == 0:
            print("gpumd running successfully!")
        else:
            raise Exception("ERROR! gpumd run error!")
        os.chdir(cwd)
        # collect result

    def set_work_files(self):
        if os.path.exists(self.input_param.working_dir):
            shutil.rmtree(self.input_param.working_dir)
        os.makedirs(self.input_param.working_dir)
        # copy files to work dir
        copy_file(self.input_param.potential_file, 
                  os.path.join(self.input_param.working_dir, os.path.basename(self.input_param.potential_file)))
        copy_file(self.input_param.run_in_file, 
                  os.path.join(self.input_param.working_dir, os.path.basename(self.input_param.run_in_file)))
        # if the md init file is not xyz format, convert it
        self.copy_md_xyz_file(self.input_param.md_init_config_file, os.path.join(self.input_param.working_dir, "model.xyz"))

    def copy_md_xyz_file(self, source_file:str, target_file:str):
        if "config".upper() in self.input_param.md_init_config_file.upper():
            # atom.config to xyz format
            atomconfig2xyz(source_file, target_file)

        elif "outcar".upper() in self.input_param.md_init_config_file.upper():
            POSCAR_OUTCAR2xyz(source_file, target_file, "OUTCAR")

        elif "poscar".upper() in self.input_param.md_init_config_file.upper():
            POSCAR_OUTCAR2xyz(source_file, target_file, "POSCAR")

        elif "xyz".upper() in self.input_param.md_init_config_file.upper():
            # copy xyz file
            copy_file(source_file, target_file)

        