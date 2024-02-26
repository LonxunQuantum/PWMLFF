from src.user.input_param import InputParam
from src.pre_data.nep_gen_data import convert_mvmfiles_to_xyz
import os, shutil
import random

class NepNetwork:
    def __init__(self, input_param:InputParam):
        self.input_param = input_param
        if self.input_param.seed is not None:
            random.seed(self.input_param.seed)
            
    def generate_data(self):
        # movements to extended xyz format
        convert_mvmfiles_to_xyz(self.input_param.file_paths.raw_path, 
                                os.getcwd(),
                                self.input_param.file_paths.nep_train_xyz_path, 
                                self.input_param.file_paths.nep_test_xyz_path, 
                                self.input_param.valid_shuffle, 
                                self.input_param.train_valid_ratio,
                                self.input_param.seed)
        if not os.path.exists(self.input_param.file_paths.nep_train_xyz_path):
            raise Exception("ERROR! MOVEMENTs to extended xyz format failed! Please check!")
        
    def train(self):
        # set nep.in file
        self.input_param.nep_param.to_nep_in_file(self.input_param.file_paths.nep_in_file)
        print("nep.in file generated successfully!")

        # run nep job
        cwd = os.getcwd()
        os.chdir(self.input_param.file_paths.work_dir)
        # if does not recover from last trainging, delete nep.txt and relative files
        if self.input_param.recover_train is False:
            os.system("rm *.out nep.restart nep.txt -r")
        result = os.system("nep")
        if result == 0:
            print("nep running successfully!")
        else:
            raise Exception("ERROR! nep run error!")
        os.chdir(cwd)
        # collection?

        # if self.input_param.nep_param.nep_in_file is None\
        #         or not os.path.exists(self.input_param.nep_param.nep_in_file):
        #     # if nep.in file not exist, generate it
        #     self.input_param.nep_param.to_nep_in_file(self.input_param.file_paths.nep_in_file)
        #     print("nep.in file generated successfully!")
        # else:
        #     #copy nep.in file to work_dir
        #     shutil.copy(self.input_param.nep_param.nep_in_file, self.input_param.file_paths.nep_in_file)
        #     print("Copy nep.in file from {} to {} done!".format(self.input_param.nep_param.nep_in_file, 
        #                                                         self.input_param.file_paths.nep_in_file))
        
    def inference(self):
        # set nep.in file
        self.input_param.nep_param.to_nep_in_file(self.input_param.file_paths.nep_in_file)
        print("nep.in file generated successfully!")
        # copy nep.txt file to work dir
        if os.path.exists(self.input_param.file_paths.nep_model_file):
            os.remove(self.input_param.file_paths.nep_model_file)
        shutil.copy(self.input_param.file_paths.model_load_path,
                    self.input_param.file_paths.nep_model_file)
        
        # run nep job
        cwd = os.getcwd()
        os.chdir(self.input_param.file_paths.work_dir)
        # if does not recover from last trainging, delete nep.txt and relative files
        os.system("rm *.out -r")
        result = os.system("nep")
        if result == 0:
            print("nep running successfully!")
        else:
            raise Exception("ERROR! nep run error!")
        os.chdir(cwd)
