from src.user.input_param import InputParam
from src.pre_data.nep_gen_data import convert_mvmfiles_to_xyz, convert_to_xyz
import os, shutil
import random

class NepNetwork:
    def __init__(self, input_param:InputParam):
        self.input_param = input_param
        if self.input_param.seed is not None:
            random.seed(self.input_param.seed)
    
    '''
    to nep extend xyz input file
    the input could be raw_files or pwmlff/npy files
    for pwmlff/npy files, the file structure could be (/the/paths/train /the/paths/valid) or (the/paths upper npy files)
    
    for train:
        convert to train.xyz and test.xyz
    
    for inference:
        convert to test.xyz
    description: 
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def generate_data(self):
        is_valid = True if self.input_param.inference else False
        is_append = False

        if len(self.input_param.file_paths.raw_path) == 0 and\
            len(self.input_param.file_paths.datasets_path) == 0:
            print("There is not configs need to be converted!")
            return
        if len(self.input_param.file_paths.raw_path) > 0:
            is_append = True
            convert_to_xyz( input_list      =self.input_param.file_paths.raw_path,
                            input_format    =self.input_param.format,
                            save_dir        =os.getcwd(),
                            train_save_path =self.input_param.file_paths.nep_train_xyz_path, 
                            valid_save_path =self.input_param.file_paths.nep_test_xyz_path,
                            valid_shuffle   =self.input_param.valid_shuffle,
                            ratio           =self.input_param.train_valid_ratio, 
                            is_valid        =is_valid,
                            is_append       =False,
                            seed            =self.input_param.seed
                            # trainDataPath   =self.input_param.file_paths.trainDataPath, 
                            # validDataPath   =self.input_param.file_paths.validDataPath
            )
        
        if len(self.input_param.file_paths.datasets_path) > 0:
            # the write patten is 'a'
            convert_to_xyz( input_list      =self.input_param.file_paths.datasets_path,
                            input_format    ="pwmlff/npy",
                            save_dir        =os.getcwd(),
                            train_save_path =self.input_param.file_paths.nep_train_xyz_path, 
                            valid_save_path =self.input_param.file_paths.nep_test_xyz_path,
                            valid_shuffle   =self.input_param.valid_shuffle,
                            ratio           =self.input_param.train_valid_ratio, 
                            is_valid        =is_valid,
                            is_append       =is_append,
                            seed            =self.input_param.seed
                            # trainDataPath   =self.input_param.file_paths.trainDataPath, 
                            # validDataPath   =self.input_param.file_paths.validDataPath
            )

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
        # write nep.in file
        if not os.path.exists(self.input_param.file_paths.test_dir):
            os.makedirs(self.input_param.file_paths.test_dir)
        target_nep_in = os.path.join(self.input_param.file_paths.test_dir, self.input_param.file_paths.nep_in_file)
        self.input_param.nep_param.to_nep_in_file(target_nep_in)
        print("nep.in file generated successfully!")
        # copy nep.txt file to work dir
        target_nep_txt = os.path.join(self.input_param.file_paths.test_dir, self.input_param.file_paths.nep_model_file)

        shutil.copy(self.input_param.file_paths.model_load_path, target_nep_txt)
        # copy train.xyz file to work dir

        target_train_txt = os.path.join(self.input_param.file_paths.test_dir, self.input_param.file_paths.nep_train_xyz_path)
        shutil.copy(os.path.join(os.getcwd(), self.input_param.file_paths.nep_train_xyz_path), target_train_txt)
                
        # run nep job
        cwd = os.getcwd()
        os.chdir(self.input_param.file_paths.test_dir)
        # if does not recover from last trainging, delete nep.txt and relative files
        os.system("rm *.out -r")
        result = os.system("nep")
        if result == 0:
            print("nep inference running successfully!")
        else:
            raise Exception("ERROR! nep run error!")
        os.chdir(cwd)
