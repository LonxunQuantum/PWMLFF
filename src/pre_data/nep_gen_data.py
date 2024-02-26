from utils.random_utils import random_index
import os

from pwdata.movement import MOVEMENT
from pwdata.extendedxyz import save_to_extxyz

'''
description: 
    convert movements to train.xyz and valid.xyz 
param {list} mvm_file_list
param {str} save_dir 
param {str} train_save_path "train.xyz"
param {str} valid_save_path "valid.xyz"
param {bool} valid_shuffle
param {float} ratio
param {int} seed
return {*}
author: wuxingxing
'''
def convert_mvmfiles_to_xyz(mvm_file_list:list, save_dir:str, train_save_path:str, valid_save_path:str, valid_shuffle:bool=False, ratio:float=0.2, seed:int=None):
    # if the save_file exists before, delete it
    if os.path.exists(os.path.join(save_dir, train_save_path)):
        os.remove(os.path.join(save_dir, train_save_path))
    if os.path.exists(os.path.join(save_dir, valid_save_path)):
        os.remove(os.path.join(save_dir, valid_save_path))
        
    for mvm in mvm_file_list:
        image_data = MOVEMENT(mvm)
        image_nums = len(image_data.get())
        if image_nums == 0:
            raise Exception("ERROR! The input movement file is empty file, please check {}".format(mvm))
        train_indexs, valid_indexs = random_index(image_nums, ratio, valid_shuffle, seed)
        train_images = []
        valid_images = []
        for index in train_indexs:
            train_images.append(image_data.image_list[index])
        for index in valid_indexs:
            valid_images.append(image_data.image_list[index])
        
        assert(len(train_images) + len(valid_images)) == len(image_data.get())
        
        save_to_extxyz(train_images, save_dir, train_save_path, write_patthen='a')
        save_to_extxyz(valid_images, save_dir, valid_save_path, write_patthen='a')


