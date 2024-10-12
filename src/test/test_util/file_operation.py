import os
import subprocess
import glob
import shutil
import re
import json
import numpy as np
import random

'''
description: 
save json_dict to save_path, if file dir is not exist, create it.
param {dict} json_dict
param {*} save_path
return {*}
author: wuxingxing
'''
def save_json_file(json_dict:dict, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    json.dump(json_dict, open(save_path, "w"), indent=4)

"""
@Description :
读取最后一行文件
@Returns     :
@Author       :wuxingxing
"""
def file_read_last_line(file_path, type_name="int"):
    if os.path.exists(file_path):
        with open(file_path, 'r') as rf:
            last_line = rf.readlines()[-1]  #the last line
            if '[]' in last_line:
                return []
            last_line = last_line.replace(" ","").split(',')
    if len(last_line) > 0 and type_name == "int":
        last_line = [int(i) for i in last_line]
    if len(last_line) > 0 and type_name == "float":
        last_line = [float(i) for i in last_line]
    return last_line

'''
description: 
param {*} file_path
param {*} data_type: 'float' or 'int'
return {*}
author: wuxingxing
'''
def file_read_lines(file_path, data_type="float"):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as rf:
            lines = rf.readlines()  #the last line
            for line in lines:
                line = re.sub('[\[\]\\n]','',line)
                if len(line) > 0 and data_type == "int":
                    line = [int(i) for i in line.split(',')]
                if len(line) > 0 and data_type == "float":
                    line = [float(i) for i in line.split(',')]
                if len(line) == 1:
                    data.append(line[0])
                else:
                    data.append(line)
    return data

'''
description: 
    load txt data
param {str} file_path
param {*} skiprows
return {*}
author: wuxingxing
'''
def read_data(file_path:str, skiprows=0):
    datas = np.loadtxt(file_path,skiprows=skiprows)
    return datas
    
'''
description: 
 save line str to file_path
param {*} file_path
param {*} line
param {*} mode :'a' or 'w', default is 'a'
return {*}
author: wuxingxing
'''
def write_to_file(file_path, line, mode='w'):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, mode) as wf:
        wf.write(line)

'''
description: 
    merge file list to one file
param {list} file_list
param {str} save_file
return {*}
author: wuxingxing
'''
def merge_files_to_one(file_list:list[str], save_file:str):
    with open(save_file, 'w') as wf:
        for index, file in enumerate(file_list):
            with open(file, 'r') as rf:
                content = rf.read()
            wf.write(content)
            # if index < len(file_list) - 1:
            #     wf.write("\n")
        
'''
description: 
    copy file from source to target
    If follow_symlinks is not set and src is a symbolic link, a new
    symlink will be created instead of copying the file it points to.
param {str} source_file
param {str} target_file
param {bool} follow_symlinks
return {*}
author: wuxingxing
'''
def copy_file(source_file:str, target_file:str, follow_symlinks:bool=True):
    if not os.path.exists(os.path.dirname(target_file)):
        os.makedirs(os.path.dirname(target_file))
    shutil.copyfile(source_file, target_file, follow_symlinks=follow_symlinks)

'''
description: 
    copy dir to target dir, if target dir is exists, delete it
    symlinks is True: for link type files, maintain link format, else copy as file.
param {str} source_dir
param {str} target_dir
param {*} symlinks
return {*}
author: wuxingxing
'''
def copy_dir(source_dir:str, target_dir:str, symlinks=True):
    if os.path.exists(target_dir):
        if os.path.islink(target_dir) or os.path.isfile(target_dir):
            os.remove(target_dir)
        else:
            shutil.rmtree(target_dir)
    if not os.path.exists(os.path.dirname(target_dir)):
        os.makedirs(os.path.dirname(target_dir))
    shutil.copytree(source_dir, target_dir, symlinks=symlinks)
    
'''
description: 
 link source file to target file, if target file is exist, replace it.
param {str} source_file
param {str} target_file
return {*}
author: wuxingxing
'''
def link_file(source_file:str, target_file:str):
    if os.path.islink(target_file):
        os.remove(target_file)
    if os.path.isfile(target_file):
        os.remove(target_file)
    os.symlink(source_file, target_file)
        
"""
@Description :
删除指定目录下所有文件(该目录不删除) / 或者删除指定文件名文件
@Returns     :
@Author       :wuxingxing
"""

def del_file(path_dir):
    if os.path.exists(path_dir) is False:
        return

    if os.path.isfile(path_dir) or os.path.islink(path_dir):
        os.remove(path_dir)
        return

    for i in os.listdir(path_dir) :
        file_path = os.path.join(path_dir, i)
        if os.path.isfile(file_path) is True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_path)
        else:
            del_file(file_path)

def del_file_list(path_list:str):
    for _path in path_list:
        del_file(_path)
        if os.path.isdir(_path):
            shutil.rmtree(_path)
        
def del_file_list_by_patten(del_dir:str, patten:str):
    file_list = glob.glob(os.path.join(del_dir, "*{}".format(patten)))
    del_file_list(file_list)

'''
description: 
    mv files under target_dir to source_dir
param {str} source_dir
param {str} target_dir
return {*}
author: wuxingxing
'''
def mv_file(source_file:str, target_file:str):
    if not os.path.exists(source_file):
        return
    if os.path.islink(target_file):
        os.remove(target_file)
    if os.path.exists(target_file):
        shutil.rmtree(target_file)
    shutil.move(source_file, target_file)

def file_shell_op(shell_cmd: str):
    subprocess.call(shell_cmd, shell=True)

'''
description: 
    add postfix to dir, example:
        for .../path/train:
            after add is:
            ../path/train-1-bk, ../path/train-2-bk, ...
param {str} source_dir
param {str} postfix_str
return {*}
author: wuxingxing
'''
def add_postfix_dir(source_dir:str, postfix_str:str):
    index = 1
    while True:
        target_dir = os.path.join(os.path.dirname(source_dir), "{}-{}-{}".format(os.path.basename(source_dir), postfix_str, index))
        if os.path.exists(target_dir):
            index += 1
        else:
            break
    return target_dir


# '''
# description: 
# param {str} source_file
# param {str} target_file
# return {*}
# author: wuxingxing
# '''
# def mv_file(source_file:str, target_file:str):
#     if os.path.islink(target_file):
#         os.remove(target_file)
#     if os.path.exists(target_file):
#         shutil.rmtree(target_file)
#     shutil.move(source_file, target_file)
    
'''
description: 
    delete dir or file if exists.
param {str} del_dir
return {*}
author: wuxingxing
'''
def del_dir(del_dir:str):
    if os.path.exists(del_dir):
        shutil.rmtree(del_dir)

'''
description: 
    Use wildcard characters to search for files in directory search_root_dir that match the wildcard characters
param {str} root_dir
param {str} current_itername
return {*}
author: wuxingxing
'''
def search_files(search_root_dir:str, template:str):
    file_list = glob.glob(os.path.join(search_root_dir, template))
    return file_list

def str_list_format(input_value):
    input_list = []
    if isinstance(input_value, str):
        input_value = input_value.replace(",", " ")
        input_value = input_value.replace(";", " ")
        input_list = input_value.strip().split()
    elif isinstance(input_value, list):
        input_list = input_value
    else:
        raise Exception("Error format of {}, it should be string or list!".format(input))
    return input_list
        
'''
description: 
    get file extension
param {str} file_name
param {*} split_char default is '.'
return {*}
author: wuxingxing
'''
def get_file_extension(file_name:str, split_char = "."):
    return file_name.split(split_char)[-1].strip()

"""
@Description :
    randomly generate n different nums of int type in the range of [start, end)
@Returns     :
@Author       :wuxingxing
"""

def get_random_nums(start, end, n):
    numsArray = set()
    while len(numsArray) < n:
        numsArray.add(random.randint(start, end-1))
    return list(numsArray)
