from typing import List, Iterable
import pandas as pd
import numpy as np
import re
import os
import sys
sys.path.append(os.getcwd())

#########################################################
#if[MemoryError: Unable to allocate *** GiB for an array]
#文件太大需要对初始数据进行split
#########################################################

# parameters need to spcify
#########################################################
# 生成训练数据(十份里的第1份)
# filename = './PWdata/MOVEMENT'    #原始文件
# split_folds = 10                  #划分成若干份
# ifold = 1                         # 生成第i份
# ifold_file = r'./PWdata/temp1'  

#########################################################
# 生成测试数据（十份里的第5份）
filename = './PWdata/MOVEMENTraw'    #原始文件
split_folds = 10                  #划分成若干份
ifold = 5                         # 生成第i份
ifold_file = r'./PWdata/temp1'  


def read(fp: str, n: int) -> Iterable[List[str]]:
    i = 0
    lines = []  # a buffer to cache lines
    with open(fp) as f:
        for line in f:
            i += 1
            lines.append(line)  # append a line
            if i >= n:
                yield lines
                # reset buffer
                i = 0
                lines.clear()
    # remaining lines
    if i > 0:
        yield lines

f = open(filename, 'r')
lines = f.readlines()
# the atom number in one image, can also obtained from parameter.py
lines0 = lines[0]
atoms_number_in_one_image = int(re.findall(r"(.*) atom", lines0)[0])
print("atoms number:", atoms_number_in_one_image)
# number of all the samples
number_of_samples = 0
for lines in lines:
    if "-----------" in lines:
        number_of_samples = number_of_samples + 1
print("image samples:", number_of_samples)
# image length
line_number = 0
with open(filename, 'r') as f:
    for line in f:
        line_number +=1
        if "------" in line:
            break 
image_len = line_number
print("image lenghth:", image_len)

# split into five different MOVEMENT files
image_index = list(np.arange(0, number_of_samples, split_folds) + ifold)
images = read(filename, image_len)
with open(ifold_file, 'w') as n:
    for index, image in enumerate(images):
        if index in image_index:
            n.writelines(image)

#########################################################
# 生成训练数据后文件重命名
# os.rename(filename, './PWdata/MOVEMENTraw')
# os.rename(ifold_file, './PWdata/MOVEMENT')

#########################################################
# 生成测试数据后文件重命名
# os.rename(ifold_file, './mytest/MOVEMENT')

