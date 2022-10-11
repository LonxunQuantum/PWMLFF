import numpy as np
#对初始数据进行permutation，需输入原始文件及排序后的文件名

#打开原文件，统计样本量
#f = open('./MOVEMENT','r') 
originfile = '/home/husiyu/data/moleculeNN/structures/MOVEMENT5'
f = open(originfile,'r')
lines = f.readlines()
number_of_samples = 0
for lines in lines:
    if "-----------" in lines:
        number_of_samples = number_of_samples + 1
print(number_of_samples)
permutated_sample_list = np.random.permutation(number_of_samples)
print(permutated_sample_list)

#打开要写入的新文件，排序后的文件
atom_len = 446
#file = r'./MOVEMENT_new'
newfile = r'/home/husiyu/software/moleculeNN/LargeSamples/transfer_nnff/data/cu/MOVEMENT5'
with open(newfile, 'a+') as newfile:                #打开要写入的文件
    for i in range(number_of_samples):
        atom_list = permutated_sample_list[i]    #乱序后的image序号
        #打开原文件
        with open(originfile, 'r') as fp:
            for i, line in enumerate(fp):
                # 要写入的image，446行
                for len in range(atom_len):
                    if i == atom_list * atom_len + len:
                        # print(line)
                        newfile.writelines(line)
