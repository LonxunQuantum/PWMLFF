#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

import default_para as pm 
"""
import use_para as pm
import parse_input
parse_input.parse_input()
"""
import numpy as np
import pandas as pd
import prepare as pp


def write_egroup_input():
    with open(os.path.join(pm.InputPath, 'egroup.in'), 'w') as f:
        f.writelines(str(pm.dwidth)+'\n')
        f.writelines(str(pm.ntypes)+'\n')
        for i in range(pm.ntypes):
            f.writelines(str(pm.b_init[i])+'\n')

def run_write_egroup():
    command = 'write_egroup.x > ./output/out_write_egroup'
    print(command)
    os.system(command)


def write_natoms_dfeat(chunk_size, shuffle=True, atom_num = 1, alive_atomic_energy = False):
    """
        put data into chunks
    """
    # max_natom = int(np.loadtxt(os.path.join(pm.OutputPath, 'max_natom')))
    max_natom = int(atom_num)
    #print (pm.dp_predict)
    # do not execute this step for dp prediction 
    if pm.dp_predict == False\
        and len(pm.sourceFileList) == 0: 
        pp.collectAllSourceFiles()
    
    f_train_natom = open(pm.f_train_natoms, 'w')
    f_test_natom = open(pm.f_test_natoms, 'w')
    kk = 0
    f_train_dfeat = {}
    f_test_dfeat = {}
    dfeat_names = {}

    for i in pm.use_Ftype:
        f_train_dfeat[i] = open(pm.f_train_dfeat+str(i), 'w')
        f_test_dfeat[i] = open(pm.f_test_dfeat+str(i), 'w')
        feat_head_tmp = pd.read_csv(os.path.join(
            pm.trainSetDir, 'trainData.txt'+'.Ftype'+str(i)), header=None).values[:, :3]
        feat_tmp = pd.read_csv(os.path.join(
            pm.trainSetDir, 'trainData.txt'+'.Ftype'+str(i)), header=None).values[:, 4:].astype(float)
        dfeat_names[i] = pd.read_csv(os.path.join(
            pm.trainSetDir, 'inquirepos'+str(i)+'.txt'), header=None).values[:, 1:].astype(int)
        if kk == 0:
            feat = feat_tmp
        else:
            # import ipdb;ipdb.set_trace()
            feat = np.concatenate((feat, feat_tmp), axis=1)
        
        kk = kk+1
    feat_all = np.concatenate((feat_head_tmp, feat), axis=1)
    # import ipdb;ipdb.set_trace()

    if alive_atomic_energy:
        egroup_all = pd.read_csv(os.path.join(
            pm.trainSetDir, 'Egroup_weight'), header=None, names=range(max_natom+2))
        egroup_all = egroup_all.fillna(0)
        egroup_all = egroup_all.values[:, :].astype(float)
        egroup_train = np.empty([0, egroup_all.shape[1]])
        egroup_test = np.empty([0, egroup_all.shape[1]])
        

    # ep_all = pd.read_csv(os.path.join(pm.trainSetDir, 'ep.txt'), header=None).values

    count = 0
    Imgcount = 0

    feat_train = np.empty([0, feat_all.shape[1]])
    feat_test = np.empty([0, feat_all.shape[1]])
    #used to randomly seperate training set and test set
    from numpy.random import choice, shuffle

    system_dir = list(set(pm.sourceFileList))

    sys_natom = {}
    sys_imgNum = {}
    #sys_train_img = {}
    #sys_test_img = {}
    img_num_base = {} 
    line_num_base = {}
    
    sys_img_list_train =  [] 
    sys_img_list_test = [] 

    img_base = 0 
    line_base = 0

    from math import floor

    # some system dependent values 
    for system in system_dir:
        
        print ("in",system)

        infodata = pd.read_csv(os.path.join(system, 'info.txt.Ftype'+str(
            pm.use_Ftype[0])), header=None, delim_whitespace=True).values[:, 0].astype(int)
        
        natom = int(infodata[1])
        ImgNum = int(infodata[2]-(len(infodata)-3))
        
        print ("natom", natom)
        print ("num image", ImgNum) 
        
        # image
        img_num_base[system] = img_base 
        img_base += ImgNum

        # line or atom 
        line_num_base[system] = line_base
        line_base += natom*ImgNum

        sys_natom[system] = natom
        sys_imgNum[system] = ImgNum

        trainImgNum = int(ImgNum*(1-pm.test_ratio))

        # chunk num for training and valid set
        train_chunk_num = floor(trainImgNum/chunk_size)
        valid_chun_num = floor(ImgNum/chunk_size) - train_chunk_num

        # shuffle images within a system. Start from 0 
        #randomIdx = [i for i in range(ImgNum)]
        if shuffle is True:
            randomIdx = choice(ImgNum,ImgNum,replace = False) 
        else:
            randomIdx = list(range(0, ImgNum))

        # (system , index within the system)
        # sys_img_list_train contains all chunks

        for i in range(train_chunk_num):
            sys_img_list_train.append([(system, j) for j in randomIdx[i:i+chunk_size]]) 

        for i in range(train_chunk_num,train_chunk_num+valid_chun_num):
            sys_img_list_test.append([(system, j) for j in randomIdx[i:i+chunk_size]])

        #sys_img_list_train += [(system, i) for i in randomIdx[:trainImgNum]]
        #sys_img_list_test += [(system, i) for i in randomIdx[trainImgNum:]]

    #print (sys_img_list_train)
    #print (sys_img_list_test)
    print (img_num_base)
    print (line_num_base)
    
    # shuffle the system-idx tuple list
    if shuffle is True:
        shuffle(sys_img_list_train)

    # print("training set")
    # for item in sys_img_list_train:   #sys_img_list_train[0][0][1]
    #     print (item)
    
    # print("valid set") 
    # for item in sys_img_list_test:
    #     print (item)

    if pm.test_ratio > 0 and pm.test_ratio < 1:
        """
            shuffle the images in all MOVEMENTs
            
            after this step 4 groups of data will be generated in fread_dfeat/NN_output
            1) dfeatname_train&test , path to dfeat binary file
            2) egroup_train&test, value of egroup 
            3) feat_train&test, value of feature
            4) natoms_train&test, number of atoms 
            
        """
        
        # re-arrange data for training set
        for chunk in sys_img_list_train:
            for system, idx_in_sys in chunk:

                # global image index within feat_all 
                Imgcount = img_num_base[system] 
                count = line_num_base[system] 
                natom = sys_natom[system]

                # natom
                f_train_natom.writelines(str(int(natom))+' '+str(int(natom))+'\n')
                
                #dfeatname 
                for mm in pm.use_Ftype:
                    f_train_dfeat[mm].writelines(str(os.path.join(system, 'dfeat.fbin.Ftype'+str(mm)))+', '+str(idx_in_sys+1)+', '+str(dfeat_names[mm][int(Imgcount+idx_in_sys), 1])+'\n')
            
                idx_start = count + natom*idx_in_sys
                idx_end = count + natom*(idx_in_sys+1)

                # feat 
                # Note: the order of feat_train.csv is not chaned yet
                feat_train = np.concatenate(
                    (feat_train, feat_all[idx_start:idx_end,:]), axis=0)
                
                #egroup
                if alive_atomic_energy:
                    egroup_train = np.concatenate(
                        (egroup_train, egroup_all[idx_start:idx_end,:]), axis=0) 
                
        # valid set
        for chunk in sys_img_list_test:
            for system, idx_in_sys in chunk: 
                
                Imgcount = img_num_base[system] 
                count = line_num_base[system] 
                natom = sys_natom[system]

                # natom
                f_test_natom.writelines(str(int(sys_natom[system]))+' '+str(int(sys_natom[system]))+'\n')

                # dfeatname 
                for mm in pm.use_Ftype:
                    f_test_dfeat[mm].writelines(str(os.path.join(system, 'dfeat.fbin.Ftype'+str(mm)))+', '+str(idx_in_sys+1)+', '+str(dfeat_names[mm][int(Imgcount+idx_in_sys), 1])+'\n')

                idx_start = count + natom*idx_in_sys
                idx_end = count + natom*(idx_in_sys+1)
                
                # feat 
                feat_test = np.concatenate(
                    (feat_test, feat_all[idx_start:idx_end,:]), axis=0)
                
                #egroup
                if alive_atomic_energy:
                    egroup_test = np.concatenate(
                        (egroup_test, egroup_all[idx_start:idx_end,:]), axis=0)

            
    """
    for system in system_dir:
        
        infodata = pd.read_csv(os.path.join(system, 'info.txt.Ftype'+str(
            pm.use_Ftype[0])), header=None, delim_whitespace=True).values[:, 0].astype(int)
        
        natom = infodata[1]

        ImgNum = infodata[2]-(len(infodata)-3)

        if pm.test_ratio > 0 and pm.test_ratio < 1:
            
            # shuffle within the movement
            trainImgNum = int(ImgNum*(1-pm.test_ratio))
                
            randomIdx = choice(ImgNum,ImgNum,replace = False) 

            trainImg = randomIdx[:trainImgNum]
            testImg = randomIdx[trainImgNum:]

            for i in trainImg:
                f_train_natom.writelines(str(int(natom))+' '+str(int(natom))+'\n')
                for mm in pm.use_Ftype:
                    f_train_dfeat[mm].writelines(str(os.path.join(system, 'dfeat.fbin.Ftype'+str(mm)))+', '+str(i+1)+', '+str(dfeat_names[mm][int(Imgcount+i), 1])+'\n')
                        
            for i in testImg:
                f_test_natom.writelines(str(int(natom))+' '+str(int(natom))+'\n')
                for mm in pm.use_Ftype:
                    f_test_dfeat[mm].writelines(str(os.path.join(system, 'dfeat.fbin.Ftype'+str(mm)))+', '+str(i+1)+', '+str(dfeat_names[mm][int(Imgcount+i), 1])+'\n')



            # image-wise? 
            feat_train = np.concatenate(
                (feat_train, feat_all[count:(count+natom*len(trainImg)), :]), axis=0)
            
            egroup_train = np.concatenate(
                (egroup_train, egroup_all[count:(count+natom*len(trainImg)), :]), axis=0)

            feat_test = np.concatenate(
                (feat_test, feat_all[(count+natom*len(trainImg)):(count+natom*ImgNum), :]), axis=0)

            egroup_test = np.concatenate(
                (egroup_test, egroup_all[(count+natom*len(trainImg)):(count+natom*ImgNum), :]), axis=0)

            count = count+natom*ImgNum
            Imgcount = Imgcount+ImgNum

        elif pm.test_ratio == 1:
            testImg = np.arange(0, ImgNum)
            for i in testImg:
                f_test_natom.writelines(str(int(natom))+' '+str(int(natom))+'\n')
                for mm in pm.use_Ftype:
                    f_test_dfeat[mm].writelines(str(os.path.join(system, 'dfeat.fbin.Ftype'+str(
                        mm)))+', '+str(i+1)+', '+str(dfeat_names[mm][int(Imgcount+i), 1])+'\n')
            feat_test = np.concatenate(
                (feat_test, feat_all[count:(count+natom*ImgNum), :]), axis=0)
            egroup_test = np.concatenate((egroup_test, egroup_all[(
                count):(count+natom*ImgNum), :]), axis=0)
            
            count = count+natom*ImgNum
            Imgcount = Imgcount+ImgNum  
    """   

    # feat_train/test.csv 
    if pm.test_ratio != 1:
        np.savetxt(pm.f_train_feat, feat_train, delimiter=',')
        if alive_atomic_energy:
            np.savetxt(pm.f_train_egroup, egroup_train, delimiter=',')
        f_train_natom.close()
        for i in pm.use_Ftype:
            f_train_dfeat[i].close()

    np.savetxt(pm.f_test_feat, feat_test, delimiter=',')

    if alive_atomic_energy:
        np.savetxt(pm.f_test_egroup, egroup_test, delimiter=',')

    f_test_natom.close()

    for i in pm.use_Ftype:
        f_test_dfeat[i].close()

def write_dR_neigh():
    # 需要生成一个自己的info文件 先用gen2b的代替
    dR_neigh = pd.read_csv(pm.dRneigh_path, header=None, delim_whitespace=True)
    count = 0

    # why source file list has duplicate in it? 
    
    for system in list(set(pm.sourceFileList)):
        infodata = pd.read_csv(os.path.join(system,'info.txt.Ftype'+str(pm.use_Ftype[0])), header=None,delim_whitespace=True).values[:,0].astype(int)
        natom = infodata[1]
        img_num = infodata[2] - (len(infodata)-3)
        tmp = img_num * natom * len(pm.atomType) * pm.maxNeighborNum
        dR_neigh_tmp = dR_neigh[count:count+tmp]

        if pm.test_ratio > 0 and pm.test_ratio < 1:
            train_img_num = int(img_num * (1 - pm.test_ratio))
            index = train_img_num * natom * len(pm.atomType) * pm.maxNeighborNum
            if count == 0:
                train_img = dR_neigh_tmp[:index]
                test_img = dR_neigh_tmp[index:]
            else:
                train_img = train_img.append(dR_neigh_tmp[:index])
                test_img = test_img.append(dR_neigh_tmp[index:])
            count += tmp
        elif pm.test_ratio == 1:
            testImg = np.arange(0, img_num)
            if count == 0:
                test_img = dR_neigh_tmp[:]
            else:
                test_img = test_img.append(dR_neigh_tmp[:])
            count += tmp

    if count != dR_neigh.shape[0]:
        raise ValueError("collected dR dimension mismatches the original")

    if pm.test_ratio != 1:
        train_img.to_csv(pm.f_train_dR_neigh, header=False, index=False)
    
    test_img.to_csv(pm.f_test_dR_neigh, header=False, index=False)
    
def seperate_data(chunk_size = 10, shuffle=True, atom_num = 1, alive_atomic_energy = False):

    print("start data seperation")

    if not os.path.isdir(pm.dir_work):
        os.system("mkdir " + pm.dir_work)
    for dirn in [pm.d_nnEi, pm.d_nnFi]:
        if not os.path.isdir(dirn):
            os.system("mkdir " + dirn)

    if alive_atomic_energy:
        write_egroup_input()
        run_write_egroup()

    write_natoms_dfeat(chunk_size, shuffle, atom_num, alive_atomic_energy)

    if (pm.dR_neigh):
        write_dR_neigh()
    
    print("data seperated")

if __name__ == '__main__':
    if not os.path.isdir(pm.dir_work):
        os.system("mkdir " + pm.dir_work)
    for dirn in [pm.d_nnEi, pm.d_nnFi]:
        if not os.path.isdir(dirn):
            os.system("mkdir " + dirn)

    write_egroup_input()
    run_write_egroup()
    write_natoms_dfeat()
    if (pm.dR_neigh):
        write_dR_neigh()
