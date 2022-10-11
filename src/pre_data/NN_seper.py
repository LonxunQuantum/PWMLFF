#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
codepath=os.path.abspath(sys.path[0])
sys.path.append(codepath+'/../src/lib')
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
#import parameters as pm
import use_para as pm
import parse_input
parse_input.parse_input()

workpath=os.path.abspath(pm.codedir)
sys.path.append(workpath)
import prepare as pp

def write_egroup_input():
    with open(os.path.join(pm.InputPath,'egroup.in'),'w') as f:
        f.writelines(str(pm.dwidth)+'\n')
        f.writelines(str(pm.ntypes)+'\n')
        for i in range(pm.ntypes):
            f.writelines(str(pm.b_init[i])+'\n')
        

def run_write_egroup():
    #command=pm.genFeatDir+"/write_egroup.x > ./output/out_write_egroup"
    command='write_egroup.x > ./output/out_write_egroup'
    os.system(command)

def write_natoms_dfeat():
    max_natom = int(np.loadtxt(os.path.join(pm.OutputPath,'max_natom')))
    pp.collectAllSourceFiles()
    # f_train_egroup = open(pm.f_train_egroup,'w')
    # f_test_egroup = open(pm.f_test_egroup,'w')
    f_train_natom = open(pm.f_train_natoms,'w')
    f_test_natom = open(pm.f_test_natoms,'w')
    # f_train_feat = open(pm.f_train_feat,'w')
    # f_test_feat = open(pm.f_test_feat,'w')
    kk=0
    f_train_dfeat={}
    f_test_dfeat={}
    dfeat_names={}
    for i in pm.use_Ftype:
        f_train_dfeat[i] = open(pm.f_train_dfeat+str(i),'w')
        f_test_dfeat[i] = open(pm.f_test_dfeat+str(i),'w')
        feat_head_tmp = pd.read_csv(os.path.join(pm.trainSetDir,'trainData.txt'+'.Ftype'+str(i)), header=None).values[:,:3]
        feat_tmp = pd.read_csv(os.path.join(pm.trainSetDir,'trainData.txt'+'.Ftype'+str(i)), header=None).values[:,4:].astype(float)
        dfeat_names[i] = pd.read_csv(os.path.join(pm.trainSetDir,'inquirepos'+str(i)+'.txt'), header=None).values[:,1:].astype(int)
        if kk==0:
            feat=feat_tmp
        else:
            feat=np.concatenate((feat,feat_tmp),axis=1)
        
        kk=kk+1
    feat_all=np.concatenate((feat_head_tmp,feat),axis=1)

    egroup_all = pd.read_csv(os.path.join(pm.trainSetDir,'Egroup_weight'), header=None,names=range(max_natom+2))
    egroup_all = egroup_all.fillna(0)
    egroup_all = egroup_all.values[:,:].astype(float)
    # print(dfeat_names.shape)
    # print(dfeat_names[8,0])
    

    count=0
    Imgcount=0
    feat_train=np.empty([0,feat_all.shape[1]])
    feat_test=np.empty([0,feat_all.shape[1]])
    egroup_train=np.empty([0,egroup_all.shape[1]])
    egroup_test=np.empty([0,egroup_all.shape[1]])
    for system in pm.sourceFileList:
        
        infodata=pd.read_csv(os.path.join(system,'info.txt.Ftype'+str(pm.use_Ftype[0])), header=None,delim_whitespace=True).values[:,0].astype(int)
        natom=infodata[1]
        ImgNum=infodata[2]-(len(infodata)-3)
        trainImgNum=int(ImgNum*(1-pm.test_ratio))
        trainImg=np.arange(0,trainImgNum)
        testImg=np.arange(trainImgNum,ImgNum)
        # with open(os.path.join(system,'info.txt'),'r') as sourceFile:
        #     sourceFile.readline()
        #     natom=int(sourceFile.readline().split()[0])
        #     ImgNum=int(sourceFile.readline().split()[0])
        # useImgs=np.arange(0,ImgNum,interval)
        
        for i in trainImg:
            f_train_natom.writelines(str(int(natom))+' '+str(int(natom))+'\n')
            # f_train_dfeat.writelines(str(os.path.join(system,'dfeat.fbin'))+', '+str(i+1)+'\n')
            for mm in pm.use_Ftype:
                f_train_dfeat[mm].writelines(str(os.path.join(system,'dfeat.fbin.Ftype'+str(mm)))+', '+str(i+1)+', '+str(dfeat_names[mm][int(Imgcount+i),1])+'\n')
            
            # feat_train=np.concatenate((feat_train,feat_all[(count+i*natom):(count+natom*(i+1)),:]),axis=0)

            # for k,line in enumerate(get_all):
            #     if k >= count+i*natom and k < count+(i+1)*natom:             
            #         f_train_feat.writelines(line)        
            #     else:           
            #         continue  
        
        for i in testImg:
            # print(i)
            f_test_natom.writelines(str(int(natom))+' '+str(int(natom))+'\n')
            # f_test_dfeat.writelines(str(os.path.join(system,'dfeat.fbin'))+', '+str(i+1)+'\n')
            for mm in pm.use_Ftype:
                f_test_dfeat[mm].writelines(str(os.path.join(system,'dfeat.fbin.Ftype'+str(mm)))+', '+str(i+1)+', '+str(dfeat_names[mm][int(Imgcount+i),1])+'\n')
            # feat_test=np.concatenate((feat_test,feat_all[(count+i*natom):(count+natom*(i+1)),:]),axis=0)
            # for k,line in enumerate(get_all):
            #     if k >= count+i*natom and k < count+(i+1)*natom:             
            #         f_test_feat.writelines(line)        
            #     else:           
            #         continue
        feat_train=np.concatenate((feat_train,feat_all[count:(count+natom*len(trainImg)),:]),axis=0)
        feat_test=np.concatenate((feat_test,feat_all[(count+natom*len(trainImg)):(count+natom*ImgNum),:]),axis=0)


        egroup_train = np.concatenate((egroup_train,egroup_all[count:(count+natom*len(trainImg)),:]),axis=0)
        egroup_test = np.concatenate((egroup_test,egroup_all[(count+natom*len(trainImg)):(count+natom*ImgNum),:]),axis=0)

        # for k,line in enumerate(get_all):
        #     m=int((k-count)/natom)
        #     if m in trainImg:
        #         f_train_feat.writelines(line)
        #     if m in testImg:
        #         f_test_feat.writelines(line)   
            # if k >= count+i*natom and k < count+(i+1)*natom:             
                     
            # else:           
            #     continue

        count=count+natom*ImgNum
        Imgcount=Imgcount+ImgNum

    np.savetxt(pm.f_train_feat, feat_train, delimiter=',')
    np.savetxt(pm.f_test_feat, feat_test, delimiter=',')
    np.savetxt(pm.f_train_egroup, egroup_train, delimiter=',')
    np.savetxt(pm.f_test_egroup, egroup_test, delimiter=',')

    f_train_natom.close()
    f_test_natom.close()
    # f_train_feat.close()
    # f_test_feat.close()
    for i in pm.use_Ftype:
        f_train_dfeat[i].close()
        f_test_dfeat[i].close()

if __name__=='__main__':
    if not os.path.isdir(pm.dir_work):
        os.system("mkdir " + pm.dir_work)
    for dirn in [pm.d_nnEi, pm.d_nnFi]:
        if not os.path.isdir(dirn):
            os.system("mkdir " + dirn)

    write_egroup_input()
    run_write_egroup()

    write_natoms_dfeat()
