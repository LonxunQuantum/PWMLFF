#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil


import numpy as np
import pandas as pd

import default_para as pm 

# import preparatory_work as ppw


def readFittingParameters():
    
    # if not pm.isCalcFeat:
    #     ppw.loadFeatCalcInfo()
        
    # pm.fortranFitFeatNum0=pm.realFeatNum*np.ones((pm.atomTypeNum,),np.int32)
    # pm.fortranFitFeatNum2=(pm.fortranFitFeatNum0*0.7).astype(np.int32)
    
    if pm.fortranGrrRefNum is None:
        pm.fortranGrrRefNum=np.array(pm.numOfCaseOfAllTypes*pm.fortranGrrRefNumRate,dtype=np.int32)
    else:
        if len(pm.fortranGrrRefNum)!=pm.atomTypeNum:
            raise ValueError("pm.fortranGrrRefNum should be a array whose len is pm.atomTypeNum!")
        pm.fortranGrrRefNum=np.array(pm.fortranGrrRefNum)
    pm.fortranGrrRefMinNum=np.where(pm.numOfCaseOfAllTypes<pm.fortranGrrRefMinNum,pm.numOfCaseOfAllTypes,pm.fortranGrrRefMinNum)
    pm.fortranGrrRefMaxNum=np.where(pm.numOfCaseOfAllTypes<pm.fortranGrrRefMaxNum,pm.numOfCaseOfAllTypes,pm.fortranGrrRefMaxNum)
    mask=pm.fortranGrrRefNum<pm.fortranGrrRefMinNum
    pm.fortranGrrRefNum[mask]=pm.fortranGrrRefMinNum[mask]
    mask=pm.fortranGrrRefNum>pm.fortranGrrRefMaxNum
    pm.fortranGrrRefNum[mask]=pm.fortranGrrRefMaxNum[mask]
    pm.fortranFitAtomRadii=np.array(pm.fortranFitAtomRadii)
    pm.fortranFitAtomRepulsingEnergies=np.array(pm.fortranFitAtomRepulsingEnergies)

def makeFitDirAndCopySomeFiles():
    
    if os.name!='posix':
        raise NotImplementedError("Can't run fitting automatically out of Linux os!")    
    
    # liuliping: no "templ"
    sourceDir=os.path.join(pm.fortranFitSourceDir,'fread_dfeat')
    
    if sourceDir==pm.fitModelDir:
        return
    
    if os.path.exists(pm.fitModelDir) and os.path.isfile(pm.fitModelDir):
        os.remove(pm.fitModelDir)
    
    if not os.path.exists(pm.fitModelDir):
        os.makedirs(pm.fitModelDir)

    #midCommand=pm.fortranFitSourceDir.replace('/','\/')
    for fileName in ['calculate_error.py', 'cur_input.py', 'GPR_fit_force_para.py', 'pca_input.py', 'plotvar.py', 'run_cur.py', 'run_pca.py',]:
        fromFilePath=os.path.join(sourceDir,fileName)
        toFilePath=os.path.join(pm.fitModelDir,fileName)
        shutil.copy(fromFilePath,toFilePath)        
        # liuliping: deprecate makefile using in calculations
        #command="sed -i 's/\.\./"+midCommand+"/g' "+toFilePath
        #os.system(command)

   
def writeFitInput():
    #fitInputPath=os.path.join(pm.fitModelDir,'fit.input')
    '''
        2, 200    ! ntype,m_neigh 
        6   2.0   0.0   ! iat,rad,wp (vdw)
        29  2.0   0.0   !  iat,rad,wp
        0.9, 0.0, 0.1, 0.00001   ! w_E,w_E0,w_F,delta
    '''
    # natom=200
    m_neigh=pm.maxNeighborNum
    # n_image=200
    with open(pm.fitInputPath_lin,'w') as fitInput:
        fitInput.write(str(len(pm.atomType))+', '+str(m_neigh)+', '+\
                       '      ! ntype,m_neighb \n')
        for i in range(pm.atomTypeNum):
            line=str(pm.atomType[i])+', '+str(float(pm.fortranFitAtomRadii[i]))+', '+\
                 str(pm.fortranFitAtomRepulsingEnergies[i])+'       ! itype, rad_atom,wp_atom\n'
            fitInput.write(line)
        # fitInput.write(str(pm.fortranGrrKernelAlpha)+', '+str(pm.fortranGrrKernalDist0)+'            ! alpha,dist0 (for kernel)\n')
        fitInput.write(str(pm.fortranFitWeightOfEnergy)+', '+str(pm.fortranFitWeightOfEtot)+', '+str(pm.fortranFitWeightOfForce)+\
                       ', '+str(pm.fortranFitRidgePenaltyTerm)+'        ! E_weight ,Etot_weight, F_weight, delta\n')
        fitInput.write(str(pm.fortranFitDwidth)+'        ! dwidth\n')


def FeatCollectIn():
    #fitInputPath=os.path.join(pm.fitModelDir,'fit.input')
    '''
        1        ! iflag_PCA
        2        ! nfeat_type
        1        ! iFtype(1): 1 means 2b
        2        ! iFtype(2): 2 means 3b
        2        ! ntype
        6        ! iat of first type
        29       ! iat of second type 
    '''
    # natom=200
    # m_neigh=pm.maxNeighborNum
    # n_image=200
    with open(pm.featCollectInPath,'w') as fitInput:
        fitInput.write(str(pm.iflag_PCA)+'        ! iflag_PCA \n')
        fitInput.write(str(pm.nfeat_type)+'        ! nfeat_type \n')
        for i in range(pm.nfeat_type):
            line = str(pm.use_Ftype[i]) + '        ! iFtype \n'
            fitInput.write(line)
        fitInput.write(str(pm.atomTypeNum)+'        ! ntype \n')
        for i in range(pm.atomTypeNum):
            line=str(pm.atomType[i])+'        ! iat \n'
            fitInput.write(line)


def copyData():
    locationFromPath=os.path.join(pm.trainSetDir,'location')
    locationToPath=os.path.join(pm.fitModelDir,'location')
    # trainDataFromPath=os.path.join(pm.trainSetDir,'trainData.txt')
    # trainDataToPath=os.path.join(pm.fitModelDir,'trainData.txt')
    
    '''
    if os.path.exists(locationToPath):
        os.remove(locationToPath)
    
    if os.path.exists(trainDataToPath):
        os.remove(trainDataToPath)
    '''

    shutil.copy(locationFromPath,locationToPath)
    # shutil.copy(trainDataFromPath,trainDataToPath)
 

def calcBreakPoint(a):
    a[a<1.0e-12]=0.0
    b=a[:-1]/a[1:] 
    b[a[1:]>10.0]=0.0
    b[np.isnan(b)]=0.0
    return b.argmax()+1


def calcFitFeatNum2AndPenaltyTerm():   
    
    command='make all -C'+pm.fitModelDir
    print(command)
    os.system(command)
    
    command='make pca -C'+pm.fitModelDir    
    print(command)
    os.system(command)
    minSingularValue=float('inf')
    for i in range(1,pm.atomTypeNum+1):
        pcaEigenFilePath=os.path.join(pm.fitModelDir,'PCA_eigen_feat.'+str(i))
        singularValuesOfOneType=pd.read_csv(pcaEigenFilePath,header=None,delim_whitespace=True,usecols=(1,))
        singularValuesOfOneType=np.array(singularValuesOfOneType)
        pm.fortranFitFeatNum2[i-1]=calcBreakPoint(singularValuesOfOneType)
        minSingularValue=min(minSingularValue,singularValuesOfOneType[pm.fortranFitFeatNum2[i-1]-1])
    if pm.isDynamicFortranFitRidgePenaltyTerm:
        pm.fortranFitRidgePenaltyTerm=float(min(pm.fortranFitRidgePenaltyTerm,minSingularValue*0.1))
    

                       
def runFit():
    # liuliping: no makefile
    # command='make pca -C'+pm.fitModelDir
    command = 'feat_collect_PCA.r feat_collect.in; '
    print(command)
    current_dir = os.getcwd()
    os.system(command)
    
    if pm.isFitLinModel:
        # liuliping: no makefile
        #command='make lin -C'+pm.fitModelDir
        command1 = 'fit_lin_forceMM.r' # feat_PV.1 needed
        command2 = 'calc_lin_forceMM.r' # linear_fitB.ntype needed
        current_dir = os.getcwd()

        print(command1)
        if not os.path.exists(pm.fitModelDir+'/feat_PV.1'):
            print('ERROR. command fit_lin_forceMM.r needs feat_PV.1')
            sys.exit(-1)
        os.system(command1)

        print(command2)
        if not os.path.exists(pm.fitModelDir+'/linear_fitB.ntype'):
            print('ERROR. command calc_lin_forceMM.r needs linear_fitB.ntype')
            sys.exit(-1)
        os.system(command2)

        shutil.copy(pm.fitInputPath_lin,pm.linFitInputBakPath)
    

def fit_vdw():
    makeFitDirAndCopySomeFiles()
    # readFittingParameters()
    copyData()
    writeFitInput()
    FeatCollectIn()
    # liuliping: deprecate makefile, use bash commands
    current_dir = os.getcwd()
    command1 = 'feat_collect_PCA.r; '
    command2 = 'fit_vdw.r; '
    print(command1)
    os.system(command1)
    print(command2)
    os.system(command2)
    

def fit():
    # makeFitDirAndCopySomeFiles()
    # readFittingParameters()
    copyData()
    writeFitInput()
    FeatCollectIn()
    # calcFitFeatNum2AndPenaltyTerm()
    runFit()
    

if __name__=='__main__':   
    input('Press Enter to quit test:')
