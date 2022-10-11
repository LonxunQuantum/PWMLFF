#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil


import numpy as np
import pandas as pd

import use_para as pm
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
    
    sourceDir=os.path.join(pm.fortranFitSourceDir,'fread_dfeat')
    
    if sourceDir==pm.fitModelDir:
        return
    
    if os.path.exists(pm.fitModelDir) and os.path.isfile(pm.fitModelDir):
        os.remove(pm.fitModelDir)
    
    if not os.path.exists(pm.fitModelDir):
        os.makedirs(pm.fitModelDir)
    
    
    midCommand=pm.fortranFitSourceDir.replace('/','\/')
    for fileName in ['makefile','run_pca.py','run_cur.py']:
        fromFilePath=os.path.join(sourceDir,fileName)
        toFilePath=os.path.join(pm.fitModelDir,fileName)
        shutil.copy(fromFilePath,toFilePath)        
        command="sed -i 's/\.\./"+midCommand+"/g' "+toFilePath
        os.system(command)
   
    
    
    
   
def writeFitInput():
    #fitInputPath=os.path.join(pm.fitModelDir,'fit.input')

    natom=200
    m_neigh=pm.maxNeighborNum
    n_image=200
    with open(pm.fitInputPath,'w') as fitInput:
        fitInput.write(str(len(pm.atomType))+', '+str(natom)+', '+str(m_neigh)+', '+\
                       str(n_image)+'      ! ntype,natom,m_neighb,nimage\n')
        for i in range(pm.atomTypeNum):
            line=str(pm.atomType[i])+', '+str(int(pm.fortranFitFeatNum0[i]))+', '+str(int(pm.fortranFitFeatNum2[i]))+\
                 ', '+str(int(pm.fortranGrrRefNum[i]))+', '+str(float(pm.fortranFitAtomRadii[i]))+', '+\
                 str(pm.fortranFitAtomRepulsingEnergies[i])+'       ! itype, nfeat0,nfeat2,ref_num,rad_atom,wp_atom\n'
            fitInput.write(line)
        fitInput.write(str(pm.fortranGrrKernelAlpha)+', '+str(pm.fortranGrrKernalDist0)+'            ! alpha,dist0 (for kernel)\n')
        fitInput.write(str(pm.fortranFitWeightOfEnergy)+', '+str(pm.fortranFitWeightOfEtot)+', '+str(pm.fortranFitWeightOfForce)+\
                       ', '+str(pm.fortranFitRidgePenaltyTerm)+'        ! E_weight ,Etot_weight, F_weight, delta\n')
        fitInput.write(str(pm.fortranFitDwidth)+'        ! dwidth\n')
                       

def copyData():
    locationFromPath=os.path.join(pm.trainSetDir,'location')
    locationToPath=os.path.join(pm.fitModelDir,'location')
    trainDataFromPath=os.path.join(pm.trainSetDir,'trainData.txt')
    trainDataToPath=os.path.join(pm.fitModelDir,'trainData.txt')
    
    '''
    if os.path.exists(locationToPath):
        os.remove(locationToPath)
    
    if os.path.exists(trainDataToPath):
        os.remove(trainDataToPath)
    '''

    shutil.copy(locationFromPath,locationToPath)
    shutil.copy(trainDataFromPath,trainDataToPath)
 

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
    
    command='make pca -C'+pm.fitModelDir
    print(command)
    os.system(command)
    
    if pm.isFitLinModel:
        command='make lin -C'+pm.fitModelDir
        print(command)
        os.system(command)
        shutil.copy(pm.featCalcInfoPath,pm.linModelCalcInfoPath)
        shutil.copy(pm.fitInputPath,pm.linFitInputBakPath)
    
    if pm.isFitGrrModel:
        command='make gpr -C'+pm.fitModelDir
        print(command)
        os.system(command)
        shutil.copy(pm.featCalcInfoPath,pm.grrModelCalcInfoPath)
        shutil.copy(pm.fitInputPath,pm.grrFitInputBakPath)
    
def fit():
    makeFitDirAndCopySomeFiles()
    # readFittingParameters()
    copyData()
    writeFitInput()
    # calcFitFeatNum2AndPenaltyTerm()
    # writeFitInput()
    runFit()



    

if __name__=='__main__':   
    input('Press Enter to quit test:')
