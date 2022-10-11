#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import numpy as np
import cupy as cp
import use_para as pm
# import preparatory_work as ppw
from md_image import MdImage
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
# from ase.md.nvtberendsen import NVTBerendsen as NVT
# from ase.md.nptberendsen import NPTBerendsen as NPT
# from ase.optimize import BFGS
from ase import units

from calc_lin import calc_lin
from calc_vv import calc_vv

# from minilib.get_util_info import getGpuInfo


class MdRunner():
    
    def __init__( \
                self,
                imageFileDir=pm.mdImageFileDir,\
                isFollow=pm.isFollowMd,\
                calcModel=pm.mdCalcModel,\
                isCheckVar=pm.isMdCheckVar,\
                isReDistribute=pm.isReDistribute,\
                imageIndex=pm.mdStartImageIndex,\
                velocityDistributionModel=pm.velocityDistributionModel,\
                stepTime=pm.mdStepTime,\
                startTemperature=pm.mdStartTemperature,\
                runModel=pm.mdRunModel,\
                endTemperature=pm.mdEndTemperature,\
                nvtTaut=pm.mdNvtTaut,\
                isOnTheFly=pm.isOnTheFlyMd,\
                isTrajAppend=pm.isTrajAppend,\
                isNewMovementAppend=pm.isNewMovementAppend,\
                trajInterval=pm.mdTrajIntervalStepNum,\
                logInterval=pm.mdLogIntervalStepNum,\
                newMovementInterval=pm.mdNewMovementIntervalStepNum,\
                isProfile=pm.isMdProfile
                ):
        
        if calcModel=='lin':
            calc=calc_lin
            # ppw.loadFeatCalcInfo(pm.linModelCalcInfoPath)
            # shutil.copy(pm.linFitInputBakPath,pm.fitInputPath)
        elif calcModel=='vv':
            calc=calc_vv
            # ppw.loadFeatCalcInfo(pm.grrModelCalcInfoPath)
            # shutil.copy(pm.grrFitInputBakPath,pm.fitInputPath)
        else:
            raise NotImplementedError(calcModel+" has't been implemented!")
        
        if isFollow:
            shutil.move(os.path.join(imageFileDir,'last_atom.config'),os.path.join(imageFileDir,'atom.config'))
        
        self.dir=os.path.abspath(imageFileDir)
        self.mdDir=os.path.join(self.dir,'md')
        if not os.path.exists(self.mdDir):
            os.mkdir(self.mdDir)
        elif os.path.isfile(self.mdDir):
            print("Warning: md is a file in the same dir of image config file, this md file will be removed")
            os.remove(self.mdDir)
            os.mkdir(self.mdDir)       
        
        self.atoms=MdImage.fromDir(imageFileDir,calc,isCheckVar,isReDistribute,imageIndex,isProfile)
        if isReDistribute and velocityDistributionModel.lower()=='maxwellboltzmann':
            MaxwellBoltzmannDistribution(self.atoms,startTemperature*units.kB)
        else:
            raise NotImplementedError("Only allow redistribute velocities and apply MaxwellBoltzmannDistribution!")

        self.isProfile=isProfile
        self.name=os.path.basename(self.dir)    
        self.logFilePath=os.path.join(self.mdDir,self.name+'_log.txt')
        self.trajFilePath=os.path.join(self.mdDir,self.name+'.extxyz')
        self.newMovementPath=os.path.join(self.mdDir,'MOVEMENT')
        self.atomConfigSavePath=os.path.join(self.dir,'last_atom.config')
        self.errorImageLogPath=os.path.join(self.mdDir,self.name+'_errorLog.txt')
        self.profileTxtPath=os.path.join(self.mdDir,self.name+'_profile.txt')
        self.trajInterval=trajInterval
        self.logInterval=logInterval
        self.newMovementInterval=newMovementInterval
    
        if (not isTrajAppend) and os.path.exists(self.trajFilePath):
            os.remove(self.trajFilePath)
        if (not isNewMovementAppend) and os.path.exists(self.newMovementPath):
            os.remove(self.newMovementPath)
        
        self.logFile=open(self.logFilePath,'w')
        self.errorImageLog=open(self.errorImageLogPath,'w')
        if self.isProfile:
            self.profileTxt=open(self.profileTxtPath,'w')
        self.currentStepNum=-1

    def readpos(self,movementPath,imageIndex=0):
        
        with open(movementPath,'r') as sourceFile:
            numOfAtoms=int(sourceFile.readline().split()[0])
        
        with open(movementPath,'r') as sourceFile:
            currentIndex=-1
            while True:
                line=sourceFile.readline()
                if not line:
                    raise EOFError("The Movement file end, there is only "+str(currentIndex+1)+\
                                   " images, and the "+str(imageIndex+1)+"th image has been choosen!")
                if "Iteration" in line:
                    currentIndex+=1
                
                if currentIndex<imageIndex:
                    continue
                else:
                    cell=cp.zeros((3,3))
                    atomTypeList=[]
                    pos=cp.zeros((numOfAtoms,3))
                    while True:
                        line=sourceFile.readline()
                        if "Lattice" in line:
                            break
                    for i in range(3):
                        L=sourceFile.readline().split()
                        for j in range(3):
                            cell[i,j]=float(L[j])
                    line=sourceFile.readline()
                    for i in range(numOfAtoms):
                        L=sourceFile.readline().split()
                        atomTypeList.append(int(L[0]))
                        for j in range(3):
                            pos[i,j]=float(L[j+1])
                    break
        
        return cell,pos,atomTypeList

    def run100(self,imageIndex):
        # self.atoms=MdImage.fromDir(self.dir,self.nn,self.data_scaler,imageIndex=imageIndex)
        cell,pos,atomTypeList=self.readpos(os.path.join(self.dir,'MOVEMENT'),imageIndex=imageIndex)
        # print(pos)
        self.atoms.set_scaled_positions(cp.asnumpy(pos))
        self.atoms.set_pos_cell()

        self.currentStepNum+=1
        ek=self.atoms.get_kinetic_energy()
        ep=self.atoms.get_potential_energy()
        etot=ek+ep
        outStr=str(self.currentStepNum+1)+' '+str(etot)+' '+str(ep)+' '+str(ek)                        
        self.logFile.write(outStr+'\n')
        self.atoms.toAtomConfig(self.newMovementPath,True)

    
    def final(self):
        self.atoms.toAtomConfig(self.atomConfigSavePath)
        self.logFile.close()
        self.errorImageLog.close()
        if self.isProfile:
            self.profileTxt.close()

if __name__=='__main__':   
    input('Press Enter to quit test:')
