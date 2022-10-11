#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import use_para as pm
import numpy as np
import cupy as cp
import pandas as pd
from image import Image
from ase.cell import Cell
from ase.atoms import Atoms
from ase.constraints import (voigt_6_to_full_3x3_stress,
                             full_3x3_to_voigt_6_stress)

# from calc_feature import calc_feature
from calc_ftype1 import calc_ftype1
from calc_ftype2 import calc_ftype2
from calc_lin import calc_lin
from calc_vv import calc_vv

# from minilib.get_util_info import getGpuInfo

class MdImage(Atoms,Image):

    def __init__(self):
        pass


    @staticmethod
    def fromImage(anImage,calc=calc_lin,isCheckVar=False,isReDistribute=True):
        
        self=MdImage()
        
        Image.__init__(self,
        atomTypeList=anImage.atomTypeList,
        cell=anImage.cupyCell,
        pos=anImage.pos,
        isOrthogonalCell=anImage.isOrthogonalCell,
        isCheckSupercell=anImage.isCheckSupercell,
        atomTypeSet=anImage.atomTypeSet,
        atomCountAsType=anImage.atomCountAsType,
        atomCategoryDict=anImage.atomCategoryDict,
        force=anImage.force,
        energy=anImage.energy,
        velocity=anImage.velocity,
        ep=anImage.ep,
        dE=anImage.dE)
        
        
        Atoms.__init__(self,
        scaled_positions=cp.asnumpy(self.pos),
        cell=Cell(cp.asnumpy(self.cupyCell)),
        pbc=pm.pbc,
        numbers=self.atomTypeList)
        
        self.calc=calc
        self.isCheckVar=isCheckVar
        calc.set_paths(pm.fitModelDir)
        calc.load_model()
        calc.set_image_info(np.array(anImage.atomTypeList),True)
        # calc_feature.load_model()
        # calc_feature.set_image_info(np.array(atomTypeList),True)
        for i in range(len(pm.use_Ftype)):
            if pm.use_Ftype[i]==1:
                calc_ftype1.load_model()
                calc_ftype1.set_image_info(np.array(atomTypeList),True)
            if pm.use_Ftype[i]==2:
                calc_ftype2.load_model()
                calc_ftype2.set_image_info(np.array(atomTypeList),True)
        self.isNewStep=True
        return self


    @staticmethod
    def fromAtoms(anAtoms):
        pass

    @staticmethod
    def fromAtomConfig(atomConfigPath,calc=calc_lin,isCheckVar=False,isReDistribute=True):
        
        numOfAtoms=int(open(atomConfigPath,'r').readline())
        cell=cp.array(pd.read_csv(atomConfigPath,delim_whitespace=True,header=None,skiprows=2,nrows=3))
        data=pd.read_csv(atomConfigPath,delim_whitespace=True,header=None,skiprows=6,nrows=numOfAtoms)
        atomTypeList=list(data[0])
        pos=cp.array(data.iloc[:,1:4])
        
        self=MdImage()
        
        Image.__init__(self,
        atomTypeList=atomTypeList,
        cell=cell,
        pos=pos)
        
        
        Atoms.__init__(self,
        scaled_positions=cp.asnumpy(pos),
        cell=Cell(cp.asnumpy(cell)),
        pbc=pm.pbc,
        numbers=self.atomTypeList)
        
        self.calc=calc
        self.isCheckVar=isCheckVar
        calc.set_paths(pm.fitModelDir)
        calc.load_model()
        calc.set_image_info(np.array(atomTypeList),True)
        # calc_feature.set_paths(pm.fitModelDir)
        # calc_feature.load_model()
        # calc_feature.set_image_info(np.array(atomTypeList),True)
        for i in range(len(pm.use_Ftype)):
            if pm.use_Ftype[i]==1:
                calc_ftype1.load_model()
                calc_ftype1.set_image_info(np.array(atomTypeList),True)
            if pm.use_Ftype[i]==2:
                calc_ftype2.load_model()
                calc_ftype2.set_image_info(np.array(atomTypeList),True)
        self.isNewStep=True
        return self


    @staticmethod
    def fromMovement(movementPath,calc=calc_lin,isCheckVar=False,isReDistribute=True,imageIndex=0):
        
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
        
        self=MdImage()
        Image.__init__(self,
        atomTypeList=atomTypeList,
        cell=cell,
        pos=pos)
        
        
        Atoms.__init__(self,
        scaled_positions=cp.asnumpy(pos),
        cell=Cell(cp.asnumpy(cell)),
        pbc=pm.pbc,
        numbers=self.atomTypeList)
        
        self.calc=calc
        self.isCheckVar=isCheckVar
        calc.set_paths(pm.fitModelDir)
        calc.load_model()
        calc.set_image_info(np.array(atomTypeList),True)
        # calc_feature.load_model()
        # calc_feature.set_image_info(np.array(atomTypeList),True)
        for i in range(len(pm.use_Ftype)):
            if pm.use_Ftype[i]==1:
                calc_ftype1.load_model()
                calc_ftype1.set_image_info(np.array(atomTypeList),True)
            if pm.use_Ftype[i]==2:
                calc_ftype2.load_model()
                calc_ftype2.set_image_info(np.array(atomTypeList),True)
        self.isNewStep=True
        return self
            


    @staticmethod
    def fromDir(dirPath,calc=calc_lin,isCheckVar=False,isReDistribute=True,imageIndex=0,isProfile=None):
        
        if isProfile==None:
            isProfile=pm.isMdProfile
        
        dirList=os.listdir(dirPath)
        if 'atom.config' in dirList and os.path.isfile(os.path.join(dirPath,'atom.config')):
            atomConfigFilePath=os.path.join(dirPath,'atom.config')
            self=MdImage.fromAtomConfig(atomConfigFilePath,calc,isCheckVar,isReDistribute)
            self.dir=os.path.abspath(dirPath)            
            self.isProfile=isProfile
            if self.isProfile:
                self.calcFeatTime=0.0
                self.calcForceTime=0.0
            return self
        elif 'MOVEMENT' in dirList and os.path.isfile(os.path.join(dirPath,'MOVEMENT')):
            movementPath=os.path.join(dirPath,'MOVEMENT')
            self=MdImage.fromMovement(movementPath,calc,isCheckVar,isReDistribute,imageIndex)
            self.dir=os.path.abspath(dirPath)
            self.isProfile=isProfile
            if self.isProfile:
                self.calcFeatTime=0.0
                self.calcForceTime=0.0
            return self
        else:
            raise ValueError("There is no atom.config or MOVEMENT in this dir")



    def toAtomConfig(self,atomConfigFilePath,isAppend=False,isApplyConstraint=False):
        if isAppend:
            openMode='a'
        else:
            openMode='w'
        energies=self.get_potential_energies()
        forces=self.get_forces()
        with open(atomConfigFilePath,openMode) as atomConfigFile:
            atomConfigFile.write(str(len(self))+'\n')
            atomConfigFile.write('LATTICE\n')
            for i in range(3):
                atomConfigFile.write(str(float(self.cell[i,0]))+'  '+str(float(self.cell[i,1]))+'  '+str(float(self.cell[i,2]))+'  \n')
            atomConfigFile.write('POSITION\n')
            if not isApplyConstraint:
                lineTail='  1  1  1  '
                sPos=self.get_scaled_positions(True)
                for i in range(len(self)):
                    atomConfigFile.write(str(self.atomTypeList[i])+'  '+str(float(sPos[i,0]))+'  '+str(float(sPos[i,1]))+'  '+\
                    str(float(sPos[i,2]))+'  '+lineTail+str(energies[i])+'  '+str(forces[i,0])+'  '+str(forces[i,1])+'  '+str(forces[i,2])+'\n')
                    
            atomConfigFile.write('-'*80+'\n')

    def toTrainMovement(self,atomConfigFilePath,isAppend=False,isApplyConstraint=False):
        if isAppend:
            openMode='a'
        else:
            openMode='w'
        energies=self.get_potential_energies()
        forces=self.get_forces()
        velocities=self.get_velocities()
        ek=np.sum(self.get_kinetic_energy())
        ep=np.sum(energies)
        etot=ek+ep
        with open(atomConfigFilePath,openMode) as atomConfigFile:
            atomConfigFile.write(str(len(self))+'  atoms,Iteration (fs) =    0.2000000000E+01, Etot,Ep,Ek (eV) =   '+str(etot)+'  '+str(ep)+'   '+str(ek)+'\n')
            atomConfigFile.write(' Lattice vector (Angstrom)\n')
            for i in range(3):
                atomConfigFile.write(str(float(self.cell[i,0]))+'  '+str(float(self.cell[i,1]))+'  '+str(float(self.cell[i,2]))+'  \n')
            atomConfigFile.write(' Position (normalized), move_x, move_y, move_z\n')
            if not isApplyConstraint:
                lineTail='  1  1  1  '
                sPos=self.get_scaled_positions(True)
                for i in range(len(self)):
                    atomConfigFile.write(str(self.atomTypeList[i])+'    '+str(float(sPos[i,0]))+'    '+str(float(sPos[i,1]))+'    '+\
                    str(float(sPos[i,2]))+'   '+lineTail+'\n')        
            atomConfigFile.write('Force (-force, eV/Angstrom)\n')
            for i in range(len(self)):
                atomConfigFile.write(str(self.atomTypeList[i])+'    '+str(-forces[i,0])+'    '+str(-forces[i,1])+'    '+str(-forces[i,2])+'\n')
            atomConfigFile.write(' Velocity (bohr/fs)\n')
            for i in range(len(self)):
                atomConfigFile.write(str(self.atomTypeList[i])+'    '+str(velocities[i,0])+'    '+str(velocities[i,1])+'    '+str(velocities[i,2])+'\n')
            atomConfigFile.write('Atomic-Energy, Etot(eV),E_nonloc(eV),Q_atom:dE(eV)=  0.0\n')
            for i in range(len(self)):
                atomConfigFile.write(str(self.atomTypeList[i])+'    '+str(energies[i])+' \n')
            atomConfigFile.write('-'*80+'\n')                
            
    # def calAllNeighborStruct(self,isSave=False,isCheckFile=True,rMin=pm.rMin,rCut=pm.Rc_M):
    #     Image.calAllNeighborStruct(self,isSave,isCheckFile,rMin,rCut)
    #     if not pm.isFixedMaxNeighborNumForMd:
    #         self.preMaxNeighborNum=0
    #     pm.maxNeighborNum=int(self.neighborNumOfAllAtoms.max())
    #     if pm.maxNeighborNum!=self.preMaxNeighborNum:
    #         mempool = cp.get_default_memory_pool()
    #         mempool.free_all_blocks()
    #     self.preMaxNeighborNum=pm.maxNeighborNum
    
    def set_pos_cell(self):
        
        scaled_positions=cp.array(self.get_scaled_positions(True))
        cell=cp.array(np.array(self.cell))
        if ((self.cupyCell==cell).all() and (self.pos==scaled_positions).all()):
            return
        else:
            self.cupyCell=cell
            self.pos=scaled_positions
            self.isNewStep=True
        
    
    def set_positions(self, newpositions, apply_constraint=True):
        Atoms.set_positions(self,newpositions,apply_constraint)
        self.set_pos_cell()

    def calc_feat(self):
        cell=np.asfortranarray(cp.asnumpy(self.cupyCell.T))
        pos=np.asfortranarray(self.get_scaled_positions(True).T)

        for i in range(len(pm.use_Ftype)):
            if pm.use_Ftype[i]==1:
                calc_ftype1.gen_feature(cell,pos)
                feat_tmp=np.array(calc_ftype1.feat).transpose()
                dfeat_tmp=np.array(calc_ftype1.dfeat).transpose(1,2,0,3)
                num_neigh_alltypeM=calc_ftype1.num_neigh_alltypem
                list_neigh_alltypeM = calc_ftype1.list_neigh_alltypem
            if pm.use_Ftype[i]==2:
                calc_ftype2.gen_feature(cell,pos)
                feat_tmp=np.array(calc_ftype2.feat).transpose()
                dfeat_tmp=np.array(calc_ftype2.dfeat).transpose(1,2,0,3)
                num_neigh_alltypeM=calc_ftype2.num_neigh_alltypem
                list_neigh_alltypeM = calc_ftype2.list_neigh_alltypem
            if i==0:
                feat=feat_tmp
                dfeat=dfeat_tmp
            else:
                feat=np.concatenate((feat,feat_tmp),axis=1)
                dfeat=np.concatenate((dfeat,dfeat_tmp),axis=2)
        feat=np.asfortranarray(feat.transpose())
        dfeat=np.asfortranarray(dfeat.transpose(2,0,1,3))

        return feat,dfeat,num_neigh_alltypeM,list_neigh_alltypeM

    def calcEnergiesForces(self):
        
        start=time.time()
        
        '''
        if pm.cudaGpuOrder is not None:
            print("Before calc feat, used gpu memory and maxNeighborNum:")
            print(getGpuInfo()[2][pm.cudaGpuOrder],pm.maxNeighborNum)
        '''
        if hasattr(self,'neighborDistanceVectArrayOfAllAtoms'):
            del self.neighborDistanceVectArrayOfAllAtoms
#TODO:
        cell=np.asfortranarray(cp.asnumpy(self.cupyCell.T))
        pos=np.asfortranarray(self.get_scaled_positions(True).T)

        feat,dfeat,num_neigh_alltypeM,list_neigh_alltypeM=self.calc_feat()

        #print("cal feat time: ",time.time()-start)
        if self.isProfile:
            self.calcFeatTime+=time.time()-start
            star=time.time()


        self.calc.cal_energy_force(feat,dfeat,num_neigh_alltypeM,list_neigh_alltypeM,cell,pos)

        self.etot=float(self.calc.etot_pred)
        self.energies=np.array(self.calc.energy_pred)
        self.forces=-np.array(self.calc.force_pred).transpose()
        istatCalc=int(self.calc.istat)
        errorMsgCalc=calc_lin.error_msg.tostring().decode('utf-8').rstrip()
        if istatCalc!=0:
            raise ValueError(errorMsgCalc)
        self.isNewStep=False
        
        
        '''
        if pm.cudaGpuOrder is not None:
            print("After calc feat, used gpu memory and maxNeighborNum:")
            print(getGpuInfo()[2][pm.cudaGpuOrder],pm.maxNeighborNum)   
        '''
        #print("cal force time: ",time.time()-start)
        if self.isProfile:
            self.calcForceTime+=time.time()-start
            print(self.calcFeatTime,self.calcForceTime)
        
    def get_potential_energy(self,force_consistent=False):
        
        self.set_pos_cell()
        if self.isNewStep:
            # print('calc in ',sys._getframe().f_code.co_name)
            self.calcEnergiesForces()        
        return self.etot
    
    def get_potential_energies(self):
        
        self.set_pos_cell()
        if self.isNewStep:
            # print('calc in ',sys._getframe().f_code.co_name)
            self.calcEnergiesForces()        
        return self.energies
        
    def get_forces(self,apply_constraint=True, md=False):
        '''
        暂时还未考虑constraint!
        '''
        self.set_pos_cell()
        if self.isNewStep:
            # print('calc in ',sys._getframe().f_code.co_name)
            self.calcEnergiesForces()        
        return self.forces

    def calc_only_energy(self):
        
        start=time.time()
        self.set_pos_cell()
        '''
        if pm.cudaGpuOrder is not None:
            print("Before calc feat, used gpu memory and maxNeighborNum:")
            print(getGpuInfo()[2][pm.cudaGpuOrder],pm.maxNeighborNum)
        '''
        if hasattr(self,'neighborDistanceVectArrayOfAllAtoms'):
            del self.neighborDistanceVectArrayOfAllAtoms
#TODO:
        cell=np.asfortranarray(cp.asnumpy(self.cupyCell.T))
        pos=np.asfortranarray(self.get_scaled_positions(True).T)
      
        feat,dfeat,num_neigh_alltype,list_neigh_alltype=self.calc_feat()

        #print("cal feat time: ",time.time()-start)
        if self.isProfile:
            self.calcFeatTime+=time.time()-start
            star=time.time()
 
        self.calc.cal_only_energy(feat,dfeat,num_neigh_alltype,list_neigh_alltype,cell,pos)

        etot=float(self.calc.etot_pred)
        # energies=np.array(self.calc.energy_pred)
    
        istatCalc=int(self.calc.istat)
        errorMsgCalc=calc_lin.error_msg.tostring().decode('utf-8').rstrip()
        if istatCalc!=0:
            raise ValueError(errorMsgCalc)
        
        #print("cal force time: ",time.time()-start)
        if self.isProfile:
            self.calcForceTime+=time.time()-start
            print(self.calcFeatTime,self.calcForceTime)
        
        return etot


    def calculate_numerical_stress(self, atoms,d=1e-6, voigt=True):
        """Calculate numerical stress using finite difference."""

        stress = np.zeros((3, 3), dtype=float)

        cell = atoms.cell.copy()
        V = atoms.get_volume()
        for i in range(3):
            x = np.eye(3)
            x[i, i] += d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        
            eplus = atoms.calc_only_energy()

            x[i, i] -= 2 * d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            # self.isNewStep=True
            eminus = atoms.calc_only_energy()

            stress[i, i] = (eplus - eminus) / (2 * d * V)
            x[i, i] += d

            j = i - 2
            x[i, j] = d
            x[j, i] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            # self.isNewStep=True
            eplus = atoms.calc_only_energy()

            x[i, j] = -d
            x[j, i] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            # self.isNewStep=True
            eminus = atoms.calc_only_energy()

            stress[i, j] = (eplus - eminus) / (4 * d * V)
            stress[j, i] = stress[i, j]
        atoms.set_cell(cell, scale_atoms=True)
        self.set_pos_cell()
        # self.isNewStep=False
        if voigt:
            return stress.flat[[0, 4, 8, 5, 2, 1]]
        else:
            return stress


    def get_stress(self, voigt=True, apply_constraint=True,
                   include_ideal_gas=False):
        """Calculate stress tensor.

        Returns an array of the six independent components of the
        symmetric stress tensor, in the traditional Voigt order
        (xx, yy, zz, yz, xz, xy) or as a 3x3 matrix.  Default is Voigt
        order.

        The ideal gas contribution to the stresses is added if the
        atoms have momenta and ``include_ideal_gas`` is set to True.
        """

        # if self._calc is None:
        #     raise RuntimeError('Atoms object has no calculator.')

        stress = self.calculate_numerical_stress(self,voigt=voigt)
        shape = stress.shape

        if shape == (3, 3):
            # Convert to the Voigt form before possibly applying
            # constraints and adding the dynamic part of the stress
            # (the "ideal gas contribution").
            stress = full_3x3_to_voigt_6_stress(stress)
        else:
            assert shape == (6,)

        if apply_constraint:
            for constraint in self.constraints:
                if hasattr(constraint, 'adjust_stress'):
                    constraint.adjust_stress(self, stress)

        # Add ideal gas contribution, if applicable
        if include_ideal_gas and self.has('momenta'):
            stresscomp = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
            p = self.get_momenta()
            masses = self.get_masses()
            invmass = 1.0 / masses
            invvol = 1.0 / self.get_volume()
            for alpha in range(3):
                for beta in range(alpha, 3):
                    stress[stresscomp[alpha, beta]] -= (
                        p[:, alpha] * p[:, beta] * invmass).sum() * invvol

        if voigt:
            return stress
        else:
            return voigt_6_to_full_3x3_stress(stress)
            # return stress

    # def get_stresses(self, include_ideal_gas=False, voigt=True):
    #     """Calculate the stress-tensor of all the atoms.

    #     Only available with calculators supporting per-atom energies and
    #     stresses (e.g. classical potentials).  Even for such calculators
    #     there is a certain arbitrariness in defining per-atom stresses.

    #     The ideal gas contribution to the stresses is added if the
    #     atoms have momenta and ``include_ideal_gas`` is set to True.
    #     """
    #     # if self._calc is None:
    #     #     raise RuntimeError('Atoms object has no calculator.')
    #     stresses = self._calc.get_stresses(self)

    #     # make sure `stresses` are in voigt form
    #     if np.shape(stresses)[1:] == (3, 3):
    #         stresses_voigt = [full_3x3_to_voigt_6_stress(s) for s in stresses]
    #         stresses = np.array(stresses_voigt)

    #     # REMARK: The ideal gas contribution is intensive, i.e., the volume
    #     # is divided out. We currently don't check if `stresses` are intensive
    #     # as well, i.e., if `a.get_stresses.sum(axis=0) == a.get_stress()`.
    #     # It might be good to check this here, but adds computational overhead.

    #     if include_ideal_gas and self.has('momenta'):
    #         stresscomp = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
    #         if hasattr(self._calc, 'get_atomic_volumes'):
    #             invvol = 1.0 / self._calc.get_atomic_volumes()
    #         else:
    #             invvol = self.get_global_number_of_atoms() / self.get_volume()
    #         p = self.get_momenta()
    #         invmass = 1.0 / self.get_masses()
    #         for alpha in range(3):
    #             for beta in range(alpha, 3):
    #                 stresses[:, stresscomp[alpha, beta]] -= (
    #                     p[:, alpha] * p[:, beta] * invmass * invvol)
    #     if voigt:
    #         return stresses
    #     else:
    #         stresses_3x3 = [voigt_6_to_full_3x3_stress(s) for s in stresses]
    #         return np.array(stresses_3x3)
        
     

if __name__=='__main__':   
    input('Press Enter to quit test:')
