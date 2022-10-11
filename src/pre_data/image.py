#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
此module可能是暂时的,有可能最后放到parameters.py或者preprocess_data.py这样的文件中

此module的作用主要是解决从指定的文件夹中搜集所有的MOVMENT文件
然后每一个MOVEMENT文件按照从属的文件夹名给出一个system名字
更重要的是,要建立system和iamge的结构
'''
import use_para as pm
import os
import time
import numpy as np
import cupy as cp
import pandas as pd
#from ase.cell import Cell


class Image():
    '''
    建立一个image的class,只支持从一个System的实例用一个整数index建立一个image

    
    Parameters
    ---------------------
    fromSystem:                          System对象,源体系
    IndexInSystem:                       int对象,该图像在源体系中的序号



    Variable Attributes
    ---------------------    
    fromSystem:                       System对象,存储源体系的信息
    numOfAtoms:                       int对象,存储源体系,也是该图像的原子个数
    indexInSystem:                    int对象,存储了该图像在源体系中的序号
    atomTypeList:                     list对象,int,numOfAtoms长度,存储源体系,也是该图像中每个原子所属的原子种类
    atomTypeSet:                      tuple对象,int, 存储了源体系,也是该图像中所包含的所有原子种类的信息
    atomCountAsType:                  cp.array对象,int,长度和atomTypeSet一致,存储了源体系,也是该图像对应的每种原子的数目的信息
    isOrthogonalCell:                 bool对象,存储了源体系,也是该图像中cell是否是’不变且是三个方向基矢两两互相垂直’的信息
    isCheckSupercell:                 bool对象,存储了源体系,也是该图像中是否需要在后续计算近邻结构信息时考虑更大的超胞里有两个相同位置的原子在同一个原子近邻的情况的信息
    cupyCell:                         cp.array对象,float,3*3,存储整个体系中所有Image的cell的信息
    pos:                              cp.array对象,float,numOfAtoms*3, 存储该图像中所有原子的position的信息
    force:                            cp.array对象,float,numOfAtoms*3, 存储该图像中所有原子的force的信息
    velocity:                         cp.array对象,float,numOfAtoms*3, 存储该图像中所有原子的velocity的信息
    energy:                           cp.array对象,float,numOfAtoms, 存储该图像中所有原子的energy的信息
    ep:                               cp.array对象,float,1,存储该图像中Ep的信息
    dE:                               cp.array对象,float,1,存储该图像中dE的信息
    basicFunc:                        function对象,存储了源体系,也是该图像使用的basicFunc,默认是用cosBasciFunc
    basicDfunc:                       function对象,存储了源体系,也是该图像使用的basicDfunc,默认是用cosBasicDfunc
    structDataFilePath:               str对象,存储该图像中部分近邻结构信息的文件路径,为在self.fromSystem.structDataDir下的一个'.npy'文件,文件名为图像在体系中的序号
    featFilePath:                     str对象,存储该图像中feat信息的文件路径,为在self.fromSystem.featDir下的一个'.npy'文件,文件名为图像在体系中的序号
    dfeatFilePath:                    str对象,存储该图像中dfeat信息的文件路径,为在self.fromSystem.dfeatDir下的一个'.npy'文件,文件名为图像在体系中的序号



    Method Attributes
    ---------------------


    Internal Method Attributes
    ---------------------

      
    '''
    def __init__(self,
        atomTypeList,
        cell,
        pos,
        isOrthogonalCell=None,
        isCheckSupercell=None,
        atomTypeSet=None,
        atomCountAsType=None,        
        atomCategoryDict=None,
        force=None,
        energy=None,
        velocity=None,
        ep=None,
        dE=None):
        '''
        初始化,从源体系中继承相应的一部分数据        

        确定存储部分近邻结构数据、feat数据、dfeat数据的三个文件的路径


        Determine Attributes Directly
        ---------------------
        fromSystem:                       System对象,存储源体系的信息
        numOfAtoms:                       int对象,存储源体系,也是该图像的原子个数
        indexInSystem:                    int对象,存储了该图像在源体系中的序号
        atomTypeList:                     list对象,int,numOfAtoms长度,存储源体系,也是该图像中每个原子所属的原子种类
        atomTypeSet:                      tuple对象,int, 存储了源体系,也是该图像中所包含的所有原子种类的信息
        atomCountAsType:                  cp.array对象,int,长度和atomTypeSet一致,存储了源体系,也是该图像对应的每种原子的数目的信息
        isOrthogonalCell:                 bool对象,存储了源体系,也是该图像中cell是否是’不变且是三个方向基矢两两互相垂直’的信息
        isCheckSupercell:                 bool对象,存储了源体系,也是该图像中是否需要在后续计算近邻结构信息时考虑更大的超胞里有两个相同位置的原子在同一个原子近邻的情况的信息
        cupyCell:                         cp.array对象,float,3*3,存储整个体系中所有Image的cell的信息
        pos:                              cp.array对象,float,numOfAtoms*3, 存储该图像中所有原子的position的信息
        force:                            cp.array对象,float,numOfAtoms*3, 存储该图像中所有原子的force的信息
        velocity:                         cp.array对象,float,numOfAtoms*3, 存储该图像中所有原子的velocity的信息
        energy:                           cp.array对象,float,numOfAtoms, 存储该图像中所有原子的energy的信息
        ep:                               cp.array对象,float,1,存储该图像中Ep的信息
        dE:                               cp.array对象,float,1,存储该图像中dE的信息
        basicFunc:                        function对象,存储了源体系,也是该图像使用的basicFunc,默认是用cosBasciFunc
        basicDfunc:                       function对象,存储了源体系,也是该图像使用的basicDfunc,默认是用cosBasicDfunc
        structDataFilePath:               str对象,存储该图像中部分近邻结构信息的文件路径,为在self.fromSystem.structDataDir下的一个'.npy'文件,文件名为图像在体系中的序号
        featFilePath:                     (废弃)str对象,存储该图像中feat信息的文件路径,为在self.fromSystem.featDir下的一个'.npy'文件,文件名为图像在体系中的序号
        dfeatFilePath:                    (废弃)str对象,存储该图像中dfeat信息的文件路径,为在self.fromSystem.dfeatDir下的一个'.npy'文件,文件名为图像在体系中的序号

        '''
        self.atomTypeList=atomTypeList
        self.numOfAtoms=len(self.atomTypeList)
        #self.atomCountAsType=atomCountAsType
        self.atomCategoryDict=atomCategoryDict
        self.cupyCell=cell
        self.pos=pos
        self.isOrthogonalCell=isOrthogonalCell
        self.isCheckSupercell=isCheckSupercell
        self.force=force
        self.energy=energy
        self.velocity=velocity
        self.ep=ep
        self.dE=dE

        if (self.isCheckSupercell is None) or (self.isOrthogonalCell is None):
            self.checkCell()
        
        if atomTypeSet is None:
            self.atomTypeSet=tuple(set(self.atomTypeList))
        else:
            self.atomTypeSet=atomTypeSet
        
        if atomCountAsType is None:
            self.atomCountAsType=cp.array([self.atomTypeList.count(i) for i in self.atomTypeSet])
        else:
            self.atomCountAsType=atomCountAsType
            
        if atomCategoryDict is None:
            self.atomCategoryDict={}
            for atomType in pm.atomType:
                self.atomCategoryDict[atomType]=cp.where(cp.array(self.atomTypeList)==atomType)[0]
        

        
        '''
        self.fromSystem=fromSystem
        self.numOfAtoms=fromSystem.numOfAtoms
        self.indexInSystem=indexInSystem
        self.atomTypeList=fromSystem.atomTypeList
        self.atomTypeSet=fromSystem.atomTypeSet
        self.atomCountAsType=fromSystem.atomCountAsType
        self.isOrthogonalCell=fromSystem.isOrthogonalCell
        self.isCheckSupercell=fromSystem.isCheckSupercell
        self.cupyCell=fromSystem.allCell[indexInSystem]
        self.pos=fromSystem.allPos[indexInSystem]
        self.force=fromSystem.allForce[indexInSystem]
        self.energy=fromSystem.allEnergy[indexInSystem]
        self.velocity=fromSystem.allVelocity[indexInSystem]
        self.ep=fromSystem.allEp[indexInSystem]
        self.dE=fromSystem.allDE[indexInSystem]

        self.basicFunc=fromSystem.basicFunc
        self.basicDfunc=fromSystem.basicDfunc
        
        self.structDataFilePath=os.path.join(fromSystem.structDataDir,str(indexInSystem)+'.npy')
        '''
        #self.featFilePath=os.path.join(fromSystem.featDir,str(indexInSystem)+'.npy')
        #self.dfeatFilePath=os.path.join(fromSystem.dfeatDir,str(indexInSystem)+'.npy')
       
    def checkCell(self):
        '''
        若又是正交格子,检查是否三个方向基矢长度是否都大于2倍rCut
        若以上三者都符合,则可以不必考虑近邻有一个原子出现两次的情况


        Parameters
        ---------------------        
        None
        

        Returns
        ---------------------        
        None
        

        Determine Attributes Directly
        ---------------------      
        isConstantCell:                   bool对象,存储整个体系中cell是否不变的信息
        isOrthogonalCell:                 bool对象,存储了整个体系中cell是否是’不变且是三个方向基矢两两互相垂直’的信息
        isCheckSupercell:                 bool对象,存储了是否需要在后续计算近邻结构信息时考虑更大的超胞里有两个相同位置的原子在同一个原子近邻的情况的信息
        '''
        if self.isCheckSupercell is not None:
            return
        self.isOrthogonalCell=False
        self.isCheckSupercell=True
        cosAng=cp.zeros((3,))
        #sinAng=cp.zeros((3,))
        #areaVect=cp.zeros((3,))
        #self.heightVect=cp.zeros((3))
        if True:
            latticeConstants=cp.linalg.norm(self.cupyCell,axis=1)
            volume=cp.abs(cp.linalg.det(self.cupyCell))
            idx0=cp.arange(3)
            idxm=(idx0-1)%3
            idxp=(idx0+1)%3
            cosAng[idx0]=cp.sum(self.cupyCell[idxm]*self.cupyCell[idxp],axis=1)/(latticeConstants[idxm]*latticeConstants[idxp])
            '''
            cosAng[0]=cp.sum(self.allCell[0][1]*self.allCell[0][2])/latticeConstants[1]/latticeConstants[2]
            cosAng[1]=cp.sum(self.allCell[0][0]*self.allCell[0][2])/latticeConstants[0]/latticeConstants[2]
            cosAng[2]=cp.sum(self.allCell[0][1]*self.allCell[0][0])/latticeConstants[1]/latticeConstants[0]
            '''
            sinAng=cp.sqrt(1.0-cosAng*cosAng)
            areaVect=latticeConstants[idxm]*latticeConstants[idxp]*sinAng[idx0]
            '''
            areaVect[0]=latticeConstants[1]*latticeConstants[2]*sinAng[0]
            areaVect[1]=latticeConstants[0]*latticeConstants[2]*sinAng[1]
            areaVect[2]=latticeConstants[1]*latticeConstants[0]*sinAng[2]
            '''
            self.heightVect=volume/areaVect
        if abs(cosAng).max()<0.0001:
            self.isOrthogonalCell=True
        if self.heightVect.min()>pm.Rc_M*2.0:
            self.isCheckSupercell=False
        

    def calDistanceVectArray(self,isShiftZeroPoint=False,shiftZeroPointVect=cp.array([0.0,0.0,0.0])):
        '''
        得到任意两个原子的位矢差的矩阵,维数为numOfAtoms*numOfAtoms*3


        Parameters
        ---------------------        
        isShiftZeroPoint:                 bool对象,决定是否有零点偏移矢量,默认为False,事实上目前不起作用
        shiftZeroPointVect:               cp.array对象,float, 1*3, 零点偏移矢量,事实上目前不起作用
        

        Returns
        ---------------------        
        None
        

        Determine Attributes Directly
        ---------------------
        distanceVectArray:                cp.array对象,float, numOfAtoms*numOfAtoms*3, 存储了直角坐标系,以A为单位的所有原子两两距离矢量,(i,j)=>Rj-Ri,i为中心原子序号,j为近邻原子序号
        '''
        pos=self.pos.copy()
        if isShiftZeroPoint:
            pos=cp.matmul(pos,self.cupyCell)
            pos+=shiftPointVect
            invCell=cp.linalg.inv(self.cupyCell)
            pos=cp.matmul(pos,invCell)

            
        pos%=1.0
        pos%=1.0
            
        self.distanceVectArray=pos[cp.newaxis,:,:]-pos[:,cp.newaxis,:]
        self.distanceVectArray[self.distanceVectArray>0.5]=self.distanceVectArray[self.distanceVectArray>0.5]-1.0
        self.distanceVectArray[self.distanceVectArray<-0.5]=self.distanceVectArray[self.distanceVectArray<-0.5]+1.0
        self.distanceVectArray=cp.matmul(self.distanceVectArray,self.cupyCell)



    def calCellNeighborStruct(self,shiftVect=cp.zeros((3,)),shiftOrder=0,rMin=pm.rMin,rCut=pm.Rc_M):
        '''
        此方法用于计算Image中self.distanceVectArray移动一个固定的矢量后的所有原子的近邻结构
        此函数会自动调用之前计算的原子近邻的结构数据并更新之


        Parameters
        ---------------------        
        shiftVect:                        cp.array对象,计算近邻结构时在self.distanceVectArray上附加的矢量,默认值为零矢量,只于self.isCheckSupercell为真时起作用
        shiftOrder:                       int对象,若shiftVect不为零矢量时,此矢量在self.shiftVects中的序号
        

        Returns
        ---------------------        
        None
        

        Operate Attributes Directly(Not Determine)
        ---------------------
        isNeighborMask:                    cp.array对象,bool, numOfAtoms*numOfAtoms*3, 存储了原子两两之间是否相邻的信息,此信息是屡次附加不同shiftVect之后累计的结果
        neighborNumOfAllAtoms:             cp.array对象,int, numOfAtoms, 存储了图像中各个原子近邻原子的数目的信息,屡次附加不同shiftVect之后累计的结果
        neighborIndexOfCenterAtoms:        cp.array对象,int, 一维,和self.neighborIndexOfNeighborAtoms一起存储了图像中所有近邻原子对的序数对,屡次附加不同shiftVect之后累计的结果
        neighborIndexOfNeighborAtoms:      cp.array对象,int, 一维,和self.neighborIndexOfCenterAtoms一起存储了图像中所有近邻原子对的序数对,屡次附加不同shiftVect之后累计的结果
        
        '''
        rMinSquare=rMin*rMin
        rCutSquare=rCut*rCut
        if not shiftOrder:
            distanceVectArray=self.distanceVectArray            
        else:
            distanceVectArray=self.distanceVectArray+shiftVect      #此处很奇怪，如果先赋值self.distanceVectArray，再用if判断是否用+=shiftVect则会出现奇怪的错误，值得研究      
        distanceSquareArray=cp.sum(distanceVectArray**2,axis=2)
        isNeighborMask=(distanceSquareArray>rMinSquare) * (distanceSquareArray<rCutSquare)
        distanceVectArray[~isNeighborMask]=0.0
        distanceSquareArray[~isNeighborMask]=0.0
        neighborNumOfAllAtoms=isNeighborMask.sum(axis=1)
        #print(self.indexInSystem,shiftOrder,int(neighborNumOfAllAtoms.max()),shiftVect)


        
        self.isNeighborMask=self.isNeighborMask+isNeighborMask
        
        self.neighborNumOfAllAtoms=self.neighborNumOfAllAtoms+neighborNumOfAllAtoms

        neighborIndexOfCenterAtoms,neighborIndexOfNeighborAtoms=cp.where(isNeighborMask)
        if shiftOrder:
            neighborIndexOfNeighborAtoms+=shiftOrder*self.numOfAtoms
        
        self.neighborIndexOfCenterAtoms=cp.concatenate((self.neighborIndexOfCenterAtoms,neighborIndexOfCenterAtoms))
        self.neighborIndexOfNeighborAtoms=cp.concatenate((self.neighborIndexOfNeighborAtoms,neighborIndexOfNeighborAtoms))

        #print(self.neighborIndexOfCenterAtoms.shape,self.neighborIndexOfNeighborAtoms.shape)



    def calAllNeighborStruct(self,isSave=False,isCheckFile=True,rMin=pm.rMin,rCut=pm.Rc_M):
        '''
        此方法用于计算整个Image的近邻结构数据,但不包括最后的近邻列表和近邻原子位矢差矩阵的数据
        之所以有此方法是为了更快地搜集可以确定最大近邻原子数的数据
        而后两者在搜集上述数据时可以不涉及,尤其近邻列表的矩阵尺寸就需要最大近邻原子数这个参数

        
        Parameters
        ---------------------        
        isSave:                           bool对象,决定计算部分近邻信息后是否存储到文件, 默认值为False
        isCheckFile:                      boll对象,决定是否检查存储部分近邻信息的文件是否已经存在并读取已有文件,默认值为True
        rMin:                             float对象, 允许的近邻最小距离,默认值为pm.rMin,
        rMax:                             float对象, 允许的近邻最大距离,默认值为pm.rMax
        

        Returns
        ---------------------        
        None
        

        Determine Attributes Directly
        ---------------------
        shiftVects:                        cp.array对象,float, 7*3, 0矢量,以及self.cupyCell的六个方向单位矢量,只有在self.isCheckSupercell为真时会计算
        isNeighborMask:                    cp.array对象,bool, numOfAtoms*numOfAtoms*3, 存储了原子两两之间是否相邻的信息
        neighborNumOfAllAtoms:             cp.array对象,int, numOfAtoms, 存储了图像中各个原子近邻原子的数目的信息
        neighborIndexOfCenterAtoms:        cp.array对象,int, 一维,和self.neighborIndexOfNeighborAtoms一起存储了图像中所有近邻原子对的序数对
        neighborIndexOfNeighborAtoms:      cp.array对象,int, 一维,和self.neighborIndexOfCenterAtoms一起存储了图像中所有近邻原子对的序数对


        By self.calDistanceVectArray
        ---------------------        
        distanceVectArray:                 cp.array对象,float, numOfAtoms*numOfAtoms*3, 存储了直角坐标系,以A为单位的所有原子两两距离矢量,(i,j)=>Rj-Ri,i为中心原子序号,j为近邻原子序号
        '''


        if isCheckFile and os.path.isfile(self.structDataFilePath):
            self.shiftVects,self.distanceVectArray,self.isNeighborMask,self.neighborNumOfAllAtoms,self.neighborIndexOfCenterAtoms,self.neighborIndexOfNeighborAtoms=np.load(self.structDataFilePath,None,True)
            return

        
        self.isNeighborMask=False
        self.neighborNumOfAllAtoms=0
        self.neighborIndexOfCenterAtoms=cp.array([],cp.int32)        #此二处必须加上dtype,否则默认dtype是cp.float64
        self.neighborIndexOfNeighborAtoms=cp.array([],cp.int32)      #
        
        self.calDistanceVectArray()
        self.shiftVects=None
        self.calCellNeighborStruct()

        
        
            

        if self.isCheckSupercell:
            self.shiftVects=cp.zeros((7,3))
            self.shiftVects[1]=-self.cupyCell[0]
            self.shiftVects[2]=self.cupyCell[0]
            self.shiftVects[3]=-self.cupyCell[1]
            self.shiftVects[4]=self.cupyCell[1]
            self.shiftVects[5]=-self.cupyCell[2]
            self.shiftVects[6]=self.cupyCell[2]
            if self.isOrthogonalCell:
                for i in range(3):
                    if cp.sum(self.cupyCell[i]**2)<4.0*rCut*rCut:
                        self.calCellNeighborStruct(-self.cupyCell[i],i*2+1)
                        self.calCellNeighborStruct(self.cupyCell[i],i*2+2)
            else:
                for i in range(1,7):
                    self.calCellNeighborStruct(self.shiftVects[i],i)

       
            

        if isSave:
            structData=np.array((self.shiftVects,self.distanceVectArray,self.isNeighborMask,self.neighborNumOfAllAtoms,self.neighborIndexOfCenterAtoms,self.neighborIndexOfNeighborAtoms))
            np.save(self.structDataFilePath,structData,True)


    


        '''

            

        
        '''

    # def getMaxNeighborNum(self,isSaveStructData=False):
    #     '''
    #     计算一个Image的最大近邻原子数,不需要储存,返回之即可


    #     Parameters
    #     ---------------------        
    #     isSaveStructData:                 bool对象,决定计算部分近邻信息后是否存储到文件, 默认值为False
        

    #     Returns
    #     ---------------------        
    #     int(self.neighborNumOfAllAtoms.max()):  int对象,该图像中的最大近邻原子数
        
        

    #     By self.calAllNeighborStruct
    #     ---------------------
    #     shiftVects:                        cp.array对象,float, 7*3, 0矢量,以及self.cupyCell的六个方向单位矢量,只有在self.isCheckSupercell为真时会计算
    #     isNeighborMask:                    cp.array对象,bool, numOfAtoms*numOfAtoms*3, 存储了原子两两之间是否相邻的信息
    #     neighborNumOfAllAtoms:             cp.array对象,int, numOfAtoms, 存储了图像中各个原子近邻原子的数目的信息
    #     neighborIndexOfCenterAtoms:        cp.array对象,int, 一维,和self.neighborIndexOfNeighborAtoms一起存储了图像中所有近邻原子对的序数对
    #     neighborIndexOfNeighborAtoms:      cp.array对象,int, 一维,和self.neighborIndexOfCenterAtoms一起存储了图像中所有近邻原子对的序数对


    #     By self.calDistanceVectArray Indirectly
    #     ---------------------        
    #     distanceVectArray:                 cp.array对象,float, numOfAtoms*numOfAtoms*3, 存储直角坐标,以A为单位的所有原子两两距离矢量,(i,j)=>Rj-Ri,i为中心原子序号,j为近邻原子序号
    #     '''

    #     self.calAllNeighborStruct(isSaveStructData)
    #     return int(self.neighborNumOfAllAtoms.max())
        
        


    # def calAllNeighborInfo(self,isCheckFile=True):
    #     '''
    #     此方法用于计算整个Image的所有数据,包括最后的近邻列表和近邻原子位矢差矩阵等数据
    #     因为要用到最大近邻原子数和工作的所有原子类型列表等数据
    #     所以此函数的调用必须在初步处理了工作的全局变量才能开始
    #     此函数应只作为为计算feat和dfeat用


    #     Parameters
    #     ---------------------        
    #     isCheckFile:                            bool对象,决定是否检查存储部分近邻信息的文件是否已经存在并读取已有文件,默认值为True
        

    #     Returns
    #     ---------------------        
    #     None


    #     Determine Attributes Directly
    #     ---------------------
    #     neighborListOfAllAtoms:                 cp.array对象,int, numOfAtoms*pm.maxNeighborNum, (i,j)=> i是中心原子序号, 若(i,j)元素不为0,则是近邻原子序号+1
    #     neighborDistanceVectArrayOfAllAtoms:    cp.array对象,float, numOfAtoms*pm.maxNeighborNum*3, 所有近邻原子对(i,j)=> Rj-Ri,i为中心原子序号,j为近邻原子序号
    #     maskDictOfAllAtomTypesInNeighborList:   dict对象,int=>numOfAtoms*pm.maxNeighborNum的cp.array,从pm.atomTypeSet中的每种原子种类映射到self.neighborListOfAllAtoms中近邻是否此种原子
    #     neighborDistanceArrayOfAllAtoms:        cp.array对象,float, numOfAtoms*pm.maxNeighborNum,所有近邻原子对(i,j)=> |Rj-Ri|,i为中心原子序号,j为近邻原子序号
    #     neighborUnitVectArrayOfAllAtoms:        cp.array对象,float, numOfAtoms*pm.maxNeighborNum*3,所有近邻原子对(i,j)=> (Rj-Ri)/|Rj-Ri|,i为中心原子序号,j为近邻原子序号
    #     abDistanceArray:                        cp.array对象,float, numOfAtoms*pm.maxNeighborNum*pm.maxNeighborNum, (i,j1,jb2)=>所有原子的近邻原子两两间的距离矢量       
    #     abUnitVectArray:                        cp.array对象,float, numOfAtoms*pm.maxNeighborNum*pm.maxNeighborNum*3, (i,j1,jb2)=>所有原子的近邻原子两两间的距离矢量的单位矩阵
    #     basic2bFeatArrayOfAllNeighborPairs:     cp.array对象,float, numOfAtoms*pm.maxNeighborNum*pm.mulNumOf2bFeat,self.basicFunc(self.neighborDistanceArrayOfAllAtoms,pm.mFV2b(缩写))
    #     basic3bFeatArrayOfAllNeighborPairs:     cp.array对象,float, numOfAtoms*pm.maxNeighborNum*pm.mulNumOf3bFeat,self.basicFunc(self.neighborDistanceArrayOfAllAtoms,pm.mFV3b(缩写))
    #     basic3bFeatArrayOfAllABpairs:           cp.array对象,float,numOfAtoms*pm.maxNeighborNum*pm.maxNeighborNum*pm.mulNumOf3bFeat,self.basicFunc(self.abDistanceArrayOfAllAtoms,pm.mFV3b)
        

    #     By self.calAllNeighborStruct
    #     ---------------------
    #     shiftVects:                             cp.array对象,float, 7*3, 0矢量,以及self.cupyCell的六个方向单位矢量,只有在self.isCheckSupercell为真时会计算
    #     isNeighborMask:                         cp.array对象,bool, numOfAtoms*numOfAtoms*3, 存储了原子两两之间是否相邻的信息
    #     neighborNumOfAllAtoms:                  cp.array对象,int, numOfAtoms, 存储了图像中各个原子近邻原子的数目的信息
    #     neighborIndexOfCenterAtoms:             cp.array对象,int, 一维,和self.neighborIndexOfNeighborAtoms一起存储了图像中所有近邻原子对的序数对
    #     neighborIndexOfNeighborAtoms:           cp.array对象,int, 一维,和self.neighborIndexOfCenterAtoms一起存储了图像中所有近邻原子对的序数对


    #     By self.calDistanceVectArray Indirectly
    #     ---------------------        
    #     distanceVectArray:                      cp.array对象,float, numOfAtoms*numOfAtoms*3, 存储直角坐标,以A为单位的所有原子两两距离矢量,(i,j)=>Rj-Ri,i为中心原子序号,j为近邻原子序号
    #     '''       


    #     if hasattr(self,'neighborDistanceVectArrayOfAllAtoms'):
    #         return

        
    #     self.calAllNeighborStruct(isSave=False,isCheckFile=isCheckFile)
    #     self.neighborAtomOrderInNeighborListOfCenterAtoms=cp.concatenate([cp.arange(int(self.neighborNumOfAllAtoms[index])) for index in range(self.numOfAtoms)])

    #     argsort=cp.argsort(self.neighborIndexOfCenterAtoms)
    #     self.neighborIndexOfCenterAtoms=self.neighborIndexOfCenterAtoms[argsort]
    #     self.neighborIndexOfNeighborAtoms=self.neighborIndexOfNeighborAtoms[argsort]


        
    #     self.neighborListOfAllAtoms=-cp.ones((self.numOfAtoms,1+pm.maxNeighborNum),dtype=cp.int)  #此处必得有dtype,否则默认是cp.float64类型. 而zeros_like或者ones_like函数的dtype默认和模板array一致
    #     self.neighborDistanceVectArrayOfAllAtoms=cp.zeros((self.numOfAtoms,pm.maxNeighborNum,3))
    #     self.maskDictOfAllAtomTypesInNeighborList={}


    #     #计算近邻原子列表self.neighborListOfAllAtoms以及近邻矢量矩阵self.neighborDistanceVectArrayOfAllAtoms
    #     if self.isCheckSupercell:     #此为在有考虑大超胞的情况下的计算
    #         self.neighborDistanceVectArrayOfAllAtoms[self.neighborIndexOfCenterAtoms,self.neighborAtomOrderInNeighborListOfCenterAtoms]=\
    #                 self.distanceVectArray[self.neighborIndexOfCenterAtoms,self.neighborIndexOfNeighborAtoms%self.numOfAtoms]+self.shiftVects[self.neighborIndexOfNeighborAtoms//self.numOfAtoms]
    #         for indexOfAtom in range(self.numOfAtoms):
    #             neighborList=self.neighborIndexOfNeighborAtoms[self.neighborIndexOfCenterAtoms==indexOfAtom]%self.numOfAtoms     #%self.numOfAtoms)  #此处是否要用到%self.numOfAtoms需要再议…………
    #             self.neighborListOfAllAtoms[indexOfAtom,1:1+neighborList.shape[0]]=neighborList                  #两种情况下是否要在第一列加上中心原子序数也需要再议

    #     else:                       #此为在没有考虑大超胞的情况下的计算
    #         self.neighborDistanceVectArrayOfAllAtoms[self.neighborIndexOfCenterAtoms,self.neighborAtomOrderInNeighborListOfCenterAtoms]=self.distanceVectArray[self.neighborIndexOfCenterAtoms,self.neighborIndexOfNeighborAtoms]
    #         for indexOfAtom in range(self.numOfAtoms):
    #             neighborList=self.neighborIndexOfNeighborAtoms[self.neighborIndexOfCenterAtoms==indexOfAtom]
    #             self.neighborListOfAllAtoms[indexOfAtom,1:1+neighborList.shape[0]]=neighborList        
    #     self.neighborListOfAllAtoms[:,0]=cp.arange(self.numOfAtoms)
            


    #     '''
    #     #此为初始版本，其效率令人难以忍受，需要30+秒的时间，需要尝试提高效率
    #     #此段为计算每个中心原子在其近邻原子的近邻列表中的序号self.centerAtomOrderInNeighborListOfNeighborAtoms:
    #     self.centerAtomOrderInNeighborListOfNeighborAtoms=-cp.ones_like(self.neighborListOfAllAtoms)
    #     for i in range(self.numOfAtoms):
    #         for j in range(pm.maxNeighborNum):
    #             if self.neighborListOfAllAtoms[i,j]>-1:
    #                 k=self.neighborListOfAllAtoms[i,j]
    #                 k_line=k%self.numOfAtoms
    #                 k_shiftOrder=k//self.numOfAtoms
    #                 k_invShiftOrder=k_shiftOrder-min(k_shiftOrder,1)*(-1)**(k_shiftOrder%2)
    #                 k_inv=k_invShiftOrder*self.numOfAtoms+i
    #                 invOrder=cp.where(self.neighborListOfAllAtoms[k_line]==k_inv)[0][0]
    #                 self.centerAtomOrderInNeighborListOfNeighborAtoms[i,j]=invOrder
    #                 pass

                    
    #     #上一步计算出来的结果实际上是个矩阵，并不适合使用，下面实际将之变成可以供后续使用的一维数组，和self.neighborIndexOfCenterAtoms，self.neighborIndexOfNeighborAtoms一样长度
    #     #实际上，为了便于使用,在这一步也需要将self.neighborIndexOfNeighborAtoms对self.numOfAtoms取余
    #     #self.neighborIndexOfNeighborAtoms=self.neighborIndexOfNeighborAtoms%self.numOfAtoms      #后续的方法仍然要用到这个
    #     self.centerAtomOrderInNeighborListOfNeighborAtoms=self.centerAtomOrderInNeighborListOfNeighborAtoms[self.centerAtomOrderInNeighborListOfNeighborAtoms>-1]
    #     '''


    #     '''
    #     #第二版的方法仍然让人难以忍受其速度，所以寻找第三版完全不需要用到循环，全程矩阵操作的办法！
    #     #完全采用另外一种方法计算self.centerAtomOrderInNeighborListOfNeighborAtoms
    #     #这一次直接将之设置成一维的，直接利用cp.where进行计算
    #     #self.centerAtomOrderInNeighborListOfNeighborAtomsSlow=self.centerAtomOrderInNeighborListOfNeighborAtoms.copy()
    #     self.centerAtomOrderInNeighborListOfNeighborAtoms=-cp.ones_like(self.neighborIndexOfNeighborAtoms)
    #     k_shiftOrder=self.neighborIndexOfNeighborAtoms//self.numOfAtoms
    #     k_invShiftOrder=k_shiftOrder-cp.where(k_shiftOrder<1,k_shiftOrder,1)*(-1)**(k_shiftOrder%2)
    #     k_inv=k_invShiftOrder*self.numOfAtoms+self.neighborIndexOfCenterAtoms
    #     neighborIndexOfNeighborAtomsMod=self.neighborIndexOfNeighborAtoms%self.numOfAtoms
    #     for i in range(len(self.centerAtomOrderInNeighborListOfNeighborAtoms)):
    #         self.centerAtomOrderInNeighborListOfNeighborAtoms[i]=cp.where(self.neighborListOfAllAtoms[neighborIndexOfNeighborAtomsMod[i]]==k_inv[i])[0][0]
    #     #cp.where((self.neighborIndexOfCenterAtoms==self.neighborIndexOfNeighborAtoms[i]%self.numOfAtoms)*(self.neighborIndexOfNeighborAtoms==k_inv[i]))[0][0]
    #     #int(np.argwhere((cp.asnumpy(self.neighborIndexOfCenterAtoms)==int(self.neighborIndexOfNeighborAtoms[i]%self.numOfAtoms))*(cp.asnumpy(self.neighborIndexOfNeighborAtoms)==int(k_inv[i]))))            

    #     #self.centerAtomOrderInNeighborListOfNeighborAtoms=self.neighborAtomOrderInNeighborListOfCenterAtoms[self.centerAtomOrderInNeighborListOfNeighborAtoms]
    #     self.neighborIndexOfNeighborAtoms=neighborIndexOfNeighborAtomsMod
    #     '''

    #     neighborIndexOfNeighborAtomsMod=self.neighborIndexOfNeighborAtoms%self.numOfAtoms
    #     neighborIndexArgsort=((self.neighborIndexOfCenterAtoms<<16)+self.neighborIndexOfNeighborAtoms).argsort()        
    #     invNeighborIndexOfNeighborAtoms=self.neighborIndexOfNeighborAtoms//self.numOfAtoms
    #     invNeighborIndexOfNeighborAtoms==invNeighborIndexOfNeighborAtoms-cp.where(invNeighborIndexOfNeighborAtoms<1,invNeighborIndexOfNeighborAtoms,1)*(-1)**(invNeighborIndexOfNeighborAtoms%2)
    #     invNeighborIndexOfNeighborAtoms=invNeighborIndexOfNeighborAtoms*self.numOfAtoms+self.neighborIndexOfCenterAtoms        
    #     invNeighborIndexInvArgsort=((neighborIndexOfNeighborAtomsMod<<16)+invNeighborIndexOfNeighborAtoms).argsort().argsort()
    #     self.centerAtomOrderInNeighborListOfNeighborAtoms=self.neighborAtomOrderInNeighborListOfCenterAtoms[neighborIndexArgsort][invNeighborIndexInvArgsort]
    #     self.neighborIndexOfNeighborAtoms=neighborIndexOfNeighborAtomsMod
            

    #     '''
    #     for i in range(self.numOfAtoms):
    #         mask=self.neighborListOfAllAtoms[i]>-1
    #         k=self.neighborListOfAllAtoms[i][mask]
    #         k_line=k%self.numOfAtoms
    #         k_shiftOrder=k//self.numOfAtoms
    #         k_invShiftOrder=k_shiftOrder-cp.where(k_shiftOrder<1,k_shiftOrder,1)*(-1)**k_shiftOrder
    #         k_inv=k_invShiftOrder*self.numOfAtoms+i
    #         invOrder=cp.where(self.neighborListOfAllAtoms[k_line]==k_inv)[0][0]
    #         self.centerAtomOrderInNeighborListOfNeighborAtoms[i][mask]=invOrder
    #     '''

                

    #     #计算每种原子的mask,以一个len(pm.atomTypeSet) or pm.atomTypeNum*self.numOfAtoms*(1+pm.maxNeighborNum)的mask矩阵来将存储所有的结果,可命名为self.allTypeNeighborMaskArray
    #     #然后原来的self.maskDictOfAllAtomTypesInNeighborList可以是指向这个Array部分结果的矩阵
    #     self.allTypeNeighborMaskArray=cp.zeros((pm.atomTypeNum,self.numOfAtoms,1+pm.maxNeighborNum),dtype=cp.bool)
    #     self.allTypeNeighborMaskArray[:,:,0]=True
    #     atomTypeListArray=cp.array(self.atomTypeList)
    #     for i in range(pm.atomTypeNum):
    #         neighborAtomType=pm.atomType[i]
    #         self.allTypeNeighborMaskArray[i][self.neighborIndexOfCenterAtoms,1+self.neighborAtomOrderInNeighborListOfCenterAtoms]=(atomTypeListArray[self.neighborIndexOfNeighborAtoms]==neighborAtomType)
    #         self.maskDictOfAllAtomTypesInNeighborList[neighborAtomType]=self.allTypeNeighborMaskArray[i,:,1:]


    #     '''
    #     #此为原来计算self.maskDictOfAllAtomTypesInNeighobrList的方法，由于在后续计算中不适用,被废除
    #     for neighborAtomType in pm.atomTypeSet:
    #         if hasattr(cp,'isin'):
    #             self.maskDictOfAllAtomTypesInNeighborList[neighborAtomType]=(cp.asarray((self.neighborListOfAllAtoms>-1)*(cp.isin(self.neighborListOfAllAtoms%self.numOfAtoms,self.fromSystem.atomCategoryDict[neighborAtomType]))))
    #         else:
    #             self.maskDictOfAllAtomTypesInNeighborList[neighborAtomType]=(cp.asarray((self.neighborListOfAllAtoms>-1)*\
    #                         cp.asarray(np.isin(cp.asnumpy(self.neighborListOfAllAtoms%self.numOfAtoms),cp.asnumpy(self.fromSystem.atomCategoryDict[neighborAtomType])))))
    #     '''
        
        
        

            
            


    #     self.neighborDistanceArrayOfAllAtoms=cp.sqrt(cp.sum(self.neighborDistanceVectArrayOfAllAtoms**2,axis=2))
    #     self.neighborUnitVectArrayOfAllAtoms=cp.zeros_like(self.neighborDistanceVectArrayOfAllAtoms)
    #     mask=(self.neighborDistanceArrayOfAllAtoms>0)
    #     self.neighborUnitVectArrayOfAllAtoms[mask]=self.neighborDistanceVectArrayOfAllAtoms[mask]/cp.expand_dims(self.neighborDistanceArrayOfAllAtoms[mask],-1)   #不同维数矩阵的自动broad_cast计算似乎要求维数少的那一方的所有维数作为最后几维

        

    #     abDistanceVectArray=self.neighborDistanceVectArrayOfAllAtoms[:,:,cp.newaxis,:]-self.neighborDistanceVectArrayOfAllAtoms[:,cp.newaxis,:,:] #此处究竟哪个在前哪个在后合适上需要研究

    #     mask=self.neighborDistanceArrayOfAllAtoms==0
    #     abDistanceVectArray[mask]=0
    #     abDistanceVectArray.transpose(0,2,1,3)[mask]=0
    #     self.abDistanceArray=cp.sqrt(cp.sum(abDistanceVectArray**2,axis=3))
        
    #     #mask=(self.abDistanceArray<pm.rMin)+(self.abDistanceArray>pm.rCut)
    #     #abDistanceVectArray[mask]=0
    #     #self.abDistanceArray[mask]=0

    #     self.abUnitVectArray=cp.zeros((self.numOfAtoms,pm.maxNeighborNum,pm.maxNeighborNum,3))
    #     mask=self.abDistanceArray>0
    #     self.abUnitVectArray[mask]=abDistanceVectArray[mask]/cp.expand_dims(self.abDistanceArray[mask],-1)


    #     self.basic2bFeatArrayOfAllNeighborPairs=self.basicFunc(self.neighborDistanceArrayOfAllAtoms,pm.mulFactorVectOf2bFeat)#此处对整个近邻距离矩阵用basicFunc计算一个结果储存,是2bFeat相关
    #     self.basic3bFeatArrayOfAllNeighborPairs=self.basicFunc(self.neighborDistanceArrayOfAllAtoms,pm.mulFactorVectOf3bFeat)#此处对整个近邻距离矩阵用basicFunc计算一个结果储存
    #     self.basic3bFeatArrayOfAllABpairs=self.basicFunc(self.abDistanceArray,pm.mulFactorVectOf3bFeat,pm.Rcut2)  #此处对整个的三体的ab距离矩阵用basicFunc计算一个结果储存,是3bFeat相关,2b和3b不一样
        

    #     basic2bDfeatArrayOfAllNeighborPairs=self.basicDfunc(self.neighborDistanceArrayOfAllAtoms,pm.mulFactorVectOf2bFeat)
    #     self.allNeighborPairs2bDfeatArray=basic2bDfeatArrayOfAllNeighborPairs[:,:,:,cp.newaxis]*self.neighborUnitVectArrayOfAllAtoms[:,:,cp.newaxis,:]
    #     #self.allNeighborPairs2bDfeatArray=cp.einsum('ija,ijr->ijar',basic2bDfeatArrayOfAllNeighborPairs,self.neighborUnitVectArrayOfAllAtoms)
    #     basic3bDfeatArrayOfAllNeighborPairs=self.basicDfunc(self.neighborDistanceArrayOfAllAtoms,pm.mulFactorVectOf3bFeat)
    #     self.allNeighborPairs3bDfeatArray=basic3bDfeatArrayOfAllNeighborPairs[:,:,:,cp.newaxis]*self.neighborUnitVectArrayOfAllAtoms[:,:,cp.newaxis,:]
    #     #self.allNeighborPairs3bDfeatArray=cp.einsum('ija,ijr->ijar',basic3bDfeatArrayOfAllNeighborPairs,self.neighborUnitVectArrayOfAllAtoms)
    #     basic3bDfeatArrayOfAllABpairs=self.basicDfunc(self.abDistanceArray,pm.mulFactorVectOf3bFeat,pm.Rcut2)
    #     self.allABpairs3bDfeatArray=basic3bDfeatArrayOfAllABpairs[:,:,:,:,cp.newaxis]*self.abUnitVectArray[:,:,:,cp.newaxis,:]
    #     #self.allABpairs3bDfeatArray=cp.einsum('ijka,ijkr->ijkar',basic3bDfeatArrayOfAllABpairs,self.abUnitVectArray)
        
       


    
if __name__=='__main__':   
    input('Press Enter to quit test:')
