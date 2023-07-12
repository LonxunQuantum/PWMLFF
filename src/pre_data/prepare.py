import os
import default_para as pm
import numpy as np
import numpy as cp
import pandas as pd
import re
import math

#added this line 

def pdFloatFormat(x):
    li1=re.split('E+', str(x),flags=re.IGNORECASE)
    li2=re.split('E-', str(x),flags=re.IGNORECASE)
    if len(li1)>1 or len(li2)>1:        
        if len(li1)>1:            
            n1=len(li1[0].replace('.',''))-1
            n2=int(li1[1])
            nx=n2-n1
            n=0 if nx>0 else nx
        else:
            n1=len(li2[0].replace('.',''))-1   
            n2=int(li2[1])
            n=n1+n2
        print(x,n)
        x2=('{:.'+str(n)+'f}').format(x)
    else:
        x2=str(x)
    return x2

def collectAllSourceFiles(workDir=pm.trainSetDir,sourceFileName='MOVEMENT'):
    '''
    搜索工作文件夹，得到所有MOVEMENT文件的路径，并将之存储在pm.sourceFileList中
    
    Determine parameters:
    ---------------------
    pm.sourceFileList:            List对象，罗列了所有MOVEMENT文件的路径        
    '''
    if not os.path.exists(workDir):
        raise FileNotFoundError(workDir+'  is not exist!')
    for path,dirList,fileList in os.walk(workDir):
        if sourceFileName in fileList:
            #pm.sourceFileList.append(os.path.abspath(path))
            # use relative path
            pm.sourceFileList.append(path)
    

def savePath(featSaveForm='C'):
    '''
    save path to file
    '''
    featSaveForm=featSaveForm.upper()
    pm.numOfSystem=len(pm.sourceFileList)
    with open(pm.fbinListPath,'w') as fbinList:
        fbinList.write(str(pm.numOfSystem)+'\n')
        #fbinList.write(str(os.path.abspath(pm.trainSetDir))+'\n')
        # use relative path
        fbinList.write(str(pm.trainSetDir)+'\n')
        for system in pm.sourceFileList:
            fbinList.write(str(system)+'\n')

def combineMovement():
    '''
    combine MOVEMENT file
    '''
    #with open(os.path.join(os.path.abspath(pm.trainSetDir),'MOVEMENTall'), 'w') as outfile:     
    with open(os.path.join(pm.trainSetDir,'MOVEMENTall'), 'w') as outfile:     
        # Iterate through list 
        for names in pm.sourceFileList:     
            # Open each file in read mode 
            with open(os.path.join(names,'MOVEMENT')) as infile:     
                # read the data from file1 and 
                # file2 and write it in file3 
                outfile.write(infile.read()) 
    
            # Add '\n' to enter data of file2 
            # from next line 
            outfile.write("\n")

def movementUsed():
    '''
    index images not used
    '''
    badImageNum=0
    for names in pm.sourceFileList:
        # image=np.loadtxt(os.path.join(os.path.abspath(names),'info.txt'))
        #image=pd.read_csv(os.path.join(os.path.abspath(names),'info.txt.Ftype'+str(pm.use_Ftype[0])), header=None,delim_whitespace=True).values[:,0].astype(int)
        image=pd.read_csv(os.path.join(names,'info.txt.Ftype'+str(pm.use_Ftype[0])), header=None,delim_whitespace=True).values[:,0].astype(int)
        badimage=image[3:]
        badImageNum=badImageNum+len(badimage)
    
    with open(os.path.join(pm.trainSetDir,'imagesNotUsed'), 'w') as outfile:     
        # Iterate through list 
        outfile.write(str(badImageNum)+'  \n')
        index=0
        
        for names in pm.sourceFileList:
            image=np.loadtxt(os.path.join(names,'info.txt'))
            badimage=image[3:]
            numOfImage=image[2]
            for i in range(len(badimage)):
                outfile.write(str(badimage[i]+index)+'\n')
            index=index+numOfImage

def writeGenFeatInput():
    
    """
        do this chunk for some fortran executables
    """
    if True:
        '''
            gen_2b_feature.in
                6.0, 200   !  Rc_M, m_neigh
                2          ! ntype 
                6          ! iat-type 
                5.0,3.0,2,0.2,0.5      !Rc,Rm,iflag_grid,fact_base,dR1
                30       ! n3b1, n3b2 
                29          ! iat-type 
                6.0,3.0,2,0.2,0.5      !Rc,Rm,iflag_grid,fact_base,dR1
                40       !  n3b1, n3b2 
                0.3    ! E_tolerance (eV)  
                2     ! iflag_ftype:1,2,3 (three different ways for 3b feature, different sin peak spans) 
                1     ! recalc_grid, 0 read from file, 1 recalc 
        '''
        #gen 2b feature input
        with open(pm.Ftype1InputPath,'w') as GenFeatInput:
            GenFeatInput.write(str(pm.Rc_M)+', '+str(pm.maxNeighborNum)+'             !  Rc_M, m_neigh \n')
            GenFeatInput.write(str(len(pm.atomType))+'               ! ntype \n')
            for i in range(pm.atomTypeNum):
                GenFeatInput.write(str(pm.atomType[i])+'              ! iat-type \n')
                GenFeatInput.write(str(pm.Rc_M)+','+str(pm.Rc_min)+','+str(pm.Ftype1_para['iflag_grid'][i])+','+str(pm.Ftype1_para['fact_base'][i])+','+\
                    str(pm.Ftype1_para['dR1'][i])+'      !Rc,Rm,iflag_grid,fact_base,dR1 \n')
                GenFeatInput.write(str(pm.Ftype1_para['numOf2bfeat'][i])+'              ! n2b \n')
            # GenFeatInput.write(str(pm.maxNeighborNum)+'      ! m_neigh \n')
            GenFeatInput.write(str(pm.E_tolerance)+'    ! E_tolerance  \n')
            GenFeatInput.write(str(pm.Ftype1_para['iflag_ftype'])+'    ! iflag_ftype \n')
            GenFeatInput.write(str(pm.recalc_grid)+'    ! recalc_grid, 0 read from file, 1 recalc \n')

    for ftype in pm.use_Ftype:
        #if ftype == 1:
        #print ("atom type from pp:",pm.atomType)
        # some program will use the gen_2b_feature.in to get some parameters.
        if ftype==1:
            '''
            gen_2b_feature.in
                6.0, 200   !  Rc_M, m_neigh
                2          ! ntype 
                6          ! iat-type 
                5.0,3.0,2,0.2,0.5      !Rc,Rm,iflag_grid,fact_base,dR1
                30       ! n3b1, n3b2 
                29          ! iat-type 
                6.0,3.0,2,0.2,0.5      !Rc,Rm,iflag_grid,fact_base,dR1
                40       !  n3b1, n3b2 
                0.3    ! E_tolerance (eV)  
                2     ! iflag_ftype:1,2,3 (three different ways for 3b feature, different sin peak spans) 
                1     ! recalc_grid, 0 read from file, 1 recalc 
            '''
            #gen 2b feature input
            with open(pm.Ftype1InputPath,'w') as GenFeatInput:
                GenFeatInput.write(str(pm.Rc_M)+', '+str(pm.maxNeighborNum)+'             !  Rc_M, m_neigh \n')
                GenFeatInput.write(str(len(pm.atomType))+'               ! ntype \n')
                for i in range(pm.atomTypeNum):
                    GenFeatInput.write(str(pm.atomType[i])+'              ! iat-type \n')
                    GenFeatInput.write(str(pm.Rc_M)+','+str(pm.Rc_min)+','+str(pm.Ftype1_para['iflag_grid'][i])+','+str(pm.Ftype1_para['fact_base'][i])+','+\
                        str(pm.Ftype1_para['dR1'][i])+'      !Rc,Rm,iflag_grid,fact_base,dR1 \n')
                    GenFeatInput.write(str(pm.Ftype1_para['numOf2bfeat'][i])+'              ! n2b \n')
                # GenFeatInput.write(str(pm.maxNeighborNum)+'      ! m_neigh \n')
                GenFeatInput.write(str(pm.E_tolerance)+'    ! E_tolerance  \n')
                GenFeatInput.write(str(pm.Ftype1_para['iflag_ftype'])+'    ! iflag_ftype \n')
                GenFeatInput.write(str(pm.recalc_grid)+'    ! recalc_grid, 0 read from file, 1 recalc \n')
        if ftype == 2:
            '''
            gen_3b_feature.in
                6.0, 200   !  Rc_M, m_neigh
                2          ! ntype 
                6          ! iat-type 
                5.0,8.0,3.0,2,0.2,0.5,0.5      !Rc,Rc2,Rm,iflag_grid,fact_base,dR1,dR2 
                2,3       ! n3b1, n3b2 
                29          ! iat-type 
                5.5,10.5,3.0,2,0.2,0.5,0.5      !Rc,Rc2,Rm,iflag_grid,fact_base,dR1,dR2 
                3,4       !  n3b1, n3b2 
                0.3    ! E_tolerance (eV)  
                2     ! iflag_ftype:1,2,3 (three different ways for 3b feature, different sin peak spans) 
                0     ! recalc_grid, 0 read from file, 1 recalc 
            '''
            #gen 3b feature input
            with open(pm.Ftype2InputPath,'w') as GenFeatInput:
                GenFeatInput.write(str(pm.Rc_M)+', '+str(pm.maxNeighborNum)+'             !  Rc_M, m_neigh \n')
                GenFeatInput.write(str(len(pm.atomType))+'               ! ntype \n')
                for i in range(pm.atomTypeNum):
                    GenFeatInput.write(str(pm.atomType[i])+'          ! iat-type \n')        
                    GenFeatInput.write(str(pm.Rc_M)+','+str(pm.Rc_M)+','+str(pm.Rc_min)+','+str(pm.Ftype2_para['iflag_grid'][i])+','+str(pm.Ftype2_para['fact_base'][i])+','+\
                        str(pm.Ftype2_para['dR1'][i])+','+str(pm.Ftype2_para['dR2'][i])+'      !Rc,Rc2,Rm,iflag_grid,fact_base,dR1,dR2 \n')
                    GenFeatInput.write(str(pm.Ftype2_para['numOf3bfeat1'][i])+','+str(pm.Ftype2_para['numOf3bfeat2'][i])+'       ! n3b1, n3b2 \n')
                # GenFeatInput.write(str(pm.maxNeighborNum)+'      ! m_neigh \n')
                GenFeatInput.write(str(pm.E_tolerance)+'    ! E_tolerance  \n')
                GenFeatInput.write(str(pm.Ftype2_para['iflag_ftype'])+'    ! iflag_ftype \n')
                GenFeatInput.write(str(pm.recalc_grid)+'    ! recalc_grid, 0 read from file, 1 recalc \n')
        if ftype == 3:
            '''
            gen_2bgauss_feature.in
                5.4, 150         ! Rc_m, m_neigh
                1                ! ntype
                1                ! atomic number of first atom type
                5.4              ! Rc
                8                ! n2b, and next n2b lines are paras
                    1.0 1.0      ! r1, w1
                    1.5 1.0      ! r2, w2
                    2.0 1.0      !
                    2.0 2.0      !
                    2.0 3.0      !
                    2.5 1.0      !
                    3.0 1.5      !
                    4.0 2.0      !
                0.3              ! E_tol
            '''
            #gen 2bgauss feature input
            with open(pm.FtypeiiInputPath[ftype],'w') as GenFeatInput:
                GenFeatInput.write(str(pm.Rc_M)+', '+str(pm.maxNeighborNum)+'             !  Rc_M, m_neigh \n')
                GenFeatInput.write(str(len(pm.atomType))+'               ! ntype \n')
                for i in range(pm.atomTypeNum):
                    Rc = pm.Ftype3_para['Rc'][i]
                    n2b = pm.Ftype3_para['n2b'][i]
                    nw = len(pm.Ftype3_para['w'])
                    GenFeatInput.write(str(pm.atomType[i])+'          ! iat-type \n')        
                    GenFeatInput.write(str(Rc)+ '    !Rc\n')
                    GenFeatInput.write(str(n2b)+'       ! n2b \n')
                    tmp_grid0 = np.linspace(1.0, Rc, math.ceil(n2b/nw))
                    tmp_grid = np.kron(tmp_grid0, np.ones(nw)) 
                    
                    for j in range(pm.Ftype3_para['n2b'][i]):
                        wj = pm.Ftype3_para['w'][j%nw]
                        GenFeatInput.write("%10.6f %10.6f\n" % (tmp_grid[j], wj))
                GenFeatInput.write(str(pm.E_tolerance)+'    ! E_tolerance  \n')
                '''
                    for j in range(pm.Ftype3_para['n2b'][i]):
                        GenFeatInput.write(str(pm.Ftype3_para['r'][i][j]) + '  ' + str(pm.Ftype3_para['w'][i][j]) + ' ! ri, wi\n')
                '''

        if ftype == 4: # 3bcos
            '''
            gen_3bcos_feature.in
                5.4, 150         ! Rc_m, m_neigh
                1                ! ntype
                14               ! atomic number of first atom type, 14 for Si
                5.4              ! Rc
                12               ! n3b, and next n3b lines are paras
                  1.0 1.0 1.0    ! zeta1, w1, lamda1
                  1.0 1.0 -1.0   ! zeta2, w2, lamda2
                  2.0 1.0 1.0    !
                  2.0 1.0 -1.0   !
                  1.0 2.0 1.0    !
                  1.0 2.0 -1.0   !
                  2.0 2.0 1.0    !
                  2.0 2.0 -1.0   !
                  1.0 3.0 1.0    !
                  1.0 3.0 -1.0   !
                  2.0 3.0 1.0    !
                  2.0 3.0 -1.0   !
                0.3              ! E_tol
            '''
            #gen 3bcos feature input
            with open(pm.FtypeiiInputPath[ftype],'w') as GenFeatInput:
                GenFeatInput.write('%10.4f  %5d              ! Rc_M, m_neigh \n' % (pm.Rc_M, pm.maxNeighborNum))
                GenFeatInput.write('%10d                     ! ntype \n' % (len(pm.atomType)))
                for i in range(pm.atomTypeNum):
                    # pre, create a list
                    zeta = pm.Ftype4_para['zeta'][i]
                    w = pm.Ftype4_para['w'][i]
                    lambda_ = [1.0, -1.0] # lambda is a inner-python name, use lambda_ here, yes ugly
                    tmp_list = []
                    for ii in range(len(zeta)):
                        for jj in range(len(w)):
                            for kk in range(2):
                                tmp_list.append([zeta[ii], w[jj], lambda_[kk]])
                    # pre, end 
                    GenFeatInput.write('%10d                     ! iat-type \n' % (pm.atomType[i]))        
                    GenFeatInput.write('%10.4f                     ! Rc \n' % (pm.Ftype4_para['Rc'][i]))
                    GenFeatInput.write('%10d                     ! n3b \n' % (pm.Ftype4_para['n3b'][i]))
                    for j in range(pm.Ftype4_para['n3b'][i]):
                        GenFeatInput.write('%10.4f  %10.4f  %5.2f  ! zeta, width, lambda\n' % (tmp_list[j][0], tmp_list[j][1], tmp_list[j][2]))
                GenFeatInput.write('%10.4f                     ! E_tolerance  \n' % (pm.E_tolerance))

        if ftype == 5: # MTP
            '''
            gen_MTP_feature.in
                5.4, 150                                     ! Rc_M, m_neigh
                1                                            ! ntype
                14                                           ! iat-type
                5.4, 0.5                                     ! Rc, Rm
                5                                            ! MTP_line
                1, 4, 0, ( )                                 ! num_tensor, mu, v, (tensor_ind)
                2, 3,3, 0,0, ( ), ( )                        ! num=2, mu1,mu2, v1,v2, (tensor1), tensor2)
                2, 3,3, 1,1, ( 21 ), ( 11 )                  ! 
                2, 3,3, 2,2, ( 21, 22 ), ( 11, 12 )          ! 
                3, 2,2,2, 2,1,1 ( 21, 31 ), ( 11 ), ( 12 )   ! 
                0.3                                          ! E_tol
            '''
            #gen MTP feature input
            with open(pm.FtypeiiInputPath[ftype],'w') as GenFeatInput:
                GenFeatInput.write(str(pm.Rc_M)+', '+str(pm.maxNeighborNum)+'                            ! Rc_M, m_neigh \n')
                GenFeatInput.write(str(len(pm.atomType))+'                                         ! ntype \n')
                for i in range(pm.atomTypeNum):
                    GenFeatInput.write(str(pm.atomType[i])+'                                      ! iat-type \n')        
                    GenFeatInput.write(str(pm.Ftype5_para['Rc'][i]) + '  ' + str(pm.Ftype5_para['Rm'][i]) + 
                                        '                               ! Rc, Rm\n')
                    GenFeatInput.write(str(pm.Ftype5_para['n_MTP_line'][i])+
                                        '                                          ! n_MTP_line \n')
                    for j in range(pm.Ftype5_para['n_MTP_line'][i]):
                        GenFeatInput.write(str(pm.Ftype5_para['tensors'][i][j]) + '   !num_tensor, {mu}, {nu}, (tensor_ind)\n')
                GenFeatInput.write(str(pm.E_tolerance)+'    ! E_tolerance  \n')

        if ftype == 6: # SNAP
            '''
            gen_SNAP_feature.in
                5.4, 150    ! Rc_M, m_neigh
                1           ! ntype
                14          ! iat-type
                5.4         ! Rc
                3, 2        ! J, n_w_line
                1.0, 0.3    ! w1, w2 (the first w_line)
                0.3, 1.0    ! w1, w2 # if number of atomtype > 1, you need more lines before E_tol
                0.3         ! E_tol
            '''
            #gen SNAP feature input
            with open(pm.FtypeiiInputPath[ftype],'w') as GenFeatInput:
                GenFeatInput.write(str(pm.Rc_M)+', '+str(pm.maxNeighborNum)+'                            ! Rc_M, m_neigh \n')
                GenFeatInput.write(str(len(pm.atomType))+'                                         ! ntype \n')
                for i in range(pm.atomTypeNum):
                    GenFeatInput.write(str(pm.atomType[i])+'                                      ! iat-type \n')        
                    GenFeatInput.write(str(pm.Ftype6_para['Rc'][i])  + '                                  ! Rc \n')
                    GenFeatInput.write(str(pm.Ftype6_para['J'][i])+ '  '+ str(pm.Ftype6_para['n_w_line'][i]) + 
                                         '                                          ! n_MTP_line \n')
                    for j in range(pm.Ftype6_para['n_w_line'][i]):
                        GenFeatInput.write(str(pm.Ftype6_para['w1'][i][j]) + '  ' + str(pm.Ftype6_para['w2'][i][j]) + '  ' + str(pm.Ftype6_para['w3'][i][j]) + 
                                            '                                          ! w1, w2, w3 \n')
                GenFeatInput.write(str(pm.E_tolerance)+'    ! E_tolerance  \n')

        if ftype == 7: # deepMD1
            '''
            gen_deepMD1_feature.in
                5.4, 150             ! Rc_M, m_neigh
                1                    ! ntype
                14                   ! iat[1]
                5.4, 3.0, 1.0        ! Rc, Rc2, Rm
                4, 1.0               ! M, weight_r
                0.3                  ! E_tol
            '''
            #gen deepMD1 feature input
            with open(pm.FtypeiiInputPath[ftype],'w') as GenFeatInput:

                GenFeatInput.write(str(pm.Ftype7_para['Rc'][0])+', '+str(pm.maxNeighborNum)+'                            ! Rc_M, m_neigh \n')
                GenFeatInput.write(str(len(pm.atomType))+'                                         ! ntype \n')
                
                for i in range(pm.atomTypeNum):
                    GenFeatInput.write(str(pm.atomType[i])+'                                      ! iat-type \n')        
                    GenFeatInput.write(str(pm.Ftype7_para['Rc'][i]) +  '  ' + str(pm.Ftype7_para['Rc2'][i]) + '  ' + 
                                       str(pm.Ftype7_para['Rm'][i])  +
                                       '                                  ! Rc, Rc2, Rm \n')
                    GenFeatInput.write(str(pm.Ftype7_para['M'][i]) + '  ' + str(pm.Ftype7_para['weight_r'][i]) +
                                        '                                    ! M1, weight_r\n')
                    # write M2  
                    GenFeatInput.write(str(pm.Ftype7_para['M2'][i]) + '  ' + str(pm.Ftype7_para['weight_r'][i]) + '                       ! M2, weight_r\n')
                    
                GenFeatInput.write(str(pm.E_tolerance)+'    ! E_tolerance  \n')

        if ftype == 8: # deepMD2
            '''
            gen_deepMD2_feature.in
                5.4, 150             ! Rc_M, m_neigh
                1                    ! ntype
                14                   ! iat[1]
                5.4                  ! Rc
                8, 1.0               ! M, weight_r (M lines)
                    1.0  1.0         ! rg, w: gauussian position and width
                    1.5  1.0
                    2.0  1.0
                    2.0  2.0
                    2.0  3.0
                    2.5  1.0
                    3.0  1.0
                    4.0  2.0
                0.3                  ! E_tol
            '''
            #gen deepMD2 feature input
            with open(pm.FtypeiiInputPath[ftype],'w') as GenFeatInput:
                GenFeatInput.write(str(pm.Rc_M)+', '+str(pm.maxNeighborNum)+'                            ! Rc_M, m_neigh \n')
                GenFeatInput.write(str(len(pm.atomType))+'                                         ! ntype \n')
                if type(pm.Ftype8_para['w'][0]) == list:
                    print('Feature type 8, error, use default [1.0, 1.5, 2.0, 2.5] in this calculation')
                    print('Ftype8 key "w" should be a list of number, not list of list')
                    pm.Ftype8_para['w'] = [1.0, 1.5, 2.0, 2.5]
                for i in range(pm.atomTypeNum):
                    Rc = pm.Ftype8_para['Rc'][i]
                    n2b = pm.Ftype8_para['M'][i]
                    nw = len(pm.Ftype8_para['w'])
                    GenFeatInput.write(str(pm.atomType[i])+'                                      ! iat-type \n')        
                    GenFeatInput.write(str(pm.Ftype8_para['Rc'][i]) +
                                        '                                  ! Rc \n')
                    GenFeatInput.write(str(pm.Ftype8_para['M'][i]) + '  ' + str(pm.Ftype8_para['weight_r'][i]) +
                                        '                                    ! M, weight_r    \n')
                    tmp_grid0 = np.linspace(1.0, Rc, math.ceil(n2b/nw))
                    tmp_grid = np.kron(tmp_grid0, np.ones(nw))
                    for j in range(pm.Ftype8_para['M'][i]):
                        wj = pm.Ftype8_para['w'][j%nw]
                        GenFeatInput.write("%10.6f %10.6f\n" % (tmp_grid[j], wj))
                GenFeatInput.write(str(pm.E_tolerance)+'    ! E_tolerance  \n')


def writeFitInput():
    #fitInputPath=os.path.join(pm.fitModelDir,'fit.input')

    natom=200
    m_neigh=pm.maxNeighborNum
    n_image=200
    with open(pm.fitInputPath_lin,'w') as fitInput:
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
        fitInput.write(str(pm.dwidth)+'\n')
                        

def readFeatnum():
    collectAllSourceFiles()
    featnum=0
    for i in pm.use_Ftype:
        with open(os.path.join(pm.sourceFileList[0],'info.txt.Ftype'+str(i)),'r') as sourceFile:
            featnum=featnum+int(sourceFile.readline().split()[0])
    
    pm.realFeatNum=featnum
    pm.nFeats=np.array([pm.realFeatNum,pm.realFeatNum,pm.realFeatNum])
            # pm.fortranFitFeatNum2[i]=pm.fortranFitFeatNum0[i]
    # pm.fortranFitFeatNum0=pm.realFeatNum*np.ones((pm.atomTypeNum,),np.int32)
    # pm.fortranFitFeatNum2=(pm.fortranFitFeatNum0*1.0).astype(np.int32)
    
def calFeatGrid():
    '''
    首先应该从设置文件中读取所有的用户设定

    Determine parameters:
    ---------------------
    mulFactorVectOf2bFeat:    一维pm.mulNumOf2bFeat长度的cp.array,用于计算pm.mulNumOf2bFeat个二体feat的相应参数
    pm.mulFactorVectOf3bFeat:    一维pm.mulNumOf3bFeat长度的cp.array,用于计算pm.mulNumOf3bFeat个三体feat的相应参数
    pm.weightOfDistanceScaler:   标量实数，basic函数中对输入距离矩阵进行Scaler的权重w
    pm.biasOfDistanceScaler：    标量实数，basic函数中对输入距离矩阵进行Scaler的偏置b 
    '''    
    mulFactorVectOf2bFeat={}
    mulFactorVectOf3bFeat1={}
    mulFactorVectOf3bFeat2={}
    h2b={}
    h3b1={}
    h3b2={}

    for itype in range(pm.atomTypeNum):
        mulFactorVectOf2bFeat[itype]=((cp.logspace(cp.log10(1.0),cp.log10(9.0),pm.Ftype1_para['numOf2bfeat'][itype]+2)[1:-1]-5.0)/4.0 +1)*pm.Rc_M/2.0
        mulFactorVectOf3bFeat1[itype]=((cp.logspace(cp.log10(1.0),cp.log10(9.0),pm.Ftype2_para['numOf3bfeat1'][itype]+2)[1:-1]-5.0)/4.0 +1)*pm.Rc_M/2.0
        mulFactorVectOf3bFeat2[itype]=((cp.logspace(cp.log10(1.0),cp.log10(9.0),pm.Ftype2_para['numOf3bfeat2'][itype]+2)[1:-1]-5.0)/4.0 +1)*pm.Rc_M/2.0
        h2b[itype]=(pm.Rc_M-float(mulFactorVectOf2bFeat[itype].max()))
        h3b1[itype]=(pm.Rc_M-float(mulFactorVectOf3bFeat1[itype].max()))
        h3b2[itype]=(pm.Rc_M-float(mulFactorVectOf3bFeat2[itype].max()))

        with open(os.path.join(pm.OutputPath,'grid2b_type3.'+str(itype+1)),'w') as f:
            
            f.write(str(pm.Ftype1_para['numOf2bfeat'][itype])+' \n')
            for i in range(pm.Ftype1_para['numOf2bfeat'][itype]):
                left=mulFactorVectOf2bFeat[itype][i]-h2b[itype]
                right=mulFactorVectOf2bFeat[itype][i]+h2b[itype]
                f.write(str(i)+'  '+str(left)+'  '+str(right)+' \n')

        with open(os.path.join(pm.OutputPath,'grid3b_cb12_type3.'+str(itype+1)),'w') as f:
            
            f.write(str(pm.Ftype2_para['numOf3bfeat1'][itype])+' \n')
            for i in range(pm.Ftype2_para['numOf3bfeat1'][itype]):
                left=mulFactorVectOf3bFeat1[itype][i]-h3b1[itype]
                right=mulFactorVectOf3bFeat1[itype][i]+h3b1[itype]
                f.write(str(i)+'  '+str(left)+'  '+str(right)+' \n')

        with open(os.path.join(pm.OutputPath,'grid3b_b1b2_type3.'+str(itype+1)),'w') as f:
            
            f.write(str(pm.Ftype2_para['numOf3bfeat2'][itype])+' \n')
            for i in range(pm.Ftype2_para['numOf3bfeat2'][itype]):
                left=mulFactorVectOf3bFeat2[itype][i]-h3b2[itype]
                right=mulFactorVectOf3bFeat2[itype][i]+h3b2[itype]
                f.write(str(i)+'  '+str(left)+'  '+str(right)+' \n')

def r_feat_csv(f_feat):
    """ read feature and energy from pandas data
    """
    df   = pd.read_csv(f_feat,header=None,index_col=False,dtype=pm.tf_dtype)
    itypes = df[1].values.astype(int)
    engy = df[2].values 
    feat = df.drop([0,1,2],axis=1).values 
    engy = engy.reshape([engy.size,1])
    return itypes,feat,engy

def r_egroup_csv(f_egroup):
    """ read feature and energy from pandas data
    """
    df   = pd.read_csv(f_egroup,header=None,index_col=False,dtype=pm.tf_dtype)
    egroup = df[0].values
    divider = df[1].values
    egroup_weight = df.drop([0,1],axis=1).values 
    egroup = egroup.reshape([egroup.size,1])
    divider = divider.reshape([divider.size,1])
    # itypes = df[1].values.astype(int)
    # engy = df[2].values
    # feat = df.drop([0,1,2],axis=1).values 
    # engy = engy.reshape([engy.size,1])
    return egroup,divider,egroup_weight

def writeVdwInput(fit_model_dir, vdw_input):
    num_atom_type = vdw_input['ntypes']
    vdw_filename = os.path.join(fit_model_dir,'vdw_fitB.ntype')
    f_vdw = open(vdw_filename, 'w')
    f_vdw.write('%d %d\n' % (vdw_input['ntypes'], vdw_input['nterms']))
    for i in range(num_atom_type):   # loop i for atomtypes
        f_vdw.write('%d %f %f ' % (vdw_input['atom_type'][i], vdw_input['rad'][i], vdw_input['e_ave'][i]))
        for j in range(len(vdw_input['wp'][i])):  # loop j for pm.ntypes*pm.nterms
            f_vdw.write(' %f ' % vdw_input['wp'][i][j])
        f_vdw.write('\n')
    f_vdw.close()


def prepare_novdw():
    # liuliping auto vdw_fitB.ntype
    print('no vdw, auto creating zero vdw_fitB.ntype file')
    if not os.path.exists(pm.fitModelDir):
        os.makedirs(pm.fitModelDir)
    strength_rad = 0.0
    if pm.isFitVdw == True:
        strength_rad = 1.0
    vdw_input = {
        'ntypes': pm.ntypes,
        'nterms': 1,
        'atom_type': pm.atomType,
        'rad': [strength_rad for i in range(pm.ntypes)],
        'e_ave': [500.0 for i in range(pm.ntypes)],
        'wp': [ [0.8 for i in range(pm.ntypes*1)] for i in range(pm.ntypes)]
    }
    writeVdwInput(pm.fitModelDir, vdw_input)
    # vdw input end


def prepare_dir_info():
    fout = open('input/info_dir', 'w')
    fout.write(pm.fitModelDir+'/\n')
    fout.close()
    

if __name__ == "__main__":
    pass
