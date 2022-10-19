import numpy as np
import os
import torch
import torch.nn.functional as F


isCalcFeat=False
isFitLinModel=False
isClassify=False
isRunMd=False                                   
isRunMd_nn=False
isFollowMd=False                                
isFitVdw=False
isRunMd100_nn=False
isRunMd100=False
add_force=False     # for NN md

# torch new parameters
is_nn_do_profile = False
isNNpretrain =False
isNNfinetuning=True
train_data_path = r'./train_data/final_train'
test_data_path = r'./train_data/final_test'
f_train_dR_neigh=''
f_test_dR_neigh=''
dR_neigh=False
#dRneigh_path = r'./'
DCNLayers = 5

nFeatures=42


torch_dtype = 'float64'

feature_dtype = 'float64'   # how the feature data files are stored
                            # take effect when you call gen_data.py
                            
training_dtype = 'float64'  # 1) feature data are casted to specified dtype
                            #    during training
                            # 2) model parameter dtype during training
                            # take effect when you call train.py

inference_dtype = 'float64' # 1) feature data are casted to specified dtype
                            #    during inference
                            # 2) model parameter dtype during inference
                            #
                            # NOTE: not implemented yet
# torch new parameters end 2021-12-30

######################## MD100 
isNewMd100=False
imodel = 1    # 1:linear;  2:VV;   3:NN;
md_num_process = 1   # mpirun -n ${md_num_process} main_MD.x
is_md100_egroup = False      # This will cost much time if True. Python is SLOW
# All pics will saved to png files, and, if X11_fig=True, it will also show in X11 GUI.
is_md100_show_X11_fig = False


#************** Dir **********************
prefix = r'./'
trainSetDir = r'./PWdata'
fortranFitSourceDir=r'/home/liuliping/program/nnff/git_version/src/fit'
#codedir=r'/home/liuliping/program/nnff/git_version/src/lib'
fitModelDir = r'./fread_dfeat'
#genFeatDir = '/home/buyu/MLFF/new-repulsive/ML_FRAME_WORK_vdw/gen_feature'
#genFeatDir = r'./gen_feature'
#mdImageFileDir='/home/buyu/MLFF/MD/AlHbulk'
mdImageFileDir=r'./MD'                              #设置md的初始image的文件所在的文件夹  default:'.'
#dR_neigh=False
pbc = True

dRneigh_path = trainSetDir + r'/dRneigh.dat'
dp_predict = False
#************** DP network parameters ********************

DP_cfg = {
    'embeding_net': {
        'network_size': [16, 32, 64], # 第一维表示输入的维度
	'bias': True,
	'resnet_dt': True,
	'activation': F.softplus,    #torch.sigmoid,
	},
    'fitting_net': {
	'network_size': [120, 120, 120, 1],
	'activation': F.softplus,    #torch.sigmoid,
	'bias': True,
	}
}

DP_cfg_dp = {
    'embeding_net': {
        'network_size': [25, 50, 100], 
	'bias': True,
	'resnet_dt': False,
	'activation': torch.tanh,
	},
    'fitting_net': {
	'network_size': [240, 240, 240, 1],
	'activation': torch.tanh,
	'resnet_dt': True,
	'bias': True,
	}
}

DP_cfg_dp_kf = {
    'embeding_net': {
        'network_size': [25, 25, 25], 
	'bias': True,
	'resnet_dt': False,
	'activation': torch.tanh,
	},
    'fitting_net': {
	'network_size': [50, 50, 50, 1],
	'activation': torch.tanh,
	'resnet_dt': True,
	'bias': True,
	}
}

DP_cfg_dp_kf_mini = {
    'embeding_net': {
        'network_size': [5, 5, 5], 
	'bias': True,
	'resnet_dt': False,
	'activation': torch.tanh,
	},
    'fitting_net': {
	'network_size': [10, 10, 10, 1],
	'activation': torch.tanh,
	'resnet_dt': True,
	'bias': True,
	}
}

MLFF_dmirror_cfg1 = [
	('linear', nFeatures, 30, True),
	('activation',),
	('linear', 30, 60, True),
	('activation',),
	('linear', 60, 1, True)
	]

#********* for gen_feature.in *********************

atomType=[14]
maxNeighborNum=100

iflag_PCA=0

Rc_M=6.0                     # max of Rcut
Rc_min = 5.8

Ftype_name={1:'gen_2b_feature', 2:'gen_3b_feature',
            3:'gen_2bgauss_feature', 4:'gen_3bcos_feature',
            5:'gen_MTP_feature', 6:'gen_SNAP_feature',
            7:'gen_deepMD1_feature', 8:'gen_deepMD2_feature',
            }
# Ftype2_name='gen_3b_feature'
#use_Ftype=[1,2,3,4,5,6,7,8]
use_Ftype=[1]
use_GKalman = False
use_LKalman = False
use_SKalman = False


kalman_prefac_Ei = 1.0
kalman_prefac_Etot = 1.0
kalman_prefac_Force = 1.0

is_scale = False
use_storage_scaler = False
storage_scaler = False

#used to control if dfeat_sparse is used
is_dfeat_sparse = True

n_epoch = 100
itype_Ei_mean = [0 for i in range(10)]
batch_size = 1  # only support bn=1 so far

nfeat_type=len(use_Ftype)

Ftype1_para={               #2b
    'numOf2bfeat':[24 for tmp in range(10)],       # [itpye1,itype2]
    'Rc':[Rc_M for tmp in range(10)],
    'Rm':[Rc_min for tmp in range(10)],
    'iflag_grid':[3 for tmp in range(10)],                      # 1 or 2 or 3
    'fact_base':[0.2 for tmp in range(10)],
    'dR1':[0.5 for tmp in range(10)],
    'iflag_ftype':3       # same value for different types, iflag_ftype:1,2,3 when 3, iflag_grid must be 3
}

Ftype2_para={             # 3b
    'numOf3bfeat1':[3 for tmp in range(10)],
    'numOf3bfeat2':[3 for tmp in range(10)],
    'Rc':[Rc_M for tmp in range(10)],
    'Rc2':[Rc_M for tmp in range(10)],
    'Rm':[Rc_min for tmp in range(10)],

    'iflag_grid':[3 for tmp in range(10)],                      # 1 or 2 or 3
    'fact_base':[0.2 for tmp in range(10)],

    'dR1':[0.5 for tmp in range(10)],
    'dR2':[0.5 for tmp in range(10)],
    'iflag_ftype':3   # same value for different types, iflag_ftype:1,2,3 when 3, iflag_grid must be 3
}

Ftype3_para={           # 2bgauss
    'Rc':[5.4 for tmp in range(10)],     # number of elements in Rc = num atom type
    'n2b':[6 for tmp in range(10)],       # number of elements in n2b = num atom type
    'w': [1.0, 1.5, 2.0],
    # 1/w^2 is the \eta in formula, and w is the width of gaussian fuction
}

Ftype4_para={           # 3bcos
    'Rc':[5.4 for tmp in range(10)],     # number of elements in Rc = num atom type
    'n3b':[20 for tmp in range(10)],
    # n3b must be less than n_zeta * n_w * n_lambda, by default, n3b < 7*10*2=126
    # it is better if n3b is divisible by (n_w*2)
    'zeta': [ (2.0 ** np.array(range(20))).tolist() for tmp in range(10)],  # feature changed
    'w':    [ [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] for tmp in range(10)],  # feature changed
    # 1/w^2 is the \eta in formula, and w is the width of gaussian fuction
    # 'lambda':[ [1.0,-1.0] for tmp in range(10)], # lambda === [1.0, -1.0]
    
}
Ftype5_para={           # MTP
    'Rc':[5.4 for tmp in range(10)],     # number of elements in Rc = num atom type
    'Rm':[0.5  for tmp in range(10)],     # number of elements in Rc = num atom type
    'n_MTP_line': [5 for tmp in range(10)], # 5~14
    'tensors':[
                [
                  '1, 4, 0, ( )                              ',
                  '2, 3,3, 0,0, ( ), ( )                     ',
                  '2, 3,3, 1,1, ( 21 ), ( 11 )               ',
                  '2, 3,3, 2,2, ( 21, 22 ), ( 11, 12 )       ',
                  '3, 2,2,2, 2,1,1 ( 21, 31 ), ( 11 ), ( 12 )',
                  '3, 3,3,3, 2,1,1 ( 21, 31 ), ( 11 ), ( 12 )',
                  '3, 2,2,2, 3,2,1 ( 21, 22, 31 ), ( 11, 12 ), ( 13 )',
                  '3, 3,3,3, 3,2,1 ( 21, 22, 31 ), ( 11, 12 ), ( 13 )',
                  '3, 2,2,2, 4,2,2 ( 21, 22, 31, 32 ), ( 11, 12 ), ( 13, 14 )',
                  '3, 3,3,3, 4,2,2 ( 21, 22, 31, 32 ), ( 11, 12 ), ( 13, 14 )',
                  '4, 2,2,2,2 3,1,1,1 ( 21, 31, 41 ), ( 11 ), ( 12 ), ( 13 )',
                  '4, 3,3,3,3 3,1,1,1 ( 21, 31, 41 ), ( 11 ), ( 12 ), ( 13 )',
                  '4, 2,2,2,2 4,2,1,1 ( 21, 22, 31, 41 ), ( 11, 12 ), ( 13 ), ( 14 )',
                  '4, 3,3,3,3 4,2,1,1 ( 21, 22, 31, 41 ), ( 11, 12 ), ( 13 ), ( 14 )',
                ] for tmp in range(10)
              ],
    }
Ftype6_para={
    'Rc':[5.4 for tmp in range(10)],     # number of elements in Rc = num atom type
    'J' :[3.0 for tmp in range(10)],
    'n_w_line': [2 for tmp in range(10)],
    'w1':[ [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]  for tmp in range(10)],  # shape(w1) = (ntype, n_w_line)
    'w2':[ [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]  for tmp in range(10) ],
    }
Ftype7_para={
    'Rc':[5.4  for tmp in range(10)],     # number of elements in Rc = num atom type
    'Rc2':[3.0  for tmp in range(10)],
    'Rm':[1.0  for tmp in range(10)],
    'M': [20  for tmp in range(10)],
    'weight_r': [1.0  for tmp in range(10)],
    }

Ftype8_para={
    'Rc':[5.4  for tmp in range(10)],     # number of elements in Rc = num atom type
    'M':[8  for tmp in range(10)],
    'weight_r':[1.0  for tmp in range(10)],
    'w':[1.0, 1.5, 2.0, 2.5 ],
    }

kfnn_trainEtot = True
kfnn_trainEi = False
kfnn_trainForce = True 
kfnn_trainEgroup = False

E_tolerance=0.3
# iflag_ftype=3        # Seems like, this should be in the Ftype1/2_para        # 2 or 3 or 4 when 4, iflag_grid must be 3
recalc_grid=1                      # 0:read from file or 1: recalc
#----------------------------------------------------
rMin=0.0
#************** cluster input **********************
kernel_type=2             # 1 is exp(-(dd/width)**alpha), 2 is 1/(dd**alpha+k_dist0**alpha)
use_lpp=False
lppNewNum=3               # new feature num lpp generate. you can adjust more lpp parameters by editing feat_LPP.py. also see explains in it
lpp_n_neighbors=5
lpp_weight='adjacency'    # 'adjacency' or 'heat'
lpp_weight_width=1.0
alpha0=1.0
k_dist0=0.01
DesignCenter=False
ClusterNum=[3,2]
#-----------------------------------------------

#******** for fit.input *******************************

fortranFitAtomRepulsingEnergies=[0.000 for tmp in range(10)]            #fortran fitting时对每种原子设置的排斥能量的大小，此值必须设置，无default值！(list_like)
fortranFitAtomRadii=[1.0 for tmp in range(10)]                        #fortran fitting时对每种原子设置的半径大小，此值必须设置，无default值！(list_like)
fortranFitWeightOfEnergy=0.0                    #fortran fitting时最后fit时各个原子能量所占的权重(linear和grr公用参数)  default:0.9
fortranFitWeightOfEtot=0.5                      #fortran fitting时最后fit时Image总能量所占的权重(linear和grr公用参数)  default:0.0
fortranFitWeightOfForce=0.5                     #fortran fitting时最后fit时各个原子所受力所占的权重(linear和grr公用参数)  default:0.1
fortranFitRidgePenaltyTerm=0.0001               #fortran fitting时最后岭回归时所加的对角penalty项的大小(linear和grr公用参数)  default:0.0001
fortranFitDwidth = 3.0
''' examples for vdw
fortranFitVdw = {
        'ntypes': 1
        'nterms': 1,
        'atom_type': atomType,
        'rad': [0.0 for i in range(ntypes)],
        'e_ave': [500.0 for i in range(ntypes)],
        'wp': [ [0.8 for i in range(ntypes*1)] for i in range(ntypes)]
    }
'''
#----------------------------------------------------

#*********************** for MD **********************

#以下部分为md设置的参数
mdCalcModel='lin'                               #运行md时，计算energy和force所用的fitting model，‘lin' or 'vv'
mdRunModel='nvt'                                #md运行时的模型,'nve' or 'nvt' or 'npt' or 'opt', default:'nve'
mdStepNum=10                                  #md运行的步数,default:1000
mdStepTime=1                                  #md运行时一步的时长(fs), default:1.0
mdStartTemperature=300                          #md运行时的初始温度
mdEndTemperature=300                            #md运行采用'nvt'模型时，稳定温度(or npt)
mdNvtTaut=0.1*1000                               #md运行采用'nvt'模型时，Berendsen温度对的时间常数 (or npt)
mdOptfmax=0.05
mdOptsteps=10

isTrajAppend=False                              #traj文件是否采用新文件还是接续上次的文件  default:False
isNewMovementAppend=False                       #md输出的movement文件是采用新文件还是接续上次的文件  default:False
mdTrajIntervalStepNum=50
mdLogIntervalStepNum=10
mdNewMovementIntervalStepNum=10
mdStartImageIndex=0                             #若初始image文件为MOVEMENT,初始的image的编号  default:0

isOnTheFlyMd=False                              #是否采用on-the-fly md,暂时还不起作用  default:False
isFixedMaxNeighborNumForMd  =False                #是否使用固定的maxNeighborNum值，默认为default,若为True，应设置mdMaxNeighborNum的值
mdMaxNeighborNum=None                           #采用固定的maxNeighborNum值时，所应该采用的maxNeighborNum值(目前此功能不可用)

isMdCheckVar=False                               #若采用 'grr' model时，是否计算var  default:False
isReDistribute=True                             #md运行时是否重新分配初速度，目前只是重新分配, default:True
velocityDistributionModel='MaxwellBoltzmann'    #md运行时,重新分配初速度的方案,目前只有'MaxwellBoltzmann',default:MaxwellBoltzmann

isMdProfile=False

#-------------------------------------------------------
#********************* NN_related ***************
# device related

gpu_mem  = 0.9       # tensorflow used gpu memory
cuda_dev = '0'       # unoccupied gpu, using 'nvidia-smi' cmd
cupyFeat=True
tf_dtype = 'float64' # dtype of tensorflow trainning, 'float32' faster than 'float64'

test_ratio = 0.2
is_rand_seper = False 

#================================================================================
# NN model related
activation_func='softplus'     # could choose 'softplus' and 'elup1' now
ntypes=len(atomType)

nLayers = 3 
nodeDim = [15,15,1]
nNodes = np.array([[15,15],[15,15],[1,1]])
b_init=np.array([0.0 for i in range(10)])      # energy of one atom, for different types, just a rough value
#nLayers = 4
#nNodes = np.array([[16,],[64,],[32,],[1,]])
dwidth = 3.0

#================================================================================
# trainning
train_continue = False     #是否接着训练
progressbar = False
flag_plt = False
train_stage = 1      # only 1 or 2, 1 is begining training from energy and then force+energy, 2 is directly training from force+energy
train_verb = 0

learning_rate= 1e-3
batch_size = 1
rtLossE      = 0.8     # weight for energy, NN fitting 各个原子能量所占的权重
rtLossF      = 0.2     # weight for force, NN fitting 各个原子所受力所占的权重
rtLossEtot      = 0.2     # weight for Etot, NN fitting 各个原子所受力所占的权重
bias_corr = True
#epochs_pretrain = 1001
epochs_alltrain = 1001     # energy 训练循环次数
epochs_Fi_train = 11       # force+energy 训练循环次数

iFi_repeat      = 1
eMAE_err = 0.01 # eV
fMAE_err = 0.02 # eV/Ang


#************* no need to edit ****************************
#fortranFitAtomTypeNum=0                        #fortran fitting时原子所属种类数目(linear和grr公用参数)  default:0(废弃，不需要)
# fortranFitFeatNum0=None                         #fortran fitting时输入的feat的数目(linear和grr公用参数)  default:None
# fortranFitFeatNum2=None                         #fortran fitting时PCA之后使用的feat的数目(linear和grr公用参数)  此值目前已经不需要设置
isDynamicFortranFitRidgePenaltyTerm=False       #fortran fitting时最后岭回归时所加的对角penalty项的大小是否根据PCA最小的奇异值调整 default:False
fortranGrrRefNum=[800,1000]                           #fortran grr fitting时每种原子所采用的ref points数目,若设置应为类数组   default:None
fortranGrrRefNumRate=0.1                        #fortran grr fitting时每种原子选择ref points数目所占总case数目的比率   default:0.1
fortranGrrRefMinNum=1000                        #fortran grr fitting时每种原子选择ref points数目的下限数目，若case数低于此数，则为case数
fortranGrrRefMaxNum=3000                        #fortran grr fitting时每种原子选择ref points数目的上限数目，若设定为None，则无上限(不建议)
fortranGrrKernelAlpha=1                         #fortran grr fitting时kernel所用超参数alpha
fortranGrrKernalDist0=3.0                       #fortran grr fitting时kernel所用超参数dist0
realFeatNum=111

#-----------------------------------------------


trainSetDir=os.path.abspath(trainSetDir)
#genFeatDir=os.path.abspath(genFeatDir)
fortranFitSourceDir=os.path.abspath(fortranFitSourceDir)
fbinListPath=os.path.join(trainSetDir,'location')
sourceFileList=[]
InputPath=os.path.abspath('./input/')
OutputPath=os.path.abspath('./output/')
Ftype1InputPath=os.path.join('./input/',Ftype_name[1]+'.in')
Ftype2InputPath=os.path.join('./input/',Ftype_name[2]+'.in')
FtypeiiInputPath={i:'' for i in range(1,9)}  # python-dictionary, i = 1,2,3,4,5,6,7,8
for i in range(1,9):
    FtypeiiInputPath[i]=os.path.join('./input/',Ftype_name[i]+'.in')
featCollectInPath=os.path.join(fitModelDir,'feat_collect.in')
fitInputPath_lin=os.path.join(fitModelDir,'fit_linearMM.input')
fitInputPath2_lin=os.path.join(InputPath,'fit_linearMM.input')
featCollectInPath2=os.path.join(InputPath,'feat_collect.in')

if fitModelDir is None:
    fitModelDir=os.path.join(fortranFitSourceDir,'fread_dfeat')
else:
    fitModelDir=os.path.abspath(fitModelDir)
linModelCalcInfoPath=os.path.join(fitModelDir,'linear_feat_calc_info.txt')
linFitInputBakPath=os.path.join(fitModelDir,'linear_fit_input.txt')

f_atoms=os.path.join(mdImageFileDir,'atom.config')
atomTypeNum=len(atomType)
nFeats=np.array([realFeatNum,realFeatNum,realFeatNum])
dir_work = os.path.join(fitModelDir,'NN_output/')

f_train_feat = os.path.join(dir_work,'feat_train.csv')
f_test_feat = os.path.join(dir_work,'feat_test.csv')

f_train_natoms = os.path.join(dir_work,'natoms_train.csv')
f_test_natoms = os.path.join(dir_work,'natoms_test.csv')        

f_train_dfeat = os.path.join(dir_work,'dfeatname_train.csv')
f_test_dfeat  = os.path.join(dir_work,'dfeatname_test.csv')

f_train_dR_neigh = os.path.join(dir_work,'dR_neigh_train.csv')
f_test_dR_neigh  = os.path.join(dir_work,'dR_neigh_test.csv')

f_train_force = os.path.join(dir_work,'force_train.csv')
f_test_force  = os.path.join(dir_work,'force_test.csv')

f_train_egroup = os.path.join(dir_work,'egroup_train.csv')
f_test_egroup  = os.path.join(dir_work,'egroup_test.csv')

f_train_ep = os.path.join(dir_work,'ep_train.csv')
f_test_ep  = os.path.join(dir_work,'ep_test.csv')

d_nnEi  = os.path.join(dir_work,'NNEi/')
d_nnFi  = os.path.join(dir_work,'NNFi/')

f_Einn_model   = d_nnEi+'allEi_final.ckpt'
f_Finn_model   = d_nnFi+'Fi_final.ckpt'

f_data_scaler = d_nnFi+'data_scaler.npy'
f_Wij_np  = d_nnFi+'Wij.npy'
