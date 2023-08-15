"""
    module for Deep Neural Network 
    2022.8
"""
import os,sys
import shutil
import subprocess
import pathlib

codepath = str(pathlib.Path(__file__).parent.resolve())

#for model.mlff 
sys.path.append(codepath+'/../model')

#for default_para, data_loader_2type dfeat_sparse
sys.path.append(codepath+'/../pre_data')

#for optimizer
sys.path.append(codepath+'/..')
sys.path.append(codepath+'/../aux')
sys.path.append(codepath+'/../lib')
sys.path.append(codepath+'/../..')

import torch

import random
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import math

import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

import time
from optimizer.KFWrapper_wu import KFOptimizerWrapper
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

# src/aux 
from src.aux.opts import opt_values 
from src.aux.feat_modifier import feat_modifier

import pickle

"""
    customized modules 
"""

from model.MLFF_wu import MLFFNet

# from optimizer.kalmanfilter import GKalmanFilter, LKalmanFilter, SKalmanFilter
from optimizer.GKF import GKFOptimizer
from optimizer.LKF import LKFOptimizer

import default_para as pm

# get_torch_data for nn single system training
from src.pre_data.data_loader_2type import get_torch_data
#get_torch_data_hybrid for nn multi hybrid systems training
from src.pre_data.data_loader_2type_wu import MovementHybridDataset, get_torch_data_hybrid

from src.pre_data.dfeat_sparse import dfeat_raw 

from src.pre_data.nn_hybird_mlff import get_cluster_dirs, make_work_dir, mv_featrues, mv_file

from utils.file_operation import write_line_to_file
class nn_network:
    
    """
        class that wraps the training model and the data

        network/network parameters 
        training/validation data
        training method 
        
        Using default_para is purely a legacy problem

    """        

    def __init__(   self,
                    # some must-haves
                    # model related argument
                    nn_layer_config = None,
                    atom_type = None, 
                    num_feature = None, 
                    feature_type = None, 
                    # data related arguments    
                    scale = True,
                    store_scaler = True,
                    device = "cpu", 
                    # optimizer related arguments 
                    
                    # Rcuts 
                    Rmax = 5.0,
                    Rmin = 0.5, 
                
                    max_neigh_num = 100, 
                    precision = "float64", 
                    kalman_type = "GKF",      # or: "LKF", "layerwise", "selected"  
                    session_dir = "record",  # directory that saves the model
                    
                    # for force update
                    nselect = 24,
                    distributed = False, 
                    group_size = 6,
                    
                    # for LKF
                    block_size = 5120, 
                    # training label related arguments
                    is_trainForce = True, 
                    is_trainEi = False,
                    is_trainEgroup = False,
                    is_trainEtot = True,
                    is_trainViral = False,  #not realilzed
                    batch_size = 1, 
                    is_movement_weighted = False,
                    is_dfeat_sparse = True,

                    n_epoch = None, 
                    
                    recover = False,
                    
                    kf_prefac_etot = 2.0,
                    kf_prefac_force = 1.0,
                    
                    # custom feature parameters
                    custom_feat_1 = None, 
                    custom_feat_2 = None, 
                    custom_feat_7 = None, 

                    dbg = False, 
                    hybrid = False,
                    inference = False
                ):
        self.workers = 1
        """
            Global parameters of the class.  
            Some member data are only "declared" here, and assignment requires further operations 
            Frequently used control parameter can be reset by class.set_xx() functions
        """
        
        """
            Frequently-used control parameters that are directly passed in as arguments:        
        """     
        
        # for hybird training
        self.hybird = hybrid
        self.inference = inference
        if self.inference is True:
            # pm.test_ratio = 1
            store_scaler = False
            batch_size = 1
            recover = True

        # initializing the opt class 
        self.opts = opt_values() 
        self.dbg = dbg

        # passing some input arguments to default_para.py
        # at the end of the day, one no longer needs parameters.py

        pm.atomType = atom_type
        self.atom_type = atom_type
        # scaling options. 
        pm.is_scale = scale 
        pm.storage_scaler = store_scaler

        # recover training. Need to load both scaler and model 
        pm.use_storage_scaler = recover  
        self.opts.opt_recover_mode = recover

        pm.maxNeighborNum = max_neigh_num 
        
        # setting NN network configuration 
        if nn_layer_config == None: 
            #node of each layer
            pm.nodeDim = [15,15,1]
            #raise Exception("network configuration of NN is missing")
        else:
            #node of each layer
            pm.nodeDim = nn_layer_config
        
        if atom_type == None:
            raise Exception("atom types not specifed")

        pm.atomTypeNum = len(pm.atomType)       #this step is important. Unexpected error
        pm.ntypes = len(pm.atomType)
        
        #number of layer
        pm.nLayer = len(pm.nodeDim)    
        # passing working_dir to opts.session_name 
        self.set_session_dir(session_dir)

        # pm.is_dfeat_sparse is True as default
        if not is_dfeat_sparse:
            pm.is_dfeat_sparse = False 

        # network for each type of element
        tmp = []
        
        for layer in range(pm.nLayer):
            tmp_layer = [pm.nodeDim[layer] for atom in pm.atomType]
            tmp.append(tmp_layer.copy()) 

        pm.nNodes = np.array(tmp) 
        
        ########################
        # feature set feature type 
        pm.use_Ftype = sorted(feature_type)

        # update nfeat_type 
        pm.nfeat_type = len(pm.use_Ftype)
        
        self.feat_mod = feat_modifier()     
        
        """
            usage:
            traienr.f_mdfr.set_feat1_xxxx([])
        """ 
        
        # set final layer bias for NN. 
        # ##################################
        #        UNDER CONSTRUCTION 
        # pm.itype_Ei_mean = self.get_b_init()   
        
        # label to be trained 
        self.is_trainForce = is_trainForce
        self.is_trainEi = is_trainEi
        self.is_trainEgroup = is_trainEgroup
        self.is_trainEtot = is_trainEtot
        self.is_trainViral = is_trainViral # not relized
        #prefactors in kfnn
        self.kf_prefac_Etot = kf_prefac_etot
        self.kf_prefac_Ei = 1.0
        self.kf_prefac_F  = kf_prefac_force
        self.kf_prefac_Egroup  = 1.0
        
        # decay rate of kf prefactor
        #self.kf_decay_rate = kf_decay_rate

        # set the kind of KF
        self.kalman_type = kalman_type 

        # parameters for KF
        self.kalman_lambda = 0.98                    
        self.kalman_nue =  0.99870

        self.precision = precision
        
        # set feature precision 
        pm.feature_dtype = self.precision
        
        # set training precision. Overarching 
        if (self.precision == 'float64'):
            print("Training: set default dtype to double")
            torch.set_default_dtype(torch.float64)

        elif (self.precision == 'float32'):
            print("Training: set default dtype to single")
            torch.set_default_dtype(torch.float32)

        else:
            #self.opts.error("Training: unsupported dtype: %s" %self.opts.opt_dtype)
            raise RuntimeError("Training: unsupported dtype: %s" %self.opts.opt_dtype)  

        # set training device
        self.device = None 

        if device == "cpu":
            self.device = torch.device('cpu')
        elif device == "cuda":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            raise Exception("device type not supported")
        
        print ("device used",self.device)

        # set random seed
        torch.manual_seed(self.opts.opt_rseed)
        torch.cuda.manual_seed(self.opts.opt_rseed)
        
        # set print precision
        torch.set_printoptions(precision = 12)
        
        self.patience = 100000 
        # movement-wise weight in training
        self.is_movement_weighted = is_movement_weighted 
        self.movement_weights = None
        self.movement_idx = [] 

        if self.is_movement_weighted: 
            self.set_movement_weights() 
        else:
            self.movement_weights = {} 
        
        # common training parameters 
        if n_epoch is None:
            self.n_epoch = 25
        else:
            self.n_epoch = n_epoch
        
        self.batch_size = batch_size
        
        self.min_loss = np.inf
        self.epoch_print = 1 
        self.iter_print = 1 
        
        """
            for torch built-in optimizers and schedulers 
        """
        self.momentum = self.opts.opt_momentum
        self.REGULAR_wd = self.opts.opt_regular_wd
        self.LR_base = self.opts.opt_lr
        self.LR_gamma = self.opts.opt_gamma
        self.LR_step = self.opts.opt_step
        
        """
        if (self.opts.opt_follow_mode == True):
            self.opts.opt_model_file = self.opts.opt_model_dir+self.opts.opt_net_cfg+'.pt'
        """ 

        """
            training data. 
            Below are placeholders
        """ 
        # anything other than dfeat
        self.loader_train = None
        self.loader_valid = None

        #sparse dfeat from fortran 
        self.dfeat = None 
        #self.dfeat_train = None
        #self.dfeat_valid = None
        #
        self.scaler = None

        """   
            models 
        """
        self.optimizer = None 
        self.model = None
        self.scheduler = None

        # starting epoch 
        self.start_epoch = 1

        # path for loading previous model 
        self.load_model_path = self.opts.opt_model_dir+'latest.pt' 

        self.nselect = nselect 
        self.distributed = distributed
        self.group_size = group_size
        
        self.block_size = block_size 
        
        # R cut for feature 
        pm.Rc_M = Rmax
        pm.Rc_min = Rmin

        # load custom feature 
        if custom_feat_7 is not None:
            pm.Ftype7_para = custom_feat_7.copy()
        
        if custom_feat_1 is not None: 
            pm.Ftype1_para = custom_feat_1.copy()
        
        if custom_feat_2 is not None:
            pm.Ftype2_para = custom_feat_2.copy()
            
            
    def generate_data_main(self,chunk_size=1, shuffle=False):
        if self.hybird:
            self.do_hybord_feat_generate(chunk_size=1, shuffle=False)
        else:
            self.generate_data()

    def do_hybord_feat_generate(self, chunk_size=10, shuffle=True):
        root_dir = os.getcwd()
        tmp_work_dir = os.path.join(root_dir, "PWdata/gen_feat_dir")
        final_data_dir = os.path.join(root_dir, "train_data")
        if os.path.exists(tmp_work_dir) is True:
            shutil.rmtree(tmp_work_dir)
        if os.path.exists(final_data_dir) is True:
            shutil.rmtree(final_data_dir)
        # 对所有的MOVEMENT文件根据元素类型、原子数进行分类
        work_dir_list = get_cluster_dirs(root_dir)
        # 对同类型的MOVEMENT 构建临时目录，用于生成该类型的MOVEMENT的feature
        sub_work_dir = make_work_dir(tmp_work_dir, work_dir_list)
        cwd_path = os.getcwd()
        for index, sub_dir in enumerate(sub_work_dir):
            os.chdir(sub_dir)
            # reset pm parameters
            atom_type = work_dir_list[os.path.basename(sub_dir)]['types']
            self._reset_pm_params(sub_dir, atom_type)
            # clean old .dat files otherwise will be appended 
            if os.path.exists("PWdata/MOVEMENTall"):
                print ("cleaning old data")
                subprocess.run([
                    "rm -rf fread_dfeat output input train_data plot_data PWdata/*.txt PWdata/trainData.* PWdata/location PWdata/Egroup_weight PWdata/MOVEMENTall"
                    ],shell=True)
                subprocess.run(["find PWdata/ -name 'dfeat*' | xargs rm -rf"],shell=True)
                subprocess.run(["find PWdata/ -name 'info*' | xargs rm -rf"],shell=True)
            # calculating feature 
            from src.pre_data.mlff_wu import calc_feat
            calc_feat()
            # seperate training set and valid set
            import src.pre_data.seper_wu as seper 
            seper.seperate_data(chunk_size = chunk_size, shuffle=shuffle)
            # save as .npy
            import src.pre_data.gen_data_wu as gen_data
            gen_data.write_data()

            # move features 
            mv_file(os.path.dirname(pm.train_data_path), os.path.join(final_data_dir, os.path.basename(sub_dir)))
        
        # copy feat_info and vdw_fitB.ntype
        fread_dfeat_dir = os.path.join(root_dir, "fread_dfeat")
        if os.path.exists(fread_dfeat_dir) is False:
            os.makedirs(fread_dfeat_dir)
        mv_file(os.path.join(sub_work_dir[0], "fread_dfeat/feat.info"), os.path.join(fread_dfeat_dir, "feat.info"))
        mv_file(os.path.join(sub_work_dir[0], "fread_dfeat/vdw_fitB.ntype"), os.path.join(fread_dfeat_dir, "vdw_fitB.ntype"))
        os.chdir(root_dir)
        self._reset_pm_params(root_dir, self.atom_type)

    '''
    description: reset PM paramters
    param {*} self
    param {*} work_dir, new work_dir
    param {*} atom_type, new atom type
    return {*}
    author: wuxingxing
    '''
    def _reset_pm_params(self, work_dir, atom_type):
        pm.sourceFileList = []
        # pm.atomType = atom_type
        # pm.atomTypeNum = len(pm.atomType)       #this step is important. Unexpected error
        # pm.ntypes = len(pm.atomType)
        # reset dirs
        pm.train_data_path = os.path.join(work_dir, r'train_data/final_train')
        pm.test_data_path =os.path.join(work_dir,  r'train_data/final_test')
        pm.prefix = work_dir
        pm.trainSetDir = os.path.join(work_dir, r'PWdata')
        # fortranFitSourceDir=r'/home/liuliping/program/nnff/git_version/src/fit'
        pm.fitModelDir = os.path.join(work_dir, r'fread_dfeat')
        pm.dRneigh_path = pm.trainSetDir + r'dRneigh.dat'

        pm.trainSetDir=os.path.abspath(pm.trainSetDir)
        #genFeatDir=os.path.abspath(genFeatDir)
        pm.fbinListPath=os.path.join(pm.trainSetDir,'location')
        sourceFileList=[]
        pm.InputPath=os.path.abspath('./input/')
        pm.OutputPath=os.path.abspath('./output/')
        pm.Ftype1InputPath=os.path.join('./input/',pm.Ftype_name[1]+'.in')
        pm.Ftype2InputPath=os.path.join('./input/',pm.Ftype_name[2]+'.in')

        pm.featCollectInPath=os.path.join(pm.fitModelDir,'feat_collect.in')
        pm.fitInputPath_lin=os.path.join(pm.fitModelDir,'fit_linearMM.input')
        pm.fitInputPath2_lin=os.path.join(pm.InputPath,'fit_linearMM.input')
        pm.featCollectInPath2=os.path.join(pm.InputPath,'feat_collect.in')

        pm.linModelCalcInfoPath=os.path.join(pm.fitModelDir,'linear_feat_calc_info.txt')
        pm.linFitInputBakPath=os.path.join(pm.fitModelDir,'linear_fit_input.txt')

        pm.dir_work = os.path.join(pm.fitModelDir,'NN_output/')

        pm.f_train_feat = os.path.join(pm.dir_work,'feat_train.csv')
        pm.f_test_feat = os.path.join(pm.dir_work,'feat_test.csv')

        pm.f_train_natoms = os.path.join(pm.dir_work,'natoms_train.csv')
        pm.f_test_natoms = os.path.join(pm.dir_work,'natoms_test.csv')        

        pm.f_train_dfeat = os.path.join(pm.dir_work,'dfeatname_train.csv')
        pm.f_test_dfeat  = os.path.join(pm.dir_work,'dfeatname_test.csv')

        pm.f_train_dR_neigh = os.path.join(pm.dir_work,'dR_neigh_train.csv')
        pm.f_test_dR_neigh  = os.path.join(pm.dir_work,'dR_neigh_test.csv')

        pm.f_train_force = os.path.join(pm.dir_work,'force_train.csv')
        pm.f_test_force  = os.path.join(pm.dir_work,'force_test.csv')

        pm.f_train_egroup = os.path.join(pm.dir_work,'egroup_train.csv')
        pm.f_test_egroup  = os.path.join(pm.dir_work,'egroup_test.csv')

        pm.f_train_ep = os.path.join(pm.dir_work,'ep_train.csv')
        pm.f_test_ep  = os.path.join(pm.dir_work,'ep_test.csv')

        pm.d_nnEi  = os.path.join(pm.dir_work,'NNEi/')
        pm.d_nnFi  = os.path.join(pm.dir_work,'NNFi/')

        pm.f_Einn_model   = pm.d_nnEi+'allEi_final.ckpt'
        pm.f_Finn_model   = pm.d_nnFi+'Fi_final.ckpt'

        pm.f_data_scaler = pm.d_nnFi+'data_scaler.npy'
        pm.f_Wij_np  = pm.d_nnFi+'Wij.npy'


    """
        ============================================================
        =================data preparation functions=================
        ============================================================ 
    """
    def generate_data(self, chunk_size = 10):
        """
            defualt chunk size set to 10 
        """
        #pm.isCalcFeat = True 
        #import mlff 
        #pm.isCalcFeat = False 

        # clean old .dat files otherwise will be appended 
        if os.path.exists("PWdata/MOVEMENTall"):
            print ("cleaning old data")
            subprocess.run([
                "rm -rf fread_dfeat output input train_data plot_data PWdata/*.txt PWdata/trainData.* PWdata/location PWdata/Egroup_weight PWdata/MOVEMENTall"
                ],shell=True)
            subprocess.run(["find PWdata/ -name 'dfeat*' | xargs rm -rf"],shell=True)
            subprocess.run(["find PWdata/ -name 'info*' | xargs rm -rf"],shell=True)

        # calculating feature 
        from src.pre_data.mlff_wu import calc_feat
        calc_feat()

        # seperate training set and valid set
        import src.pre_data.seper_wu as seper 
        seper.seperate_data(chunk_size = chunk_size)
        
        # save as .npy
        import src.pre_data.gen_data_wu as gen_data
        gen_data.write_data()
        
    """
        ============================================================
        =================training related functions=================
        ============================================================ 
    """ 

    # def set_epoch_num(self, input_num):
    #     self.n_epoch = input_num    

    def set_movement_weights(self, input_mvt_w): 

        """
            setting movement weight. Can be used as a initializer.
            A MAP: 
            absolute image index in MOVEMENTall -> the corresponding prefactor             
        """  
        mvmt_idx = [] 

        # dfeat file that contains the MOVEMENT dir the img index 
        tgt_file = pm.f_train_dfeat + str(pm.use_Ftype[0])
        values = pd.read_csv(tgt_file, header=None, encoding= 'unicode_escape').values

        get_mvt_name = lambda x: x[0].split('/')[-2]  

        st_idx = 0 
        mvt_name_prev = get_mvt_name(values[0])

        for idx, line in enumerate(values):

            # movement name 
            mvt_name = get_mvt_name(line) 

            # encounter a new movement file
            if mvt_name != mvt_name_prev:

                mvmt_idx.append([(st_idx,idx), mvt_name])
                st_idx = idx 
                mvt_name_prev = mvt_name 

        mvmt_idx.append([(st_idx,idx+1), mvt_name])
        self.movement_idx = mvmt_idx  

        #the map of movement name (i.e. the name of directory) and its weight
        self.movement_weights = input_mvt_w

    def scale_hybrid(self, train_data:MovementHybridDataset, valid_data:MovementHybridDataset, data_num:int):
        feat_train, feat_valid = None, None
        shape_train, shape_valid = [0], [0]
        for data_index in range(data_num):
            feat_train = train_data.feat[data_index] if feat_train is None else np.concatenate((feat_train,train_data.feat[data_index]),0)
            feat_valid = valid_data.feat[data_index] if feat_valid is None else np.concatenate((feat_valid,valid_data.feat[data_index]),0)
            shape_train.append(train_data.feat[data_index].shape[0] + sum(shape_train))
            shape_valid.append(valid_data.feat[data_index].shape[0] + sum(shape_valid))

        # scale feat
        if pm.use_storage_scaler:
            scaler_path = self.opts.opt_session_dir + 'scaler.pkl'
            print("loading scaler from file",scaler_path)
            self.scaler = load(scaler_path)
            print("transforming feat with loaded scaler")
            feat_train = self.scaler.transform(feat_train)
            feat_valid = self.scaler.transform(feat_valid)
        else:
            self.scaler = MinMaxScaler()
            feat_train = self.scaler.fit_transform(feat_train)
            feat_valid = self.scaler.transform(feat_valid)
        for i in range(len(shape_train)-1):
            train_data.feat[data_index] = feat_train[shape_train[i]:shape_train[i+1]]
            valid_data.feat[data_index] = feat_valid[shape_valid[i]:shape_valid[i+1]]
        
        # scale dfeat
        if pm.is_dfeat_sparse == False:
            for data_index in range(data_num):
                trans = lambda x : x.transpose(0, 1, 3, 2) 
                train_data.dfeat[data_index] = trans(trans(train_data.dfeat[data_index]) * self.scaler.scale_) 
                valid_data.dfeat[data_index] = trans(trans(valid_data.dfeat[data_index]) * self.scaler.scale_)           
    
        if pm.storage_scaler:
            pickle.dump(self.scaler, open(self.opts.opt_session_dir+"scaler.pkl",'wb'))
            print ("scaler.pkl saved to:",self.opts.opt_session_dir)

        return train_data, valid_data
    
    def scale(self,train_data,valid_data):

        """
            if pm.is_scale = True
        """
        
        if pm.use_storage_scaler:
            
            scaler_path = self.opts.opt_session_dir + 'scaler.pkl'
            print("loading scaler from file",scaler_path)
            self.scaler = load(scaler_path) 

            print("transforming feat with loaded scaler")
            train_data.feat = self.scaler.transform(train_data.feat)
            valid_data.feat = self.scaler.transform(valid_data.feat)

        else:
            # generate new scaler 
            print("using new scaler")
            
            self.scaler = MinMaxScaler()

            train_data.feat = self.scaler.fit_transform(train_data.feat)
            valid_data.feat = self.scaler.transform(valid_data.feat)

            if pm.storage_scaler:
                pickle.dump(self.scaler, open(self.opts.opt_session_dir+"scaler.pkl",'wb'))
                print ("scaler.pkl saved to:",self.opts.opt_session_dir)

        #atom index within this image, neighbor index, feature index, spatial dimension   
        if pm.is_dfeat_sparse == False: 
            
            trans = lambda x : x.transpose(0, 1, 3, 2) 

            print("transforming dense dfeat with loaded scaler")

            train_data.dfeat = trans(trans(train_data.dfeat) * self.scaler.scale_) 
            valid_data.dfeat = trans(trans(valid_data.dfeat) * self.scaler.scale_)

    def load_data(self):

        """
            In default, training data is not shuffled, and should not be.
        """
        # load anything other than dfeat
        
        self.set_nFeature()  

        torch_train_data = get_torch_data(pm.train_data_path)
        torch_valid_data = get_torch_data(pm.test_data_path)

        # scaler saved to self.scaler
        if pm.is_scale:
            self.scale(torch_train_data, torch_valid_data)
        
        assert self.scaler != None, "scaler is not correctly saved"

        """
            Note! When using sparse dfeat, shuffle must be False. 
            sparse dfeat class only works under the batch order before calling Data.DataLoader
        """
        self.loader_train = Data.DataLoader(torch_train_data, batch_size=self.batch_size, shuffle = True)
        self.loader_valid = Data.DataLoader(torch_valid_data, batch_size=self.batch_size, shuffle = False)

        assert self.loader_train != None, "training data (except dfeat) loading fails"
        assert self.loader_valid != None, "validation data (except dfeat) loading fails"

        # load sparse dfeat. This is the default setting 
        if pm.is_dfeat_sparse == True:     

            self.dfeat = dfeat_raw( input_dfeat_record_path_train = pm.f_train_dfeat, 
                                    input_feat_path_train = pm.f_train_feat,
                                    input_natoms_path_train = pm.f_train_natoms,

                                    input_dfeat_record_path_valid = pm.f_test_dfeat, 
                                    input_feat_path_valid = pm.f_test_feat,
                                    input_natoms_path_valid = pm.f_test_natoms,

                                    scaler = self.scaler)

            self.dfeat.load() 

    def load_data_hybrid(self, data_shuffle=False):

        """
            In default, training data is not shuffled, and should not be.
        """
        # load anything other than dfeat
        
        self.set_nFeature()
        train_data_path = os.path.dirname(pm.train_data_path)
        data_dirs = os.listdir(train_data_path)
        # data_dirs = sorted(data_dirs, key=lambda x: len(x.split('_')), reverse = True)
        torch_train_data = get_torch_data_hybrid(train_data_path, data_dirs, data_type = os.path.basename(pm.train_data_path), \
                                                 atom_type = pm.atomType, is_dfeat_sparse=pm.is_dfeat_sparse)
        self.energy_shift = torch_train_data.energy_shift

        torch_valid_data = get_torch_data_hybrid(train_data_path, data_dirs, data_type = os.path.basename(pm.test_data_path), \
                                                 atom_type = pm.atomType, is_dfeat_sparse=pm.is_dfeat_sparse)
        
        # scaler saved to self.scaler
        if pm.is_scale:
            torch_train_data, torch_valid_data = self.scale_hybrid(torch_train_data, torch_valid_data, len(data_dirs))
        
        assert self.scaler != None, "scaler is not correctly saved"

        #************************add by wuxing for LKF optimizer******************
        train_sampler = None
        val_sampler = None

        self.loader_train = torch.utils.data.DataLoader(
            torch_train_data,
            batch_size=self.batch_size,
            shuffle=data_shuffle, #(train_sampler is None)
            num_workers=self.workers, 
            pin_memory=True,
            sampler=train_sampler,
        )

        self.loader_valid = torch.utils.data.DataLoader(
            torch_valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers, 
            pin_memory=True,
            sampler=val_sampler,
        )
        
        """
            Note! When using sparse dfeat, shuffle must be False. 
            sparse dfeat class only works under the batch order before calling Data.DataLoader
        """
        #commit by wuxing
        # self.loader_train = Data.DataLoader(torch_train_data, batch_size=self.batch_size, shuffle = False)
        # self.loader_valid = Data.DataLoader(torch_valid_data, batch_size=self.batch_size, shuffle = False)
        #************end ***************#
        assert self.loader_train != None, "training data (except dfeat) loading fails"
        assert self.loader_valid != None, "validation data (except dfeat) loading fails"

        # load sparse dfeat. This is the default setting 
        if pm.is_dfeat_sparse == True:     
            self.dfeat = dfeat_raw( input_dfeat_record_path_train = pm.f_train_dfeat, 
                                    input_feat_path_train = pm.f_train_feat,
                                    input_natoms_path_train = pm.f_train_natoms,

                                    input_dfeat_record_path_valid = pm.f_test_dfeat, 
                                    input_feat_path_valid = pm.f_test_feat,
                                    input_natoms_path_valid = pm.f_test_natoms,

                                    scaler = self.scaler)

            self.dfeat.load() 


    def set_model(self, start_epoch = 1, model_name = None):
        """
            set the model 
        """ 
        # automatically calculate mean atomic energy
        if self.hybird is True:
            pm.itype_Ei_mean = self.energy_shift
        else:
            pm.itype_Ei_mean = self.set_b_init()
        self.model = MLFFNet(device = self.device)
        
        self.model.to(self.device)

        # THIS IS IMPORTANT
        self.start_epoch = start_epoch

        # load previous model if needed 
        if (self.opts.opt_recover_mode == True):
            
            if (self.opts.opt_session_name == ''):
                raise RuntimeError("session not specified for the recover mode. Use     _dir")
            
            if model_name is None:
                # use lattest.pt as default 
                load_model_path = self.opts.opt_model_dir+'latest.pt' 
            else:
                load_model_path = self.opts.opt_model_dir + model_name

            print ("load model from:",load_model_path)

            #self.load_model_path = self.opts.opt_model_dir+'better.pt'

            checkpoint = torch.load(load_model_path, map_location = self.device)

            self.model.load_state_dict(checkpoint['model'])

            self.start_epoch = checkpoint['epoch'] + 1 
        
        print("network initialized")
        
    def set_optimizer(self):

        """
            initialize optimzer 
        """

        #if self.use_GKalman or self.use_LKalman or self.use_SKalman:
        if self.kalman_type is not None: 
            """     
                use Kalman filter 
            """
            #if self.use_GKalman == True:
            if self.kalman_type == "GKF": 
                # self.optimizer = GKalmanFilter( self.model, 
                #                                 kalman_lambda = self.kalman_lambda, 
                #                                 kalman_nue = self.kalman_nue, 
                #                                 device = self.device)
                self.optimizer = GKFOptimizer(
                    self.model.parameters(), self.kalman_lambda, self.kalman_nue, self.device, self.precision
                )

            elif self.kalman_type == "LKF": 
                # self.optimizer = LKalmanFilter( 
                #                                 self.model, 
                #                                 kalman_lambda = self.kalman_lambda, 
                #                                 kalman_nue = self.kalman_nue, 
                #                                 device = self.device, 
                #                                 nselect = self.nselect, 
                #                                 groupsize = self.group_size, 
                #                                 blocksize = self.block_size, 
                #                                 fprefactor = self.opts.opt_fprefactor)
                self.optimizer = LKFOptimizer(
                    self.model.parameters(), self.kalman_lambda, self.kalman_nue, self.block_size
                )
            
            # elif self.kalman_type == "selected": 
            #     self.optimizer = SKalmanFilter( self.model, 
            #                                     kalman_lambda = self.kalman_lambda,
            #                                     kalman_nue = self.kalman_nue, 
            #                                     device = self.device)

            else: 
                raise Exception("input Kalman filter type is not supported")
        else:   
            """   
                use torch's built-in optimizer 
            """
            model_parameters = self.model.parameters()

            if (self.opts.opt_optimizer == 'SGD'):
                self.optimizer = optim.SGD(model_parameters, lr=self.LR_base, momentum=self.momentum, weight_decay=self.REGULAR_wd)

            elif (self.opts.opt_optimizer == 'ASGD'):
                self.optimizer = optim.ASGD(model_parameters, lr=self.LR_base, weight_decay=self.REGULAR_wd)

            elif (self.opts.opt_optimizer == 'RPROP'):
                self.optimizer = optim.Rprop(model_parameters, lr=self.LR_base, weight_decay = self.REGULAR_wd)

            elif (self.opts.opt_optimizer == 'RMSPROP'):
                self.optimizer = optim.RMSprop(model_parameters, lr=self.LR_base, weight_decay = self.REGULAR_wd, momentum = self.momentum)

            elif (self.opts.opt_optimizer == 'ADAG'):
                self.optimizer = optim.Adagrad(model_parameters, lr = self.LR_base, weight_decay = self.REGULAR_wd)

            elif (self.opts.opt_optimizer == 'ADAD'):
                self.optimizer = optim.Adadelta(model_parameters, lr = self.LR_base, weight_decay = self.REGULAR_wd)

            elif (self.opts.opt_optimizer == 'ADAM'):
                self.optimizer = optim.Adam(model_parameters, lr = self.LR_base, weight_decay = self.REGULAR_wd)
            elif (self.opts.opt_optimizer == 'ADAMW'):
                self.optimizer = optim.AdamW(model_parameters, lr = self.LR_base)

            elif (self.opts.opt_optimizer == 'ADAMAX'):
                self.optimizer = optim.Adamax(model_parameters, lr = self.LR_base, weight_decay = self.REGULAR_wd)

            elif (self.opts.opt_optimizer == 'LBFGS'):
                self.optimizer = optim.LBFGS(self.model.parameters(), lr = self.LR_base)

            else:
                raise RuntimeError("unsupported optimizer: %s" %self.opts.opt_optimizer)  

            # set scheduler
            self.set_scheduler() 
        
    def set_scheduler(self):

        # user specific LambdaLR lambda function
        lr_lambda = lambda epoch: self.LR_gamma ** epoch

        opt_scheduler = self.opts.opt_scheduler 

        if (opt_scheduler == 'LAMBDA'):
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = lr_lambda)
        elif (opt_scheduler == 'STEP'):
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = self.LR_step, gamma = self.LR_gamma)
        elif (opt_scheduler == 'MSTEP'):
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.ptimizer, milestones = self.opts.opt_LR_milestones, gamma = self.LR_gamma)
        elif (opt_scheduler == 'EXP'):
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma= self.LR_gamma)
        elif (opt_scheduler == 'NONE'):
            pass
        else:   
            raise RuntimeError("unsupported scheduler: %s" %opt_scheduler)
    
    def train(self):
        """    
            trianing method for the class 
        """ 
        iter = 0 
        iter_valid = 0 

        # set the log files
        iter_train_log = self.opts.opt_session_dir+'iter_loss.dat'
        f_iter_train_log = open(iter_train_log, 'w')
        epoch_train_log = self.opts.opt_session_dir+'epoch_loss.dat'
        f_epoch_train_log = open(epoch_train_log, 'w')
        iter_valid_log = self.opts.opt_session_dir+'iter_loss_valid.dat'
        f_iter_valid_log = open(iter_valid_log, 'w')
        epoch_valid_log =  self.opts.opt_session_dir + 'epoch_loss_valid.dat'
        f_epoch_valid_log = open(epoch_valid_log, 'w')
        
        # Define the lists based on the training type
        iter_train_lists = ["iter", "loss"]
        iter_valid_lists = ["iter", "loss"]
        epoch_train_lists = ["epoch", "loss"]
        epoch_valid_lists = ["epoch", "loss"]

        if self.is_trainEtot:
            iter_train_lists.append("RMSE_Etot")
            epoch_train_lists.append("RMSE_Etot")
            iter_valid_lists.append("RMSE_Etot")
            epoch_valid_lists.append("RMSE_Etot")
        if self.is_trainEi:
            iter_train_lists.append("RMSE_Ei")
            epoch_train_lists.append("RMSE_Ei")
            iter_valid_lists.append("RMSE_Ei")
            epoch_valid_lists.append("RMSE_Ei")
        if self.is_trainEgroup:
            iter_train_lists.append("RMSE_Egroup")
            epoch_train_lists.append("RMSE_Egroup")
            iter_valid_lists.append("RMSE_Egroup")
            epoch_valid_lists.append("RMSE_Egroup")
        if self.is_trainForce:
            iter_train_lists.append("RMSE_F")
            epoch_train_lists.append("RMSE_F")
            iter_valid_lists.append("RMSE_F")
            epoch_valid_lists.append("RMSE_F")

        if self.kalman_type is None:
            iter_valid_lists.extend(["lr"])

        print_width = {
            "iter": 5,
            "epoch": 5,
            "loss": 18,
            "RMSE_Etot": 18,
            "RMSE_Ei": 18,
            "RMSE_Egroup": 18,
            "RMSE_F": 18,
            "lr": 18
        }

        iter_train_format = "".join(["%{}s".format(print_width[i]) for i in iter_train_lists])
        iter_valid_format = "".join(["%{}s".format(print_width[i]) for i in iter_valid_lists])
        epoch_train_format = "".join(["%{}s".format(print_width[i]) for i in epoch_train_lists])
        epoch_valid_format = "".join(["%{}s".format(print_width[i]) for i in epoch_valid_lists])

        # write the header
        f_iter_train_log.write("%s\n" % (iter_train_format % tuple(iter_train_lists)))
        f_iter_valid_log.write("%s\n" % (iter_valid_format % tuple(iter_valid_lists)))
        f_epoch_train_log.write("%s\n" % (epoch_train_format % tuple(epoch_train_lists)))
        f_epoch_valid_log.write("%s\n" % (epoch_valid_format % tuple(epoch_valid_lists)))

        for epoch in range(self.start_epoch, self.n_epoch + 1):
            
            timeEpochStart = time.time()

            if (epoch == self.n_epoch):
                last_epoch = True
            else:
                last_epoch = False
            
            print("<-------------------------  epoch %d  ------------------------->" %(epoch))
            
            """
                ========== training starts ==========
            """ 
            
            nr_total_sample = 0
            loss = 0.
            loss_Etot = 0.
            loss_Ei = 0.
            loss_F = 0.
            loss_Egroup = 0.0 
            # this line code should go out?
            KFOptWrapper = KFOptimizerWrapper(
                self.model, self.optimizer, self.nselect, self.group_size, self.distributed, "torch"
             )
            
            self.model.train()
            # 重写一下训练这部分
            for i_batch, sample_batches in enumerate(self.loader_train):
                
                nr_batch_sample = sample_batches['input_feat'].shape[0]

                global_step = (epoch - 1) * len(self.loader_train) + i_batch * nr_batch_sample
                
                real_lr = self.adjust_lr(iter_num = global_step)

                if self.not_KF():
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = real_lr * (self.batch_size ** 0.5)
                natoms_sum = sample_batches['natoms_img'][0, 0].item()  
                # use sparse feature 
                if pm.is_dfeat_sparse == True:
                    # Error this function not realized
                    #sample_batches['input_dfeat']  = dfeat_train.transform(i_batch)
                    sample_batches['input_dfeat']  = self.dfeat.transform(i_batch,"train")

                if self.not_KF():
                    # non-KF t
                    batch_loss, batch_loss_Etot, batch_loss_Ei, batch_loss_F = \
                        self.train_img(sample_batches, self.model, self.optimizer, nn.MSELoss(), last_epoch, real_lr)
                else:
                    # KF 
                    batch_loss, batch_loss_Etot, batch_loss_Ei, batch_loss_F, batch_loss_Egroup = \
                        self.train_kalman_img(sample_batches, self.model, KFOptWrapper, nn.MSELoss(), last_epoch, 0.001)
                                
                iter += 1

                f_iter_train_log = open(iter_train_log, 'a')

                # Write the training log line to the log file
                iter_train_log_line = "%5d%18.10e" % (
                    iter,
                    batch_loss,
                )

                if self.is_trainEtot:
                    iter_train_log_line += "%18.10e" % (
                        math.sqrt(batch_loss_Etot)/natoms_sum
                    )
                if self.is_trainEi:
                    iter_train_log_line += "%18.10e" % (
                        math.sqrt(batch_loss_Ei)
                    )
                if self.is_trainEgroup:
                    iter_train_log_line += "%18.10e" % (
                        math.sqrt(batch_loss_Egroup)
                    )
                if self.is_trainForce:
                    iter_train_log_line += "%18.10e" % (
                        math.sqrt(batch_loss_F)
                    )
                if self.kalman_type is None:
                    iter_train_log_line += "%18.10e" % (
                        real_lr,
                    )

                f_iter_train_log.write("%s\n" % (iter_train_log_line))
                f_iter_train_log.close()
                
                loss += batch_loss.item() * nr_batch_sample

                loss_Etot += batch_loss_Etot.item() * nr_batch_sample
                loss_Ei += batch_loss_Ei.item() * nr_batch_sample
                loss_F += batch_loss_F.item() * nr_batch_sample
                loss_Egroup += batch_loss_Egroup.item() * nr_batch_sample 

                nr_total_sample += nr_batch_sample
                 
            loss /= nr_total_sample
            loss_Etot /= nr_total_sample
            loss_Ei /= nr_total_sample
            loss_F /= nr_total_sample
            loss_Egroup /= nr_total_sample  

            RMSE_Etot = loss_Etot ** 0.5
            RMSE_Ei = loss_Ei ** 0.5
            RMSE_F = loss_F ** 0.5
            RMSE_Egroup = loss_Egroup ** 0.5 

            print("epoch_loss = %.10f (RMSE_Etot = %.12f, RMSE_Ei = %.12f, RMSE_F = %.12f, RMSE_Eg = %.12f)" \
                %(loss, RMSE_Etot, RMSE_Ei, RMSE_F, RMSE_Egroup))
            """
            epoch_err_log = self.opts.opt_session_dir+'epoch_loss.dat'

            if epoch == 1:
                f_epoch_err_log = open(epoch_err_log, 'w')
                f_epoch_err_log.write('epoch\t loss\t RMSE_Etot\t RMSE_Ei\t RMSE_F\t RMSE_Eg\n')
                f_epoch_err_log.close()
                
            f_epoch_err_log = open(epoch_err_log, 'a')
            f_epoch_err_log.write('%d %e %e %e %e %e \n'%(epoch, loss, RMSE_Etot, RMSE_Ei, RMSE_F, RMSE_Egroup))
            f_epoch_err_log.close() 
            """
            f_epoch_train_log = open(epoch_train_log, 'a')

            # Write the training log line to the log file
            epoch_train_log_line = "%5d%18.10e" % (
                epoch,
                loss,
            )

            if self.is_trainEtot:
                epoch_train_log_line += "%18.10e" % (
                    RMSE_Etot
                )
            if self.is_trainEi:
                epoch_train_log_line += "%18.10e" % (
                    RMSE_Ei
                )
            if self.is_trainEgroup:
                epoch_train_log_line += "%18.10e" % (
                    RMSE_Egroup
                )
            if self.is_trainForce:
                epoch_train_log_line += "%18.10e" % (
                    RMSE_F
                )
            
            f_epoch_train_log.write("%s\n" % (epoch_train_log_line))
            f_epoch_train_log.close()
            
            if self.not_KF():
                """
                    for built-in optimizer only 
                """
                opt_scheduler = self.opts.opt_scheduler  

                if (opt_scheduler == 'OC'):
                    pass 
                elif (opt_scheduler == 'PLAT'):
                    self.scheduler.step(loss)

                elif (opt_scheduler == 'LR_INC'):
                    self.LinearLR(optimizer=self.optimizer, base_lr=self.LR_base, target_lr=pm.opt_LR_max_lr, total_epoch=self.n_epoch, cur_epoch=epoch)

                elif (opt_scheduler == 'LR_DEC'):
                    self.LinearLR(optimizer=self.optimizer, base_lr=self.LR_base, target_lr=pm.opt_LR_min_lr, total_epoch=self.n_epoch, cur_epoch=epoch)

                elif (opt_scheduler == 'NONE'):
                    pass

                else:
                    self.scheduler.step()

            """
                ========== validation starts ==========
            """ 
            
            nr_total_sample = 0
            valid_loss = 0.
            valid_loss_Etot = 0.
            valid_loss_Ei = 0.
            valid_loss_F = 0.
            valid_loss_Egroup = 0.0
            
            for i_batch, sample_batches in enumerate(self.loader_valid):
                
                iter_valid +=1 

                natoms_sum = sample_batches['natoms_img'][0, 0].item()
                nr_batch_sample = sample_batches['input_feat'].shape[0]

                if pm.is_dfeat_sparse == True:
                    sample_batches['input_dfeat']  = self.dfeat.transform(i_batch,"valid")

                if sample_batches['input_dfeat'] == "aborted":
                    continue 

                valid_error_iter, batch_loss_Etot, batch_loss_Ei, batch_loss_F, batch_loss_Egroup = self.valid_img(sample_batches, self.model, nn.MSELoss())

                # n_iter = (epoch - 1) * len(loader_valid) + i_batch + 1
                valid_loss += valid_error_iter * nr_batch_sample

                valid_loss_Etot += batch_loss_Etot * nr_batch_sample
                valid_loss_Ei += batch_loss_Ei * nr_batch_sample
                valid_loss_F += batch_loss_F * nr_batch_sample
                valid_loss_Egroup += batch_loss_Egroup * nr_batch_sample
                
                nr_total_sample += nr_batch_sample
                """
                f_err_log = self.opts.opt_session_dir+'iter_loss_valid.dat'

                if iter_valid == 1:
                    fid_err_log = open(f_err_log, 'w')
                    fid_err_log.write('iter\t loss\t RMSE_Etot\t RMSE_Ei\t RMSE_F\t lr\n')
                    fid_err_log.close() 

                fid_err_log = open(f_err_log, 'a')
                fid_err_log.write('%d %e %e %e %e %e \n'%(iter, batch_loss, math.sqrt(batch_loss_Etot)/natoms_sum, math.sqrt(batch_loss_Ei), math.sqrt(batch_loss_F), real_lr))
                fid_err_log.close() 
                """
                f_iter_valid_log = open(iter_valid_log, 'a')

                # Write the valid log line to the log file
                iter_valid_log_line = "%5d%18.10e" % (
                    iter,
                    batch_loss,
                )

                if self.is_trainEtot:
                    iter_valid_log_line += "%18.10e" % (
                        math.sqrt(batch_loss_Etot)/natoms_sum
                    )
                if self.is_trainEi:
                    iter_valid_log_line += "%18.10e" % (
                        math.sqrt(batch_loss_Ei)
                    )
                if self.is_trainEgroup:
                    iter_valid_log_line += "%18.10e" % (
                        math.sqrt(batch_loss_Egroup)
                    )
                if self.is_trainForce:
                    iter_valid_log_line += "%18.10e" % (
                        math.sqrt(batch_loss_F)
                    )
                if self.kalman_type is None:
                    iter_train_log_line += "%18.10e" % (
                        real_lr,
                    )

                f_iter_valid_log.write("%s\n" % (iter_valid_log_line))
                f_iter_valid_log.close()

            #end for

            # epoch loss update
            valid_loss /= nr_total_sample
            valid_loss_Etot /= nr_total_sample
            valid_loss_Ei /= nr_total_sample
            valid_loss_F /= nr_total_sample
            valid_loss_Egroup /= nr_total_sample

            valid_RMSE_Etot = valid_loss_Etot ** 0.5
            valid_RMSE_Ei = valid_loss_Ei ** 0.5
            valid_RMSE_F = valid_loss_F ** 0.5
            valid_RMSE_Egroup = valid_loss_Egroup ** 0.5
                
            print("valid_loss = %.10f (valid_RMSE_Etot = %.12f, valid_RMSE_Ei = %.12f, valid_RMSE_F = %.12f, valid_RMSE_Egroup = %.12f)" \
                     %(valid_loss, valid_RMSE_Etot, valid_RMSE_Ei, valid_RMSE_F, valid_RMSE_Egroup))
            """
            f_err_log =  self.opts.opt_session_dir + 'epoch_loss_valid.dat'
            
            if not os.path.exists(f_err_log):
                fid_err_log = open(f_err_log, 'w')
                fid_err_log.write('epoch\t valid_RMSE_Etot\t valid_RMSE_Ei\t valid_RMSE_F\t valid_RMSE_Eg \n')
                fid_err_log.close() 

            fid_err_log = open(f_err_log, 'a')
            fid_err_log.write('%d %e %e %e %e\n'%(epoch, valid_RMSE_Etot, valid_RMSE_Ei, valid_RMSE_F, valid_RMSE_Egroup))
            fid_err_log.close() 
            """
            f_epoch_valid_log = open(epoch_valid_log, 'a')

            # Write the valid log line to the log file
            epoch_valid_log_line = "%5d%18.10e" % (
                epoch,
                valid_loss,
            )

            if self.is_trainEtot:
                epoch_valid_log_line += "%18.10e" % (
                    valid_RMSE_Etot
                )
            if self.is_trainEi:
                epoch_valid_log_line += "%18.10e" % (
                    valid_RMSE_Ei
                )
            if self.is_trainEgroup:
                epoch_valid_log_line += "%18.10e" % (
                    valid_RMSE_Egroup
                )
            if self.is_trainForce:
                epoch_valid_log_line += "%18.10e" % (
                    valid_RMSE_F
                )

            f_epoch_valid_log.write("%s\n" % (epoch_valid_log_line))
            f_epoch_valid_log.close()

            """  
                save model 
            """

            if self.not_KF():
                state = {'model': self.model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
            else:
                state = {'model': self.model.state_dict(), 'epoch': epoch}
            
            latest_path = self.opts.opt_model_dir + "latest.pt"
            torch.save(state, latest_path) 

            if self.not_KF():
                if epoch % 10 == 0: 
                    current_model_path = self.opts.opt_model_dir + str(epoch) + '.pt'
                    torch.save(state, current_model_path)
            else:
                current_model_path = self.opts.opt_model_dir + str(epoch) + '.pt'
                torch.save(state, current_model_path)

            timeEpochEnd = time.time()

            print("time of epoch %d: %f s" %(epoch, timeEpochEnd - timeEpochStart))

    def load_and_train(self, data_shuffle=False):
        
        # transform data
        if self.hybird:
            self.load_data_hybrid(data_shuffle=data_shuffle)
        else:
            self.load_data()
        # initialize the network
        self.set_model()
        
        # initialize the optimizer and related scheduler
        self.set_optimizer()

        # set epoch number for training
        # self.set_epoch_num()

        # start training
        if self.inference is True:
            self.do_inference()
            return
        
        self.train()


    def train_img(self, sample_batches, model, optimizer, criterion, last_epoch, real_lr):
        """   
            single image traing for non-Kalman 
        """
        if (self.precision == 'float64'):
            Ei_label = Variable(sample_batches['output_energy'][:,:,:].double().to(self.device))
            Force_label = Variable(sample_batches['output_force'][:,:,:].double().to(self.device))   #[40,108,3]
            Egroup_label = Variable(sample_batches['input_egroup'].double().to(self.device))
            input_data = Variable(sample_batches['input_feat'].double().to(self.device), requires_grad=True)
            dfeat = Variable(sample_batches['input_dfeat'].double().to(self.device))  #[40,108,100,42,3]
            egroup_weight = Variable(sample_batches['input_egroup_weight'].double().to(self.device))
            divider = Variable(sample_batches['input_divider'].double().to(self.device))
            # Ep_label = Variable(sample_batches['output_ep'][:,:,:].double().to(device))
            
        elif (self.precision == 'float32'):
            Ei_label = Variable(sample_batches['output_energy'][:,:,:].float().to(self.device))
            Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(self.device))   #[40,108,3]
            Egroup_label = Variable(sample_batches['input_egroup'].float().to(self.device))
            input_data = Variable(sample_batches['input_feat'].float().to(self.device), requires_grad=True)
            dfeat = Variable(sample_batches['input_dfeat'].float().to(self.device))  #[40,108,100,42,3]
            egroup_weight = Variable(sample_batches['input_egroup_weight'].float().to(self.device))
            divider = Variable(sample_batches['input_divider'].float().to(self.device))
            # Ep_label = Variable(sample_batches['output_ep'][:,:,:].float().to(device))

        else:
            #error("train(): unsupported opt_dtype %s" %self.opts.opt_dtype)
            raise RuntimeError("train(): unsupported opt_dtype %s" %self.precision)  


        atom_number = Ei_label.shape[1]
        Etot_label = torch.sum(Ei_label, dim=1)
        neighbor = Variable(sample_batches['input_nblist'].int().to(self.device))  # [40,108,100]
        ind_img = Variable(sample_batches['ind_image'].int().to(self.device))
        natoms_img = Variable(sample_batches['natoms_img'].int().to(self.device))    

        Etot_predict, Ei_predict, Force_predict, Egroup_predict = self.model(input_data, dfeat, neighbor, natoms_img, egroup_weight, divider)
        
        optimizer.zero_grad()

        loss_Etot = torch.zeros([1,1],device = self.device)
        loss_Ei = torch.zeros([1,1],device = self.device)
        loss_F = torch.zeros([1,1],device = self.device)
        loss_Egroup = 0

        # update loss with repsect to the data used
        if self.is_trainEi:
            loss_Ei = criterion(Ei_predict, Ei_label)
        if self.is_trainEtot:
            loss_Etot = criterion(Etot_predict, Etot_label)
        if self.is_trainForce:
            loss_F = criterion(Force_predict, Force_label)

        start_lr = self.opts.opt_lr
        
        w_f = 1 if self.is_trainForce == True else 0
        w_e = 1 if self.is_trainEtot == True else 0
        w_ei = 1 if self.is_trainEi == True else 0
        w_eg = 0 

        loss, pref_f, pref_e = self.get_loss_func(start_lr, real_lr, w_f, loss_F, w_e, loss_Etot, w_eg, loss_Egroup, w_ei, loss_Ei, natoms_img[0, 0].item())

        # using a total loss to update weights 
        loss.backward()

        self.optimizer.step()
        
        return loss, loss_Etot, loss_Ei, loss_F

    def train_kalman_img(self,sample_batches, model, KFOptWrapper :KFOptimizerWrapper, criterion, last_epoch, real_lr):
        """
            why setting precision again? 
        """
        """
            **********************************************************************
        """
        Ei_label = Variable(sample_batches['output_energy'][:,:,:].to(self.device))
        Force_label = Variable(sample_batches['output_force'][:,:,:].to(self.device))   #[40,108,3]
        Egroup_label = Variable(sample_batches['input_egroup'].to(self.device))
        input_data = Variable(sample_batches['input_feat'].to(self.device), requires_grad=True)

        dfeat = Variable(sample_batches['input_dfeat'].to(self.device))  #[40,108,100,42,3]
        
        egroup_weight = Variable(sample_batches['input_egroup_weight'].to(self.device))
        divider = Variable(sample_batches['input_divider'].to(self.device))

        #atom_number = Ei_label.shape[1]
        Etot_label = torch.sum(Ei_label, dim=1)
        neighbor = Variable(sample_batches['input_nblist'].int().to(self.device))  # [40,108,100]
        #ind_img = Variable(sample_batches['ind_image'].int().to(self.device))
        natoms_img = Variable(sample_batches['natoms_img'].int().to(self.device))
        atom_type = Variable(sample_batches['atom_type'].int().to(self.device))
        """
            **********************************************************************
        """
        if self.is_trainEgroup:
            kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, egroup_weight, divider]
        else:
            kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, None, None]
        #        KFOptWrapper.update_energy(kalman_inputs, Etot_label)
        #        KFOptWrapper.update_force(kalman_inputs, Force_label, 2)
        # choosing what data are used for W update. Defualt are Etot and Force
        if self.is_trainEtot: 
            # kalman.update_energy(kalman_inputs, Etot_label, update_prefactor = self.kf_prefac_Etot)
            Etot_predict = KFOptWrapper.update_energy(kalman_inputs, Etot_label, self.kf_prefac_Etot, train_type = "NN")

        if self.is_trainEi:
            Ei_predict = KFOptWrapper.update_ei(kalman_inputs,Ei_label, update_prefactor = self.kf_prefac_Ei, train_type = "NN")     

        if self.is_trainEgroup:
            # kalman.update_egroup(kalman_inputs, Egroup_label)
            Egroup_predict = KFOptWrapper.update_egroup(kalman_inputs, Egroup_label, self.kf_prefac_Egroup, train_type = "NN")

        # if Egroup does not participate in training, the output of Egroup_predict will be None
        if self.is_trainForce:
            Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = KFOptWrapper.update_force(
                    kalman_inputs, Force_label, self.kf_prefac_F, train_type = "NN")

        # if self.is_trainForce is True:
        #     if self.is_trainEgroup is True:
        #         Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = KFOptWrapper.update_force(
        #             kalman_inputs, Force_label, self.kf_prefac_F)
        #     else:
        #         Etot_predict, Ei_predict, Force_predict, Virial_predict = KFOptWrapper.update_force(
        #             kalman_inputs, Force_label, self.kf_prefac_F)

        # if self.is_trainEi:
        #     kalman.update_ei(kalman_inputs,Ei_label, update_prefactor = self.kf_prefac_Ei)     
        # if self.is_trainForce:
        #     kalman.update_force(kalman_inputs, Force_label, update_prefactor = self.kf_prefac_F)

        #kalman.update_ei_and_force(kalman_inputs,Ei_label,Force_label,update_prefactor = 0.1)
        
        # Etot_predict, Ei_predict, Force_predict, _, _ = model(input_data, dfeat, neighbor, natoms_img, egroup_weight, divider)
        Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(kalman_inputs[0], kalman_inputs[1], kalman_inputs[2], kalman_inputs[3], kalman_inputs[4], kalman_inputs[5], kalman_inputs[6])

        # if self.is_trainEgroup:
        #     Egroup_predict = torch.zeros_like(Ei_predict)
        #     Egroup_predict = model.get_egroup(Ei_predict,egroup_weight,divider) 

        # dtype same as torch.default
        loss_Etot = torch.zeros([1,1],device = self.device)
        loss_Ei = torch.zeros([1,1], device = self.device)
        loss_F = torch.zeros([1,1],device = self.device)
        loss_egroup = torch.zeros([1,1],device = self.device)

        """
            update loss only for used labels  
            At least 2 flags should be true. 
        """

        if self.is_trainEi:
            loss_Ei = criterion(Ei_predict, Ei_label)
        
        if self.is_trainEtot:
            loss_Etot = criterion(Etot_predict, Etot_label)

        if self.is_trainForce:
            loss_F = criterion(Force_predict, Force_label)

        if self.is_trainEgroup:
            loss_egroup = criterion(Egroup_label,Egroup_predict)

        loss = loss_F + loss_Etot + loss_Ei + loss_egroup 
        
        print("RMSE_Etot = %.12f, RMSE_Ei = %.12f, RMSE_Force = %.12f, RMSE_Egroup = %.12f" %(loss_Etot ** 0.5, loss_Ei ** 0.5, loss_F ** 0.5, loss_egroup**0.5))
        
        del Ei_label
        del Force_label
        del Egroup_label
        del input_data
        del dfeat
        del egroup_weight
        del divider
        del Etot_label
        del neighbor
        del natoms_img

        return loss, loss_Etot, loss_Ei, loss_F, loss_egroup

    def valid_img(self,sample_batches, model, criterion):
        """
            ******************* load *********************
        """
        Ei_label = Variable(sample_batches['output_energy'][:,:,:].to(self.device))
        Force_label = Variable(sample_batches['output_force'][:,:,:].to(self.device))   #[40,108,3]
        Egroup_label = Variable(sample_batches['input_egroup'].to(self.device))
        input_data = Variable(sample_batches['input_feat'].to(self.device), requires_grad=True)
        dfeat = Variable(sample_batches['input_dfeat'].to(self.device))  #[40,108,100,42,3]
        egroup_weight = Variable(sample_batches['input_egroup_weight'].to(self.device))
        divider = Variable(sample_batches['input_divider'].to(self.device))
        
        neighbor = Variable(sample_batches['input_nblist'].int().to(self.device))  # [40,108,100]
        natoms_img = Variable(sample_batches['natoms_img'].int().to(self.device))  # [40,108,100]
        atom_type = Variable(sample_batches['atom_type'].int().to(self.device))

        error=0
        atom_number = Ei_label.shape[1]
        Etot_label = torch.sum(Ei_label, dim=1)
        
        # model.train()
        self.model.eval()
        if self.is_trainEgroup:
            kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, egroup_weight, divider]
        else:
            kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, None, None]

        Etot_predict, Ei_predict, Force_predict, Egroup_predict, _ = model(kalman_inputs[0], kalman_inputs[1], kalman_inputs[2], kalman_inputs[3], kalman_inputs[4], kalman_inputs[5], kalman_inputs[6])
        
        if self.dbg is True:
            print("Etot predict")
            print(Etot_predict)
            
            print("Force predict")
            print(Force_predict)

        # Egroup_predict = torch.zeros_like(Ei_predict)

        # if self.is_trainEgroup:
        #     Egroup_predict = self.model.get_egroup(Ei_predict,egroup_weight,divider)

        # Egroup_predict = model.get_egroup(Ei_predict, egroup_weight, divider) 

        """
        loss_F = criterion(Force_predict, Force_label)
        loss_Etot = criterion(Etot_predict, Etot_label)
        loss_Ei = criterion(Ei_predict, Ei_label)
        """
        
        loss_Etot = torch.zeros([1,1],device=self.device)
        loss_Ei = torch.zeros([1,1],device=self.device)
        loss_F = torch.zeros([1,1],device = self.device)
        loss_egroup = torch.zeros([1,1],device = self.device)
        
        # update loss with repsect to the data used
        if self.is_trainEi:
            loss_Ei = criterion(Ei_predict, Ei_label)
        
        if self.is_trainEtot:
            loss_Etot = criterion(Etot_predict, Etot_label)

        if self.is_trainForce:
            loss_F = criterion(Force_predict, Force_label)

        if self.is_trainEgroup:
            loss_egroup = criterion(Egroup_label,Egroup_predict)

        error = float(loss_F.item()) + float(loss_Etot.item()) + float(loss_Ei.item()) + float(loss_egroup.item())

        # del Ei_label
        # del Force_label
        # del Egroup_label
        # del input_data
        # del dfeat
        # del egroup_weight
        # del divider
        # del neighbor
        # del natoms_img  
        
        return error, loss_Etot, loss_Ei, loss_F, loss_egroup

    def do_inference(self):

        train_lists = ["img_idx", "Etot_lab", "Etot_pre", "Ei_lab", "Ei_pre", "Force_lab", "Force_pre"]
        train_lists.extend(["RMSE_Etot", "RMSE_Etot_per_atom", "RMSE_Ei", "RMSE_F"])
        if self.is_trainEgroup:
            train_lists.append("RMSE_Egroup")
        res_pd = pd.DataFrame(columns=train_lists)

        inf_dir = os.path.join(self.opts.opt_session_dir, "inference")
        if os.path.exists(inf_dir) is True:
            shutil.rmtree(inf_dir)
        os.mkdir(inf_dir)
        res_pd_save_path = os.path.join(inf_dir, "inference_info.csv")
        inf_force_save_path = os.path.join(inf_dir,"inference_force.txt")
        lab_force_save_path = os.path.join(inf_dir,"label_force.txt")
        inf_energy_save_path = os.path.join(inf_dir,"inference_energy.txt")
        lab_energy_save_path = os.path.join(inf_dir,"label_energy.txt")
        inf_Ei_save_path = os.path.join(inf_dir,"inference_Ei.txt")
        lab_Ei_save_path = os.path.join(inf_dir,"label_Ei.txt")

        for i_batch, sample_batches in enumerate(self.loader_train):
            if pm.is_dfeat_sparse == True:
                sample_batches['input_dfeat']  = self.dfeat.transform(i_batch,"valid")
            if sample_batches['input_dfeat'] == "aborted":
                continue 

            Ei_label = Variable(sample_batches['output_energy'][:,:,:].to(self.device))
            Force_label = Variable(sample_batches['output_force'][:,:,:].to(self.device))   #[40,108,3]
            Egroup_label = Variable(sample_batches['input_egroup'].to(self.device))
            input_data = Variable(sample_batches['input_feat'].to(self.device), requires_grad=True)
            dfeat = Variable(sample_batches['input_dfeat'].to(self.device))  #[40,108,100,42,3]
            egroup_weight = Variable(sample_batches['input_egroup_weight'].to(self.device))
            divider = Variable(sample_batches['input_divider'].to(self.device))
            
            neighbor = Variable(sample_batches['input_nblist'].int().to(self.device))  # [40,108,100]
            natoms_img = Variable(sample_batches['natoms_img'].int().to(self.device))  # [40,108,100]
            atom_type = Variable(sample_batches['atom_type'].int().to(self.device))
            Etot_label = torch.sum(Ei_label, dim=1)
            
            self.model.eval()
            if self.is_trainEgroup:
                kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, egroup_weight, divider]
            else:
                kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, None, None]

            Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = self.model(kalman_inputs[0], kalman_inputs[1], kalman_inputs[2], kalman_inputs[3], kalman_inputs[4], kalman_inputs[5], kalman_inputs[6])
            
            # mse
            criterion = nn.MSELoss()
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_F_val = criterion(Force_predict, Force_label)
            loss_Ei_val = criterion(Ei_predict, Ei_label)
            if self.is_trainEgroup is True:
                loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
            # rmse
            Etot_rmse = loss_Etot_val ** 0.5
            etot_atom_rmse = Etot_rmse / natoms_img[0][0]
            Ei_rmse = loss_Ei_val ** 0.5
            F_rmse = loss_F_val ** 0.5

            res_list = [i_batch, float(Etot_label), float(Etot_predict), \
                        float(Ei_label.abs().mean()), float(Ei_predict.abs().mean()), \
                        float(Force_label.abs().mean()), float(Force_predict.abs().mean()),\
                        float(Etot_rmse), float(etot_atom_rmse), float(Ei_rmse), float(F_rmse)]
            res_pd.loc[res_pd.shape[0]] = res_list
           
            #''.join(map(str, list(np.array(Force_predict.flatten().cpu().data))))
            ''.join(map(str, list(np.array(Force_predict.flatten().cpu().data))))
            write_line_to_file(inf_force_save_path, \
                               ' '.join(np.array(Force_predict.flatten().cpu().data).astype('str')), "a")
            write_line_to_file(lab_force_save_path, \
                               ' '.join(np.array(Force_label.flatten().cpu().data).astype('str')), "a")
            
            write_line_to_file(inf_Ei_save_path, \
                               ' '.join(np.array(Ei_predict.flatten().cpu().data).astype('str')), "a")
            write_line_to_file(lab_Ei_save_path, \
                               ' '.join(np.array(Ei_label.flatten().cpu().data).astype('str')), "a")
            
            write_line_to_file(inf_energy_save_path, \
                               ' '.join(np.array(Etot_label.flatten().cpu().data).astype('str')), "a")
            write_line_to_file(lab_energy_save_path, \
                               ' '.join(np.array(Etot_predict.flatten().cpu().data).astype('str')), "a")
            
        res_pd.to_csv(res_pd_save_path)
        
        inference_cout = ""
        inference_cout += "For {} images: \n".format(res_pd.shape[0])
        inference_cout += "Avarage REMSE of Etot: {} \n".format(res_pd['RMSE_Etot'].mean())
        inference_cout += "Avarage REMSE of Etot per atom: {} \n".format(res_pd['RMSE_Etot_per_atom'].mean())
        inference_cout += "Avarage REMSE of Ei: {} \n".format(res_pd['RMSE_Ei'].mean())
        inference_cout += "Avarage REMSE of RMSE_F: {} \n".format(res_pd['RMSE_F'].mean())
        if self.is_trainEgroup:
            inference_cout += "Avarage REMSE of RMSE_Egroup: {} \n".format(res_pd['RMSE_Egroup'].mean())
        if self.is_trainViral:  #not realized
            inference_cout += "Avarage REMSE of RMSE_virial: {} \n".format(res_pd['RMSE_virial'].mean())
            inference_cout += "Avarage REMSE of RMSE_virial_per_atom: {} \n".format(res_pd['RMSE_virial_per_atom'].mean())

        inference_cout += "\nMore details can be found under the file directory:\n{}\n".format(inf_dir)
        print(inference_cout)
        return

    """ 
        ============================================================
        ====================MD related functions====================
        ============================================================ 
    """
    def gen_lammps_config(self):
        """
            generate lammps config from POSCAR
            Default name is POSCAR
        """ 
        
        import poscar2lammps
        poscar2lammps.p2l() 
        

    def extract_model_para(self, model_name = None):
        """
            extracting network parameters and scaler values for fortran MD routine
        """
        from src.aux.extract_nn import read_wij, read_scaler 

        if model_name is None:  
            load_model_path = self.opts.opt_model_dir + 'latest.pt' 
        else:
            load_model_path = model_name

        print ("extracting parameters from:", load_model_path)

        read_wij(load_model_path)

        load_scaler_path = self.opts.opt_session_dir + "scaler.pkl"

        print ("extracting scaler values from:", load_scaler_path) 

        read_scaler(load_scaler_path)
        
    def extract_force_field(self, name= "myforcefield.ff", model_name = None):
        
        from extract_ff import extract_ff

        self.extract_model_para(model_name= model_name)
        extract_ff(name = name, model_type = 3)
                

    def run_md(self, init_config = "atom.config", md_details = None, num_thread = 1,follow = False):
        
        mass_table = {  1:1.007,2:4.002,3:6.941,4:9.012,5:10.811,6:12.011,
                        7:14.007,8:15.999,9:18.998,10:20.18,11:22.99,12:24.305,
                        13:26.982,14:28.086,15:30.974,16:32.065,17:35.453,
                        18:39.948,19:39.098,20:40.078,21:44.956,22:47.867,
                        23:50.942,24:51.996,25:54.938,26:55.845,27:58.933,
                        28:58.693,29:63.546,30:65.38,31:69.723,32:72.64,33:74.922,
                        34:78.96,35:79.904,36:83.798,37:85.468,38:87.62,39:88.906,
                        40:91.224,41:92.906,42:95.96,43:98,44:101.07,45:102.906,46:106.42,
                        47:107.868,48:112.411,49:114.818,50:118.71,51:121.76,52:127.6,
                        53:126.904,54:131.293,55:132.905,56:137.327,57:138.905,58:140.116,
                        59:140.908,60:144.242,61:145,62:150.36,63:151.964,64:157.25,65:158.925,
                        66:162.5,67:164.93,68:167.259,69:168.934,70:173.054,71:174.967,72:178.49,
                        73:180.948,74:183.84,75:186.207,76:190.23,77:192.217,78:195.084,
                        79:196.967,80:200.59,81:204.383,82:207.2,83:208.98,84:210,85:210,86:222,
                        87:223,88:226,89:227,90:232.038,91:231.036,92:238.029,93:237,94:244,
                        95:243,96:247,97:247,98:251,99:252,100:257,101:258,102:259,103:262,104:261,105:262,106:266}
        
        # remove existing MOVEMENT file for not 
        if follow == False:
            os.system('rm -f MOVEMENT')     
        
        if md_details is None:  
            raise Exception("md detail is missing")
        
        md_detail_line = str(md_details)[1:-1]+"\n"
        
        if os.path.exists(init_config) is not True: 
            raise Exception("initial config for MD is not found")
        
        # preparing md.input 
        f = open('md.input', 'w')
        f.write(init_config+"\n")

        f.write(md_detail_line) 
        f.write('F\n')
        f.write("3\n")     # imodel=1,2,3.    {1:linear;  2:VV;   3:NN;}
        f.write('1\n')               # interval for MOVEMENT output
        f.write('%d\n' % len(pm.atomType)) 
            
        # write mass 
        for i in range(len(pm.atomType)):
            #f.write('%d %f\n' % (pm.atomType[i], 2*pm.atomType[i]))
            f.write('%d %f\n' % (pm.atomType[i], mass_table[pm.atomType[i]]))
        f.close()    
        
        # creating md.input for main_MD.x 
        command = r'mpirun -n ' + str(num_thread) + r' main_MD.x'
        print (command)
        subprocess.run(command, shell=True) 
        
    """
        ============================================================
        ===================auxiliary functions======================
        ============================================================ 
    """

    def use_global_kalman(self):
        self.kalman_type = "GKF"

    def use_layerwise_kalman(self):
        self.kalman_type = "LKF"

    def set_kalman_lambda(self,val):
        self.kalman_lambda = val  
    # set prefactor 
    def set_kalman_nue(self,val):
        self.kalman_nue = val

    def set_kf_prefac_Etot(self,val):
        self.kf_prefac_Etot = val

    def set_kf_prefac_Ei(self,val):
        self.kf_prefac_Ei = val 

    def set_kf_prefac_F(self,val):
        self.kf_prefac_F = val 

    def set_kf_prefac_Egroup(self,val):
        self.kf_prefac_Egroup = val 

    # set label to train 
    def set_train_force(self,val):
        self.is_trainForce = val 

    def set_train_Ei(self,val):
        self.is_trainEi = val

    def set_train_Etot(self,val):
        self.is_trainEtot = val 

    def set_train_Egroup(self,val):
        self.is_trainEgroup = val   

    def set_load_model_path(self,val): 
        self.load_model_path = val 

    def set_max_neigh_num(self,val):
        pm.maxNeighborNum = val 

    def not_KF(self):

        if self.kalman_type is None: 
            return True
        else:
            return False
    
    def set_b_init(self):

        """
            get mean atomic energy (Ei) for each type automatically

        """
        type_dict = {} 
        result = []

        tgt_dir = os.getcwd() + r"/train_data/final_train/"        
        
        num_atom = np.load(tgt_dir+"ind_img.npy")[1]

        # atom type list of a image
        type_list = np.load(tgt_dir+"itypes.npy")[0:num_atom]
        atomic_energy_list = np.load(tgt_dir+"engy_scaled.npy")[0:num_atom]
        
        for atom, energy in zip(type_list, atomic_energy_list):
            if atom not in type_dict:
                type_dict[atom] = [energy]
            else:
                type_dict[atom].append(energy)
        
        for atom in pm.atomType:
            result.append(np.mean(type_dict[atom]))

        print ("initial bias for atoms:", result)

        return result.copy()         
        
    def set_session_dir(self,session_dir):

        print("models and other information will be saved in:",'/'+session_dir)

        self.opts.opt_session_name = session_dir
        self.opts.opt_session_dir = './'+self.opts.opt_session_name+'/'
        self.opts.opt_logging_file = self.opts.opt_session_dir+'train.log'
        self.opts.opt_model_dir = self.opts.opt_session_dir+'model/'

        if not os.path.exists(self.opts.opt_session_dir):
            os.makedirs(self.opts.opt_session_dir) 
        if not os.path.exists(self.opts.opt_model_dir):
            os.makedirs(self.opts.opt_model_dir)

    def set_nFeature(self):
        """    
            obtain number of feature from fread_dfeat/feat.info
        """
        from os import path
        
        tgt = "fread_dfeat/feat.info" 

        if not path.exists(tgt):
            raise Exception("feature information file feat.info is not generated")

        f = open(tgt,'r')

        raw = f.readlines()[-1].split()

        pm.nFeatures = sum([int(item) for item in raw])

        print("number of features:",pm.nFeatures)
    
    def print_feat_para(self):
        # print feature parameter 
        
        for feat_idx in pm.feature_type:
            name  = "" 
            
            print(name)
            print(getattr(pm,name))
            
        pass 

    # print starting info
    def print_parameters(self):

        """
            print all infos at the beginning 
        """
        self.opts.summary("")
        self.opts.summary("#########################################################################################")
        self.opts.summary("#            ___          __                         __      __  ___       __  ___      #")
        self.opts.summary("#      |\ | |__  |  |    |__) |  | |\ | |\ | | |\ | / _`    /__`  |   /\  |__)  |       #")
        self.opts.summary("#      | \| |___ |/\|    |  \ \__/ | \| | \| | | \| \__>    .__/  |  /~~\ |  \  |       #")
        self.opts.summary("#                                                                                       #")
        self.opts.summary("#########################################################################################")
        self.opts.summary("") 

        print("Training: set default dtype to float64: %s" %self.opts.opt_dtype)
        print("Training: rseed = %s" %self.opts.opt_rseed) 
        print("Training: session = %s" %self.opts.opt_session_name)
        print("Training: run_id = %s" %self.opts.opt_run_id)
        print("Training: journal_cycle = %d" %self.opts.opt_journal_cycle)
        print("Training: follow_mode = %s" %self.opts.opt_follow_mode)
        print("Training: recover_mode = %s" %self.opts.opt_recover_mode)
        print("Training: network = %s" %self.opts.opt_net_cfg)
        print("Training: model_dir = %s" %self.opts.opt_model_dir)
        print("Training: model_file = %s" %self.opts.opt_model_file)
        print("Training: activation = %s" %self.opts.opt_act)
        print("Training: optimizer = %s" %self.opts.opt_optimizer)
        print("Training: momentum = %.16f" %self.opts.opt_momentum)
        print("Training: REGULAR_wd = %.16f" %self.opts.opt_regular_wd)
        print("Training: scheduler = %s" %self.opts.opt_scheduler)
        print("Training: n_epoch = %d" %self.opts.opt_epochs)
        print("Training: LR_base = %.16f" %self.opts.opt_lr)
        print("Training: LR_gamma = %.16f" %self.opts.opt_gamma)
        print("Training: LR_step = %d" %self.opts.opt_step)
        print("Training: batch_size = %d" %self.opts.opt_batch_size)

        # scheduler specific options
        print("Scheduler: opt_LR_milestones = %s" %self.opts.opt_LR_milestones)
        print("Scheduler: opt_LR_patience = %s" %self.opts.opt_LR_patience)
        print("Scheduler: opt_LR_cooldown = %s" %self.opts.opt_LR_cooldown)
        print("Scheduler: opt_LR_total_steps = %s" %self.opts.opt_LR_total_steps)
        print("Scheduler: opt_LR_max_lr = %s" %self.opts.opt_LR_max_lr)
        print("Scheduler: opt_LR_min_lr = %s" %self.opts.opt_LR_min_lr)
        print("Scheduler: opt_LR_T_max = %s" %self.opts.opt_LR_T_max)
        print("scheduler: opt_autograd = %s" %self.opts.opt_autograd)

    # calculate loss 
    def get_loss_func(self,start_lr, real_lr, has_fi, lossFi, has_etot, loss_Etot, has_egroup, loss_Egroup, has_ei, loss_Ei, natoms_sum):
        start_pref_egroup = 0.02
        limit_pref_egroup = 1.0
        start_pref_F = 1000  #1000
        limit_pref_F = 1.0
        start_pref_etot = 0.02   
        limit_pref_etot = 1.0
        start_pref_ei = 0.02
        limit_pref_ei = 1.0

        pref_fi = has_fi * (limit_pref_F + (start_pref_F - limit_pref_F) * real_lr / start_lr)
        pref_etot = has_etot * (limit_pref_etot + (start_pref_etot - limit_pref_etot) * real_lr / start_lr)
        pref_egroup = has_egroup * (limit_pref_egroup + (start_pref_egroup - limit_pref_egroup) * real_lr / start_lr)
        pref_ei = has_ei * (limit_pref_ei + (start_pref_ei - limit_pref_ei) * real_lr / start_lr)

        l2_loss = 0
        
        if has_fi==1:
            l2_loss += pref_fi * lossFi      # * 108
        if has_etot==1:
            l2_loss += 1./natoms_sum * pref_etot * loss_Etot  # 1/108 = 0.009259259259, 1/64=0.015625
        if has_egroup==1:
            l2_loss += pref_egroup * loss_Egroup
        if has_ei==1:
            l2_loss += pref_ei * loss_Ei
        
        return l2_loss, pref_fi, pref_etot

    #update learning rate at iter_num
    def adjust_lr(self,iter_num,  stop_lr=3.51e-8):

        start_lr = self.opts.opt_lr 

        stop_step = 1000000
        decay_step=5000
        decay_rate = np.exp(np.log(stop_lr/start_lr) / (stop_step/decay_step)) #0.9500064099092085
        real_lr = start_lr * np.power(decay_rate, (iter_num//decay_step))
        return real_lr  

    # implement a linear scheduler
    def LinearLR(self,optimizer, base_lr, target_lr, total_epoch, cur_epoch):

        lr = base_lr - (base_lr - target_lr) * (float(cur_epoch) / float(total_epoch))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    
    def evaluate(self,num_thread = 1):
        """
            evaluate a model w.r.t AIMD
            put a MOVEMENT in /MD and run MD100 
        """
        # features input for main_MD.x
        
        if not os.path.exists("input"):
            os.mkdir("input")
            
        import prepare as pp
        pp.writeGenFeatInput()

        os.system('rm -f MOVEMENT')

        if not os.path.exists("MD/MOVEMENT"):
            raise Exception("MD/MOVEMENT not found")

        import md100
        md100.run_md100(imodel = 3, atom_type = pm.atomType, num_process = num_thread)

    def plot_evaluation(self, plot_elem, save_data):
                
        if not os.path.exists("MOVEMENT"):
            raise Exception("MOVEMENT not found. It should be force field MD result")

        import plot_evaluation
        # plot_evaluation.plot()

        if self.is_trainEi or self.is_trainEgroup:
            plot_ei = True
        else:
            plot_ei = False

        plot_evaluation.plot_new(atom_type = pm.atomType, plot_elem = plot_elem, save_data = save_data, plot_ei = plot_ei)
    
    """
        ======================================================================
        ===================user-defined debugging functions===================
        ======================================================================  
    """


    def mydebug():
        return "hahaha"
