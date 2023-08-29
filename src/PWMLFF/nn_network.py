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
import horovod.torch as hvd

import time
import numpy as np
import pandas as pd
import torch.nn as nn
import math

import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

import time
from optimizer.KFWrapper import KFOptimizerWrapper
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

import pickle

from model.MLFF import MLFFNet
import default_para as pm

# get_torch_data for nn single system training
from src.pre_data.data_loader_2type import get_torch_data
#get_torch_data_hybrid for nn multi hybrid systems training
from src.pre_data.data_loader_2type_nn_hybrid import MovementHybridDataset, get_torch_data_hybrid
from src.pre_data.dfeat_sparse import dfeat_raw 
from src.pre_data.nn_mlff_hybrid import get_cluster_dirs, make_work_dir, mv_featrues, copy_file

from src.PWMLFF.nn_param_extract import load_scaler_from_checkpoint
from utils.file_operation import write_line_to_file, smlink_file
from src.user.model_param import DpParam
from src.aux.plot_nn_inference import plot
# from optimizer.kalmanfilter import GKalmanFilter, LKalmanFilter, SKalmanFilter
from src.optimizer.GKF import GKFOptimizer
from src.optimizer.LKF import LKFOptimizer

"""
    class that wraps the training model and the data
    network/network parameters 
    training/validation data
    training method 
    Using default_para is purely a legacy problem
"""  
class nn_network:
    def __init__(self, dp_params: DpParam):
        self.workers = 1
        # Global parameters of the class.  
        # Some member data are only "declared" here, and assignment requires further operations 
        # Frequently used control parameter can be reset by class.set_xx() functions
        
        # Frequently-used control parameters that are directly passed in as arguments:        
        # initializing the opt class 
        # self.opts = opt_values()
        self.dp_params = dp_params
        # passing some input arguments to default_para.py
        # at the end of the day, one no longer needs parameters.py
        pm.atomType = self.dp_params.atom_type
        self.atom_type = self.dp_params.atom_type
        pm.maxNeighborNum = self.dp_params.max_neigh_num 
        pm.nodeDim = self.dp_params.model_param.fitting_net.network_size
        pm.atomTypeNum = len(pm.atomType)       #this step is important. Unexpected error
        pm.ntypes = len(pm.atomType)
        #number of layer
        pm.nLayer = len(pm.nodeDim)
        # pm.is_dfeat_sparse is True as default
        pm.is_dfeat_sparse = self.dp_params.is_dfeat_sparse 
        # network for each type of element
        tmp = []
        for layer in range(pm.nLayer):
            tmp_layer = [pm.nodeDim[layer] for atom in pm.atomType]
            tmp.append(tmp_layer.copy()) 
        pm.nNodes = np.array(tmp) 
        # feature set feature type 
        pm.use_Ftype = sorted(self.dp_params.descriptor.feature_type)
        for ftype  in pm.use_Ftype:
            ftype_key = "{}".format(ftype)
            if ftype_key == '1':
                pm.Ftype1_para = self.dp_params.descriptor.feature_dict[ftype_key]
            elif ftype_key == '2':
                pm.Ftype2_para = self.dp_params.descriptor.feature_dict[ftype_key]
            elif ftype_key == '3':
                pm.Ftype3_para = self.dp_params.descriptor.feature_dict[ftype_key]
            elif ftype_key == '4':
                pm.Ftype4_para = self.dp_params.descriptor.feature_dict[ftype_key]
            elif ftype_key == '5':
                pm.Ftype5_para = self.dp_params.descriptor.feature_dict[ftype_key]
            elif ftype_key == '6':
                pm.Ftype6_para = self.dp_params.descriptor.feature_dict[ftype_key]
            elif ftype_key == '7':
                pm.Ftype7_para = self.dp_params.descriptor.feature_dict[ftype_key]
            elif ftype_key == '8':
                pm.Ftype8_para = self.dp_params.descriptor.feature_dict[ftype_key]

        # update nfeat_type 
        pm.nfeat_type = len(pm.use_Ftype)
        # self.feat_mod = feat_modifier() 

        pm.feature_dtype = self.dp_params.precision
        # set training precision. Overarching 
        if (self.dp_params.precision == 'float64'):
            print("Training: set default dtype to double")
            torch.set_default_dtype(torch.float64)
        elif (self.dp_params.precision == 'float32'):
            print("Training: set default dtype to single")
            torch.set_default_dtype(torch.float32)
        else:
            raise RuntimeError("Training: unsupported dtype: %s" %self.dp_params.precision)  

        if self.dp_params.hvd:
            hvd.init()
            self.dp_params.gpu = hvd.local_rank()

        # set training device
        if torch.cuda.is_available():
            if self.dp_params.gpu:
                print("Use GPU: {} for training".format(self.dp_params.gpu))
                self.device = torch.device("cuda:{}".format(self.dp_params.gpu))
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print ("device used",self.device)
        # set random seed
        if self.dp_params.seed is not None:
            torch.manual_seed(self.dp_params.seed)
            torch.cuda.manual_seed(self.dp_params.seed)
        # set print precision
        torch.set_printoptions(precision = 12)

        # anything other than dfeat
        self.loader_train = None
        self.loader_valid = None

        #sparse dfeat from fortran 
        self.dfeat = None 
        self.scaler = None

        self.optimizer = None 
        self.model = None
        self.scheduler = None

        # R cut for feature 
        pm.Rc_M = self.dp_params.descriptor.Rmax
        pm.Rc_min = self.dp_params.descriptor.Rmin

        # # load custom feature 
        # if custom_feat_7 is not None:
        #     pm.Ftype7_para = custom_feat_7.copy()
        # if custom_feat_1 is not None: 
        #     pm.Ftype1_para = custom_feat_1.copy()
        # if custom_feat_2 is not None:
        #     pm.Ftype2_para = custom_feat_2.copy()
            
    def generate_data(self, chunk_size=1, shuffle=False):
        if self.dp_params.inference:
            gen_feature_data = self.dp_params.file_paths.test_dir
            movement_path = self.dp_params.file_paths.test_movement_path
        else:
            gen_feature_data = self.dp_params.file_paths.train_dir
            movement_path = self.dp_params.file_paths.train_movement_path
            
        if os.path.exists(gen_feature_data) is True: # work_dir/feature dir
            shutil.rmtree(gen_feature_data)

        global alive_atomic_energy      # Declare is_real_Ep as a global variable
        command = 'grep Atomic-Energy ' + movement_path[0] + ' | head -n 1'
        print('running-shell-command: ' + command)
        result = subprocess.run(command, stdout=subprocess.PIPE, encoding='utf-8', shell=True)
        if 'Atomic-Energy' in result.stdout:
            alive_atomic_energy = True
        else:
            alive_atomic_energy = False
            if self.dp_params.optimizer_param.train_ei or self.dp_params.optimizer_param.train_egroup:
                raise Exception("Error! Atomic-Energy not found in movement file, please check!")

        # classify all MOVEMENT files based on atom type and atom type nums
        work_dir_list, atom_num = get_cluster_dirs(movement_path)
        # copy the movement files to the working path by cluster result
        sub_work_dir = make_work_dir(gen_feature_data, 
                                    self.dp_params.file_paths.trainSetDir, #'PWdata'
                                    self.dp_params.file_paths.movement_name, #"MOVEMENT"
                                    work_dir_list)
        
        for index, sub_dir in enumerate(sub_work_dir):
            os.chdir(sub_dir)
            # reset pm parameters
            atom_type = work_dir_list[os.path.basename(sub_dir)]['types']
            self._reset_pm_params(sub_dir, atom_type)
            # calculating feature 
            from src.pre_data.mlff import calc_feat
            calc_feat()
            # seperate training set and valid set
            import src.pre_data.seper as seper 
            seper.seperate_data(chunk_size = chunk_size, shuffle = shuffle, atom_num = atom_num, alive_atomic_energy = alive_atomic_energy)
            # save as .npy
            import src.pre_data.gen_data as gen_data
            gen_data.write_data(alive_atomic_energy = alive_atomic_energy)
    
        # copy feat_info and vdw_fitB.ntype
        fread_dfeat_dir = os.path.join(gen_feature_data, "fread_dfeat") #target dir
        if os.path.exists(fread_dfeat_dir):
            shutil.rmtree(fread_dfeat_dir)
        os.makedirs(fread_dfeat_dir)
        copy_file(os.path.join(sub_work_dir[0], "fread_dfeat/feat.info"), os.path.join(fread_dfeat_dir, "feat.info"))
        copy_file(os.path.join(sub_work_dir[0], "fread_dfeat/vdw_fitB.ntype"), os.path.join(fread_dfeat_dir, "vdw_fitB.ntype"))
        # copy input dir 
        input_dir = os.path.join(gen_feature_data, "input") #target dir
        source_input_dir = os.path.join(sub_work_dir[0], "input")
        if os.path.exists(input_dir) is False:
            shutil.copytree(source_input_dir, input_dir)
        os.chdir(self.dp_params.file_paths.work_dir)
        self._reset_pm_params(self.dp_params.file_paths.work_dir, self.atom_type)
        
        return gen_feature_data


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

    def generate_data_single(self, chunk_size = 10, shuffle=False):
        """
            defualt chunk size set to 10 
        """
        # clean old .dat files otherwise will be appended 
        if os.path.exists("PWdata/MOVEMENTall"):
            print ("cleaning old data")
            subprocess.run([
                "rm -rf fread_dfeat output input train_data plot_data PWdata/*.txt PWdata/trainData.* PWdata/location PWdata/Egroup_weight PWdata/MOVEMENTall"
                ],shell=True)
            subprocess.run(["find PWdata/ -name 'dfeat*' | xargs rm -rf"],shell=True)
            subprocess.run(["find PWdata/ -name 'info*' | xargs rm -rf"],shell=True)
        # calculating feature 
        from src.pre_data.mlff import calc_feat
        calc_feat()
        # seperate training set and valid set
        import src.pre_data.seper as seper 
        seper.seperate_data(chunk_size = chunk_size, shuffle=shuffle)
        # save as .npy
        import src.pre_data.gen_data as gen_data
        gen_data.write_data()
        
    def scale_hybrid(self, train_data:MovementHybridDataset, valid_data:MovementHybridDataset, data_num:int):
        feat_train, feat_valid = None, None
        shape_train, shape_valid = [0], [0]
        for data_index in range(data_num):
            feat_train = train_data.feat[data_index] if feat_train is None else np.concatenate((feat_train,train_data.feat[data_index]),0)
            feat_valid = valid_data.feat[data_index] if feat_valid is None else np.concatenate((feat_valid,valid_data.feat[data_index]),0)
            shape_train.append(train_data.feat[data_index].shape[0] + sum(shape_train))
            shape_valid.append(valid_data.feat[data_index].shape[0] + sum(shape_valid))

        # scale feat
        if  self.scaler is not None:
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

        if os.path.exists(self.dp_params.file_paths.model_store_dir) is False:
            os.makedirs(self.dp_params.file_paths.model_store_dir)
        pickle.dump(self.scaler, open(os.path.join(self.dp_params.file_paths.model_store_dir, "scaler.pkl"),'wb'))
        print ("scaler.pkl saved to:",self.dp_params.file_paths.model_store_dir)

        return train_data, valid_data
    
    def scale(self,train_data,valid_data):
        if pm.use_storage_scaler:
            scaler_path = self.dp_params.file_paths.model_store_dir + 'scaler.pkl'
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

    def load_data_hybrid(self, data_shuffle=False, alive_atomic_energy=False):
        # load anything other than dfeat
         
        if self.dp_params.inference:
            feature_paths = self.dp_params.file_paths.test_feature_path
        else:
            feature_paths = self.dp_params.file_paths.train_feature_path
        self.set_nFeature(feature_paths)
        data_list = []
        for feature_path in feature_paths:
            data_dirs = os.listdir(feature_path)
            for data_dir in data_dirs:
                if os.path.exists(os.path.join(feature_path, data_dir, "train_data", "final_train")):
                    data_list.append(os.path.join(feature_path, data_dir, "train_data"))
        # data_dirs = sorted(data_dirs, key=lambda x: len(x.split('_')), reverse = True)
        torch_train_data = get_torch_data_hybrid(data_list, "final_train", alive_atomic_energy, \
                                                 atom_type = pm.atomType, is_dfeat_sparse=pm.is_dfeat_sparse)
        self.energy_shift = torch_train_data.energy_shift

        torch_valid_data = get_torch_data_hybrid(data_list, "final_test", alive_atomic_energy, \
                                                 atom_type = pm.atomType, is_dfeat_sparse=pm.is_dfeat_sparse)
        
        if self.dp_params.inference:
            if os.path.exists(self.dp_params.file_paths.model_load_path):
                self.scaler = load_scaler_from_checkpoint(self.dp_params.file_paths.model_load_path)
            elif os.path.exists(self.dp_params.file_paths.model_save_path):
                self.scaler = load_scaler_from_checkpoint(self.dp_params.file_paths.model_save_path)
            else:
                raise Exception("Error! Load scaler from checkpoint: {}".format(self.dp_params.file_paths.model_load_path))

        torch_train_data, torch_valid_data = self.scale_hybrid(torch_train_data, torch_valid_data, len(data_list))
        
        assert self.scaler != None, "scaler is not correctly saved"

        train_sampler = None
        val_sampler = None

        self.loader_train = torch.utils.data.DataLoader(
            torch_train_data,
            batch_size=self.dp_params.optimizer_param.batch_size,
            shuffle=data_shuffle, #(train_sampler is None)
            num_workers=self.dp_params.workers, 
            pin_memory=True,
            sampler=train_sampler,
        )

        self.loader_valid = torch.utils.data.DataLoader(
            torch_valid_data,
            batch_size=self.dp_params.optimizer_param.batch_size,
            shuffle=False,
            num_workers=self.dp_params.workers, 
            pin_memory=True,
            sampler=val_sampler,
        )
        """
            Note! When using sparse dfeat, shuffle must be False. 
            sparse dfeat class only works under the batch order before calling Data.DataLoader
        """
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

    def set_model_optimizer(self, start_epoch = 1, model_name = None):
        # initialize model 
        pm.itype_Ei_mean = self.energy_shift
        self.model = MLFFNet(device = self.device)
        self.model.to(self.device)

        # initialize optimzer 
        opt_optimizer = self.dp_params.optimizer_param.opt_name
        # if self.use_GKalman or self.use_LKalman or self.use_SKalman:
        if opt_optimizer == "LKF":
            self.optimizer = LKFOptimizer(
                self.model.parameters(),
                self.dp_params.optimizer_param.kalman_lambda,
                self.dp_params.optimizer_param.kalman_nue,
                self.dp_params.optimizer_param.block_size
            )
        elif opt_optimizer == "GKF":
            self.optimizer = GKFOptimizer(
                self.model.parameters(),
                self.dp_params.optimizer_param.kalman_lambda,
                self.dp_params.optimizer_param.kalman_nue
            )
        else: #use torch's built-in optimizer 
            model_parameters = self.model.parameters()
            self.momentum = self.dp_params.optimizer_param.momentum
            self.REGULAR_wd = self.dp_params.optimizer_param.weight_decay
            self.LR_base = self.dp_params.optimizer_param.learning_rate
            self.LR_gamma = self.dp_params.optimizer_param.gamma
            self.LR_step = self.dp_params.optimizer_param.step
            if (opt_optimizer == 'SGD'):
                self.optimizer = optim.SGD(model_parameters, lr=self.LR_base, momentum=self.momentum, weight_decay=self.REGULAR_wd)
            elif (opt_optimizer == 'ASGD'):
                self.optimizer = optim.ASGD(model_parameters, lr=self.LR_base, weight_decay=self.REGULAR_wd)
            elif (opt_optimizer == 'RPROP'):
                self.optimizer = optim.Rprop(model_parameters, lr=self.LR_base, weight_decay = self.REGULAR_wd)
            elif (opt_optimizer == 'RMSPROP'):
                self.optimizer = optim.RMSprop(model_parameters, lr=self.LR_base, weight_decay = self.REGULAR_wd, momentum = self.momentum)
            elif (opt_optimizer == 'ADAG'):
                self.optimizer = optim.Adagrad(model_parameters, lr = self.LR_base, weight_decay = self.REGULAR_wd)
            elif (opt_optimizer == 'ADAD'):
                self.optimizer = optim.Adadelta(model_parameters, lr = self.LR_base, weight_decay = self.REGULAR_wd)
            elif (opt_optimizer == 'ADAM'):
                self.optimizer = optim.Adam(model_parameters, lr = self.LR_base, weight_decay = self.REGULAR_wd)
            elif (opt_optimizer == 'ADAMW'):
                self.optimizer = optim.AdamW(model_parameters, lr = self.LR_base)
            elif (opt_optimizer == 'ADAMAX'):
                self.optimizer = optim.Adamax(model_parameters, lr = self.LR_base, weight_decay = self.REGULAR_wd)
            elif (opt_optimizer == 'LBFGS'):
                self.optimizer = optim.LBFGS(self.model.parameters(), lr = self.LR_base)
            else:
                raise RuntimeError("unsupported optimizer: %s" %opt_optimizer)  
            # set scheduler
            self.set_scheduler() 

        # optionally resume from a checkpoint
        if self.dp_params.recover_train or os.path.exists(self.dp_params.file_paths.model_load_path):
            if self.dp_params.recover_train:    #recover from last training
                model_path = self.dp_params.file_paths.model_save_path  # .../checkpoint.pth.tar
                print("model recover from the checkpoint: {}".format(model_path))
            else: # resume model specified by user
                model_path = self.dp_params.file_paths.model_load_path
                print("model resume from the checkpoint: {}".format(model_path))
            if os.path.isfile(model_path):
                print("=> loading checkpoint '{}'".format(model_path))
                if not torch.cuda.is_available():
                    checkpoint = torch.load(model_path,map_location=torch.device('cpu') )
                elif self.dp_params.gpu is None:
                    checkpoint = torch.load(model_path)
                elif torch.cuda.is_available():
                    # Map model to be loaded to specified single gpu.
                    loc = "cuda:{}".format(self.dp_params.gpu)
                    checkpoint = torch.load(model_path, map_location=loc)
                # start afresh
                self.dp_params.optimizer_param.start_epoch = checkpoint["epoch"] + 1
                self.model.load_state_dict(checkpoint["state_dict"])
                if "optimizer" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                # scheduler.load_state_dict(checkpoint["scheduler"])
                print("=> loaded checkpoint '{}' (epoch {})"\
                      .format(model_path, checkpoint["epoch"]))
            else:
                print("=> no checkpoint found at '{}'".format(model_path))
        print("network initialized")

    def set_scheduler(self):

        # user specific LambdaLR lambda function
        lr_lambda = lambda epoch: self.LR_gamma ** epoch

        opt_scheduler = self.dp_params.optimizer_param.scheduler

        if opt_scheduler == 'LAMBDA':
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = lr_lambda)
        elif opt_scheduler == 'STEP':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = self.LR_step, gamma = self.LR_gamma)
        elif opt_scheduler == 'MSTEP':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = None, gamma = self.LR_gamma)
        elif opt_scheduler == 'EXP':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma= self.LR_gamma)
        elif opt_scheduler is None:
            pass
        else:   
            raise RuntimeError("unsupported scheduler: %s" %opt_scheduler)
    
    def train(self):
        """    
            trianing method for the class 
        """ 
        iter = 0 
        iter_valid = 0 
        smlink_file(self.dp_params.file_paths.model_store_dir, \
                    os.path.join(self.dp_params.file_paths.json_dir, os.path.basename(self.dp_params.file_paths.model_store_dir)))
        # set the log files
        iter_train_log = os.path.join(self.dp_params.file_paths.model_store_dir, "iter_train.dat")
        f_iter_train_log = open(iter_train_log, 'w')
        epoch_train_log = os.path.join(self.dp_params.file_paths.model_store_dir, "epoch_train.dat")
        f_epoch_train_log = open(epoch_train_log, 'w')
        iter_valid_log = os.path.join(self.dp_params.file_paths.model_store_dir, "iter_valid.dat")
        f_iter_valid_log = open(iter_valid_log, 'w')
        epoch_valid_log =  os.path.join(self.dp_params.file_paths.model_store_dir, "epoch_valid.dat")
        f_epoch_valid_log = open(epoch_valid_log, 'w')
        
        # Define the lists based on the training type
        iter_train_lists = ["iter", "loss"]
        iter_valid_lists = ["iter", "loss"]
        epoch_train_lists = ["epoch", "loss"]
        epoch_valid_lists = ["epoch", "loss"]

        if self.dp_params.optimizer_param.train_energy:
            iter_train_lists.append("RMSE_Etot_per_atom")
            epoch_train_lists.append("RMSE_Etot_per_atom")
            iter_valid_lists.append("RMSE_Etot_per_atom")
            epoch_valid_lists.append("RMSE_Etot_per_atom")
        if self.dp_params.optimizer_param.train_ei:
            iter_train_lists.append("RMSE_Ei")
            epoch_train_lists.append("RMSE_Ei")
            iter_valid_lists.append("RMSE_Ei")
            epoch_valid_lists.append("RMSE_Ei")
        if self.dp_params.optimizer_param.train_egroup:
            iter_train_lists.append("RMSE_Egroup")
            epoch_train_lists.append("RMSE_Egroup")
            iter_valid_lists.append("RMSE_Egroup")
            epoch_valid_lists.append("RMSE_Egroup")
        if self.dp_params.optimizer_param.train_force:
            iter_train_lists.append("RMSE_F")
            epoch_train_lists.append("RMSE_F")
            iter_valid_lists.append("RMSE_F")
            epoch_valid_lists.append("RMSE_F")

        if "KF" not in self.dp_params.optimizer_param.opt_name:# adam or sgd optimizer need learning rate
            iter_valid_lists.extend(["lr"])

        print_width = {
            "iter": 5,
            "epoch": 5,
            "loss": 18,
            "RMSE_Etot_per_atom": 21,
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

        for epoch in range(self.dp_params.optimizer_param.start_epoch, self.dp_params.optimizer_param.epochs + 1):
            timeEpochStart = time.time()
            last_epoch = True if epoch == self.dp_params.optimizer_param.epochs else False
            print("<-------------------------  epoch %d  ------------------------->" %(epoch))
            nr_total_sample = 0
            loss = 0.
            loss_Etot = 0.
            loss_Etot_per_atom = 0.
            loss_Ei = 0.
            loss_F = 0.
            loss_Egroup = 0.0 
            # this line code should go out?
            KFOptWrapper = KFOptimizerWrapper(
                self.model, self.optimizer, 
                self.dp_params.optimizer_param.nselect, self.dp_params.optimizer_param.groupsize, 
                self.dp_params.hvd, "hvd"
             )
            
            self.model.train()
            # 重写一下训练这部分
            for i_batch, sample_batches in enumerate(self.loader_train):
                
                nr_batch_sample = sample_batches['input_feat'].shape[0]
                if "KF" not in self.dp_params.optimizer_param.opt_name:
                    global_step = (epoch - 1) * len(self.loader_train) + i_batch * nr_batch_sample
                    real_lr = self.adjust_lr(iter_num = global_step)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = real_lr * (self.dp_params.optimizer_param.batch_size ** 0.5)
                natoms_sum = sample_batches['natoms_img'][0, 0].item()  
                # use sparse feature 
                if pm.is_dfeat_sparse == True:
                    # Error this function not realized
                    #sample_batches['input_dfeat']  = dfeat_train.transform(i_batch)
                    sample_batches['input_dfeat']  = self.dfeat.transform(i_batch,"train")

                if "KF" not in self.dp_params.optimizer_param.opt_name:
                    # non-KF t
                    batch_loss, batch_loss_Etot, batch_loss_Ei, batch_loss_F = \
                        self.train_img(sample_batches, self.model, self.optimizer, nn.MSELoss(), last_epoch, real_lr)
                else:
                    # KF 
                    batch_loss, batch_loss_Etot, batch_loss_Ei, batch_loss_F, batch_loss_Egroup = \
                        self.train_kalman_img(sample_batches, self.model, KFOptWrapper, nn.MSELoss())
                                
                iter += 1

                f_iter_train_log = open(iter_train_log, 'a')

                # Write the training log line to the log file
                iter_train_log_line = "%5d%18.10e" % (iter, batch_loss)

                if self.dp_params.optimizer_param.train_energy:
                    iter_train_log_line += "%18.10e" % (math.sqrt(batch_loss_Etot)/natoms_sum)
                if self.dp_params.optimizer_param.train_ei:
                    iter_train_log_line += "%18.10e" % (math.sqrt(batch_loss_Ei))
                if self.dp_params.optimizer_param.train_egroup:
                    iter_train_log_line += "%18.10e" % (math.sqrt(batch_loss_Egroup))
                if self.dp_params.optimizer_param.train_force:
                    iter_train_log_line += "%18.10e" % ( math.sqrt(batch_loss_F))
                if "KF" not in self.dp_params.optimizer_param.opt_name:
                    iter_train_log_line += "%18.10e" % (real_lr)

                f_iter_train_log.write("%s\n" % (iter_train_log_line))
                f_iter_train_log.close()
                
                loss += batch_loss.item() * nr_batch_sample

                loss_Etot += batch_loss_Etot.item() * nr_batch_sample
                loss_Etot_per_atom += math.sqrt(batch_loss_Etot)/natoms_sum * nr_batch_sample

                loss_Ei += batch_loss_Ei.item() * nr_batch_sample
                loss_F += batch_loss_F.item() * nr_batch_sample
                loss_Egroup += batch_loss_Egroup.item() * nr_batch_sample 

                nr_total_sample += nr_batch_sample
                 
            loss /= nr_total_sample
            loss_Etot /= nr_total_sample
            loss_Etot_per_atom /= nr_total_sample
            loss_Ei /= nr_total_sample
            loss_F /= nr_total_sample
            loss_Egroup /= nr_total_sample  

            RMSE_Etot = loss_Etot ** 0.5
            RMSE_Ei = loss_Ei ** 0.5
            RMSE_F = loss_F ** 0.5
            RMSE_Egroup = loss_Egroup ** 0.5 

            print("epoch_loss = %.10f (RMSE_Etot_per_atom = %.12f, RMSE_Ei = %.12f, RMSE_F = %.12f, RMSE_Eg = %.12f)" \
                %(loss, loss_Etot_per_atom, RMSE_Ei, RMSE_F, RMSE_Egroup))

            f_epoch_train_log = open(epoch_train_log, 'a')

            # Write the training log line to the log file
            epoch_train_log_line = "%5d%18.10e" % (epoch, loss,)

            if self.dp_params.optimizer_param.train_energy:
                epoch_train_log_line += "%18.10e" % (loss_Etot_per_atom)
            if self.dp_params.optimizer_param.train_ei:
                epoch_train_log_line += "%18.10e" % (RMSE_Ei)
            if self.dp_params.optimizer_param.train_egroup:
                epoch_train_log_line += "%18.10e" % (RMSE_Egroup)
            if self.dp_params.optimizer_param.train_force:
                epoch_train_log_line += "%18.10e" % (RMSE_F)
            
            f_epoch_train_log.write("%s\n" % (epoch_train_log_line))
            f_epoch_train_log.close()
            
            if "KF" not in self.dp_params.optimizer_param.opt_name:
                """
                    for built-in optimizer only 
                """
                opt_scheduler = self.dp_params.optimizer_param.scheduler

                if (opt_scheduler == 'OC'):
                    pass 
                elif (opt_scheduler == 'PLAT'):
                    self.scheduler.step(loss)

                elif (opt_scheduler == 'LR_INC'):
                    self.LinearLR(optimizer=self.optimizer, base_lr=self.LR_base, target_lr=pm.opt_LR_max_lr, total_epoch=self.dp_params.optimizer_param.epochs, cur_epoch=epoch)

                elif (opt_scheduler == 'LR_DEC'):
                    self.LinearLR(optimizer=self.optimizer, base_lr=self.LR_base, target_lr=pm.opt_LR_min_lr, total_epoch=self.dp_params.optimizer_param.epochs, cur_epoch=epoch)

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
            valid_loss_Etot_pre_atom = 0.
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

                valid_loss += valid_error_iter * nr_batch_sample

                valid_loss_Etot += batch_loss_Etot * nr_batch_sample
                valid_loss_Etot_pre_atom += math.sqrt(batch_loss_Etot)/natoms_sum  * nr_batch_sample
                valid_loss_Ei += batch_loss_Ei * nr_batch_sample
                valid_loss_F += batch_loss_F * nr_batch_sample
                valid_loss_Egroup += batch_loss_Egroup * nr_batch_sample
                
                nr_total_sample += nr_batch_sample
                f_iter_valid_log = open(iter_valid_log, 'a')

                # Write the valid log line to the log file
                iter_valid_log_line = "%5d%18.10e" % (iter, batch_loss,)

                if self.dp_params.optimizer_param.train_energy:
                    iter_valid_log_line += "%18.10e" % (math.sqrt(batch_loss_Etot)/natoms_sum)
                if self.dp_params.optimizer_param.train_ei:
                    iter_valid_log_line += "%18.10e" % (math.sqrt(batch_loss_Ei))
                if self.dp_params.optimizer_param.train_egroup:
                    iter_valid_log_line += "%18.10e" % (math.sqrt(batch_loss_Egroup))
                if self.dp_params.optimizer_param.train_force:
                    iter_valid_log_line += "%18.10e" % (math.sqrt(batch_loss_F))
                if "KF" not in self.dp_params.optimizer_param.opt_name:
                    iter_train_log_line += "%18.10e" % (real_lr)

                f_iter_valid_log.write("%s\n" % (iter_valid_log_line))
                f_iter_valid_log.close()

            # epoch loss update
            valid_loss /= nr_total_sample
            valid_loss_Etot /= nr_total_sample
            valid_loss_Etot_pre_atom /= nr_total_sample
            valid_loss_Ei /= nr_total_sample
            valid_loss_F /= nr_total_sample
            valid_loss_Egroup /= nr_total_sample

            valid_RMSE_Etot = valid_loss_Etot ** 0.5
            valid_RMSE_Ei = valid_loss_Ei ** 0.5
            valid_RMSE_F = valid_loss_F ** 0.5
            valid_RMSE_Egroup = valid_loss_Egroup ** 0.5
                
            print("valid_loss = %.10f (valid_RMSE_Etot_pre_atom = %.12f, valid_RMSE_Ei = %.12f, valid_RMSE_F = %.12f, valid_RMSE_Egroup = %.12f)" \
                     %(valid_loss, valid_loss_Etot_pre_atom, valid_RMSE_Ei, valid_RMSE_F, valid_RMSE_Egroup))
   
            f_epoch_valid_log = open(epoch_valid_log, 'a')

            # Write the valid log line to the log file
            epoch_valid_log_line = "%5d%18.10e" % (epoch, valid_loss)

            if self.dp_params.optimizer_param.train_energy:
                epoch_valid_log_line += "%18.10e" % (valid_loss_Etot_pre_atom)
            if self.dp_params.optimizer_param.train_ei:
                epoch_valid_log_line += "%18.10e" % (valid_RMSE_Ei)
            if self.dp_params.optimizer_param.train_egroup:
                epoch_valid_log_line += "%18.10e" % (valid_RMSE_Egroup)
            if self.dp_params.optimizer_param.train_force:
                epoch_valid_log_line += "%18.10e" % (valid_RMSE_F)

            f_epoch_valid_log.write("%s\n" % (epoch_valid_log_line))
            f_epoch_valid_log.close()

            # save model
            if not self.dp_params.hvd or (self.dp_params.hvd and hvd.rank() == 0):
                if self.dp_params.file_paths.save_p_matrix:
                    self.save_checkpoint(
                        {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "optimizer":self.optimizer.state_dict(),
                        "scaler": self.scaler
                        },
                        self.dp_params.file_paths.model_name,
                        self.dp_params.file_paths.model_store_dir,
                    )
                else: 
                    self.save_checkpoint(
                        {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "scaler": self.scaler
                        },
                        self.dp_params.file_paths.model_name,
                        self.dp_params.file_paths.model_store_dir,
                    )
                
            timeEpochEnd = time.time()
            print("time of epoch %d: %f s" %(epoch, timeEpochEnd - timeEpochStart))

    def save_checkpoint(self, state, filename, prefix):
        filename = os.path.join(prefix, filename)
        torch.save(state, filename)

    def load_and_train(self):
        # transform data
        self.load_data_hybrid(data_shuffle=self.dp_params.data_shuffle, alive_atomic_energy=alive_atomic_energy)
        # else:
        #     self.load_data()
        # initialize the network
        self.set_model_optimizer()
        # initialize the optimizer and related scheduler
        if self.dp_params.inference:
            self.do_inference()
        else:
            self.train()

    def train_img(self, sample_batches, model, optimizer, criterion, last_epoch, real_lr):
        # single image traing for non-Kalman 
        if (self.dp_params.precision == 'float64'):
            Ei_label = Variable(sample_batches['output_energy'][:,:,:].double().to(self.device))
            Force_label = Variable(sample_batches['output_force'][:,:,:].double().to(self.device))   #[40,108,3]
            Egroup_label = Variable(sample_batches['input_egroup'].double().to(self.device))
            input_data = Variable(sample_batches['input_feat'].double().to(self.device), requires_grad=True)
            dfeat = Variable(sample_batches['input_dfeat'].double().to(self.device))  #[40,108,100,42,3]
            egroup_weight = Variable(sample_batches['input_egroup_weight'].double().to(self.device))
            divider = Variable(sample_batches['input_divider'].double().to(self.device))
            # Ep_label = Variable(sample_batches['output_ep'][:,:,:].double().to(device))
        elif (self.dp_params.precision == 'float32'):
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
            raise RuntimeError("train(): unsupported opt_dtype %s" %self.dp_params.precision)  

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
        if self.dp_params.optimizer_param.train_ei:
            loss_Ei = criterion(Ei_predict, Ei_label)
        if self.dp_params.optimizer_param.train_energy:
            loss_Etot = criterion(Etot_predict, Etot_label)
        if self.dp_params.optimizer_param.train_force:
            loss_F = criterion(Force_predict, Force_label)

        start_lr = self.dp_params.optimizer_param.learning_rate
        
        w_f = 1 if self.dp_params.optimizer_param.train_force == True else 0
        w_e = 1 if self.dp_params.optimizer_param.train_energy == True else 0
        w_ei = 1 if self.dp_params.optimizer_param.train_ei == True else 0
        w_eg = 0 

        loss, pref_f, pref_e = self.get_loss_func(start_lr, real_lr, w_f, loss_F, w_e, loss_Etot, w_eg, loss_Egroup, w_ei, loss_Ei, natoms_img[0, 0].item())

        # using a total loss to update weights 
        loss.backward()

        self.optimizer.step()
        
        return loss, loss_Etot, loss_Ei, loss_F

    def train_kalman_img(self,sample_batches, model, KFOptWrapper :KFOptimizerWrapper, criterion):
        """
            why setting precision again? 
        """
        Ei_label = Variable(sample_batches['output_energy'][:,:,:].to(self.device))
        Force_label = Variable(sample_batches['output_force'][:,:,:].to(self.device))   #[40,108,3]
        input_data = Variable(sample_batches['input_feat'].to(self.device), requires_grad=True)

        dfeat = Variable(sample_batches['input_dfeat'].to(self.device))  #[40,108,100,42,3]
        
        if alive_atomic_energy:
            Egroup_label = Variable(sample_batches['input_egroup'].to(self.device))
            egroup_weight = Variable(sample_batches['input_egroup_weight'].to(self.device))
            divider = Variable(sample_batches['input_divider'].to(self.device))

        #atom_number = Ei_label.shape[1]
        Etot_label = torch.sum(Ei_label, dim=1)
        neighbor = Variable(sample_batches['input_nblist'].int().to(self.device))  # [40,108,100]
        #ind_img = Variable(sample_batches['ind_image'].int().to(self.device))
        natoms_img = Variable(sample_batches['natoms_img'].int().to(self.device))
        atom_type = Variable(sample_batches['atom_type'].int().to(self.device))

        if self.dp_params.optimizer_param.train_egroup:
            kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, egroup_weight, divider]
        else:
            kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, None, None]

        if self.dp_params.optimizer_param.train_energy: 
            # kalman.update_energy(kalman_inputs, Etot_label, update_prefactor = self.dp_params.optimizer_param.pre_fac_etot)
            Etot_predict = KFOptWrapper.update_energy(kalman_inputs, Etot_label, self.dp_params.optimizer_param.pre_fac_etot, train_type = "NN")
            
        if self.dp_params.optimizer_param.train_ei:
            Ei_predict = KFOptWrapper.update_ei(kalman_inputs,Ei_label, update_prefactor = self.dp_params.optimizer_param.pre_fac_ei, train_type = "NN")     

        if self.dp_params.optimizer_param.train_egroup:
            # kalman.update_egroup(kalman_inputs, Egroup_label)
            Egroup_predict = KFOptWrapper.update_egroup(kalman_inputs, Egroup_label, self.dp_params.optimizer_param.pre_fac_egroup, train_type = "NN")

        # if Egroup does not participate in training, the output of Egroup_predict will be None
        if self.dp_params.optimizer_param.train_force:
            Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = KFOptWrapper.update_force(
                    kalman_inputs, Force_label, self.dp_params.optimizer_param.pre_fac_force, train_type = "NN")

        Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(kalman_inputs[0], kalman_inputs[1], kalman_inputs[2], kalman_inputs[3], kalman_inputs[4], kalman_inputs[5], kalman_inputs[6])

        # dtype same as torch.default
        loss_Etot = torch.zeros([1,1],device = self.device)
        loss_Ei = torch.zeros([1,1], device = self.device)
        loss_F = torch.zeros([1,1],device = self.device)
        loss_egroup = torch.zeros([1,1],device = self.device)

        # update loss only for used labels  
        # At least 2 flags should be true. 
        if self.dp_params.optimizer_param.train_ei:
            loss_Ei = criterion(Ei_predict, Ei_label)
        
        if self.dp_params.optimizer_param.train_energy:
            loss_Etot = criterion(Etot_predict, Etot_label)

        if self.dp_params.optimizer_param.train_force:
            loss_F = criterion(Force_predict, Force_label)

        if self.dp_params.optimizer_param.train_egroup:
            loss_egroup = criterion(Egroup_label,Egroup_predict)

        loss = loss_F + loss_Etot + loss_Ei + loss_egroup 
        
        print("RMSE_Etot_per_atom = %.12f, RMSE_Ei = %.12f, RMSE_Force = %.12f, RMSE_Egroup = %.12f" %(loss_Etot ** 0.5 / natoms_img[0, 0].item() , loss_Ei ** 0.5, loss_F ** 0.5, loss_egroup**0.5))
        
        del Ei_label
        del Force_label
        del input_data
        del dfeat
        if alive_atomic_energy:
            del Egroup_label
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
        input_data = Variable(sample_batches['input_feat'].to(self.device), requires_grad=True)
        dfeat = Variable(sample_batches['input_dfeat'].to(self.device))  #[40,108,100,42,3]
        if alive_atomic_energy:
            Egroup_label = Variable(sample_batches['input_egroup'].to(self.device))
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
        if self.dp_params.optimizer_param.train_egroup:
            kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, egroup_weight, divider]
        else:
            kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, None, None]

        Etot_predict, Ei_predict, Force_predict, Egroup_predict, _ = model(kalman_inputs[0], kalman_inputs[1], kalman_inputs[2], kalman_inputs[3], kalman_inputs[4], kalman_inputs[5], kalman_inputs[6])
        
        loss_Etot = torch.zeros([1,1],device=self.device)
        loss_Ei = torch.zeros([1,1],device=self.device)
        loss_F = torch.zeros([1,1],device = self.device)
        loss_egroup = torch.zeros([1,1],device = self.device)
        
        # update loss with repsect to the data used
        if self.dp_params.optimizer_param.train_ei:
            loss_Ei = criterion(Ei_predict, Ei_label)
        
        if self.dp_params.optimizer_param.train_energy:
            loss_Etot = criterion(Etot_predict, Etot_label)

        if self.dp_params.optimizer_param.train_force:
            loss_F = criterion(Force_predict, Force_label)

        if self.dp_params.optimizer_param.train_egroup:
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

        train_lists = ["img_idx", "RMSE_Etot", "RMSE_Etot_per_atom", "RMSE_Ei", "RMSE_F"]
        if self.dp_params.optimizer_param.train_egroup:
            train_lists.append("RMSE_Egroup")
        res_pd = pd.DataFrame(columns=train_lists)

        inf_dir = self.dp_params.file_paths.test_dir
        if os.path.exists(inf_dir) is True:
            shutil.rmtree(inf_dir)
        os.mkdir(inf_dir)
        res_pd_save_path = os.path.join(inf_dir, "inference_loss.csv")
        inf_force_save_path = os.path.join(inf_dir,"inference_force.txt")
        lab_force_save_path = os.path.join(inf_dir,"dft_force.txt")
        inf_energy_save_path = os.path.join(inf_dir,"inference_total_energy.txt")
        lab_energy_save_path = os.path.join(inf_dir,"dft_total_energy.txt")
        if alive_atomic_energy:
            inf_Ei_save_path = os.path.join(inf_dir,"inference_atomic_energy.txt")
            lab_Ei_save_path = os.path.join(inf_dir,"dft_atomic_energy.txt")
        inference_path = os.path.join(inf_dir,"inference_summary.txt") 

        for i_batch, sample_batches in enumerate(self.loader_train):
            if pm.is_dfeat_sparse == True:
                sample_batches['input_dfeat']  = self.dfeat.transform(i_batch,"valid")
            if sample_batches['input_dfeat'] == "aborted":
                continue 

            Ei_label = Variable(sample_batches['output_energy'][:,:,:].to(self.device))
            Force_label = Variable(sample_batches['output_force'][:,:,:].to(self.device))   #[40,108,3]
            input_data = Variable(sample_batches['input_feat'].to(self.device), requires_grad=True)
            dfeat = Variable(sample_batches['input_dfeat'].to(self.device))  #[40,108,100,42,3]
            if alive_atomic_energy:
                Egroup_label = Variable(sample_batches['input_egroup'].to(self.device))
                egroup_weight = Variable(sample_batches['input_egroup_weight'].to(self.device))
                divider = Variable(sample_batches['input_divider'].to(self.device))
            
            neighbor = Variable(sample_batches['input_nblist'].int().to(self.device))  # [40,108,100]
            natoms_img = Variable(sample_batches['natoms_img'].int().to(self.device))  # [40,108,100]
            atom_type = Variable(sample_batches['atom_type'].int().to(self.device))
            Etot_label = torch.sum(Ei_label, dim=1)
            
            self.model.eval()
            if self.dp_params.optimizer_param.train_egroup:
                kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, egroup_weight, divider]
            else:
                kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, None, None]

            Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = self.model(kalman_inputs[0], kalman_inputs[1], kalman_inputs[2], kalman_inputs[3], kalman_inputs[4], kalman_inputs[5], kalman_inputs[6])
            
            # mse
            criterion = nn.MSELoss()
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_F_val = criterion(Force_predict, Force_label)
            loss_Ei_val = criterion(Ei_predict, Ei_label)
            if self.dp_params.optimizer_param.train_egroup is True:
                loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
            # rmse
            Etot_rmse = loss_Etot_val ** 0.5
            etot_atom_rmse = Etot_rmse / natoms_img[0][0]
            Ei_rmse = loss_Ei_val ** 0.5
            F_rmse = loss_F_val ** 0.5

            res_list = [i_batch, float(Etot_rmse), float(etot_atom_rmse), float(Ei_rmse), float(F_rmse)]
            if self.dp_params.optimizer_param.train_egroup:
                res_list.append(float(loss_Egroup_val))
            res_pd.loc[res_pd.shape[0]] = res_list
           
            #''.join(map(str, list(np.array(Force_predict.flatten().cpu().data))))
            ''.join(map(str, list(np.array(Force_predict.flatten().cpu().data))))
            write_line_to_file(inf_force_save_path, \
                               ' '.join(np.array(Force_predict.flatten().cpu().data).astype('str')), "a")
            write_line_to_file(lab_force_save_path, \
                               ' '.join(np.array(Force_label.flatten().cpu().data).astype('str')), "a")
            if alive_atomic_energy:
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
        if alive_atomic_energy:
            inference_cout += "Avarage REMSE of Ei: {} \n".format(res_pd['RMSE_Ei'].mean())
        inference_cout += "Avarage REMSE of RMSE_F: {} \n".format(res_pd['RMSE_F'].mean())
        if self.dp_params.optimizer_param.train_egroup:
            inference_cout += "Avarage REMSE of RMSE_Egroup: {} \n".format(res_pd['RMSE_Egroup'].mean())
        # if self.dp_params.optimizer_param.train_virial:  #not realized
        #     inference_cout += "Avarage REMSE of RMSE_virial: {} \n".format(res_pd['RMSE_virial'].mean())
        #     inference_cout += "Avarage REMSE of RMSE_virial_per_atom: {} \n".format(res_pd['RMSE_virial_per_atom'].mean())

        inference_cout += "\nMore details can be found under the file directory:\n{}\n".format(inf_dir)
        print(inference_cout)

        if alive_atomic_energy:
            if self.dp_params.optimizer_param.train_ei or self.dp_params.optimizer_param.train_egroup:
                plot_ei = True
            else:
                plot_ei = False
        else:
            plot_ei = False
            
        plot(inf_dir, plot_ei = plot_ei)

        with open(inference_path, 'w') as wf:
            wf.writelines(inference_cout)
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
    
    def set_nFeature(self, feature_path):
        # obtain number of feature from fread_dfeat/feat.info
        if self.dp_params.inference:
            feat_info = os.path.join(feature_path[0], "fread_dfeat/feat.info")
        else:
            feat_info = os.path.join(feature_path[0], "fread_dfeat/feat.info")
        f = open(feat_info,'r')
        raw = f.readlines()[-1].split()
        pm.nFeatures = sum([int(item) for item in raw])
        print("number of features:",pm.nFeatures)
    
    # calculate loss 
    def get_loss_func(self,start_lr, real_lr, has_fi, lossFi, has_etot, loss_Etot, has_egroup, loss_Egroup, has_ei, loss_Ei, natoms_sum):
        start_pref_egroup =self.dp_params.optimizer_param.start_pre_fac_egroup  # 0.02
        start_pref_F =self.dp_params.optimizer_param.start_pre_fac_force  # 1000  #1000
        start_pref_etot =self.dp_params.optimizer_param.start_pre_fac_etot # 0.02   
        start_pref_ei =self.dp_params.optimizer_param.start_pre_fac_ei # 0.02

        limit_pref_egroup =self.dp_params.optimizer_param.end_pre_fac_egroup  # 1.0
        limit_pref_F =self.dp_params.optimizer_param.end_pre_fac_force # 1.0
        limit_pref_etot =self.dp_params.optimizer_param.end_pre_fac_etot # 1.0
        limit_pref_ei =self.dp_params.optimizer_param.end_pre_fac_ei # 1.0

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
    def adjust_lr(self,iter_num):
        stop_lr= self.dp_params.optimizer_param.stop_lr #3.51e-8
        start_lr = self.dp_params.optimizer_param.learning_rate 
        stop_step = self.dp_params.optimizer_param.stop_step # 1000000
        decay_step= self.dp_params.optimizer_param.decay_step # 5000
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
        if self.dp_params.optimizer_param.train_ei or self.dp_params.optimizer_param.train_egroup:
            plot_ei = True
        else:
            plot_ei = False
        plot_evaluation.plot_new(atom_type = pm.atomType, plot_elem = plot_elem, save_data = save_data, plot_ei = plot_ei)

