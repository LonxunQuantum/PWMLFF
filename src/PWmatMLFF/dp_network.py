"""
    module for Deep Potential Network 

    L. Wang, 2022.8
"""
from ast import For, If, NodeTransformer
import os,sys
import pathlib
from xml.sax.handler import feature_external_ges

from pre_data.default_para import Rc

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

from statistics import mode 
from turtle import Turtle, update
import torch

import random
import time
import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt
import torch.nn as nn
import math

import torch.nn.functional as F
from torch.nn.modules import loss
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from loss.AutomaticWeightedLoss import AutomaticWeightedLoss
import torch.utils.data as Data
from torch.autograd import Variable

import time
import getopt

from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from sklearn.feature_selection import VarianceThreshold

# src/aux 
from opts import opt_values 
from feat_modifier import feat_modifier

import pickle
import logging

"""
    customized modules 
"""
from model.dp import DP
from optimizer.kalmanfilter_dp import GKalmanFilter, LKalmanFilter, SKalmanFilter, L1KalmanFilter, L2KalmanFilter

import default_para as pm

from data_loader_2type_dp import MovementDataset, get_torch_data
from scalers import DataScalers
from utils import get_weight_grad
from dfeat_sparse import dfeat_raw 

class dp_network:
    
    def __init__(   self,
                    # some must-haves
                    # model related argument
                    atom_type = None, 
                    feature_type = [1], # only 1 can be used. 
                    # data related arguments    
                    device = "cpu", 
                    # optimizer related arguments 
                    
                    max_neigh_num = 100, 
                    # "g", "l_0","l_1", "l_2", "s" 
                    optimizer = None,
                    kalman_type = "l_1",     # optimal version    
                    
                    session_dir = "record",  # directory that saves the model
                    n_epoch = 25, 
                    batch_size = 4, 

                    # paras for l-kalman 
                    select_num = 24,
                    block_size = 5120,
                    group_size = 6,

                    # training label related arguments
                    is_trainForce = True, 
                    is_trainEi = False,
                    is_trainEgroup = False,
                    is_trainEtot = True,

                    is_movement_weighted = False,
                    
                    # inital values for network config
                    
                    embedding_net_config = None,
                    fitting_net_config = None, 

                    recover = False,
                    
                    # smooth function calculation 
                    Rmin = None,
                    Rmax = None
                    ):
        
        # parsing command line args 
        self.opts = opt_values()  
        
        # feature para modifier
        self.feat_mod = feat_modifier 
        
        if Rmin is not None:
            pm.Rm = Rmin

        if Rmax is not None:
            pm.Rc = Rmax
        # scaling
        # recover training. Need to load both scaler and model 
        pm.use_storage_scaler = recover  
        self.opts.opt_recover_mode = recover

        pm.maxNeighborNum = max_neigh_num 
        
        if atom_type == None:
            raise Exception("atom types not specifed")         
        
        pm.atomType = atom_type 

        # these two variables are the same thing. Crazy. 
        pm.atomTypeNum = len(atom_type)
        pm.ntypes = len(atom_type)

        pm.use_Ftype = feature_type 
        pm.dR_neigh = True
        
        # setting network config 
        self.network_config = None 

        # setting kalman 
        self.kalman_type = kalman_type
            
        if kalman_type is None:
            self.network_config = pm.DP_cfg_dp 
        else:
            self.network_config = pm.DP_cfg_dp_kf               

        # passed-in configs 
        if embedding_net_config is not None:
            print("overwritting embedding network config ",self.network_config['embeding_net']["network_size"], " with ", embedding_net_config) 
            self.network_config['embeding_net']["network_size"] = embedding_net_config

        if fitting_net_config is not None: 
            print("overwritting fitting network config ",self.network_config['fitting_net']["network_size"], " with ", fitting_net_config) 
            self.network_config['fitting_net']["network_size"] = fitting_net_config

        print ("network config used for training:")
        print (self.network_config)
        
        # label to be trained 
        self.is_trainForce = is_trainForce
        self.is_trainEi = is_trainEi
        self.is_trainEgroup = is_trainEgroup
        self.is_trainEtot = is_trainEtot
        
        # prefactor in kf 
        self.kf_prefac_Etot = 1.0  
        self.kf_prefac_Ei = 1.0
        self.kf_prefac_F  = 1.0
        self.kf_prefac_Egroup  = 1.0

        self.kalman_lambda = 0.98                    
        self.kalman_nue =  0.99870
        
        # for layerwise kalman 
        self.select_num = select_num
        self.block_size = block_size
        self.group_size = group_size

        if (self.opts.opt_dtype == 'float64'):
            #print("Training: set default dtype to float64")
            torch.set_default_dtype(torch.float64)
        elif (self.opts.opt_dtype == 'float32'):
            #print("Training: set default dtype to float32")
            torch.set_default_dtype(torch.float32)
        else:
            self.opts.error("Training: unsupported dtype: %s" %self.opts.opt_dtype)
            raise RuntimeError("Training: unsupported dtype: %s" %self.opts.opt_dtype)

        # setting device
        self.device = None
        if device == "cpu":
            self.device = torch.device('cpu')
        elif device == "cuda":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            raise Exception("device type not supported")

        # set random seed
        torch.manual_seed(self.opts.opt_rseed)
        torch.cuda.manual_seed(self.opts.opt_rseed)
        
        # set print precision
        torch.set_printoptions(precision = 12)

        self.patience = 100000
        
        # common training parameters
        self.n_epoch = n_epoch
        
        # batch size = 1 for kalman 
        if kalman_type is not None:
            self.batch_size = 1
        else:
            self.batch_size = batch_size 
            # optimizer other than ADAM
            if optimizer is not None:
                self.opts.opt_optimizer = optimizer 

        self.min_loss = np.inf
        self.epoch_print = 1 
        self.iter_print = 1 
        
        # set session directory 
        self.set_session_dir(session_dir)

        """
            for torch built-in optimizers and schedulers 
        """
        self.momentum = self.opts.opt_momentum
        self.REGULAR_wd = self.opts.opt_regular_wd
        self.LR_base = self.opts.opt_lr
        self.LR_gamma = self.opts.opt_gamma
        self.LR_step = self.opts.opt_step
        
        """
            training data. 
            Below are placeholders
        """ 
        # anything other than dfeat
        self.loader_train = None
        self.loader_valid = None
        self.stat = None
        
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
        self.scheduler = None
        self.model = None

        # starting epoch 
        self.start_epoch = 1

        # path for loading previous model 
        self.load_model_path = self.opts.opt_model_dir+'latest.pt' 

        self.momentum = self.opts.opt_momentum
        self.REGULAR_wd = self.opts.opt_regular_wd
        self.n_epoch = self.opts.opt_epochs
        self.LR_base = self.opts.opt_lr
        self.LR_gamma = self.opts.opt_gamma
        self.LR_step = self.opts.opt_step
        
        print ("device:",self.device)
        
        

    """
        ============================================================
        =================data preparation functions=================
        ============================================================ 
    """

    def generate_data(self):
        
        """
            Calculate features.
            ONLY feat #1 for DP 
        """
        import calc_feat_dp 
        import seper  
        import gen_dpdata
        
        # calc
        calc_feat_dp.calc_feat() 

        # seperate training and validation set
        seper.main()
        
        # save to ./train_data  
        gen_dpdata.main()
        
        return 
        
    """ 
        ============================================================
        ===================auxiliary functions======================
        ============================================================ 
    """
    def not_KF(self):
        if self.kalman_type is None:
            return True
        else:
            return False

    def set_epoch_num(self, input_num):
        self.n_epoch = input_num    
    
    def set_batch_size(self,val):
        self.batch_size = val

    def set_device(self,val):
        
        if val == "cpu":
            self.device = torch.device('cpu')
        elif val == "cuda":
            self.device = torch.device('cuda')
        else:
            raise Exception("unrecognizable device")
        
    def set_session_dir(self,working_dir):
        
        print("models and other information will be saved in:",'/'+working_dir)
        self.opts.opt_session_name = working_dir
        self.opts.opt_session_dir = './'+self.opts.opt_session_name+'/'
        self.opts.opt_logging_file = self.opts.opt_session_dir+'train.log'
        self.opts.opt_model_dir = self.opts.opt_session_dir+'model/'

        #tensorboard_base_dir = self.opt_session_dir+'tensorboard/'
        if not os.path.exists(self.opts.opt_session_dir):
            os.makedirs(self.opts.opt_session_dir) 
        if not os.path.exists(self.opts.opt_model_dir):
            os.makedirs(self.opts.opt_model_dir)

    # for layerwsie KF 
    def set_select_num(self,val):
        self.select_num = val 
    
    def set_block_size(self,val):
        self.block_size = val 

    def set_group_size(self,val):
        self.group_size = val 

    # set prefactor 
    def set_kf_prefac_Etot(self,val):
        self.kf_prefac_Etot = val

    """
    def set_kf_prefac_Ei(self,val):
        self.kf_prefac_Ei = val 
    """

    def set_kf_prefac_F(self,val):
        self.kf_prefac_F = val 
    """
    def set_kf_prefac_Egroup(self,val):
        self.kf_prefac_Egroup = val 
    """
    
    # set label to be trained 

    """
    def set_train_force(self,val):
        self.is_trainForce = val 

    def set_train_Ei(self,val):
        self.is_trainEi = val

    def set_train_Etot(self,val):
        self.is_trainEtot = val 

    def set_train_Egroup(self,val):
        self.is_trainEgroup = val  
    """
    def set_fitting_net_config(self,val):
        self.network_config['fitting_net']["network_size"] = val

    def set_embedding_net_config(self,val):
        self.network_config['embeding_net']["network_size"] = val
    
    def set_Rmin(self,val):
        pm.Rm = val 

    def set_Rmax(self,val):
        pm.Rc = val
    

    def print_feat_para(self):
        # print feature parameter 
        for feat_idx in pm.use_Ftype:    
            name  = "Ftype"+str(feat_idx)+"_para"   
            print(name,":")
            print(getattr(pm,name)) 
    
    
    
    """
        ============================================================
        =================training related functions=================
        ============================================================ 
    """ 
    
    def load_data(self):
        
        image_num_stat = 1

        train_data_path = pm.train_data_path
        torch_train_data = get_torch_data(train_data_path)
        
        davg, dstd, ener_shift = torch_train_data.get_stat(image_num=image_num_stat)
        self.stat = [davg, dstd, ener_shift]
        
        if os.path.exists("davg.npy") or os.path.exists("dstd.npy"):
            print("sclaer files already exsit and will be used in the following calculations")

        else:
            print("creating new scaler files: davg.npy, dstd.npy")
            np.save("./davg.npy", davg)
            np.save("./dstd.npy", dstd)
        
        valid_data_path = pm.test_data_path
        torch_valid_data = get_torch_data(valid_data_path, False)

        self.loader_train = Data.DataLoader(torch_train_data, batch_size=self.batch_size, shuffle=False)
        self.loader_valid = Data.DataLoader(torch_valid_data, batch_size=self.batch_size, shuffle=False)
        
        #self.loader_valid = Data.DataLoader(torch_train_data, batch_size=self.batch_size, shuffle=False)
        
        print("data loaded")

    def set_model(self,start_epoch = 1, model_name = None):
        """
            specify network_name for loading model 
        """
        self.model = DP(self.network_config, self.opts.opt_act, self.device, self.stat, self.opts.opt_magic)
        
        self.model.to(self.device)



        # load previous model if needed 
        if (self.opts.opt_recover_mode == True): 
            
            if (self.opts.opt_session_name == ''):
                raise RuntimeError("session not specified for the recover mode. Use set_session_dir")

            if model_name is None:
                # use lattest.pt as default 
                load_model_path = self.opts.opt_model_dir+'latest.pt' 
            else:
                load_model_path = self.opts.opt_model_dir + model_name

            print ("load network from:",load_model_path)
            
            checkpoint = torch.load(load_model_path, map_location = self.device)

            self.model.load_state_dict(checkpoint['model'])

            self.start_epoch = checkpoint['epoch'] + 1 

        print("network initialized")

    def set_optimizer(self):

        """
            initialize optimzer 

            kalman_type = "g", "l_0","l_1", "l_2", "s" 
        """ 

        if self.kalman_type == "g":
            # global KF
            self.optimizer = GKalmanFilter(self.model, kalman_lambda=self.kalman_lambda, kalman_nue=self.kalman_nue, device=self.device)

            print ("optimizer: global KF")

        elif self.kalman_type == "l_0":

            self.optimizer = LKalmanFilter(self.model, kalman_lambda=self.kalman_lambda, kalman_nue=self.kalman_nue, device=self.device,
                                        nselect=self.select_num, groupsize=self.group_size, blocksize=self.block_size, fprefactor=1.0)
            print ("optimizer: layerwise KF V0")
        elif self.kalman_type == "l_1":
            
            self.optimizer = L1KalmanFilter(self.model, kalman_lambda=self.kalman_lambda, kalman_nue=self.kalman_nue, device=self.device,
                                        nselect=self.select_num, groupsize=self.group_size, blocksize=self.block_size, fprefactor=1.0)
            print ("optimizer: layerwise KF V1")
        elif self.kalman_type == "l_2":

            self.optimizer = L2KalmanFilter(self.model, kalman_lambda=self.kalman_lambda, kalman_nue=self.kalman_nue, device=self.device,
                                        nselect=self.select_num, groupsize=self.group_size, blocksize=self.block_size, fprefactor=1.0)
            print ("optimizer: layerwise KF v2")
        elif self.kalman_type == "s":
            
            self.optimizer = SKalmanFilter(self.model, kalman_lambda=self.kalman_lambda, kalman_nue=self.kalman_nue, device=self.device)

        else:
            # using built-in optimizer 
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
                self.optimizer = optim.LBFGS(model_parameters, lr = self.LR_base)

            else:
                raise RuntimeError("unsupported optimizer: %s" %self.opts.opt_optimizer) 
            print ("optimizer: " + self.opts.opt_optimizer)
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
        
    def LinearLR(optimizer, base_lr, target_lr, total_epoch, cur_epoch):
        lr = base_lr - (base_lr - target_lr) * (float(cur_epoch) / float(total_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_lr(self,iter_num,  stop_lr=3.51e-8):

        start_lr = self.opts.opt_lr 

        stop_step = 1000000
        decay_step=5000
        decay_rate = np.exp(np.log(stop_lr/start_lr) / (stop_step/decay_step)) #0.9500064099092085
        real_lr = start_lr * np.power(decay_rate, (iter_num//decay_step))
        return real_lr  

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


    def train_img(self,sample_batches, model, optimizer, criterion, last_epoch, real_lr):
        """
            training on a single image w. built-in optimizer 
        """
        if (self.opts.opt_dtype == 'float64'):
            Ei_label = Variable(sample_batches['output_energy'][:,:,:].double().to(self.device))
            Force_label = Variable(sample_batches['output_force'][:,:,:].double().to(self.device))   #[40,108,3]
            dR_neigh_list = Variable(sample_batches['input_dR_neigh_list'].to(self.device))
            Ri = Variable(sample_batches['input_Ri'].double().to(self.device), requires_grad=True)
            Ri_d = Variable(sample_batches['input_Ri_d'].to(self.device))


        elif (self.opts.opt_dtype == 'float32'):
            Ei_label = Variable(sample_batches['output_energy'][:,:,:].float().to(self.device))
            Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(self.device))   #[40,108,3]
            dR_neigh_list = Variable(sample_batches['input_dR_neigh_list'].to(self.device))
            Ri = Variable(sample_batches['input_Ri'].double().to(self.device), requires_grad=True)
            Ri_d = Variable(sample_batches['input_Ri_d'].to(self.device))
            
        else:
            raise RuntimeError("train(): unsupported opt_dtype %s" %self.opts.opt_dtype)

        atom_number = Ei_label.shape[1]
        Etot_label = torch.sum(Ei_label, dim=1)
        neighbor = Variable(sample_batches['input_nblist'].int().to(self.device))  # [40,108,100]
        ind_img = Variable(sample_batches['ind_image'].int().to(self.device))
        natoms_img = Variable(sample_batches['natoms_img'].int().to(self.device))
        
        model.train()
        
        Etot_predict, Ei_predict, Force_predict = model(Ri, Ri_d, dR_neigh_list, natoms_img, None, None)
        
        optimizer.zero_grad()

        loss_F = criterion(Force_predict, Force_label)
        loss_Etot = criterion(Etot_predict, Etot_label)

        loss_Ei = 0.0
        loss_Egroup = 0.0
        
        start_lr = self.opts.opt_lr
        w_f = 1
        w_e = 1
        w_eg = 0
        w_ei = 0
        
        loss, pref_f, pref_e = self.get_loss_func(start_lr, real_lr, w_f, loss_F, w_e, loss_Etot, w_eg, loss_Egroup, w_ei, loss_Ei, natoms_img[0, 0].item())

        loss.backward() 
        optimizer.step()

        print("RMSE_Etot = %.10f, RMSE_Ei = %.10f, RMSE_Force = %.10f, RMSE_Egroup = %.10f" %(loss_Etot ** 0.5, loss_Ei ** 0.5, loss_F ** 0.5, loss_Egroup**0.5))

        return loss, loss_Etot, loss_Ei, loss_F


    def train_kalman_img(self,sample_batches, model, kalman, criterion, last_epoch, real_lr):
        if (self.opts.opt_dtype == 'float64'):
            Ei_label = Variable(sample_batches['output_energy'][:,:,:].double().to(self.device))
            Force_label = Variable(sample_batches['output_force'][:,:,:].double().to(self.device))   #[40,108,3]
          
            dR_neigh_list = Variable(sample_batches['input_dR_neigh_list'].to(self.device))
            Ri = Variable(sample_batches['input_Ri'].double().to(self.device), requires_grad=True)
            Ri_d = Variable(sample_batches['input_Ri_d'].to(self.device))
    

        elif (self.opts.opt_dtype == 'float32'):
            Ei_label = Variable(sample_batches['output_energy'][:,:,:].float().to(self.device))
            Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(self.device))   #[40,108,3]
      
            dR_neigh_list = Variable(sample_batches['input_dR_neigh_list'].to(self.device))
            Ri = Variable(sample_batches['input_Ri'].double().to(self.device), requires_grad=True)
            Ri_d = Variable(sample_batches['input_Ri_d'].to(self.device))
           
        else:
            
            raise RuntimeError("train(): unsupported opt_dtype %s" %self.opts.opt_dtype)

        Etot_label = torch.sum(Ei_label, dim=1)
        
        natoms_img = Variable(sample_batches['natoms_img'].int().to(self.device))

        kalman_inputs = [Ri, Ri_d, dR_neigh_list, natoms_img, None, None]

        if self.is_trainEtot: 
            kalman.update_energy(kalman_inputs, Etot_label, update_prefactor = self.kf_prefac_Etot)

        if self.is_trainForce:
            kalman.update_force(kalman_inputs, Force_label, update_prefactor = self.kf_prefac_F)
        
        """
            No Ei and Egroup update at this moment              
        """
        
        Etot_predict, Ei_predict, Force_predict = model(Ri, Ri_d, dR_neigh_list, natoms_img, None, None)
        
        loss_F = criterion(Force_predict, Force_label)
        loss_Etot = criterion(Etot_predict, Etot_label)
        loss_Ei = torch.zeros([1,1], device = self.device)
        loss_Egroup = torch.zeros([1,1], device = self.device)
        loss = loss_F + loss_Etot
        
        print("RMSE_Etot = %.10f, RMSE_Ei = %.10f, RMSE_Force = %.10f, RMSE_Egroup = %.10f" %(loss_Etot ** 0.5, loss_Ei ** 0.5, loss_F ** 0.5, loss_Egroup**0.5))

        return loss, loss_Etot, loss_Ei, loss_F, loss_Egroup

    def valid_img(self,sample_batches, model, criterion):
        if (self.opts.opt_dtype == 'float64'):
            Ei_label = Variable(sample_batches['output_energy'][:,:,:].double().to(self.device))
            Force_label = Variable(sample_batches['output_force'][:,:,:].double().to(self.device))   #[40,108,3]

            dR_neigh_list = Variable(sample_batches['input_dR_neigh_list'].to(self.device))
            Ri = Variable(sample_batches['input_Ri'].double().to(self.device), requires_grad=True)
            Ri_d = Variable(sample_batches['input_Ri_d'].to(self.device))
        
        elif (self.opts.opt_dtype == 'float32'):
            Ei_label = Variable(sample_batches['output_energy'][:,:,:].float().to(self.device))
            Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(self.device))   #[40,108,3]
            dR_neigh_list = Variable(sample_batches['input_dR_neigh_list'].to(self.device))
            Ri = Variable(sample_batches['input_Ri'].double().to(self.device), requires_grad=True)
            Ri_d = Variable(sample_batches['input_Ri_d'].to(self.device))


        else:
            raise RuntimeError("train(): unsupported opt_dtype %s" %self.opts.opt_dtype)

        natoms_img = Variable(sample_batches['natoms_img'].int().to(self.device))  # [40,108,100]

        error=0.0

        Etot_label = torch.sum(Ei_label, dim=1)
        
        # model.train()
        model.eval()
        
        Etot_predict, Ei_predict, Force_predict = model(Ri, Ri_d, dR_neigh_list, natoms_img, None, None)
        
        """
        print("********************************************************")
        print("********************************************************\n")
        print("Etot by inference:\n",Etot_predict)
        print("Force of 31st Cu atom by inference:\n" , Force_predict[0][30])
        print("********************************************************")
        print("********************************************************\n")
        """ 
        
        
        # Egroup_predict = model.get_egroup(Ei_predict, egroup_weight, divider)
        loss_F = criterion(Force_predict, Force_label)
        loss_Etot = criterion(Etot_predict, Etot_label)
        loss_Ei = criterion(Ei_predict, Ei_label)
        loss_Egroup = torch.zeros([1,1], device = self.device)

        error = float(loss_F.item()) + float(loss_Etot.item())
        
        return error, loss_Etot, loss_Ei, loss_F, loss_Egroup
        
    def train(self):
        """
            training of DP network 
        """ 
        iter = 0 
        iter_valid = 0 

        for epoch in range(self.start_epoch, self.n_epoch + 1):

            timeEpochStart = time.time()

            if (epoch == self.n_epoch):
                last_epoch = True
            else:
                last_epoch = False

            print("<----------------------------  epoch %d  ---------------------------->" %(epoch))
            
            """
                ========== training starts ==========
            """
            nr_total_sample = 0
            loss = 0.
            loss_Etot = 0.
            loss_Ei = 0.
            loss_F = 0.
            loss_Egroup = 0.0   

            for i_batch, sample_batches in enumerate(self.loader_train):

                nr_batch_sample = sample_batches['output_energy'].shape[0]

                global_step = (epoch - 1) * len(self.loader_train) + i_batch * nr_batch_sample
                
                real_lr = self.adjust_lr(iter_num = global_step)

                if self.not_KF():
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = real_lr * (self.batch_size ** 0.5)
                
                natoms_sum = sample_batches['natoms_img'][0, 0].item()  
                
                if self.not_KF():
                    # non-KF t
                    batch_loss, batch_loss_Etot, batch_loss_Ei, batch_loss_F = \
                        self.train_img(sample_batches, self.model, self.optimizer, nn.MSELoss(), last_epoch, real_lr)
                else:
                    # KF 
                    batch_loss, batch_loss_Etot, batch_loss_Ei, batch_loss_F, batch_loss_Egroup = \
                        self.train_kalman_img(sample_batches, self.model, self.optimizer, nn.MSELoss(), last_epoch, 0.001)
                
                iter += 1
                f_err_log = self.opts.opt_session_dir+'iter_loss.dat'
                
                if iter == 1:
                    fid_err_log = open(f_err_log, 'w')
                    fid_err_log.write('iter\t loss\t RMSE_Etot\t RMSE_Ei\t RMSE_F\t RMSE_Eg\n')
                    fid_err_log.close() 

                fid_err_log = open(f_err_log, 'a')
                fid_err_log.write('%d %e %e %e %e %e \n'%(iter, batch_loss, math.sqrt(batch_loss_Etot)/natoms_sum, math.sqrt(batch_loss_Ei), math.sqrt(batch_loss_F), math.sqrt(batch_loss_Egroup)))
                fid_err_log.close() 

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

            epoch_err_log = self.opts.opt_session_dir+'epoch_loss.dat'

            if epoch == 1:
                f_epoch_err_log = open(epoch_err_log, 'w')
                f_epoch_err_log.write('epoch\t loss\t RMSE_Etot\t RMSE_Ei\t RMSE_F\t RMSE_Eg\n')
                f_epoch_err_log.close()
            
            f_epoch_err_log = open(epoch_err_log, 'a')
            f_epoch_err_log.write('%d %e %e %e %e %e \n'%(epoch, loss, RMSE_Etot, RMSE_Ei, RMSE_F, RMSE_Egroup))
            f_epoch_err_log.close() 
            
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
                    self.LinearLR(optimizer=self.optimizer, base_lr=self.LR_base, target_lr=self.opts.opt_LR_max_lr, total_epoch=self.n_epoch, cur_epoch=epoch)

                elif (opt_scheduler == 'LR_DEC'):
                    self.LinearLR(optimizer=self.optimizer, base_lr=self.LR_base, target_lr=self.opts.opt_LR_min_lr, total_epoch=self.n_epoch, cur_epoch=epoch)

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
                nr_batch_sample = sample_batches['output_energy'].shape[0]

                valid_error_iter, batch_loss_Etot, batch_loss_Ei, batch_loss_F, batch_loss_Egroup = self.valid_img(sample_batches, self.model, nn.MSELoss())

                # n_iter = (epoch - 1) * len(loader_valid) + i_batch + 1
                valid_loss += valid_error_iter * nr_batch_sample

                valid_loss_Etot += batch_loss_Etot * nr_batch_sample
                valid_loss_Ei += batch_loss_Ei * nr_batch_sample
                valid_loss_F += batch_loss_F * nr_batch_sample
                valid_loss_Egroup += batch_loss_Egroup * nr_batch_sample

                nr_total_sample += nr_batch_sample

                f_err_log = self.opts.opt_session_dir+'iter_loss_valid.dat'

                if iter_valid == 1:
                    fid_err_log = open(f_err_log, 'w')
                    fid_err_log.write('iter\t loss\t RMSE_Etot\t RMSE_Ei\t RMSE_F\t lr\n')
                    fid_err_log.close() 
                
                                
                fid_err_log = open(f_err_log, 'a')
                fid_err_log.write('%d %e %e %e %e %e \n'%(iter, batch_loss, math.sqrt(batch_loss_Etot)/natoms_sum, math.sqrt(batch_loss_Ei), math.sqrt(batch_loss_F), real_lr))
                fid_err_log.close() 

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

            f_err_log =  self.opts.opt_session_dir + 'epoch_loss_valid.dat'
            
            if not os.path.exists(f_err_log):
                fid_err_log = open(f_err_log, 'w')
                fid_err_log.write('epoch\t valid_RMSE_Etot\t valid_RMSE_Ei\t valid_RMSE_F\t valid_RMSE_Eg \n')
                fid_err_log.close() 

            fid_err_log = open(f_err_log, 'a')
            fid_err_log.write('%d %e %e %e %e\n'%(epoch, valid_RMSE_Etot, valid_RMSE_Ei, valid_RMSE_F, valid_RMSE_Egroup))
            fid_err_log.close() 

            """  
                save model 
            """
            if self.not_KF():
                state = {'model': self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'epoch':epoch}
            else:
                state = {'model': self.model.state_dict(), 'epoch': epoch}
            
            latest_path = self.opts.opt_model_dir + "latest.pt"
            torch.save(state, latest_path) 
            
            # save .pt
            if self.not_KF():
                if epoch % 50 == 0: 
                    current_model_path = self.opts.opt_model_dir + str(epoch) + '.pt'
                    torch.save(state, current_model_path)
            else:
                current_model_path = self.opts.opt_model_dir + str(epoch) + '.pt'
                torch.save(state, current_model_path)

            timeEpochEnd = time.time()

            print("time of epoch %d: %f s" %(epoch, timeEpochEnd - timeEpochStart))

    def evaluate(self,num_thread = 4):
        """
            evaluate a model w.r.t AIMD
            put a MOVEMENT in /MD and run MD100 
        """

        if not os.path.exists("MD/MOVEMENT"):
            raise Exception("MD/MOVEMENT not found")
        
        import md100
        md100.run_md100(imodel = 5, atom_type = pm.atomType, num_process = num_thread) 
        
    def plot_evaluation(self):

        if not os.path.exists("MOVEMENT"):
            raise Exception("MOVEMENT not found. It should be force field MD result")

        import plot_evaluation
        plot_evaluation.plot()

    """
        parameter extraction related functions
    """
    
    def catNameEmbedingW(self,idxNet, idxLayer):
        return "embeding_net."+str(idxNet)+".weights.weight"+str(idxLayer)

    def catNameEmbedingB(self,idxNet, idxLayer):
        return "embeding_net."+str(idxNet)+".bias.bias"+str(idxLayer)

    def catNameFittingW(self,idxNet, idxLayer):
        return "fitting_net."+str(idxNet)+".weights.weight"+str(idxLayer)

    def catNameFittingB(self,idxNet, idxLayer):
        return "fitting_net."+str(idxNet)+".bias.bias"+str(idxLayer)

    def catNameFittingRes(self,idxNet, idxResNet):
        return "fitting_net."+str(idxNet)+".resnet_dt.resnet_dt"+str(idxResNet)

    def dump(self,item, f):
        raw_str = ''
        for num in item:
            raw_str += (str(float(num))+' ')
        f.write(raw_str)
        f.write('\n')

    def extract_model_para(self, model_name = None):
        """
            extract the model parameters of DP network
        """
        if model_name is None: 
            self.extract_model_name = self.opts.opt_session_name + "/model/latest.pt"
        else:
            self.extract_model_name = model_name
        

        print ("extracting network parameters from:",self.extract_model_name )
        
        
        kfdp = True if self.kalman_type is not None else False
        
        print ("using KF?",kfdp)

        netConfig = pm.DP_cfg_dp if kfdp==False else pm.DP_cfg_dp_kf

        isEmbedingNetResNet = netConfig["embeding_net"]["resnet_dt"]
        isFittingNetResNet  = netConfig["fitting_net"]["resnet_dt"]

        embedingNetSizes = netConfig['embeding_net']['network_size']
        nLayerEmbedingNet = len(embedingNetSizes)   

        print("layer number of embeding net:"+str(nLayerEmbedingNet))
        print("size of each layer"+ str(embedingNetSizes) + '\n')

        fittingNetSizes = netConfig['fitting_net']['network_size']
        nLayerFittingNet = len(fittingNetSizes)

        print("layer number of fitting net:"+str(nLayerFittingNet))
        print("size of each layer"+ str(fittingNetSizes) + '\n')

        embedingNet_output = 'embeding.net' 
        fittingNet_output = 'fitting.net'

        pt_name = self.extract_model_name
        
        #r"record/model/better.pt" # modify according to your need 

        raw = torch.load(pt_name,map_location=torch.device("cpu"))['model']

        tensor_list = list(raw.keys())

        #determining # of networks 
        nEmbedingNet = len(pm.atomType)**2  
        nFittingNet = len(pm.atomType)
            
        """
            write embedding network
        """
        f = open(embedingNet_output, 'w')

        # total number of embeding network
        f.write(str(nEmbedingNet)+'\n') 

        #layer of embeding network
        f.write(str(nLayerEmbedingNet) + '\n')

        #size of each layer

        f.write("1 ")
        for i in embedingNetSizes:
            f.write(str(i)+' ')

        f.write('\n')

        #f.writelines([str(i) for i in embedingNetSizes])
            
        print("******** converting embeding network starts ********")
        for idxNet in range(nEmbedingNet):
            print ("converting embeding network No."+str(idxNet))
            for idxLayer in range(nLayerEmbedingNet):
                print ("converting layer "+str(idxLayer) )	

                #write wij
                label_W = self.catNameEmbedingW(idxNet,idxLayer)
                for item in raw[label_W]:
                    self.dump(item,f)

                print("w matrix dim:" +str(len(raw[label_W])) +str('*') +str(len(raw[label_W][0])))

                #write bi
                label_B = self.catNameEmbedingB(idxNet,idxLayer)
                self.dump(raw[label_B][0],f)
                print ("b dim:" + str(len(raw[label_B][0])))

            print ('\n')
                #break
        f.close()

        print("******** converting embeding network ends  *********")

        """
            write fitting network
        """

        f = open(fittingNet_output, 'w')

        # total number of embeding network
        f.write(str(nFittingNet)+'\n') 

        #layer of embeding network
        f.write(str(nLayerFittingNet) + '\n')

        #size of each layer

        f.write(str(len(raw[self.catNameFittingW(0,0)]))+' ')

        for i in fittingNetSizes:
            f.write(str(i)+' ')

        f.write('\n')

        print("******** converting fitting network starts ********")
        for idxNet in range(nFittingNet):
            print ("converting fitting network No."+str(idxNet))
            for idxLayer in range(nLayerFittingNet):
                print ("converting layer "+str(idxLayer) )  

                #write wij
                label_W = self.catNameFittingW(idxNet,idxLayer)
                for item in raw[label_W]:
                    self.dump(item,f)

                print("w matrix dim:" +str(len(raw[label_W])) +str('*') +str(len(raw[label_W][0])))

                #write bi
                label_B = self.catNameFittingB(idxNet,idxLayer)
                self.dump(raw[label_B][0],f)
                print ("b dim:" + str(len(raw[label_B][0])))

            print ('\n')
                #break
        f.close()

        print("******** converting fitting network ends  *********")

        """
            writing ResNets
        """
        print("******** converting resnet starts  *********")

        if isFittingNetResNet:
            numResNet = 0

            """


            for keys in list(raw.keys()):
                tmp = keys.split('.')
                if tmp[0] == "fitting_net" and tmp[1] == '0' and tmp[2] == 'resnet_dt':
                    numResNet +=1 

            print ("number of resnet: " + str(numResNet))

            filename  = "fittingNet.resnet"

            f= open(filename, "w")
            
            f.write(str(numResNet)+"\n")

            for fittingNetIdx in range(nFittingNet):
                for resNetIdx in range(1,numResNet+1):
                    f.write(str(fittingNetSizes[i+1])+"\n")
                    label_resNet = catNameFittingRes(fittingNetIdx,resNetIdx)
                    dump(raw[label_resNet][0],f)

            """ 

            """
                The format below fits Dr.Wang's Fortran routine
            """

            for keys in list(raw.keys()):
                tmp = keys.split('.')
                if tmp[0] == "fitting_net" and tmp[1] == '0' and tmp[2] == 'resnet_dt':
                    numResNet +=1 

            print ("number of resnet: " + str(numResNet))

            filename  = "fittingNet.resnet"

            f= open(filename, "w")
            # itype: number of fitting network 
            f.write(str(nFittingNet)+'\n') 

            #nlayer: 
            f.write(str(nLayerFittingNet) + '\n')

            #dim of each layer 
            f.write(str(len(raw[self.catNameFittingW(0,0)]))+' ')

            for i in fittingNetSizes:
                f.write(str(i)+' ')	
            f.write("\n")

            for i in range(0,len(fittingNetSizes)+1):
                if (i > 1) and (i < len(fittingNetSizes)):
                    f.write("1 ")
                else:
                    f.write("0 ")

            f.write("\n")

            #f.write(str(numResNet)+"\n")

            for fittingNetIdx in range(nFittingNet):
                for resNetIdx in range(1,numResNet+1):
                    f.write(str(fittingNetSizes[resNetIdx])+"\n")
                    label_resNet = self.catNameFittingRes(fittingNetIdx,resNetIdx)   
                    self.dump(raw[label_resNet][0],f)

            f.close()

        print("******** converting resnet done *********\n")

        print("******** generating gen_dp.in  *********\n")
        
        orderedAtomList = [str(atom) for atom in pm.atomType]

        print ("ordered atom list", pm.atomType)
        
        davg = np.load("davg.npy")
        dstd = np.load("dstd.npy")

        davg_size = len(davg)
        dstd_size = len(dstd)

        assert(davg_size == dstd_size)
        assert(davg_size == len(orderedAtomList))

        f_out = open("gen_dp.in","w")

        # in default_para.py, Rc is the max cut, beyond which S(r) = 0 
        # Rm is the min cut, below which S(r) = 1

        f_out.write(str(pm.Rc) + ' ') 
        f_out.write(str(pm.maxNeighborNum)+"\n")
        f_out.write(str(dstd_size)+"\n")
        
        for i,atom in enumerate(orderedAtomList):
            f_out.write(atom+"\n")
            f_out.write(str(pm.Rc)+' '+str(pm.Rm)+'\n')

            for idx in range(4):
                f_out.write(str(davg[i][idx])+" ")
            
            f_out.write("\n")

            for idx in range(4):
                f_out.write(str(dstd[i][idx])+" ")
            f_out.write("\n")
        
        f_out.close() 

        print("******** gen_dp.in generation done *********")

    def test_dbg(self):

        """
            varying cordinate in the first image of MOVEMENT
        """
        self.generate_data()
        
        self.load_data()
        
        self.set_model(load_network = True) 


        nr_total_sample = 0

        test_loss = 0.
        test_loss_Etot = 0.
        test_loss_Ei = 0.
        test_loss_F = 0.
        
        assert self.batch_size == 1 
        
        # picking out the first image of training set 
        for i_batch, sample_batches in enumerate(self.loader_train):
            
            # ONLY handle the 1st image
            print ("testing image:",i_batch)
            natoms_sum = sample_batches['natoms_img'][0, 0].item()
            nr_batch_sample = sample_batches['output_energy'].shape[0]
            
            valid_error_iter, batch_loss_Etot, batch_loss_Ei, batch_loss_F, batch_loss_Egroup = self.valid_img(sample_batches, self.model, nn.MSELoss())

            test_loss_Etot += batch_loss_Etot.item() * nr_batch_sample
            test_loss_Ei += batch_loss_Ei.item() * nr_batch_sample
            test_loss_F += batch_loss_F.item() * nr_batch_sample

            nr_total_sample += nr_batch_sample 
            #print(valid_error_iter, batch_loss_Etot, batch_loss_Ei, batch_loss_F, batch_loss_Egroup)

            break
        
        test_loss /= nr_total_sample
        test_loss_Etot /= nr_total_sample
        test_loss_Ei /= nr_total_sample
        test_loss_F /= nr_total_sample

        test_RMSE_Etot = test_loss_Etot ** 0.5
        test_RMSE_Ei = test_loss_Ei ** 0.5
        test_RMSE_F = test_loss_F ** 0.5   
        
        print(" (test_RMSE_Etot = %.16f, test_RMSE_Ei = %.16f, test_RMSE_F = %.16f)" \
        %( test_RMSE_Etot, test_RMSE_Ei, test_RMSE_F))

    def run_md(self, init_config = "atom.config", md_details = None, num_thread = 1,follow = False):
        import subprocess 

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
        f.write("5\n")     # imodel=1,2,3.    {1:linear;  2:VV;   3:NN, 5: dp 
        f.write('1\n')               # interval for MOVEMENT output
        f.write('%d\n' % len(pm.atomType)) 
        
        for i in range(len(pm.atomType)):
            f.write('%d %d\n' % (pm.atomType[i], 2*pm.atomType[i]))
        f.close()    
        
        # creating md.input for main_MD.x 
        command = r'mpirun -n ' + str(num_thread) + r' main_MD.x'
        print (command)
        subprocess.run(command, shell=True) 

