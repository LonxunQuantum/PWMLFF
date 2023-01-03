"""
    module for Deep Potential Network 

    L. Wang, 2022.8

    updated 2022.12

"""
from ast import For, If, NodeTransformer
import os,sys
import pathlib
import argparse
import random

codepath = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(codepath)

#for model.mlff 
sys.path.append(codepath+'/../model')

#for default_para, data_loader_2type dfeat_sparse dp_mlff
sys.path.append(codepath+'/../pre_data')

#for optimizer
sys.path.append(codepath+'/..')
sys.path.append(codepath+'/../aux')
sys.path.append(codepath+'/../lib')
sys.path.append(codepath+'/../..')

from statistics import mode 
from turtle import Turtle, update
import torch
    
import time
import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt
import torch.nn as nn
import math

import torch.utils.data as Data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import signal 

import time
import warnings

from joblib import dump, load

# src/aux 
from opts import opt_values 
from feat_modifier import feat_modifier

"""
    customized modules 
"""
from model.dp_dp import DP
from optimizer.GKF import GKFOptimizer
from optimizer.LKF import LKFOptimizer
from dp_data_loader import MovementDataset
from dp_mods.dp_trainer import *

import default_para as pm

#from data_loader_2type_dp import MovementDataset, get_torch_data

def get_terminal_args():
    """
        obtaining args from terminal
    """
    parser = argparse.ArgumentParser(description="PyTorch MLFF Training")
    
    parser.add_argument(
        "--datatype",
        default="float64",
        type=str,
        help="Datatype and Modeltype default float64",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs", default=30, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=1,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=16,
        type=int,
        metavar="N",
        help="mini-batch size (default: 1), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        dest="resume",
        action="store_true",
        help="resume the latest checkpoint",
    )
    parser.add_argument(
        "-s" "--store-path",
        default="default",
        type=str,
        metavar="STOREPATH",
        dest="store_path",
        help="path to store checkpoints (default: 'default')",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://localhost:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--magic", default=2022, type=int, help="Magic number. ")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--dp", dest="dp", action="store_true", help="Whether to use DP, default False."
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    # parser.add_argument("--dummy", action="store_true", help="use fake data to benchmark")
    parser.add_argument(
        "-n", "--net-cfg", default="DeepMD_cfg_dp_kf", type=str, help="Net Arch"
    )
    parser.add_argument("--act", default="sigmoid", type=str, help="activation kind")
    
    parser.add_argument(
        "--opt", default="ADAM", type=str, help="optimizer type: LKF, GKF, ADAM, SGD"
    )
    parser.add_argument(
        "--Lambda", default=0.98, type=float, help="KFOptimizer parameter: Lambda."
    )
    parser.add_argument(
        "--nue", default=0.99870, type=float, help="KFOptimizer parameter: Nue."
    )
    parser.add_argument(
        "--blocksize", default=10240, type=int, help="KFOptimizer parameter: Blocksize."
    )
    
    parser.add_argument(
        "--nselect", default=24, type=int, help="KFOptimizer parameter: Nselect."
    )
    
    parser.add_argument(
        "--groupsize", default=6, type=int, help="KFOptimizer parameter: Groupsize."
    )
    
    args = parser.parse_args()
    
    return args


class dp_network:
    
    def __init__(   self,
                    # some must-haves
                    # model related argument
                    terminal_args = get_terminal_args(), 
                    atom_type = None, 
                    # optimizer related arguments 
                    
                    max_neigh_num = 100, 
                    # "g", "l_0","l_1", "l_2", "s" 
                    
                    kalman_type = None,     # optimal version    
                    
                    session_dir = "record",  # directory that saves the model
                    n_epoch = 25, 
                    start_epoch = 1, 
                    batch_size = 1, 
                    learning_rate = 0.001, 
                    momentum = 0.9, 
                    weight_decay = 1e-4, 
                    is_resume = False, 
                    is_evaluate = False, 
                    world_size = -1, 
                    rank = -1, 
                    dist_url = "tcp://localhost:23456", 
                    dist_backend = 'nccl', 
                    seed = None, 
                    gpu_id = None, 
                    is_distributed = False, 
                    optimizer = "ADAM",    # LKF, GKF, ADAM, SGD 
                    kalman_lambda = 0.98,
                    kalman_nue = 0.99870, 
                    # paras for l-kalman 
                    select_num = 24,
                    block_size = 5120,
                    group_size = 6,

                    # training label related arguments
                    is_trainForce = True, 
                    is_trainEi = False,
                    is_trainEgroup = False,
                    is_trainEtot = True,
                    
                    precision = "float64", 
                    train_valid_ratio = 0.8, 
                    
                    is_movement_weighted = False,
                    e_tolerance = 999.0, 
                    
                    # inital values for network config
                    embedding_net_config = None,
                    fitting_net_config = None, 

                    embedding_net_size = None, 
                    embedding_net_bias = None,
                    embedding_net_res = None, 
                    embedding_net_act = None, 

                    fitting_net_size = None, 
                    fitting_net_bias = None,
                    fitting_net_res = None, 
                    fitting_net_act = None, 

                    recover = False,
                    dataset_size = 1000, 
                    
                    workers_dataload = 4, 
                      
                    # smooth function calculation 
                    Rmin = 5.8,
                    Rmax = 6.0,
                    
                    # paras for DP network
                    M2 = 16,
                    reconnect = True  
                    
                    ):
        
        # args from terminal 
        self.terminal_args = terminal_args
        
        if atom_type == None:
            raise Exception("atom types not specifed")         
        # parsing command line args 
        
        #opt_values()  
        
        # feature para modifier
        self.feat_mod = feat_modifier 
        
        if Rmin is not None:
            pm.Rm = Rmin

        if Rmax is not None:
            pm.Rc = Rmax
        
        # scaling
        # recover training. Need to load both scaler and model 
        pm.use_storage_scaler = recover  

        pm.maxNeighborNum = max_neigh_num 

        self.atomType = atom_type 

        # these two variables are the same thing. Crazy. 
        pm.atomTypeNum = len(atom_type)
        pm.ntypes = len(atom_type)

        
        # passed-in configs 
        # the whole config is required 
        
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

        self.best_loss = 1e10
        
        # setting M2 of the intermediate matrix D 
        if M2 is not None:
            self.set_M2(M2) 
        
        # set global reconnect switch 
        self.is_reconnect = reconnect
        
        # setting device
        self.device = None
        """
        if device == "cpu":
            self.device = torch.device('cpu')
        elif device == "cuda":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            raise Exception("device type not supported")
        """
        
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

        self.min_loss = np.inf
        self.epoch_print = 1 
        self.iter_print = 1 
            
        # set session directory 
        # self.set_session_dir(session_dir) 

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
        self.load_model_path = 'latest.pt' 
        
        """
            finalizing config. 
            Create a copy 
        """
        self.config = { 'Rc_M': Rmax, 
                        'maxNeighborNum': max_neigh_num,
                        'atomType': [],

                        # no need to change frequently 
                        'E_tolerance': e_tolerance,
                        'trainSetDir': './PWdata',
                        'dRFeatureInputDir': './input',
                        'dRFeatureOutputDir': './output',
                        'trainDataPath': './train',
                        'validDataPath': './valid',
                        'ratio': train_valid_ratio,
                        'training_type': precision,

                        'M2': M2, # M2 of G matrix in DP model 
                        'datasetImageNum': dataset_size,
                        
                        'net_cfg': {'embedding_net': {'network_size': [25, 25, 25],
                        'bias': True,
                        'resnet_dt': False,
                        'activation': 'tanh'},

                        'fitting_net': {'network_size': [50, 50, 50, 1],
                        'bias': True,
                        'resnet_dt': False,
                        'activation': 'tanh'}}}
            

        self.config["atomType"] = [] 
        for idx in atom_type:
            single_dict = {'type': idx, 'Rc': Rmax, 'Rm': Rmin, 'iflag_grid': 3, 'fact_base': 0.2, 'dR1': 0.5}
            self.config["atomType"].append(single_dict.copy())

        
        is_indiv_en = lambda: embedding_net_size is None and embedding_net_bias is None and embedding_net_res is None and embedding_net_act is None
        is_indiv_fn = lambda: fitting_net_size is None and fitting_net_bias is None and fitting_net_res is None and fitting_net_act is None

        """ 
            should provide full config or partial. Otherwise use default 
            full config overrides parital config
        """
        if (embedding_net_config is not None) or (not is_indiv_en()) :
            
            if is_indiv_en():
                # load the whole config
                self.config["net_cfg"]["embedding_net"] = embedding_net_config
            else: 
                # load sperately
                if embedding_net_size is not None:
                    self.config["net_cfg"]["embedding_net"]["network_size"] = embedding_net_size
                    
                if embedding_net_bias is not None:
                    self.config["net_cfg"]["embedding_net"]["bias"] = embedding_net_bias

                if embedding_net_res is not None:
                    self.config["net_cfg"]["embedding_net"]["resnet_dt"] = embedding_net_res
                
                if embedding_net_act is not None:
                    self.config["net_cfg"]["embedding_net"]["activation"] = embedding_net_act

        # fitting net 
        if (fitting_net_config is not None) or (not is_indiv_fn()) :
            
            if is_indiv_fn():
                # load the whole config
                self.config["net_cfg"]["fitting_net"] = fitting_net_config
            else: 
                # load sperately
                if fitting_net_size is not None:
                    self.config["net_cfg"]["fitting_net"]["network_size"] = fitting_net_size
                    
                if fitting_net_bias is not None:
                    self.config["net_cfg"]["fitting_net"]["bias"] = fitting_net_bias

                if fitting_net_res is not None:
                    self.config["net_cfg"]["fitting_net"]["resnet_dt"] = fitting_net_res
                
                if fitting_net_act is not None:
                    self.config["net_cfg"]["fitting_net"]["activation"] = fitting_net_act

        # finalize terminal args
        """
            Load args with instantiation parameters. 
            The latter overrides. 
            Leave instantiation parameter blank if using terminal args 
        """
        
        
        # check if any command line arg is used     
        # DO NOT USE ARGS AND PARA AT THE SAME TIME
        if len(sys.argv) < 2:
            
            self.terminal_args.datatype = precision
            self.terminal_args.workers = workers_dataload
            self.terminal_args.epochs = n_epoch
            self.terminal_args.start_epoch = start_epoch
            self.terminal_args.batch_size = batch_size
            self.terminal_args.lr = learning_rate
            self.terminal_args.momentum = momentum
            self.terminal_args.weight_decay = weight_decay
            self.terminal_args.print_freq = 10 
            self.terminal_args.resume = is_resume
            self.terminal_args.store_path = session_dir
            self.terminal_args.evaluate = is_evaluate
            self.terminal_args.world_size = world_size
            self.terminal_args.rank = rank
            self.terminal_args.dist_url = dist_url
            #self.terminal_args.dist_backend = dist_backend, 
            self.terminal_args.seed = seed 
            self.terminal_args.magic = 2022
            self.terminal_args.gpu = gpu_id
            self.terminal_args.multiprocessing_distributed = is_distributed 
            self.terminal_args.opt = optimizer 
            self.terminal_args.Lambda = kalman_lambda 
            self.terminal_args.nue = kalman_nue 
            self.terminal_args.blocksize = block_size
            self.terminal_args.nselect = select_num
            self.terminal_args.groupsize = group_size
            
            print("using args from class instantiation")
            
        else:
            print("using args from command line")

        
    """
        data pre-processing    
    """
    def generate_data(self):
        """
            generate dp's pre-feature
        """
        import dp_mlff 
        
        dp_mlff.gen_train_data(self.config)
        dp_mlff.sepper_data(self.config)

        # Please book in advance!         
        
    """
        load and train. 
    """
    def load_and_train(self):
        """
            main()
        """
        if self.terminal_args.seed is not None:
            random.seed(self.terminal_args.seed)
            torch.manual_seed(self.terminal_args.seed)
            cudnn.deterministic = True
            cudnn.benchmark = False
            warnings.warn(
                "You have chosen to seed training. "
                "This will turn on the CUDNN deterministic setting, "
                "which can slow down your training considerably! "
                "You may see unexpected behavior when restarting "
                "from checkpoints."
            )   
        
        if self.terminal_args.gpu is not None:
            warnings.warn(
                "You have chosen a specific GPU. This will completely "
                "disable data parallelism."
            )

        if self.terminal_args.dist_url == "env://" and self.terminal_args.world_size == -1:
            self.terminal_args.world_size = int(os.environ["WORLD_SIZE"])

        self.terminal_args.distributed = self.terminal_args.world_size > 1 or self.terminal_args.multiprocessing_distributed

        if not os.path.exists(self.terminal_args.store_path):
            os.mkdir(self.terminal_args.store_path)

        if torch.cuda.is_available():
            ngpus_per_node = torch.cuda.device_count()
        else:
            ngpus_per_node = 1

        if self.terminal_args.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            self.terminal_args.world_size = ngpus_per_node * self.terminal_args.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(self.main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, self.terminal_args))
        else:
            # Simply call main_worker function
            self.main_worker(self.terminal_args.gpu, ngpus_per_node, self.terminal_args)
        

    def destory_process(self,signum, frame):
        signame = signal.Signals(signum).name
        print(
            f"Signal handler called with signal {signame} ({signum}): destory dist process"
        )
        dist.destroy_process_group()

    def main_worker(self,gpu, ngpus_per_node, args):

        args.gpu = gpu

        if args.gpu is not None:
            print("Use GPU: {} for training".format(args.gpu))

        if args.distributed:
            if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                args.rank = args.rank * ngpus_per_node + gpu
            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
            signal.signal(signal.SIGINT, self.destory_process)

        if torch.cuda.is_available():
            if args.gpu:
                device = torch.device("cuda:{}".format(args.gpu))
            else:
                device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")


        if args.datatype == "float32":
            training_type = torch.float32  # training type is weights type
        else:
            training_type = torch.float64

        # Create dataset
        train_dataset = MovementDataset(self.config, "./train")
        valid_dataset = MovementDataset(self.config, "./valid")

        # create model
        davg, dstd, ener_shift = train_dataset.get_stat()
        stat = [davg, dstd, ener_shift]
        model = DP(self.config, device, stat, args.magic)
        model = model.to(training_type)

        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            print("using CPU, this will be slow")

        elif args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if torch.cuda.is_available():
                if args.gpu is not None:
                    torch.cuda.set_device(args.gpu)
                    model.cuda(args.gpu)
                    # When using a single GPU per process and per
                    # DistributedDataParallel, we need to divide the batch size
                    # ourselves based on the total number of GPUs of the current node.
                    args.batch_size = int(args.batch_size / ngpus_per_node)
                    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                    model = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[args.gpu], find_unused_parameters=False
                    )
                else:
                    model.cuda()
                    # DistributedDataParallel will divide and allocate batch_size to all
                    # available GPUs if device_ids are not set
                    model = torch.nn.parallel.DistributedDataParallel(model)
        elif args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            model = model.to(device)
        else:
            model = model.cuda()
        
        # define loss function (criterion), optimizer, and learning rate scheduler
        criterion = nn.MSELoss().to(device)

        if args.opt == "LKF":
            optimizer = LKFOptimizer(
                model.parameters(),
                args.Lambda,
                args.nue,
                args.blocksize,
            )
        elif args.opt == "GKF":
            optimizer = GKFOptimizer(
                model.parameters(), 
                args.Lambda, 
                args.nue, device, 
                training_type
            )
        elif args.opt == "ADAM":
            optimizer = optim.Adam(model.parameters(), args.lr)
        elif args.opt == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        else:
            print("Unsupported optimizer!")
        
        # """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

        # optionally resume from a checkpoint
        if args.resume:
            file_name = os.path.join(args.store_path, "best.pth.tar")
            if os.path.isfile(file_name):
                print("=> loading checkpoint '{}'".format(file_name))
                if args.gpu is None:
                    checkpoint = torch.load(file_name)
                elif torch.cuda.is_available():
                    # Map model to be loaded to specified single gpu.
                    loc = "cuda:{}".format(args.gpu)
                    checkpoint = torch.load(file_name, map_location=loc)

                args.start_epoch = checkpoint["epoch"]
                self.best_loss = checkpoint["best_loss"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                # scheduler.load_state_dict(checkpoint["scheduler"])
                print(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        file_name, checkpoint["epoch"]
                    )
                )
            else:
                print("=> no checkpoint found at '{}'".format(file_name))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_dataset, shuffle=False, drop_last=True
            )
        else:
            train_sampler = None
            val_sampler = None
        

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
        )

        val_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler,
        )

        if args.evaluate:
            # validate(val_loader, model, criterion, args)
            valid(val_loader, model, criterion, device, args)
            return

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            train_log = os.path.join(args.store_path, "epoch_train.dat")

            f_train_log = open(train_log, "w")
            f_train_log.write(
                "epoch\t loss\t RMSE_Etot\t RMSE_Ei\t RMSE_F\t real_lr\t time\n"
            )

            valid_log = os.path.join(args.store_path, "epoch_valid.dat")
            f_valid_log = open(valid_log, "w")
            f_valid_log.write("epoch\t loss\t RMSE_Etot\t RMSE_Ei\t RMSE_F\n")

        for epoch in range(args.start_epoch, args.epochs + 1):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # train for one epoch
            if args.opt == "LKF" or args.opt == "GKF":
                real_lr = args.lr
                loss, loss_Etot, loss_Force, loss_Ei, epoch_time = train_KF(
                    train_loader, model, criterion, optimizer, epoch, device, args
                )
            else:
                loss, loss_Etot, loss_Force, loss_Ei, epoch_time, real_lr = train(
                    train_loader, model, criterion, optimizer, epoch, args.lr, device, args
                )
            # evaluate on validation set
            vld_loss, vld_loss_Etot, vld_loss_Force, vld_loss_Ei = valid(
                val_loader, model, criterion, device, args
            )

            if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
            ):
                f_train_log = open(train_log, "a")
                f_train_log.write(
                    "%d %e %e %e %e %e %s\n"
                    % (epoch, loss, loss_Etot, loss_Ei, loss_Force, real_lr, epoch_time)
                )
                f_valid_log = open(valid_log, "a")
                f_valid_log.write(
                    "%d %e %e %e %e\n"
                    % (epoch, vld_loss, vld_loss_Etot, vld_loss_Ei, vld_loss_Force)
                )

            # scheduler.step()

            # remember best loss and save checkpoint
            is_best = vld_loss < self.best_loss
            self.best_loss = min(loss, self.best_loss)

            if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
            ):
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "best_loss": self.best_loss,
                        "optimizer": optimizer.state_dict(),
                        # "scheduler": scheduler.state_dict(),
                    },
                    is_best,
                    "checkpoint.pth.tar",
                    args.store_path,
                )

        if args.distributed:
            dist.destroy_process_group()
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

    def set_M2(self,val):
        pm.dp_M2 = val 
        
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

    """
    def set_model(self,model_name = None):

        self.model = DP(self.network_config, self.opts.opt_act, self.device, self.stat, self.opts.opt_magic, is_reconnect = self.is_reconnect)
        
        self.model.to(self.device)

        if model_name is not None:
                # use lattest.pt as default 
            load_model_path = self.opts.opt_model_dir + model_name

            print ("load network from:",load_model_path)
            
            checkpoint = torch.load(load_model_path, map_location = self.device)

            self.model.load_state_dict(checkpoint['model'])

            self.start_epoch = checkpoint['epoch'] + 1 

        print("network initialized")
    """
    
    """
    def set_optimizer(self):


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
    """

    """    
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
    """ 


    def evaluate(self,num_thread = 1):
        """
            evaluate a model against AIMD
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
    
    def catNameEmbedingW(self,idxNet, idxLayer, has_module=""):
        return "{}embedding_net.".format(has_module)+str(idxNet)+".weights.weight"+str(idxLayer)

    def catNameEmbedingB(self,idxNet, idxLayer, has_module=""):
        return "{}embedding_net.".format(has_module)+str(idxNet)+".bias.bias"+str(idxLayer)

    def catNameFittingW(self,idxNet, idxLayer, has_module=""):
        return "{}fitting_net.".format(has_module)+str(idxNet)+".weights.weight"+str(idxLayer)

    def catNameFittingB(self,idxNet, idxLayer, has_module=""):
        return "{}fitting_net.".format(has_module)+str(idxNet)+".bias.bias"+str(idxLayer)

    def catNameFittingRes(self,idxNet, idxResNet, has_module=""):
        return "{}fitting_net.".format(has_module)+str(idxNet)+".resnet_dt.resnet_dt"+str(idxResNet)
        
    def dump(self,item, f):
        raw_str = ''
        for num in item:
            raw_str += (str(float(num))+' ')
        f.write(raw_str)
        f.write('\n')

    def extract_model_para(self, model_name = None):
        """
            extract the model parameters of DP network

            NEED TO ADD SESSION DIR NAME
        """
        
        if model_name is None: 
            extract_model_name = self.terminal_args.store_path + "/best.pth.tar"
        else:
            extract_model_name = model_name
        
        print ("extracting network parameters from:",extract_model_name )
        
        netConfig = self.config["net_cfg"]

        isEmbedingNetResNet = netConfig["embedding_net"]["resnet_dt"]
        isFittingNetResNet  = netConfig["fitting_net"]["resnet_dt"]

        embedingNetSizes = netConfig['embedding_net']['network_size']
        nLayerEmbedingNet = len(embedingNetSizes)   

        print("layer number of embeding net:"+str(nLayerEmbedingNet))
        print("size of each layer:"+ str(embedingNetSizes) + '\n')

        fittingNetSizes = netConfig['fitting_net']['network_size']
        nLayerFittingNet = len(fittingNetSizes)

        print("layer number of fitting net:"+str(nLayerFittingNet))
        print("size of each layer:"+ str(fittingNetSizes) + '\n')

        embedingNet_output = 'embeding.net' 
        fittingNet_output = 'fitting.net'
        
        raw = torch.load(extract_model_name,map_location=torch.device("cpu"))['state_dict']

        has_module = "module." if "module" in list(raw.keys())[0] else ""
        module_sign = True if "module" in list(raw.keys())[0] else False
        
        #determining # of networks 
        nEmbedingNet = len(self.config["atomType"])**2  
        nFittingNet = len(self.config["atomType"])

        print("number of embedding network:",nEmbedingNet)
        print("\n")
        print("number of fitting network:",nFittingNet)
        
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
            
        print("******** converting embedding network starts ********")
        for idxNet in range(nEmbedingNet):
            
            print ("converting embedding network No."+str(idxNet))

            for idxLayer in range(nLayerEmbedingNet):
                print ("converting layer "+str(idxLayer) )	

                #write wij
                label_W = self.catNameEmbedingW(idxNet,idxLayer,has_module)
                for item in raw[label_W]:
                    self.dump(item,f)

                print("w matrix dim:" +str(len(raw[label_W])) +str('*') +str(len(raw[label_W][0])))

                #write bi
                label_B = self.catNameEmbedingB(idxNet,idxLayer,has_module)
                self.dump(raw[label_B][0],f)
                print ("b dim:" + str(len(raw[label_B][0])))

            print ('\n')

        f.close()

        print("******** converting embedding network ends  *********")
        print("\n")
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
                label_W = self.catNameFittingW(idxNet,idxLayer,has_module)
                for item in raw[label_W]:
                    self.dump(item,f)

                print("w matrix dim:" +str(len(raw[label_W])) +str('*') +str(len(raw[label_W][0])))

                #write bi
                label_B = self.catNameFittingB(idxNet,idxLayer,has_module)
                self.dump(raw[label_B][0],f)
                print ("b dim:" + str(len(raw[label_B][0])))

            print ('\n')
                #break
        f.close()

        print("******** converting fitting network ends  *********")
        print("\n")
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
                if module_sign:
                    if tmp[1] == "fitting_net" and tmp[2] == '0' and tmp[3] == 'resnet_dt':
                        numResNet +=1
                else:
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
        
        orderedAtomList = [str(atom) for atom in self.atomType]
        
        # from where 
        davg = np.load(self.config["trainDataPath"]+"/davg.npy")
        dstd = np.load(self.config["trainDataPath"]+"/dstd.npy")

        davg_size = len(davg)
        dstd_size = len(dstd)

        assert(davg_size == dstd_size)
        assert(davg_size == len(orderedAtomList))
        
        f_out = open("gen_dp.in","w")

        # in default_para.py, Rc is the max cut, beyond which S(r) = 0 
        # Rm is the min cut, below which S(r) = 1

        f_out.write(str(pm.Rc) + ' ') 
        f_out.write(str(pm.maxNeighborNum)+"\n")
        f_out.write(str(pm.dp_M2)+"\n")
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
            varying coordinate in the first image of MOVEMENT
        """
        #self.generate_data()

        self.load_data()
        
        self.set_model(model_name = "latest.pt")   
        
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

