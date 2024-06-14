import os,sys
import pathlib
import random
import torch
import time
import torch.nn as nn
# import horovod.torch as hvd
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

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

import torch
import time
import torch.nn as nn
# import horovod.torch as hvd
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

from src.model.nep_net import NEP
# from src.model.dp_dp_typ_emb_Gk5 import TypeDP as Gk5TypeDP # this is Gk5 type embedding of dp
from src.optimizer.GKF import GKFOptimizer
from src.optimizer.LKF import LKFOptimizer
from src.optimizer.SNES import SNESOptimizer

from src.pre_data.nep_data_loader import MovementDataset, get_stat, gen_train_data
from src.PWMLFF.nep_mods.nep_trainer import train_KF, train, valid, save_checkpoint, predict
from src.PWMLFF.nep_mods.nep_snes_trainer import train_snes
from src.PWMLFF.dp_param_extract import load_atomtype_energyshift_from_checkpoint
from src.user.input_param import InputParam
from utils.file_operation import write_arrays_to_file, write_force_ei

class nep_network:
    def __init__(self, nep_param:InputParam):
        self.input_param = nep_param
        # self.config = self.nep_params.get_dp_net_dict()
        self.davg_dstd_energy_shift = None # davg/dstd/energy_shift from training data
        torch.set_printoptions(precision = 12)
        if self.input_param.seed is not None:
            random.seed(self.input_param.seed)
            torch.manual_seed(self.input_param.seed)

        # if self.dp_params.hvd:
        #     hvd.init()
        #     self.dp_params.gpu = hvd.local_rank()

        if torch.cuda.is_available():
            if self.input_param.gpu:
                print("Use GPU: {} for training".format(self.input_param.gpu))
                self.device = torch.device("cuda:{}".format(self.input_param.gpu))
            else:
                self.device = torch.device("cuda")
        #elif torch.backends.mps.is_available():
        #    self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if self.input_param.precision == "float32":
            self.training_type = torch.float32  # training type is weights type
        else:
            self.training_type = torch.float64

        self.criterion = nn.MSELoss().to(self.device)

    def generate_data(self):    
        """
        Generate training data for MLFF model.

        Returns:
            list: list of labels path
        """
        raw_data_path = self.input_param.file_paths.raw_path
        datasets_path = os.path.join(self.input_param.file_paths.json_dir, self.input_param.file_paths.trainSetDir)
        train_ratio = self.input_param.train_valid_ratio
        train_data_path = self.input_param.file_paths.trainDataPath
        valid_data_path = self.input_param.file_paths.validDataPath
        labels_path = gen_train_data(train_ratio, raw_data_path, datasets_path, 
                               train_data_path, valid_data_path, 
                               self.input_param.valid_shuffle, self.input_param.seed, self.input_param.format)
        return labels_path
    
    '''
    description: 
        get energy shift and max atom numbers of image from inference model/loaded model/pwdata
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def _get_stat(self):
        # data_file_config = self.nep_params.get_data_file_dict()
        if self.input_param.inference:
            if self.input_param.nep_param.nep_txt_file is not None and os.path.exists(self.input_param.nep_param.nep_txt_file):
                atom_map = self.input_param.atom_type
                energy_shift = [1.0 for _ in atom_map] # just for init model, the bias will be replaced by nep.txt params
            elif self.input_param.file_paths.model_load_path is not None and os.path.exists(self.input_param.file_paths.model_load_path):
                # load davg, dstd from checkpoint of model
                atom_map, energy_shift = load_atomtype_energyshift_from_checkpoint(self.input_param.file_paths.model_load_path)
            elif os.path.exists(self.input_param.file_paths.model_save_path):
                atom_map, energy_shift = load_atomtype_energyshift_from_checkpoint(self.input_param.file_paths.model_save_path)
            else:
                raise Exception("Erorr! Loading model for inference can not find checkpoint: \
                                \nmodel load path: {} \n or model at work path: {}\n"\
                                .format(self.input_param.file_paths.model_load_path, self.input_param.file_paths.model_save_path))
            stat_add = [atom_map, energy_shift]
            # return energy_shift, None, None 
        else:
            stat_add = None
        
        if self.input_param.file_paths.datasets_path is None or len(self.input_param.file_paths.datasets_path) == 0:# for togpumd
            return energy_shift, 100, None
        energy_shift, max_atom_nums, image_path = get_stat(self.input_param, stat_add, self.input_param.file_paths.datasets_path, 
                         self.input_param.file_paths.json_dir, self.input_param.chunk_size)
        return energy_shift, max_atom_nums, image_path
    
    def load_data(self, energy_shift, max_atom_nums):
        config = self.input_param.get_data_file_dict()
        if self.input_param.inference:
            data_paths = []
            for data_path in self.input_param.file_paths.datasets_path:
                if os.path.exists(os.path.join(os.path.join(data_path, config['trainDataPath'], "position.npy"))):
                    data_paths.append(os.path.join(os.path.join(data_path, config['trainDataPath']))) #train dir
                if os.path.exists(os.path.join(os.path.join(data_path, config['validDataPath'], "position.npy"))):
                    data_paths.append(os.path.join(os.path.join(data_path, config['validDataPath']))) #valid dir
                if os.path.exists(os.path.join(data_path, "position.npy")) > 0: # add train or valid data
                    data_paths.append(data_path)
                else:# with out data
                    pass
            train_dataset = MovementDataset(data_paths, 
                                            config, self.input_param, energy_shift, max_atom_nums)
            valid_dataset = None
        else:           
            print(self.input_param.file_paths.datasets_path) 
            train_dataset = MovementDataset([os.path.join(_, config['trainDataPath']) for _ in self.input_param.file_paths.datasets_path], 
                                            config, self.input_param, energy_shift, max_atom_nums)
            valid_dataset = MovementDataset([os.path.join(_, config['validDataPath']) 
                                             for _ in self.input_param.file_paths.datasets_path
                                             if os.path.exists(os.path.join(_, config['validDataPath']))],
                                             config, self.input_param, energy_shift, max_atom_nums)

        energy_shift, atom_map = train_dataset.get_stat()

        # if self.input_param.hvd:
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(
        #         train_dataset, num_replicas=hvd.size(), rank=hvd.rank()
        #     )
        #     val_sampler = torch.utils.data.distributed.DistributedSampler(
        #         valid_dataset, num_replicas=hvd.size(), rank=hvd.rank(), drop_last=True
        #     )
        # else:
        train_sampler = None
        val_sampler = None

        # should add a collate function for padding
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.input_param.optimizer_param.batch_size,
            shuffle=self.input_param.data_shuffle,
            num_workers=self.input_param.workers,   
            pin_memory=True,
            sampler=train_sampler,
        )
        
        if self.input_param.inference:
            val_loader = None
        else:
            val_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=self.input_param.optimizer_param.batch_size,
                shuffle=False,
                num_workers=self.input_param.workers,
                pin_memory=True,
                sampler=val_sampler,
            )
        return energy_shift, atom_map, train_loader, val_loader
    
    '''
    description:
        if davg, dstd and energy_shift not from load_data, get it from model_load_file no use code
    return {*} 
    author: wuxingxing
    '''
    # def get_stat(self):
    #     if self.davg_dstd_energy_shift is None:
    #         if os.path.exists(self.input_param.file_paths.model_load_path) is False:
    #             raise Exception("ERROR! {} is not exist when get energy shift !".format(self.input_param.file_paths.model_load_path))
    #         davg_dstd_energy_shift = load_atomtype_energyshift_from_checkpoint(self.input_param.file_paths.model_load_path)
    #     else:
    #         davg_dstd_energy_shift = self.davg_dstd_energy_shift
    #     return davg_dstd_energy_shift
    
    def load_model_optimizer(self, energy_shift):
        # create model 
        # when running evaluation, nothing needs to be done with davg.npy
        
        model = NEP(self.input_param, energy_shift)
        model = model.to(self.training_type)

        # optionally resume from a checkpoint
        checkpoint = None
        if self.input_param.recover_train:
            if self.inference and \
                self.input_param.file_paths.model_load_path is not None and \
                    os.path.exists(self.input_param.file_paths.model_load_path): # recover from user input ckpt file for inference work
                model_path = self.input_param.file_paths.model_load_path
            else: # resume model specified by user
                if self.input_param.nep_param.model_wb is None:
                    model_path = self.input_param.file_paths.model_save_path  #recover from last training for training
                else:
                    model_path = None
            if model_path is not None and os.path.isfile(model_path):
                print("=> loading checkpoint '{}'".format(model_path))
                if not torch.cuda.is_available():
                    checkpoint = torch.load(model_path,map_location=torch.device('cpu') )
                elif self.input_param.gpu is None:
                    checkpoint = torch.load(model_path)
                elif torch.cuda.is_available():
                    # Map model to be loaded to specified single gpu.
                    loc = "cuda:{}".format(self.input_param.gpu)
                    checkpoint = torch.load(model_path, map_location=loc)
                # start afresh
                if self.input_param.optimizer_param.reset_epoch:
                    self.input_param.optimizer_param.start_epoch = 1
                else:
                    self.input_param.optimizer_param.start_epoch = checkpoint["epoch"] + 1
                model.load_state_dict(checkpoint["state_dict"])
                
                # scheduler.load_state_dict(checkpoint["scheduler"])
                print("=> loaded checkpoint '{}' (epoch {})"\
                      .format(model_path, checkpoint["epoch"]))
                if "compress" in checkpoint.keys():
                    model.set_comp_tab(checkpoint["compress"])
            else:
                if model_path is not None:
                    print("=> no checkpoint found at '{}'".format(model_path))

        if not torch.cuda.is_available():
            print("using CPU, this will be slow")
            '''
        elif self.input_param.hvd:
            if torch.cuda.is_available():
                if self.input_param.gpu is not None:
                    torch.cuda.set_device(self.input_param.gpu)
                    model.cuda(self.input_param.gpu)
                    self.input_param.optimizer_param.batch_size = int(self.input_param.optimizer_param.batch_size / hvd.size())
            '''
        elif self.input_param.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.input_param.gpu)
            model = model.cuda(self.input_param.gpu)
        else:
            model = model.cuda()
            # if model.compress_tab is not None:
            #     model.compress_tab.to(device=self.device)
        # optimizer, and learning rate scheduler
        if self.input_param.optimizer_param.opt_name == "LKF":
            optimizer = LKFOptimizer(
                model.parameters(),
                self.input_param.optimizer_param.kalman_lambda,
                self.input_param.optimizer_param.kalman_nue,
                self.input_param.optimizer_param.block_size
            )
        elif self.input_param.optimizer_param.opt_name == "GKF":
            optimizer = GKFOptimizer(
                model.parameters(),
                self.input_param.optimizer_param.kalman_lambda,
                self.input_param.optimizer_param.kalman_nue
            )
        elif self.input_param.optimizer_param.opt_name == "ADAM":
            optimizer = optim.Adam(model.parameters(), 
                                   self.input_param.optimizer_param.learning_rate)
        elif self.input_param.optimizer_param.opt_name == "SGD":
            optimizer = optim.SGD(
                model.parameters(), 
                self.input_param.optimizer_param.learning_rate,
                momentum=self.input_param.optimizer_param.momentum,
                weight_decay=self.input_param.optimizer_param.weight_decay
            )
        elif self.input_param.optimizer_param.opt_name == "SNES":
                optimizer = SNESOptimizer(model, self.input_param)
        else:
            raise Exception("Error: Unsupported optimizer!")
        
        q_scaler = None
        if checkpoint is not None and "q_scaler" in checkpoint.keys(): # from model ckpt file
            q_scaler = checkpoint["q_scaler"]

        if self.input_param.optimizer_param.opt_name in ["LKF", "GKF", "ADAM", "SGD"]:
            if checkpoint is not None and "optimizer" in checkpoint.keys():
                optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint is not None and self.input_param.optimizer_param.opt_name in ["LKF"] and "optimizer" in checkpoint.keys() and 'P' in checkpoint["optimizer"]['state'][0].keys():
                load_p = checkpoint["optimizer"]['state'][0]['P']
                optimizer.set_kalman_P(load_p, checkpoint["optimizer"]['state'][0]['kalman_lambda'])
        elif self.input_param.optimizer_param.opt_name in ["SNES"]:
            # model.set_nep_cparam(c2_param = checkpoint["c1_param"] , c3_param = checkpoint["c1_param"], q_scaler=checkpoint["q_scaler"])
            # elif self.input_param.nep_param.c2_param is not None: # from nep.txt NEP.txt没有训练的细节，不能直接训练，考虑后期从nep.restart取
            #     model.set_nep_cparam(c2_param = self.input_param.nep_param.c2_param, 
            #                         c3_param = self.input_param.nep_param.c3_param, 
            #                         q_scaler = self.input_param.nep_param.q_scaler)
            if checkpoint is not None and 'm' in checkpoint.keys():
                optimizer.load_m_s(checkpoint['m'], checkpoint['s'])
        
        model.set_nep_param_device(q_scaler)
        '''
        if self.input_param.hvd:
            # after hvd.DistributedOptimizer, the matrix P willed be reset to Identity matrix
            # its because hvd.DistributedOptimizer will initialize a new object of Optimizer Class
            optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=model.named_parameters()
            )
            # Broadcast parameters from rank 0 to all other processes.
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        '''
        # model.device = optimizer._state["P"][0].device
        # set params device
        return model, optimizer

    def inference(self):
        # do inference
        energy_shift, max_atom_nums, image_path = self._get_stat()
        energy_shift, atom_map, train_loader, val_loader = self.load_data(energy_shift, max_atom_nums)
        model, optimizer = self.load_model_optimizer(energy_shift)
        start = time.time()
        res_pd, etot_label_list, etot_predict_list, ei_label_list, ei_predict_list, force_label_list, force_predict_list\
        = predict(train_loader, model, self.criterion, self.device, self.input_param)
        end = time.time()
        print("fitting time:", end - start, 's')

        # print infos
        inference_cout = ""
        inference_cout += "For {} images: \n".format(res_pd.shape[0])
        inference_cout += "Avarage REMSE of Etot: {} \n".format(res_pd['RMSE_Etot'].mean())
        inference_cout += "Avarage REMSE of Etot per atom: {} \n".format(res_pd['RMSE_Etot_per_atom'].mean())
        inference_cout += "Avarage REMSE of Ei: {} \n".format(res_pd['RMSE_Ei'].mean())
        inference_cout += "Avarage REMSE of RMSE_F: {} \n".format(res_pd['RMSE_F'].mean())
        if self.input_param.optimizer_param.train_egroup:
            inference_cout += "Avarage REMSE of RMSE_Egroup: {} \n".format(res_pd['RMSE_Egroup'].mean())
        if self.input_param.optimizer_param.train_virial:
            inference_cout += "Avarage REMSE of RMSE_virial: {} \n".format(res_pd['RMSE_virial'].mean())
            inference_cout += "Avarage REMSE of RMSE_virial_per_atom: {} \n".format(res_pd['RMSE_virial_per_atom'].mean())
        
        inference_cout += "\nMore details can be found under the file directory:\n{}\n".format(os.path.realpath(self.input_param.file_paths.test_dir))
        print(inference_cout)

        inference_path = self.input_param.file_paths.test_dir
        if os.path.exists(inference_path) is False:
            os.makedirs(inference_path)
        write_arrays_to_file(os.path.join(inference_path, "image_atom_nums.txt"), [int(len(_)/3) for _ in force_predict_list])
        write_arrays_to_file(os.path.join(inference_path, "dft_total_energy.txt"), etot_label_list)
        write_arrays_to_file(os.path.join(inference_path, "inference_total_energy.txt"), etot_predict_list)
        # for force
        write_force_ei(os.path.join(inference_path, "inference_ei_force.txt"), ei_predict_list, force_predict_list)
        write_force_ei(os.path.join(inference_path, "dft_ei_force.txt"), ei_label_list, force_label_list)

        # write_arrays_to_file(os.path.join(inference_path, "dft_atomic_energy.txt"), ei_label_list)
        # write_arrays_to_file(os.path.join(inference_path, "inference_atomic_energy.txt"), ei_predict_list)

        # write_arrays_to_file(os.path.join(inference_path, "dft_force.txt"), force_label_list)
        # write_arrays_to_file(os.path.join(inference_path, "inference_force.txt"), force_predict_list)

        res_pd.to_csv(os.path.join(inference_path, "inference_loss.csv"))

        with open(os.path.join(inference_path, "inference_summary.txt"), 'w') as wf:
            wf.writelines(inference_cout)
        return 
                
    def train(self):
        energy_shift, max_atom_nums, image_path = self._get_stat()
        energy_shift, atom_map, train_loader, val_loader = self.load_data(energy_shift, max_atom_nums)
         #energy_shift is same as energy_shift of upper; atom_map is the user input order
        model, optimizer = self.load_model_optimizer(energy_shift)
        # self.convert_to_gpumd(model) 
        if self.input_param.optimizer_param.opt_name == "SNES":
            self.input_param.optimizer_param.epochs = int(self.input_param.optimizer_param.generation/len(train_loader))
            print("Automatically adjust 'epochs' to {} based on population size (image nums {} generation {})"\
                .format(self.input_param.optimizer_param.epochs, len(train_loader), self.input_param.optimizer_param.generation))

        if not os.path.exists(self.input_param.file_paths.model_store_dir):
            os.makedirs(self.input_param.file_paths.model_store_dir)
        # if self.nep_params.model_num == 1:
        #     smlink_file(self.nep_params.file_paths.model_store_dir, os.path.join(self.nep_params.file_paths.json_dir, os.path.basename(self.nep_params.file_paths.model_store_dir)))
        
        # if not self.dp_params.hvd or (self.dp_params.hvd and hvd.rank() == 0):
        train_log = os.path.join(self.input_param.file_paths.model_store_dir, "epoch_train.dat")
        f_train_log = open(train_log, "w")
        valid_log = os.path.join(self.input_param.file_paths.model_store_dir, "epoch_valid.dat")
        f_valid_log = open(valid_log, "w")
        # Define the lists based on the training type
        train_lists = ["epoch", "loss"]
        valid_lists = ["epoch", "loss"]
        if self.input_param.optimizer_param.train_energy:
            # train_lists.append("RMSE_Etot")
            # valid_lists.append("RMSE_Etot")
            train_lists.append("RMSE_Etot_per_atom")
            valid_lists.append("RMSE_Etot_per_atom")
        if self.input_param.optimizer_param.train_ei:
            train_lists.append("RMSE_Ei")
            valid_lists.append("RMSE_Ei")
        if self.input_param.optimizer_param.train_egroup:
            train_lists.append("RMSE_Egroup")
            valid_lists.append("RMSE_Egroup")
        if self.input_param.optimizer_param.train_force:
            train_lists.append("RMSE_F")
            valid_lists.append("RMSE_F")
        if self.input_param.optimizer_param.train_virial:
            train_lists.append("RMSE_virial_per_atom")
            valid_lists.append("RMSE_virial_per_atom")
        if self.input_param.optimizer_param.opt_name == "SNES":
            train_lists.append("Loss_l1")
            train_lists.append("Loss_l2")
            # valid_lists.append("Loss_l1")
            # valid_lists.append("Loss_l2")
        if self.input_param.optimizer_param.opt_name == "LKF" or self.input_param.optimizer_param.opt_name == "GKF":
            train_lists.extend(["time"])
        else:
            train_lists.extend(["real_lr", "time"])

        train_print_width = {
            "epoch": 5,
            "loss": 18,
            "RMSE_Etot": 18,
            "RMSE_Etot_per_atom": 21,
            "RMSE_Ei": 18,
            "RMSE_Egroup": 18,
            "RMSE_F": 18,
            "RMSE_virial": 18,
            "RMSE_virial_per_atom": 23,
            "Loss_l1": 18,
            "Loss_l2": 18,
            "real_lr": 18,
            "time": 12,
        }

        train_format = "".join(["%{}s".format(train_print_width[i]) for i in train_lists])
        valid_format = "".join(["%{}s".format(train_print_width[i]) for i in valid_lists])

        f_train_log.write("%s\n" % (train_format % tuple(train_lists)))
        f_valid_log.write("%s\n" % (valid_format % tuple(valid_lists)))
        # Sij_max = 0 # this is for dp model do type embedding
        # self.convert_to_gpumd(model)
        for epoch in range(self.input_param.optimizer_param.start_epoch, self.input_param.optimizer_param.epochs + 1):
            # if self.dp_params.hvd: # this code maybe error, check when add multi GPU training. wu
            #     self.train_sampler.set_epoch(epoch)

            # train for one epoch
            time_start = time.time()
            if self.input_param.optimizer_param.opt_name == "LKF" or self.input_param.optimizer_param.opt_name == "GKF":
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom = train_KF(
                    train_loader, model, self.criterion, optimizer, epoch, self.device, self.input_param
                )

            elif self.input_param.optimizer_param.opt_name == "SNES":
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, loss_l1, loss_l2 = train_snes(
                    train_loader, model, self.criterion, optimizer, epoch, self.device, self.input_param
                )
            else:
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, real_lr = train(
                    train_loader, model, self.criterion, optimizer, epoch, \
                        self.input_param.optimizer_param.learning_rate, self.device, self.input_param
                )
            time_end = time.time()
            # self.convert_to_gpumd(model)

            # evaluate on validation set
            vld_loss, vld_loss_Etot, vld_loss_Etot_per_atom, vld_loss_Force, vld_loss_Ei, val_loss_egroup, val_loss_virial, val_loss_virial_per_atom = valid(
                val_loader, model, self.criterion, self.device, self.input_param
            )

            # if not self.dp_params.hvd or (self.dp_params.hvd and hvd.rank() == 0):

            f_train_log = open(train_log, "a")
            f_valid_log = open(valid_log, "a")

            # Write the log line to the file based on the training mode
            train_log_line = "%5d%18.10e" % (
                epoch,
                loss,
            )
            valid_log_line = "%5d%18.10e" % (
                epoch,
                vld_loss,
            )

            if self.input_param.optimizer_param.train_energy:
                # train_log_line += "%18.10e" % (loss_Etot)
                # valid_log_line += "%18.10e" % (vld_loss_Etot)
                train_log_line += "%21.10e" % (loss_Etot_per_atom)
                valid_log_line += "%21.10e" % (vld_loss_Etot_per_atom)
            if self.input_param.optimizer_param.train_ei:
                train_log_line += "%18.10e" % (loss_Ei)
                valid_log_line += "%18.10e" % (vld_loss_Ei)
            if self.input_param.optimizer_param.train_egroup:
                train_log_line += "%18.10e" % (loss_egroup)
                valid_log_line += "%18.10e" % (val_loss_egroup)
            if self.input_param.optimizer_param.train_force:
                train_log_line += "%18.10e" % (loss_Force)
                valid_log_line += "%18.10e" % (vld_loss_Force)
            if self.input_param.optimizer_param.train_virial:
                # train_log_line += "%18.10e" % (loss_virial)
                # valid_log_line += "%18.10e" % (val_loss_virial)
                train_log_line += "%23.10e" % (loss_virial_per_atom)
                valid_log_line += "%23.10e" % (val_loss_virial_per_atom)
            if self.input_param.optimizer_param.opt_name == "SNES": 
                train_log_line += "%18.10e" % (loss_l1)
                train_log_line += "%18.10e" % (loss_l2)
            if self.input_param.optimizer_param.opt_name == "LKF" or self.input_param.optimizer_param.opt_name == "GKF":
                train_log_line += "%10.4f" % (time_end - time_start)
            else:
                train_log_line += "%18.10e%10.4f" % (-1 , time_end - time_start)

            f_train_log.write("%s\n" % (train_log_line))
            f_valid_log.write("%s\n" % (valid_log_line))
        
            f_train_log.close()
            f_valid_log.close()
            
            # if not self.dp_params.hvd or (self.dp_params.hvd and hvd.rank() == 0):
            if self.input_param.optimizer_param.opt_name == "SNES":
                save_checkpoint(
                    {
                    "json_file":self.input_param.to_dict(),
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "energy_shift":energy_shift,
                    "atom_type_order": atom_map,    #atom type order of davg/dstd/energy_shift, the user input order
                    "m":optimizer.get_m(),
                    "s": optimizer.get_s(),
                    "q_scaler": model.get_q_scaler(),
                    "optimizer":optimizer.state_dict()                        
                    },
                    self.input_param.file_paths.model_name,
                    self.input_param.file_paths.model_store_dir,                    
                )

            elif self.input_param.optimizer_param.opt_name in ["LKF", "GKF"] and \
                self.input_param.file_paths.save_p_matrix:
                save_checkpoint(
                    {
                    "json_file":self.input_param.to_dict(),
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "energy_shift":energy_shift,
                    "atom_type_order": atom_map,    #atom type order of davg/dstd/energy_shift, the user input order
                    # "sij_max":Sij_max,
                    "q_scaler": model.get_q_scaler(),
                    "optimizer":optimizer.state_dict()
                    },
                    self.input_param.file_paths.model_name,
                    self.input_param.file_paths.model_store_dir,
                )
            else: # ADAM or SGD optimizer
                save_checkpoint(
                    {
                    "json_file":self.input_param.to_dict(),
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "energy_shift":energy_shift,
                    "q_scaler": model.get_q_scaler(),
                    "atom_type_order": atom_map,    #atom type order of davg/dstd/energy_shift
                    "optimizer":optimizer.state_dict()
                    # "sij_max":Sij_max
                    },
                    self.input_param.file_paths.model_name,
                    self.input_param.file_paths.model_store_dir,
                )
            self.convert_to_gpumd(model)
            

    def convert_to_gpumd(self, model:NEP, save_dir:str = None):
        model_content = self.input_param.nep_param.to_nep_in_txt()
        train_content = self.input_param.optimizer_param.snes_to_nep_txt()
        model_content += train_content
        if save_dir is None:
            save_nep_in_path = os.path.join(self.input_param.file_paths.model_store_dir, self.input_param.file_paths.nep_in_file)
            save_nep_txt_path = os.path.join(self.input_param.file_paths.model_store_dir, self.input_param.file_paths.nep_model_file)
        else:
            save_nep_in_path = os.path.join(save_dir, self.input_param.file_paths.nep_in_file)
            save_nep_txt_path = os.path.join(save_dir, self.input_param.file_paths.nep_model_file)            
        # extract parameters
        txt_head = self.input_param.nep_param.to_nep_txt()
        txt_body = model.get_nn_params()
        txt_body_str = "\n".join(map(str, txt_body))
        txt_head += txt_body_str

        with open(save_nep_in_path, 'w') as wf:
            wf.writelines(model_content)

        with open(save_nep_txt_path, 'w') as wf:
            wf.writelines(txt_head)
        print("Successfully convert to nep.in and nep.txt file.")        

    def evaluate(self,num_thread = 1):
        """
            evaluate a model against AIMD
            put a MOVEMENT in /MD and run MD100 
        """
        if not os.path.exists("MD/MOVEMENT"):
            raise Exception("MD/MOVEMENT not found")
        import md100
        md100.run_md100(imodel = 5, atom_type = self.input_param.atom_type, num_process = num_thread) 
        
    def plot_evaluation(self):
        if not os.path.exists("MOVEMENT"):
            raise Exception("MOVEMENT not found. It should be force field MD result")
        import src.aux.plot_evaluation as plot_evaluation
        plot_evaluation.plot()

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
        f.write('%d\n' % len(self.input_param.atom_type)) 
        
        for i in range(len(self.input_param.atom_type)):
            f.write('%d %d\n' % (self.input_param.atom_type[i], 2*self.input_param.atom_type[i]))
        f.close()    
        
        # creating md.input for main_MD.x 
        command = r'mpirun -n ' + str(num_thread) + r' main_MD.x'
        print (command)
        subprocess.run(command, shell=True) 

