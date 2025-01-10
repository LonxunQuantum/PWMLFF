import os
import pandas as pd
import numpy as np
import time
from enum import Enum
import torch
from torch.utils.data import Subset
from torch.autograd import Variable
from src.loss.dploss import dp_loss, adjust_lr
from src.optimizer.KFWrapper import KFOptimizerWrapper
# import horovod.torch as hvd
from torch.profiler import profile, record_function, ProfilerActivity
from src.user.input_param import InputParam
from utils.debug_operation import check_cuda_memory
from collections import defaultdict

# abandon this function
def get_ri_rid_by_cutoff(
    Ri:torch.Tensor,
    list_neigh :torch.Tensor, 
    list_neigh_type :torch.Tensor, 
    cutoff: float,
    max_neighbor_type:int
    ):
    # calculate ri_d
    mask = Ri[:, :, :, 0].abs() > 1e-5
    Ri_d = torch.zeros(Ri.shape[0], Ri.shape[1], max_neighbor_type, 4, 3, dtype=Ri.dtype, device=Ri.device)
    Ri_d[:, :, :, 0, 0][mask] = Ri[:, :, :, 1][mask] / Ri[:, :, :, 0][mask]
    Ri_d[:, :, :, 1, 0][mask] = 1
    # dy
    Ri_d[:, :, :, 0, 1][mask] = Ri[:, :, :, 2][mask] / Ri[:, :, :, 0][mask]
    Ri_d[:, :, :, 2, 1][mask] = 1
    # dz
    Ri_d[:, :, :, 0, 2][mask] = Ri[:, :, :, 3][mask] / Ri[:, :, :, 0][mask]
    Ri_d[:, :, :, 3, 2][mask] = 1 

    # 1. Ri 的第 0 列元素如果大于 cutoff 就将整行置 0
    ri_new = Ri.clone().detach()
    mask = (Ri[:, :, :, 0] > cutoff)
    ri_new[mask] = 0
    # ri_new.requires_grad_()
    # 2. 创建 ri_d：对于 ri_new 中整行置 0 的位置，对应的 Ri_d 中的元素置 0
    ri_d_new = Ri_d.clone().detach()
    ri_d_new[mask] = 0

    # 3. 创建 neigh：对于 ri_new 中整行置 0 的位置，对应的 neigh 中的元素置 0
    neigh_new = list_neigh.clone().detach()
    neigh_new[mask] = 0

    # 4. 创建 type：对于 ri_new 中整行置 0 的位置，对应的 type 中的元素置 -1
    type_new = list_neigh_type.clone().detach()
    type_new[mask] = -1
    return Ri_d, ri_new, ri_d_new, neigh_new, type_new


def print_l1_l2(model):
    params = model.parameters()
    dtype = next(params).dtype
    device = next(params).device
    L1 = torch.tensor(0.0, device=device, dtype=dtype).detach().requires_grad_(False)
    L2 = torch.tensor(0.0, device=device, dtype=dtype).detach().requires_grad_(False)
    nums_param = 0
    for p in params:
        L1 += torch.sum(torch.abs(p))
        L2 += torch.sum(p**2)
        nums_param += p.nelement()
    L1 = L1 / nums_param
    L2 = L2 / nums_param
    return L1, L2

def train(train_loader, model, criterion, optimizer, epoch, start_lr, device, args:InputParam):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    loss_Etot = AverageMeter("Etot", ":.4e", Summary.ROOT)
    loss_Etot_per_atom = AverageMeter("Etot_per_atom", ":.4e", Summary.ROOT)
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_Virial = AverageMeter("Virial", ":.4e", Summary.ROOT)
    loss_Virial_per_atom = AverageMeter("Virial_per_atom", ":.4e", Summary.ROOT)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT)
    loss_Egroup = AverageMeter("Egroup", ":.4e", Summary.ROOT)
    loss_L1 = AverageMeter("Loss_L1", ":.4e", Summary.ROOT)
    loss_L2 = AverageMeter("Loss_L2", ":.4e", Summary.ROOT)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_L1, loss_L2, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup, loss_Virial, loss_Virial_per_atom],
        prefix="Epoch: [{}]".format(epoch),
    )
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, sample_batches in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        nr_batch_sample = sample_batches["Ei"].shape[0]
        global_step = (epoch - 1) * len(train_loader) + i * nr_batch_sample
        real_lr = adjust_lr(global_step, start_lr, \
                            args.optimizer_param.stop_step, args.optimizer_param.decay_step, args.optimizer_param.stop_lr) #  stop_step, decay_step

        for param_group in optimizer.param_groups:
            param_group["lr"] = real_lr * (nr_batch_sample**0.5)

        if args.precision == "float64":
            Ei_label_cpu = sample_batches["Ei"].double()
            Etot_label_cpu = sample_batches["Etot"].double()
            Force_label_cpu = sample_batches["Force"][:, :, :].double()
            # Sij_max_cpu = sample_batches["max_ri"].double()

            if args.optimizer_param.train_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].double()
                Divider_cpu = sample_batches["Divider"].double()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].double()

            if args.optimizer_param.train_virial is True:
                Virial_label_cpu = sample_batches["Virial"].double()

            ImageDR_cpu = sample_batches["ImageDR"].double()
            ImageDR_angular_cpu = sample_batches["ImageDRAngular"].double()
            # Ri_cpu = sample_batches["Ri"].double()
            # Ri_d_cpu = sample_batches["Ri_d"].double()

        elif args.precision == "float32":
            Ei_label_cpu = sample_batches["Ei"].float()
            Etot_label_cpu = sample_batches["Etot"].float()
            Force_label_cpu = sample_batches["Force"][:, :, :].float()
            # Sij_max_cpu = sample_batches["max_ri"].float()

            if args.optimizer_param.train_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].float()
                Divider_cpu = sample_batches["Divider"].float()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].float()

            if args.optimizer_param.train_virial is True:
                Virial_label_cpu = sample_batches["Virial"].float()

            ImageDR_cpu = sample_batches["ImageDR"].float()
            ImageDR_angular_cpu = sample_batches["ImageDRAngular"].float()
            # Ri_cpu = sample_batches["Ri"].float()
            # Ri_d_cpu = sample_batches["Ri_d"].float()
        else:
            raise Exception("Error! Please specify floating point type: float32 or float64 by the parameter --datatype! ")
        
        # if max(Sij_max_cpu) > Sij_max:
        #     Sij_max = max(Sij_max_cpu)

        dR_neigh_list_cpu = sample_batches["ListNeighbor"].int()
        dR_neigh_type_list_cpu = sample_batches["ListNeighborType"].int()
        dR_neigh_list_angular_cpu = sample_batches["ListNeighborAngular"].int()
        dR_neigh_type_angular_cpu = sample_batches["ListNeighborTypeAngular"].int()
        natoms_img_cpu = sample_batches["ImageAtomNum"].int()
        atom_type_cpu = sample_batches["AtomType"].int()
        atom_type_map_cpu = sample_batches["AtomTypeMap"].int()
        # classify batchs according to their atom type and atom nums
        batch_clusters = _classify_batchs(atom_type_map_cpu, len(args.atom_type))

        for batch_indexs in batch_clusters:
            # transport data to GPU
            natoms_img = Variable(natoms_img_cpu[batch_indexs].int().to(device))
            natoms = natoms_img[0]
            
            dR_neigh_list = Variable(dR_neigh_list_cpu[batch_indexs, :natoms].int().to(device))
            dR_neigh_type_list = Variable(dR_neigh_type_list_cpu[batch_indexs, :natoms].int().to(device))
            dR_neigh_list_angular = Variable(dR_neigh_list_angular_cpu[batch_indexs, :natoms].int().to(device))
            dR_neigh_type_list_angular = Variable(dR_neigh_type_angular_cpu[batch_indexs, :natoms].int().to(device))
            # atom list of image
            atom_type = Variable(atom_type_cpu[batch_indexs].to(device))
            atom_type_map = Variable(atom_type_map_cpu[batch_indexs, :natoms].to(device))
            Ei_label = Variable(Ei_label_cpu[batch_indexs, :natoms].to(device))
            Etot_label = Variable(Etot_label_cpu[batch_indexs].to(device))
            Force_label = Variable(Force_label_cpu[batch_indexs, :natoms].to(device))  # [40,108,3]

            if args.optimizer_param.train_egroup is True:
                Egroup_label = Variable(Egroup_label_cpu[batch_indexs, :natoms].to(device))
                Divider = Variable(Divider_cpu[batch_indexs, :natoms].to(device))
                Egroup_weight = Variable(Egroup_weight_cpu[batch_indexs, :natoms, :natoms].to(device))

            if args.optimizer_param.train_virial is True:
                Virial_label = Variable(Virial_label_cpu[batch_indexs].to(device))
            
            ImageDR = Variable(ImageDR_cpu[batch_indexs, :natoms].to(device))
            ImageDR_angular = Variable(ImageDR_angular_cpu[batch_indexs, :natoms].to(device))
            
            # Ri = Variable(Ri_cpu[batch_indexs, :natoms].to(device), requires_grad=True)
            # Ri_d = Variable(Ri_d_cpu[batch_indexs, :natoms].to(device))

            batch_size = len(batch_indexs)

            if args.optimizer_param.train_egroup is True:
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                    dR_neigh_list, ImageDR, dR_neigh_type_list, \
                        dR_neigh_list_angular, ImageDR_angular, dR_neigh_type_list_angular, \
                        atom_type_map[0], atom_type[0], 0, Egroup_weight, Divider)
            else:
                # atom_type_map: we only need the first element, because it is same for each image of MOVEMENT
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                    dR_neigh_list, ImageDR, dR_neigh_type_list, \
                        dR_neigh_list_angular, ImageDR_angular, dR_neigh_type_list_angular, \
                            atom_type_map[0], atom_type[0],  0, None, None)
                    
            optimizer.zero_grad()

            loss_F_val = criterion(Force_predict, Force_label)
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_Etot_per_atom_val = loss_Etot_val/natoms/natoms
            loss_Ei_val = criterion(Ei_predict, Ei_label)
            # loss_Ei_val = 0

            #print ("Etot_predict") 
            #print (Etot_predict)
            
            #print ("Force_predict")
            #print (Force_predict)
            
            #print ("Virial_predict")
            #print (Virial_predict)
            #Ei_predict, Force_predict, Egroup_predict, Virial_predict)
            #print("Egroup_predict",Egroup_predict)

            #print("Egroup_label",Egroup_label)
            if args.optimizer_param.train_egroup is True:
                loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
            if args.optimizer_param.train_virial is True:
                # loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))  #115.415137283393
                data_mask = Virial_label[:, 9] > 0  # 判断最后一列是否大于 0
                _Virial_label = Virial_label[:, :9][data_mask]
                if data_mask.any().item():
                    loss_Virial_val = criterion(Virial_predict[data_mask], _Virial_label)
                else:
                    loss_Virial_val = torch.tensor(0.0)
                loss_Virial_per_atom_val = loss_Virial_val/natoms/natoms

            loss_val = torch.zeros_like(loss_F_val)
            
            w_f, w_e, w_v, w_eg, w_ei = 0, 0, 0, 0, 0

            if args.optimizer_param.train_force is True:
                w_f = 1.0 
                loss_val += loss_F_val
            
            if args.optimizer_param.train_energy is True:
                w_e = 1.0
                loss_val += loss_Etot_val
            
            if args.optimizer_param.train_virial is True:
                w_v = 1.0 
                loss_val += loss_Virial_val
            
            if args.optimizer_param.train_egroup is True:
                w_eg = 1.0 
                loss_val += loss_Egroup_val

            if args.optimizer_param.train_egroup is True and args.optimizer_param.train_virial is True:
                loss, _, _ = dp_loss(
                    args,
                    0.001,
                    real_lr,
                    1,
                    w_f,
                    loss_F_val,
                    w_e,
                    loss_Etot_val,
                    w_v,
                    loss_Virial_val,
                    w_eg,
                    loss_Egroup_val,
                    w_ei,
                    loss_Ei_val,
                    natoms_img[0].item(),
                )
            elif args.optimizer_param.train_egroup is True and args.optimizer_param.train_virial is False:
                loss, _, _ = dp_loss(
                    args,
                    0.001,
                    real_lr,
                    2,
                    w_f,
                    loss_F_val,
                    w_e,
                    loss_Etot_val,
                    w_eg,
                    loss_Egroup_val,
                    w_ei,
                    loss_Ei_val,
                    natoms_img[0].item(),
                )
            elif args.optimizer_param.train_egroup is False and args.optimizer_param.train_virial is True:
                loss, _, _ = dp_loss(
                    args,
                    0.001,
                    real_lr,
                    3,
                    w_f,
                    loss_F_val,
                    w_e,
                    loss_Etot_val,
                    w_v,
                    loss_Virial_val,
                    w_ei,
                    loss_Ei_val,
                    natoms_img[0].item(),
                )
            else:
                loss, _, _ = dp_loss(
                    args,
                    0.001,
                    real_lr,
                    4,
                    w_f,
                    loss_F_val,
                    w_e,
                    loss_Etot_val,
                    w_ei,
                    loss_Ei_val,
                    natoms_img[0].item(),
                )
            # import ipdb;ipdb.set_trace()
            loss.backward()
            optimizer.step()

            L1, L2 = print_l1_l2(model)
            if args.optimizer_param.lambda_2 is not None:
                loss_val += L2

            # measure accuracy and record loss
            losses.update(loss_val.item(), batch_size)
            loss_Etot.update(loss_Etot_val.item(), batch_size)
            loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
            loss_Ei.update(loss_Ei_val.item(), batch_size)
            
            loss_L1.update(L1.item(), batch_size)
            loss_L2.update(L2.item(), batch_size)

            if args.optimizer_param.train_egroup is True:
                loss_Egroup.update(loss_Egroup_val.item(), batch_size)
            if args.optimizer_param.train_virial is True:
                loss_Virial.update(loss_Virial_val.item(), _Virial_label.shape[0])
                loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), _Virial_label.shape[0])
            loss_Force.update(loss_F_val.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.optimizer_param.print_freq == 0:
            progress.display(i + 1)

    progress.display_summary(["Training Set:"])
    return (
        losses.avg,
        loss_Etot.root,
        loss_Etot_per_atom.root,
        loss_Force.root,
        loss_Ei.root,
        loss_Egroup.root,
        loss_Virial.root,
        loss_Virial_per_atom.root,
        real_lr,
        loss_L1.root,
        loss_L2.root
        # Sij_max,    
    )

def train_KF(train_loader, model, criterion, optimizer, epoch, device, args:InputParam):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    loss_Etot = AverageMeter("Etot", ":.4e", Summary.ROOT)
    loss_Etot_per_atom = AverageMeter("Etot_per_atom", ":.4e", Summary.ROOT)
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT)
    loss_Egroup = AverageMeter("Egroup", ":.4e", Summary.ROOT)
    loss_Virial = AverageMeter("Virial", ":.4e", Summary.ROOT)
    loss_Virial_per_atom = AverageMeter("Virial_per_atom", ":.4e", Summary.ROOT)
    loss_L1 = AverageMeter("Loss_L1", ":.4e", Summary.ROOT)
    loss_L2 = AverageMeter("Loss_L2", ":.4e", Summary.ROOT)    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_L1, loss_L2, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup, loss_Virial, loss_Virial_per_atom],
        prefix="Epoch: [{}]".format(epoch),
    )

    KFOptWrapper = KFOptimizerWrapper(
        model, optimizer, args.optimizer_param.nselect, args.optimizer_param.groupsize, lambda_l1 = args.optimizer_param.lambda_1, lambda_l2 = args.optimizer_param.lambda_2
    )
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, sample_batches in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # load data to cpu
        if args.precision == "float64":
            Ei_label_cpu    = sample_batches["Ei"].double()
            Etot_label_cpu  = sample_batches["Etot"].double()
            Force_label_cpu = sample_batches["Force"][:, :, :].double()
            # position_cpu    = sample_batches["Position"][:, :, :].double()
            if args.optimizer_param.train_egroup is True:
                Egroup_label_cpu  = sample_batches["Egroup"].double()
                Divider_cpu       = sample_batches["Divider"].double()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].double()

            if args.optimizer_param.train_virial is True:
                Virial_label_cpu  = sample_batches["Virial"].double()
            ImageDR_cpu = sample_batches["ImageDR"].double()
            ImageDR_angular_cpu = sample_batches["ImageDRAngular"].double()

        elif args.precision == "float32":
            Ei_label_cpu    = sample_batches["Ei"].float()
            Etot_label_cpu  = sample_batches["Etot"].float()
            Force_label_cpu = sample_batches["Force"][:, :, :].float()
            if args.optimizer_param.train_egroup is True:
                Egroup_label_cpu  = sample_batches["Egroup"].float()
                Divider_cpu       = sample_batches["Divider"].float()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].float()

            if args.optimizer_param.train_virial is True:
                Virial_label_cpu  = sample_batches["Virial"].float()
            ImageDR_cpu = sample_batches["ImageDR"].float()
            ImageDR_angular_cpu = sample_batches["ImageDRAngular"].float()
        else:
            raise Exception("Error! Please specify floating point type: float32 or float64 by the parameter --datatype! ")
        
        dR_neigh_list_cpu = sample_batches["ListNeighbor"].int()
        dR_neigh_type_list_cpu = sample_batches["ListNeighborType"].int()
        dR_neigh_list_angular_cpu = sample_batches["ListNeighborAngular"].int()
        dR_neigh_type_angular_cpu = sample_batches["ListNeighborTypeAngular"].int()
        natoms_img_cpu    = sample_batches["ImageAtomNum"].int()
        atom_type_cpu     = sample_batches["AtomType"].int()
        atom_type_map_cpu = sample_batches["AtomTypeMap"].int()
        # classify batchs according to their atom type and atom nums
        batch_clusters    = _classify_batchs(atom_type_map_cpu, len(args.atom_type))
        
        for batch_indexs in batch_clusters:
            # transport data to GPU
            natoms_img = Variable(natoms_img_cpu[batch_indexs].int().to(device))
            natoms     = natoms_img[0]
            
            dR_neigh_list = Variable(dR_neigh_list_cpu[batch_indexs, :natoms].int().to(device))
            dR_neigh_type_list = Variable(dR_neigh_type_list_cpu[batch_indexs, :natoms].int().to(device))
            dR_neigh_list_angular = Variable(dR_neigh_list_angular_cpu[batch_indexs, :natoms].int().to(device))
            dR_neigh_type_list_angular = Variable(dR_neigh_type_angular_cpu[batch_indexs, :natoms].int().to(device))
            # atom list of image
            atom_type     = Variable(atom_type_cpu[batch_indexs].to(device))
            atom_type_map = Variable(atom_type_map_cpu[batch_indexs, :natoms].to(device))
            # print("atom_type_map ", atom_type)
            Ei_label      = Variable(Ei_label_cpu[batch_indexs, :natoms].to(device))
            Etot_label    = Variable(Etot_label_cpu[batch_indexs].to(device))
            Force_label   = Variable(Force_label_cpu[batch_indexs, :natoms].to(device))  # [40,108,3]
            # position      = Variable(position_cpu[batch_indexs, :natoms].to(device))
            if args.optimizer_param.train_egroup is True:
                Egroup_label = Variable(Egroup_label_cpu[batch_indexs, :natoms].to(device))
                Divider = Variable(Divider_cpu[batch_indexs, :natoms].to(device))
                Egroup_weight = Variable(Egroup_weight_cpu[batch_indexs, :natoms, :natoms].to(device))

            if args.optimizer_param.train_virial is True:
                Virial_label = Variable(Virial_label_cpu[batch_indexs].to(device))
            ImageDR = Variable(ImageDR_cpu[batch_indexs, :natoms].to(device))
            ImageDR_angular = Variable(ImageDR_angular_cpu[batch_indexs, :natoms].to(device))
            # Ri = Variable(Ri_cpu[batch_indexs, :natoms].to(device), requires_grad=True)
            # Ri_d = Variable(Ri_d_cpu[batch_indexs, :natoms].to(device))

            batch_size = len(batch_indexs)
            if args.optimizer_param.train_egroup is True:
                kalman_inputs = [dR_neigh_list, ImageDR, dR_neigh_type_list, \
                                    dR_neigh_list_angular, ImageDR_angular, dR_neigh_type_list_angular, \
                                        atom_type_map[0], atom_type[0], 0, Egroup_weight, Divider]
            else:
                # atom_type_map: we only need the first element, because it is same for each image of MOVEMENT
                kalman_inputs = [dR_neigh_list, ImageDR, dR_neigh_type_list, \
                                    dR_neigh_list_angular, ImageDR_angular, dR_neigh_type_list_angular, \
                                        atom_type_map[0], atom_type[0],  0, None, None]

            if args.optimizer_param.train_virial is True:
                Virial_predict = KFOptWrapper.update_virial(kalman_inputs, Virial_label, args.optimizer_param.pre_fac_virial, train_type = "NEP")
            if args.optimizer_param.train_energy is True: 
                # check_cuda_memory(-1, -1, "update_energy start")
                Etot_predict = KFOptWrapper.update_energy(kalman_inputs, Etot_label, args.optimizer_param.pre_fac_etot, train_type = "NEP")
                # check_cuda_memory(-1, -1, "update_energy end")
            if args.optimizer_param.train_ei is True:
                Ei_predict = KFOptWrapper.update_ei(kalman_inputs, Ei_label, args.optimizer_param.pre_fac_ei, train_type = "NEP")

            if args.optimizer_param.train_egroup is True:
                Egroup_predict = KFOptWrapper.update_egroup(kalman_inputs, Egroup_label, args.optimizer_param.pre_fac_egroup, train_type = "NEP")

            if args.optimizer_param.train_force is True:
                # check_cuda_memory(-1, -1, "update_force start")
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = KFOptWrapper.update_force(
                    kalman_inputs, Force_label, args.optimizer_param.pre_fac_force, train_type = "NEP")
                # check_cuda_memory(-1, -1, "update_force end")
            # Force_predict = Force_label
            # Ei_predict = Ei_label
            loss_F_val = criterion(Force_predict, Force_label)
            L1, L2 = print_l1_l2(model)

            # divide by natoms 
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_Etot_per_atom_val = loss_Etot_val/natoms/natoms
            
            loss_Ei_val = criterion(Ei_predict, Ei_label)   
            if args.optimizer_param.train_egroup is True:
                loss_Egroup_val = criterion(Egroup_predict, Egroup_label)

            if args.optimizer_param.train_virial is True:
                data_mask = Virial_label[:, 9] > 0
                _Virial_label = Virial_label[:, :9][data_mask]
                if data_mask.any().item():
                    loss_Virial_val = criterion(Virial_predict[data_mask], _Virial_label)
                else:
                    loss_Virial_val = torch.tensor(0.0)
                loss_Virial_per_atom_val = loss_Virial_val/natoms/natoms

            loss_val = loss_F_val + loss_Etot_val*natoms

            if args.optimizer_param.lambda_2 is not None:
                loss_val += L2
            if args.optimizer_param.lambda_1 is not None:
                loss_val += L1
            
            # measure accuracy and record loss
            losses.update(loss_val.item(), batch_size)

            loss_L1.update(L1.item(), batch_size)
            loss_L2.update(L2.item(), batch_size)

            loss_Etot.update(loss_Etot_val.item(), batch_size)
            loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
            loss_Ei.update(loss_Ei_val.item(), batch_size)
            if args.optimizer_param.train_egroup is True:
                loss_Egroup.update(loss_Egroup_val.item(), batch_size)
            if args.optimizer_param.train_virial is True:
                loss_Virial.update(loss_Virial_val.item(), _Virial_label.shape[0])
                loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), _Virial_label.shape[0])
            loss_Force.update(loss_F_val.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.optimizer_param.print_freq == 0:
            progress.display(i + 1)
        
    """
    if args.hvd:
        losses.all_reduce()
        loss_Etot.all_reduce()
        loss_Etot_per_atom.all_reduce()
        loss_Force.all_reduce()
        loss_Ei.all_reduce()
        if args.optimizer_param.train_egroup is True:
            loss_Egroup.all_reduce()
        if args.optimizer_param.train_virial is True:
            loss_Virial.all_reduce()
            loss_Virial_per_atom.all_reduce()
        batch_time.all_reduce()
    """
    progress.display_summary(["Training Set:"])
    return losses.avg, loss_Etot.root, loss_Etot_per_atom.root, loss_Force.root, loss_Ei.root, loss_Egroup.root, loss_Virial.root, loss_Virial_per_atom.root, loss_L1.root, loss_L2.root

def _classify_batchs(atom_type_map, atom_types:int):
    mask = atom_type_map != -1
    all_atom_types = torch.arange(atom_types)
    atom_counts = []
    for idx, row in enumerate(atom_type_map):
        count = torch.bincount(row[mask[idx]], minlength=len(all_atom_types))
        atom_counts.append(count)
    atom_counts_tuples = [tuple(count.tolist()) for count in atom_counts]
    class_dict = defaultdict(list)
    for idx, count_tuple in enumerate(atom_counts_tuples):
        class_dict[count_tuple].append(idx)
    return list(class_dict.values())

def valid(val_loader, model, criterion, device, args:InputParam):
    def run_validate(loader, base_progress=0):
        end = time.time()
        for i, sample_batches in enumerate(loader):
            i = base_progress + i
            if args.precision == "float64":
                Ei_label_cpu = sample_batches["Ei"].double()
                Etot_label_cpu = sample_batches["Etot"].double()
                Force_label_cpu = sample_batches["Force"][:, :, :].double()

                if args.optimizer_param.train_egroup is True:
                    Egroup_label_cpu = sample_batches["Egroup"].double()
                    Divider_cpu = sample_batches["Divider"].double()
                    Egroup_weight_cpu = sample_batches["Egroup_weight"].double()

                if args.optimizer_param.train_virial is True:
                    Virial_label_cpu = sample_batches["Virial"].double()

                ImageDR_cpu = sample_batches["ImageDR"].double()
                ImageDR_angular_cpu = sample_batches["ImageDRAngular"].double()
                # Ri_cpu = sample_batches["Ri"].double()
                # Ri_d_cpu = sample_batches["Ri_d"].double()

            elif args.precision == "float32":
                Ei_label_cpu = sample_batches["Ei"].float()
                Etot_label_cpu = sample_batches["Etot"].float()
                Force_label_cpu = sample_batches["Force"][:, :, :].float()

                if args.optimizer_param.train_egroup is True:
                    Egroup_label_cpu = sample_batches["Egroup"].float()
                    Divider_cpu = sample_batches["Divider"].float()
                    Egroup_weight_cpu = sample_batches["Egroup_weight"].float()

                if args.optimizer_param.train_virial is True:
                    Virial_label_cpu = sample_batches["Virial"].float()

                ImageDR_cpu = sample_batches["ImageDR"].float()
                ImageDR_angular_cpu = sample_batches["ImageDRAngular"].float()
                # Ri_cpu = sample_batches["Ri"].float()
                # Ri_d_cpu = sample_batches["Ri_d"].float()
            else:
                raise Exception("Error! Please specify floating point type: float32 or float64 by the parameter --datatype! ")
            
            dR_neigh_list_cpu = sample_batches["ListNeighbor"].int()
            dR_neigh_type_list_cpu = sample_batches["ListNeighborType"].int()
            dR_neigh_list_angular_cpu = sample_batches["ListNeighborAngular"].int()
            dR_neigh_type_angular_cpu = sample_batches["ListNeighborTypeAngular"].int()
            natoms_img_cpu = sample_batches["ImageAtomNum"].int()
            atom_type_cpu = sample_batches["AtomType"].int()
            atom_type_map_cpu = sample_batches["AtomTypeMap"].int()
            # classify batchs according to their atom type and atom nums
            batch_clusters = _classify_batchs(atom_type_map_cpu, len(args.atom_type))
            for batch_indexs in batch_clusters:
                # transport data to GPU
                natoms_img = Variable(natoms_img_cpu[batch_indexs].int().to(device))
                natoms = natoms_img[0]
                
                dR_neigh_list = Variable(dR_neigh_list_cpu[batch_indexs, :natoms].int().to(device))
                dR_neigh_type_list = Variable(dR_neigh_type_list_cpu[batch_indexs, :natoms].int().to(device))
                dR_neigh_list_angular = Variable(dR_neigh_list_angular_cpu[batch_indexs, :natoms].int().to(device))
                dR_neigh_type_list_angular = Variable(dR_neigh_type_angular_cpu[batch_indexs, :natoms].int().to(device))
                # atom list of image
                atom_type = Variable(atom_type_cpu[batch_indexs].to(device))
                atom_type_map = Variable(atom_type_map_cpu[batch_indexs, :natoms].to(device))
                Ei_label = Variable(Ei_label_cpu[batch_indexs, :natoms].to(device))
                Etot_label = Variable(Etot_label_cpu[batch_indexs].to(device))
                Force_label = Variable(Force_label_cpu[batch_indexs, :natoms].to(device))  # [40,108,3]

                if args.optimizer_param.train_egroup is True:
                    Egroup_label = Variable(Egroup_label_cpu[batch_indexs, :natoms].to(device))
                    Divider = Variable(Divider_cpu[batch_indexs, :natoms].to(device))
                    Egroup_weight = Variable(Egroup_weight_cpu[batch_indexs, :natoms, :natoms].to(device))

                if args.optimizer_param.train_virial is True:
                    Virial_label = Variable(Virial_label_cpu[batch_indexs].to(device))
                
                ImageDR = Variable(ImageDR_cpu[batch_indexs, :natoms].to(device))
                ImageDR_angular = Variable(ImageDR_angular_cpu[batch_indexs, :natoms].to(device))
                # Ri = Variable(Ri_cpu[batch_indexs, :natoms].to(device), requires_grad=True)
                # Ri_d = Variable(Ri_d_cpu[batch_indexs, :natoms].to(device))

                batch_size = len(batch_indexs)
                """
                    Dim of Ri [bs, natoms, ntype*max_neigh_num, 4] 
                """ 
                # check_cuda_memory(-1, -1, "test {} batch".format(i))
                if args.optimizer_param.train_egroup is True:
                    Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                        dR_neigh_list, ImageDR, dR_neigh_type_list, \
                            dR_neigh_list_angular, ImageDR_angular, dR_neigh_type_list_angular, \
                            atom_type_map[0], atom_type[0], 0, Egroup_weight, Divider)
                else:
                    # atom_type_map: we only need the first element, because it is same for each image of MOVEMENT
                    Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                        dR_neigh_list, ImageDR, dR_neigh_type_list, \
                            dR_neigh_list_angular, ImageDR_angular, dR_neigh_type_list_angular, \
                                atom_type_map[0], atom_type[0],  0, None, None)
                                                
                #return 
                loss_F_val = criterion(Force_predict, Force_label)
                loss_Etot_val = criterion(Etot_predict, Etot_label)
                loss_Etot_per_atom_val = loss_Etot_val/natoms/natoms
                loss_Ei_val = criterion(Ei_predict, Ei_label)
                if args.optimizer_param.train_egroup is True:
                    loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
                if args.optimizer_param.train_virial is True:
                    # loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))  #115.415137283393
                    data_mask = Virial_label[:, 9] > 0  # 判断最后一列是否大于 0
                    _Virial_label = Virial_label[:, :9][data_mask]
                    if data_mask.any().item():
                        loss_Virial_val = criterion(Virial_predict[data_mask], _Virial_label)
                    else:
                        loss_Virial_val = torch.tensor(0.0)
                    loss_Virial_per_atom_val = loss_Virial_val/natoms/natoms
                
                loss_val = loss_F_val + loss_Etot_val*natoms

                # measure accuracy and record loss
                losses.update(loss_val.item(), batch_size)
                loss_Etot.update(loss_Etot_val.item(), batch_size)
                loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
                loss_Ei.update(loss_Ei_val.item(), batch_size)
                if args.optimizer_param.train_egroup is True:
                    loss_Egroup.update(loss_Egroup_val.item(), batch_size)
                if args.optimizer_param.train_virial is True:
                    loss_Virial.update(loss_Virial_val.item(), _Virial_label.shape[0])
                    loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), _Virial_label.shape[0])
                loss_Force.update(loss_F_val.item(), batch_size)
            # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.optimizer_param.print_freq == 0:
            progress.display(i + 1) 
        

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    loss_Etot = AverageMeter("Etot", ":.4e", Summary.ROOT)
    loss_Etot_per_atom = AverageMeter("Etot_per_atom", ":.4e", Summary.ROOT)
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT)
    loss_Egroup = AverageMeter("Egroup", ":.4e", Summary.ROOT)
    loss_Virial = AverageMeter("Virial", ":.4e", Summary.ROOT)
    loss_Virial_per_atom = AverageMeter("Virial_per_atom", ":.4e", Summary.ROOT)

    progress = ProgressMeter(
        len(val_loader),
        # + (
        #     args.hvd
        #     and (len(val_loader.sampler) * hvd.size() < len(val_loader.dataset))
        # ),
        [batch_time, losses, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup, loss_Virial, loss_Virial_per_atom],
        prefix="Test: ",    
    )
    
    # switch to evaluate mode
    model.eval()
    
    run_validate(val_loader)

    """
    if args.hvd and (len(val_loader.sampler) * hvd.size() < len(val_loader.dataset)):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * hvd.size(), len(val_loader.dataset)),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    if args.hvd:
        losses.all_reduce()
        loss_Etot.all_reduce()
        loss_Etot_per_atom.all_reduce()
        loss_Force.all_reduce()
        loss_Ei.all_reduce()
        if args.optimizer_param.train_virial is True:
            loss_Virial.all_reduce()
            loss_Virial_per_atom.all_reduce()
    """

    progress.display_summary(["Test Set:"])

    return losses.avg, loss_Etot.root, loss_Etot_per_atom.root, loss_Force.root, loss_Ei.root, loss_Egroup.root, loss_Virial.root, loss_Virial_per_atom.root

'''
description: 
this function is used for inference:
the output is a pandas DataFrame object
param {*} val_loader
param {*} model
param {*} criterion
param {*} device
param {*} args
return {*}
author: wuxingxing
'''
def predict(val_loader, model, criterion, device, args:InputParam, isprofile=False):
    train_lists = ["img_idx"] #"Etot_lab", "Etot_pre", "Ei_lab", "Ei_pre", "Force_lab", "Force_pre"
    train_lists.extend(["RMSE_Etot", "RMSE_Etot_per_atom", "RMSE_Ei", "RMSE_F"])
    if args.optimizer_param.train_egroup:
        train_lists.append("RMSE_Egroup")
    if args.optimizer_param.train_virial:
        train_lists.append("RMSE_virial")
        train_lists.append("RMSE_virial_per_atom")

    res_pd = pd.DataFrame(columns=train_lists)
    force_label_list = []
    force_predict_list = []
    ei_label_list = []
    ei_predict_list = []
    etot_label_list = []
    etot_predict_list = []
    model.eval()

    for i, sample_batches in enumerate(val_loader):
        # measure data loading time
        # load data to cpu
        if args.precision == "float64":
            Ei_label_cpu = sample_batches["Ei"].double()
            Etot_label_cpu = sample_batches["Etot"].double()
            Force_label_cpu = sample_batches["Force"][:, :, :].double()

            if args.optimizer_param.train_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].double()
                Divider_cpu = sample_batches["Divider"].double()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].double()

            if args.optimizer_param.train_virial is True:
                Virial_label_cpu = sample_batches["Virial"].double()

            ImageDR_cpu = sample_batches["ImageDR"].double()
            ImageDR_angular_cpu = sample_batches["ImageDRAngular"].double()
            # Ri_cpu = sample_batches["Ri"].double()
            # Ri_d_cpu = sample_batches["Ri_d"].double()

        elif args.precision == "float32":
            Ei_label_cpu = sample_batches["Ei"].float()
            Etot_label_cpu = sample_batches["Etot"].float()
            Force_label_cpu = sample_batches["Force"][:, :, :].float()

            if args.optimizer_param.train_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].float()
                Divider_cpu = sample_batches["Divider"].float()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].float()

            if args.optimizer_param.train_virial is True:
                Virial_label_cpu = sample_batches["Virial"].float()

            ImageDR_cpu = sample_batches["ImageDR"].float()
            ImageDR_angular_cpu = sample_batches["ImageDRAngular"].float()
            # Ri_cpu = sample_batches["Ri"].float()
            # Ri_d_cpu = sample_batches["Ri_d"].float()
        else:
            raise Exception("Error! Please specify floating point type: float32 or float64 by the parameter --datatype! ")
        
        dR_neigh_list_cpu = sample_batches["ListNeighbor"].int()
        dR_neigh_type_list_cpu = sample_batches["ListNeighborType"].int()
        dR_neigh_list_angular_cpu = sample_batches["ListNeighborAngular"].int()
        dR_neigh_type_angular_cpu = sample_batches["ListNeighborTypeAngular"].int()
        natoms_img_cpu = sample_batches["ImageAtomNum"].int()
        atom_type_cpu = sample_batches["AtomType"].int()
        atom_type_map_cpu = sample_batches["AtomTypeMap"].int()
        # classify batchs according to their atom type and atom nums
        batch_clusters = _classify_batchs(atom_type_map_cpu, len(args.atom_type))

        for batch_indexs in batch_clusters:
            # transport data to GPU
            natoms_img = Variable(natoms_img_cpu[batch_indexs].int().to(device))
            natoms = natoms_img[0]
            
            dR_neigh_list = Variable(dR_neigh_list_cpu[batch_indexs, :natoms].int().to(device))
            dR_neigh_type_list = Variable(dR_neigh_type_list_cpu[batch_indexs, :natoms].int().to(device))
            dR_neigh_list_angular = Variable(dR_neigh_list_angular_cpu[batch_indexs, :natoms].int().to(device))
            dR_neigh_type_list_angular = Variable(dR_neigh_type_angular_cpu[batch_indexs, :natoms].int().to(device))
            # atom list of image
            atom_type = Variable(atom_type_cpu[batch_indexs].to(device))
            atom_type_map = Variable(atom_type_map_cpu[batch_indexs, :natoms].to(device))
            Ei_label = Variable(Ei_label_cpu[batch_indexs, :natoms].to(device))
            Etot_label = Variable(Etot_label_cpu[batch_indexs].to(device))
            Force_label = Variable(Force_label_cpu[batch_indexs, :natoms].to(device))  # [40,108,3]

            if args.optimizer_param.train_egroup is True:
                Egroup_label = Variable(Egroup_label_cpu[batch_indexs, :natoms].to(device))
                Divider = Variable(Divider_cpu[batch_indexs, :natoms].to(device))
                Egroup_weight = Variable(Egroup_weight_cpu[batch_indexs, :natoms, :natoms].to(device))

            if args.optimizer_param.train_virial is True:
                Virial_label = Variable(Virial_label_cpu[batch_indexs].to(device))
            
            ImageDR = Variable(ImageDR_cpu[batch_indexs, :natoms].to(device))
            ImageDR_angular = Variable(ImageDR_angular_cpu[batch_indexs, :natoms].to(device))
            # Ri = Variable(Ri_cpu[batch_indexs, :natoms].to(device), requires_grad=True)
            # Ri_d = Variable(Ri_d_cpu[batch_indexs, :natoms].to(device))
            # batch_size = len(batch_indexs)
            if args.optimizer_param.train_egroup is True:
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                    dR_neigh_list, ImageDR, dR_neigh_type_list, \
                        dR_neigh_list_angular, ImageDR_angular, dR_neigh_type_list_angular, \
                        atom_type_map[0], atom_type[0], 0, Egroup_weight, Divider)
            else:
                # atom_type_map: we only need the first element, because it is same for each image of MOVEMENT
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                    dR_neigh_list, ImageDR, dR_neigh_type_list, \
                        dR_neigh_list_angular, ImageDR_angular, dR_neigh_type_list_angular, \
                            atom_type_map[0], atom_type[0],  0, None, None)
            # mse
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_F_val = criterion(Force_predict, Force_label)
            loss_Ei_val = criterion(Ei_predict, Ei_label)
            # loss_val = loss_F_val + loss_Etot_val
            if args.optimizer_param.train_egroup is True:
                loss_Egroup_val = criterion(Egroup_predict, Egroup_label)

            if args.optimizer_param.train_virial is True:
                loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))
                loss_Virial_per_atom_val = loss_Virial_val/natoms/natoms
            # rmse
            Etot_rmse = loss_Etot_val ** 0.5
            etot_atom_rmse = Etot_rmse / natoms
            Ei_rmse = loss_Ei_val ** 0.5
            F_rmse = loss_F_val ** 0.5

            res_list = [i, float(Etot_rmse), float(etot_atom_rmse), float(Ei_rmse), float(F_rmse)]
            #float(Etot_predict), float(Ei_label.abs().mean()), float(Ei_predict.abs().mean()), float(Force_label.abs().mean()), float(Force_predict.abs().mean()),\
            if args.optimizer_param.train_egroup:
                res_list.append(float(loss_Egroup_val))
            if args.optimizer_param.train_virial:
                res_list.append(float(loss_Virial_val))
                res_list.append(float(loss_Virial_per_atom_val))
            
            force_label_list.append(Force_label.flatten().cpu().numpy())
            force_predict_list.append(Force_predict.flatten().detach().cpu().numpy())
            ei_label_list.append(Ei_label.flatten().cpu().numpy())
            ei_predict_list.append(Ei_predict.flatten().detach().cpu().numpy())
            etot_label_list.append(float(Etot_label))
            etot_predict_list.append(float(Etot_predict))
            res_pd.loc[res_pd.shape[0]] = res_list
    
    return res_pd, etot_label_list, etot_predict_list, ei_label_list, ei_predict_list, force_label_list, force_predict_list

def calculate_scaler(val_loader, model, criterion, device, args:InputParam):
    def run_validate(loader, base_progress=0):
        for i, sample_batches in enumerate(loader):
            if i % 5 != 0:
                continue
            i = base_progress + i
            if args.precision == "float64":
                Ei_label_cpu = sample_batches["Ei"].double()
                Etot_label_cpu = sample_batches["Etot"].double()
                Force_label_cpu = sample_batches["Force"][:, :, :].double()

                if args.optimizer_param.train_egroup is True:
                    Egroup_label_cpu = sample_batches["Egroup"].double()
                    Divider_cpu = sample_batches["Divider"].double()
                    Egroup_weight_cpu = sample_batches["Egroup_weight"].double()

                if args.optimizer_param.train_virial is True:
                    Virial_label_cpu = sample_batches["Virial"].double()

                ImageDR_cpu = sample_batches["ImageDR"].double()
                ImageDR_angular_cpu = sample_batches["ImageDRAngular"].double()
                # Ri_cpu = sample_batches["Ri"].double()
                # Ri_d_cpu = sample_batches["Ri_d"].double()

            elif args.precision == "float32":
                Ei_label_cpu = sample_batches["Ei"].float()
                Etot_label_cpu = sample_batches["Etot"].float()
                Force_label_cpu = sample_batches["Force"][:, :, :].float()

                if args.optimizer_param.train_egroup is True:
                    Egroup_label_cpu = sample_batches["Egroup"].float()
                    Divider_cpu = sample_batches["Divider"].float()
                    Egroup_weight_cpu = sample_batches["Egroup_weight"].float()

                if args.optimizer_param.train_virial is True:
                    Virial_label_cpu = sample_batches["Virial"].float()

                ImageDR_cpu = sample_batches["ImageDR"].float()
                ImageDR_angular_cpu = sample_batches["ImageDRAngular"].float()
                # Ri_cpu = sample_batches["Ri"].float()
                # Ri_d_cpu = sample_batches["Ri_d"].float()
            else:
                raise Exception("Error! Please specify floating point type: float32 or float64 by the parameter --datatype! ")
            
            dR_neigh_list_cpu = sample_batches["ListNeighbor"].int()
            dR_neigh_type_list_cpu = sample_batches["ListNeighborType"].int()
            dR_neigh_list_angular_cpu = sample_batches["ListNeighborAngular"].int()
            dR_neigh_type_angular_cpu = sample_batches["ListNeighborTypeAngular"].int()
            natoms_img_cpu = sample_batches["ImageAtomNum"].int()
            atom_type_cpu = sample_batches["AtomType"].int()
            atom_type_map_cpu = sample_batches["AtomTypeMap"].int()
            # classify batchs according to their atom type and atom nums
            batch_clusters = _classify_batchs(atom_type_map_cpu, len(args.atom_type))
            for batch_indexs in batch_clusters:
                # transport data to GPU
                natoms_img = Variable(natoms_img_cpu[batch_indexs].int().to(device))
                natoms = natoms_img[0]
                
                dR_neigh_list = Variable(dR_neigh_list_cpu[batch_indexs, :natoms].int().to(device))
                dR_neigh_type_list = Variable(dR_neigh_type_list_cpu[batch_indexs, :natoms].int().to(device))
                dR_neigh_list_angular = Variable(dR_neigh_list_angular_cpu[batch_indexs, :natoms].int().to(device))
                dR_neigh_type_list_angular = Variable(dR_neigh_type_angular_cpu[batch_indexs, :natoms].int().to(device))
                # atom list of image
                atom_type = Variable(atom_type_cpu[batch_indexs].to(device))
                atom_type_map = Variable(atom_type_map_cpu[batch_indexs, :natoms].to(device))
                Ei_label = Variable(Ei_label_cpu[batch_indexs, :natoms].to(device))
                Etot_label = Variable(Etot_label_cpu[batch_indexs].to(device))
                Force_label = Variable(Force_label_cpu[batch_indexs, :natoms].to(device))  # [40,108,3]

                if args.optimizer_param.train_egroup is True:
                    Egroup_label = Variable(Egroup_label_cpu[batch_indexs, :natoms].to(device))
                    Divider = Variable(Divider_cpu[batch_indexs, :natoms].to(device))
                    Egroup_weight = Variable(Egroup_weight_cpu[batch_indexs, :natoms, :natoms].to(device))

                if args.optimizer_param.train_virial is True:
                    Virial_label = Variable(Virial_label_cpu[batch_indexs].to(device))
                
                ImageDR = Variable(ImageDR_cpu[batch_indexs, :natoms].to(device))
                ImageDR_angular = Variable(ImageDR_angular_cpu[batch_indexs, :natoms].to(device))
                # Ri = Variable(Ri_cpu[batch_indexs, :natoms].to(device), requires_grad=True)
                # Ri_d = Variable(Ri_d_cpu[batch_indexs, :natoms].to(device))

                batch_size = len(batch_indexs)
                """
                    Dim of Ri [bs, natoms, ntype*max_neigh_num, 4] 
                """ 
                # check_cuda_memory(-1, -1, "test {} batch".format(i))
                if args.optimizer_param.train_egroup is True:
                    Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                        dR_neigh_list, ImageDR, dR_neigh_type_list, \
                            dR_neigh_list_angular, ImageDR_angular, dR_neigh_type_list_angular, \
                            atom_type_map[0], atom_type[0], 0, Egroup_weight, Divider, False)
                else:
                    # atom_type_map: we only need the first element, because it is same for each image of MOVEMENT
                    Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                        dR_neigh_list, ImageDR, dR_neigh_type_list, \
                            dR_neigh_list_angular, ImageDR_angular, dR_neigh_type_list_angular, \
                                atom_type_map[0], atom_type[0],  0, None, None, False)
    # switch to evaluate mode
    model.eval()
    run_validate(val_loader)


def save_checkpoint(state, filename, prefix):
    filename = os.path.join(prefix, filename)
    torch.save(state, filename)



class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
    ROOT = 4


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.root = 0
    
    def update(self, val, n=1):
        self.val = val
        if n > 0: # for same data, such as virial, some images do not have virial datas, the n will be 0
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
            self.root = self.avg**0.5

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        # total = hvd.allreduce(total, hvd.Sum)
        # dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count
        self.root = self.avg**0.5

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        elif self.summary_type is Summary.ROOT:
            fmtstr = "{name} {root:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries), flush=True)

    def display_summary(self, entries=[" *"]):
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
