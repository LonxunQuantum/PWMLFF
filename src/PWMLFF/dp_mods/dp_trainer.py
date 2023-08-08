import os
import pandas as pd
import numpy as np
import shutil
import time
from enum import Enum
import torch
from torch.utils.data import Subset
from torch.autograd import Variable

from loss.dploss import dp_loss, adjust_lr

from optimizer.KFWrapper import KFOptimizerWrapper
import horovod.torch as hvd

from torch.profiler import profile, record_function, ProfilerActivity

def train(train_loader, model, criterion, optimizer, epoch, start_lr, device, config):
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
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup, loss_Virial, loss_Virial_per_atom],
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
        real_lr = adjust_lr(global_step, start_lr)

        for param_group in optimizer.param_groups:
            param_group["lr"] = real_lr * (nr_batch_sample**0.5)

        if config.datatype == "float64":
            Ei_label_cpu = sample_batches["Ei"].double()
            Etot_label_cpu = sample_batches["Etot"].double()
            Force_label_cpu = sample_batches["Force"][:, :, :].double()

            if config.is_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].double()
                Divider_cpu = sample_batches["Divider"].double()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].double()

            if config.is_virial is True:
                Virial_label_cpu = sample_batches["Virial"].double()

            ImageDR_cpu = sample_batches["ImageDR"].double()
            Ri_cpu = sample_batches["Ri"].double()
            Ri_d_cpu = sample_batches["Ri_d"].double()

        elif config.datatype == "float32":
            Ei_label_cpu = sample_batches["Ei"].float()
            Etot_label_cpu = sample_batches["Etot"].float()
            Force_label_cpu = sample_batches["Force"][:, :, :].float()

            if config.is_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].float()
                Divider_cpu = sample_batches["Divider"].float()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].float()

            if config.is_virial is True:
                Virial_label_cpu = sample_batches["Virial"].float()

            ImageDR_cpu = sample_batches["ImageDR"].float()
            Ri_cpu = sample_batches["Ri"].float()
            Ri_d_cpu = sample_batches["Ri_d"].float()
        else:
            raise Exception("Error! Please specify floating point type: float32 or float64 by the parameter --datatype! ")
        
        dR_neigh_list_cpu = sample_batches["ListNeighbor"].int()
        natoms_img_cpu = sample_batches["ImageAtomNum"].int()
        atom_type_cpu = sample_batches["AtomType"].int()
        # classify batchs according to their atom type and atom nums
        batch_clusters = _classify_batchs(np.array(atom_type_cpu), np.array(natoms_img_cpu))

        for batch_indexs in batch_clusters:
            # transport data to GPU
            natoms_img = Variable(natoms_img_cpu[batch_indexs].int().to(device))
            natoms_img = torch.squeeze(natoms_img, 1)
            natoms = natoms_img[0,1:].sum()
            
            dR_neigh_list = Variable(dR_neigh_list_cpu[batch_indexs, :natoms].int().to(device))
            # atom list of image
            atom_type = Variable(atom_type_cpu[batch_indexs].to(device))
            natom = natoms_img[0,0]
            Ei_label = Variable(Ei_label_cpu[batch_indexs, :natoms].to(device))
            Etot_label = Variable(Etot_label_cpu[batch_indexs].to(device))
            Force_label = Variable(Force_label_cpu[batch_indexs, :natoms].to(device))  # [40,108,3]

            if config.is_egroup is True:
                Egroup_label = Variable(Egroup_label_cpu[batch_indexs, :natoms].to(device))
                Divider = Variable(Divider_cpu[batch_indexs, :natoms].to(device))
                Egroup_weight = Variable(Egroup_weight_cpu[batch_indexs, :natoms, :natoms].to(device))

            if config.is_virial is True:
                Virial_label = Variable(Virial_label_cpu[batch_indexs].to(device))
            
            ImageDR = Variable(ImageDR_cpu[batch_indexs, :natoms].to(device))
            Ri = Variable(Ri_cpu[batch_indexs, :natoms].to(device), requires_grad=True)
            Ri_d = Variable(Ri_d_cpu[batch_indexs, :natoms].to(device))

            batch_size = Ri.shape[0]
            natom = natoms_img[0,0]
            if config.profiling:
                print("=" * 60, "Start profiling model inference", "=" * 60)
                with profile(
                    activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                    record_shapes=True,
                ) as prof:
                    with record_function("model_inference"):
                        Etot_predict, Ei_predict, Force_predict, _ = model(
                            Ri, Ri_d, dR_neigh_list, natoms_img, None, None
                        )

                print(prof.key_averages().table(sort_by="cuda_time_total"))
                print("=" * 60, "Profiling model inference end", "=" * 60)
                prof.export_chrome_trace("model_infer.json")
            else:
                if config.is_egroup is True:
                    Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                        Ri, Ri_d, dR_neigh_list, natoms_img, atom_type, ImageDR, Egroup_weight, Divider)
                else:
                    Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                        Ri, Ri_d, dR_neigh_list, natoms_img, atom_type, ImageDR, None, None)
                    
            optimizer.zero_grad()

            loss_F_val = criterion(Force_predict, Force_label)
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_Etot_per_atom_val = loss_Etot_val/natom/natom
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
            if config.is_egroup is True:
                loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
            if config.is_virial is True:
                loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))
                loss_Virial_per_atom_val = loss_Virial_val/natom/natom

            loss_val = torch.zeros_like(loss_F_val)
            
            w_f, w_e, w_v, w_eg, w_ei = 0, 0, 0, 0, 0

            if config.is_force is True:
                w_f = 1.0 
                loss_val += loss_F_val
            
            if config.is_etot is True:
                w_e = 1.0
                loss_val += loss_Etot_val
            
            if config.is_virial is True:
                w_v = 1.0 
                loss_val += loss_Virial_val
            
            if config.is_egroup is True:
                w_eg = 1.0 
                loss_val += loss_Egroup_val

            if config.is_egroup is True and config.is_virial is True:
                loss, _, _ = dp_loss(
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
                    natoms_img[0, 0].item(),
                )
            elif config.is_egroup is True and config.is_virial is False:
                loss, _, _ = dp_loss(
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
                    natoms_img[0, 0].item(),
                )
            elif config.is_egroup is False and config.is_virial is True:
                loss, _, _ = dp_loss(
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
                    natoms_img[0, 0].item(),
                )
            else:
                loss, _, _ = dp_loss(
                    0.001,
                    real_lr,
                    4,
                    w_f,
                    loss_F_val,
                    w_e,
                    loss_Etot_val,
                    w_ei,
                    loss_Ei_val,
                    natoms_img[0, 0].item(),
                )
            # import ipdb;ipdb.set_trace()
            loss.backward()
            optimizer.step()

            
            # measure accuracy and record loss
            losses.update(loss_val.item(), batch_size)
            loss_Etot.update(loss_Etot_val.item(), batch_size)
            loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
            loss_Ei.update(loss_Ei_val.item(), batch_size)
            if config.is_egroup is True:
                loss_Egroup.update(loss_Egroup_val.item(), batch_size)
            if config.is_virial is True:
                loss_Virial.update(loss_Virial_val.item(), batch_size)
                loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), batch_size)
            loss_Force.update(loss_F_val.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
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
    )


def train_KF(train_loader, model, criterion, optimizer, epoch, device, config):
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
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup, loss_Virial, loss_Virial_per_atom],
        prefix="Epoch: [{}]".format(epoch),
    )

    KFOptWrapper = KFOptimizerWrapper(
        model, optimizer, config.nselect, config.groupsize, config.hvd, "hvd"
    )
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, sample_batches in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # load data to cpu
        if config.datatype == "float64":
            Ei_label_cpu = sample_batches["Ei"].double()
            Etot_label_cpu = sample_batches["Etot"].double()
            Force_label_cpu = sample_batches["Force"][:, :, :].double()

            if config.is_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].double()
                Divider_cpu = sample_batches["Divider"].double()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].double()

            if config.is_virial is True:
                Virial_label_cpu = sample_batches["Virial"].double()

            ImageDR_cpu = sample_batches["ImageDR"].double()
            Ri_cpu = sample_batches["Ri"].double()
            Ri_d_cpu = sample_batches["Ri_d"].double()

        elif config.datatype == "float32":
            Ei_label_cpu = sample_batches["Ei"].float()
            Etot_label_cpu = sample_batches["Etot"].float()
            Force_label_cpu = sample_batches["Force"][:, :, :].float()

            if config.is_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].float()
                Divider_cpu = sample_batches["Divider"].float()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].float()

            if config.is_virial is True:
                Virial_label_cpu = sample_batches["Virial"].float()

            ImageDR_cpu = sample_batches["ImageDR"].float()
            Ri_cpu = sample_batches["Ri"].float()
            Ri_d_cpu = sample_batches["Ri_d"].float()
        else:
            raise Exception("Error! Please specify floating point type: float32 or float64 by the parameter --datatype! ")
        
        dR_neigh_list_cpu = sample_batches["ListNeighbor"].int()
        natoms_img_cpu = sample_batches["ImageAtomNum"].int()
        atom_type_cpu = sample_batches["AtomType"].int()
        # classify batchs according to their atom type and atom nums
        batch_clusters = _classify_batchs(np.array(atom_type_cpu), np.array(natoms_img_cpu))

        for batch_indexs in batch_clusters:
            # transport data to GPU
            natoms_img = Variable(natoms_img_cpu[batch_indexs].int().to(device))
            natoms_img = torch.squeeze(natoms_img, 1)
            natoms = natoms_img[0,1:].sum()
            
            dR_neigh_list = Variable(dR_neigh_list_cpu[batch_indexs, :natoms].int().to(device))
            # atom list of image
            atom_type = Variable(atom_type_cpu[batch_indexs].to(device))
            natom = natoms_img[0,0]
            Ei_label = Variable(Ei_label_cpu[batch_indexs, :natoms].to(device))
            Etot_label = Variable(Etot_label_cpu[batch_indexs].to(device))
            Force_label = Variable(Force_label_cpu[batch_indexs, :natoms].to(device))  # [40,108,3]

            if config.is_egroup is True:
                Egroup_label = Variable(Egroup_label_cpu[batch_indexs, :natoms].to(device))
                Divider = Variable(Divider_cpu[batch_indexs, :natoms].to(device))
                Egroup_weight = Variable(Egroup_weight_cpu[batch_indexs, :natoms, :natoms].to(device))

            if config.is_virial is True:
                Virial_label = Variable(Virial_label_cpu[batch_indexs].to(device))
            
            ImageDR = Variable(ImageDR_cpu[batch_indexs, :natoms].to(device))
            Ri = Variable(Ri_cpu[batch_indexs, :natoms].to(device), requires_grad=True)
            Ri_d = Variable(Ri_d_cpu[batch_indexs, :natoms].to(device))

            batch_size = len(batch_indexs)
            if config.is_egroup is True:
                kalman_inputs = [Ri, Ri_d, dR_neigh_list, natoms_img, atom_type, ImageDR, Egroup_weight, Divider]
            else:
                kalman_inputs = [Ri, Ri_d, dR_neigh_list, natoms_img, atom_type, ImageDR, None, None]

            if config.profiling:
                print("=" * 60, "Start profiling KF update energy", "=" * 60)
                with profile(
                    activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                    record_shapes=True,
                ) as prof:
                    with record_function("kf_update_energy"):
                        Etot_predict = KFOptWrapper.update_energy(kalman_inputs, Etot_label)
                print(prof.key_averages().table(sort_by="cuda_time_total"))
                print("=" * 60, "Profiling KF update energy end", "=" * 60)
                prof.export_chrome_trace("kf_update_energy.json")

                print("=" * 60, "Start profiling KF update force", "=" * 60)
                with profile(
                    activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                    record_shapes=True,
                ) as prof:
                    with record_function("kf_update_force"):
                        Etot_predict, Ei_predict, Force_predict = KFOptWrapper.update_force(
                            kalman_inputs, Force_label, 2
                        )
                print(prof.key_averages().table(sort_by="cuda_time_total"))
                print("=" * 60, "Profiling KF update force end", "=" * 60)
                
                prof.export_chrome_trace("kf_update_force.json")
            else:
                if config.is_virial is True:
                    Virial_predict = KFOptWrapper.update_virial(kalman_inputs, Virial_label, config.pre_fac_virial)

                if config.is_etot is True: 
                    Etot_predict = KFOptWrapper.update_energy(kalman_inputs, Etot_label, config.pre_fac_etot)
                
                if config.is_ei is True:
                    Ei_predict = KFOptWrapper.update_ei(kalman_inputs, Ei_label, config.pre_fac_ei)

                if config.is_egroup is True:
                    Egroup_predict = KFOptWrapper.update_egroup(kalman_inputs, Egroup_label, config.pre_fac_egroup)

 
                if config.is_force is True:
                    Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = KFOptWrapper.update_force(
                        kalman_inputs, Force_label, config.pre_fac_force)

            loss_F_val = criterion(Force_predict, Force_label)

            # divide by natom 
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_Etot_per_atom_val = loss_Etot_val/natom/natom
            
            loss_Ei_val = criterion(Ei_predict, Ei_label)   
            if config.is_egroup is True:
                loss_Egroup_val = criterion(Egroup_predict, Egroup_label)

            if config.is_virial is True:
                loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))
                loss_Virial_per_atom_val = loss_Virial_val/natom/natom
            loss_val = loss_F_val + loss_Etot_val*natom

            # measure accuracy and record loss
            losses.update(loss_val.item(), batch_size)

            loss_Etot.update(loss_Etot_val.item(), batch_size)
            loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
            loss_Ei.update(loss_Ei_val.item(), batch_size)
            if config.is_egroup is True:
                loss_Egroup.update(loss_Egroup_val.item(), batch_size)
            if config.is_virial is True:
                loss_Virial.update(loss_Virial_val.item(), batch_size)
                loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), batch_size)
            loss_Force.update(loss_F_val.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i + 1)
 
    if config.hvd:
        losses.all_reduce()
        loss_Etot.all_reduce()
        loss_Etot_per_atom.all_reduce()
        loss_Force.all_reduce()
        loss_Ei.all_reduce()
        if config.is_egroup is True:
            loss_Egroup.all_reduce()
        if config.is_virial is True:
            loss_Virial.all_reduce()
            loss_Virial_per_atom.all_reduce()
        batch_time.all_reduce()

    progress.display_summary(["Training Set:"])
    return losses.avg, loss_Etot.root, loss_Etot_per_atom.root, loss_Force.root, loss_Ei.root, loss_Egroup.root, loss_Virial.root, loss_Virial_per_atom.root

'''
description: 
classify according to atom type and the atom num of the image. 

example, for this two array, after lassified, will return:
    [[0, 6, 8], [1, 2, 3, 4, 5, 7, 9]]
natoms_img                                  atoms:
array([[76, 60, 16],                 array([[ 3, 14],
       [64,  0, 64],                        [14,  0],          
       [64,  0, 64],                        [14,  0],          
       [64,  0, 64],                        [14,  0],          
       [64,  0, 64],                        [14,  0],          
       [64,  0, 64],                        [14,  0],          
       [76, 60, 16],                        [ 3, 14],          
       [64,  0, 64],                        [14,  0],          
       [76, 60, 16],                        [ 3, 14],          
       [64,  0, 64]], dtype=int32)          [14,  0]], dtype=int32)

param {np} atoms 
param {np} img_natoms
return {*}
author: wuxingxing
'''
def _classify_batchs(atom_types: np.ndarray, img_natoms: np.ndarray):
    dicts = {}
    for i in range(atom_types.shape[0]):
        key = ""
        for k1 in atom_types[i]:
            key += "{}_".format(k1)
        key += "{}".format(img_natoms[i][0])
        if key in dicts.keys():
            dicts[key].append(i)
        else:
            dicts[key] = [i]
    return [dicts[_] for _ in dicts.keys()]

def valid(val_loader, model, criterion, device, args):
    def run_validate(loader, base_progress=0):
        end = time.time()
        for i, sample_batches in enumerate(loader):

            i = base_progress + i
            if args.datatype == "float64":
                Ei_label_cpu = sample_batches["Ei"].double()
                Etot_label_cpu = sample_batches["Etot"].double()
                Force_label_cpu = sample_batches["Force"][:, :, :].double()

                if args.is_egroup is True:
                    Egroup_label_cpu = sample_batches["Egroup"].double()
                    Divider_cpu = sample_batches["Divider"].double()
                    Egroup_weight_cpu = sample_batches["Egroup_weight"].double()

                if args.is_virial is True:
                    Virial_label_cpu = sample_batches["Virial"].double()

                ImageDR_cpu = sample_batches["ImageDR"].double()
                Ri_cpu = sample_batches["Ri"].double()
                Ri_d_cpu = sample_batches["Ri_d"].double()

            elif args.datatype == "float32":
                Ei_label_cpu = sample_batches["Ei"].float()
                Etot_label_cpu = sample_batches["Etot"].float()
                Force_label_cpu = sample_batches["Force"][:, :, :].float()

                if args.is_egroup is True:
                    Egroup_label_cpu = sample_batches["Egroup"].float()
                    Divider_cpu = sample_batches["Divider"].float()
                    Egroup_weight_cpu = sample_batches["Egroup_weight"].float()

                if args.is_virial is True:
                    Virial_label_cpu = sample_batches["Virial"].float()

                ImageDR_cpu = sample_batches["ImageDR"].float()
                Ri_cpu = sample_batches["Ri"].float()
                Ri_d_cpu = sample_batches["Ri_d"].float()
            else:
                raise Exception("Error! Please specify floating point type: float32 or float64 by the parameter --datatype! ")
            
            dR_neigh_list_cpu = sample_batches["ListNeighbor"].int()
            natoms_img_cpu = sample_batches["ImageAtomNum"].int()
            atom_type_cpu = sample_batches["AtomType"].int()

            # classify batchs according to their atom type and atom nums
            batch_clusters = _classify_batchs(np.array(atom_type_cpu), np.array(natoms_img_cpu))
            for batch_indexs in batch_clusters:
                # transport data to GPU
                natoms_img = Variable(natoms_img_cpu[batch_indexs].int().to(device))
                natoms_img = torch.squeeze(natoms_img, 1)
                natoms = natoms_img[0,1:].sum()
                
                dR_neigh_list = Variable(dR_neigh_list_cpu[batch_indexs, :natoms].int().to(device))
                # atom list of image
                atom_type = Variable(atom_type_cpu[batch_indexs].to(device))
                natom = natoms_img[0,0]

                Ei_label = Variable(Ei_label_cpu[batch_indexs, :natoms].to(device))
                Etot_label = Variable(Etot_label_cpu[batch_indexs].to(device))
                Force_label = Variable(Force_label_cpu[batch_indexs, :natoms].to(device))  # [40,108,3]

                if args.is_egroup is True:
                    Egroup_label = Variable(Egroup_label_cpu[batch_indexs, :natoms].to(device))
                    Divider = Variable(Divider_cpu[batch_indexs, :natoms].to(device))
                    Egroup_weight = Variable(Egroup_weight_cpu[batch_indexs, :natoms, :natoms].to(device))

                if args.is_virial is True:
                    Virial_label = Variable(Virial_label_cpu[batch_indexs].to(device))
                
                ImageDR = Variable(ImageDR_cpu[batch_indexs, :natoms].to(device))
                Ri = Variable(Ri_cpu[batch_indexs, :natoms].to(device), requires_grad=True)
                Ri_d = Variable(Ri_d_cpu[batch_indexs, :natoms].to(device))

                batch_size = len(batch_indexs)
                """
                    Dim of Ri [bs, natom, ntype*max_neigh_num, 4] 
                """ 
                if args.is_egroup is True:
                    Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                        Ri, Ri_d, dR_neigh_list, natoms_img, atom_type, ImageDR, Egroup_weight, Divider)
                else:
                    Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                        Ri, Ri_d, dR_neigh_list, natoms_img, atom_type, ImageDR, None, None)
                
                #return 
                loss_F_val = criterion(Force_predict, Force_label)
                loss_Etot_val = criterion(Etot_predict, Etot_label)
                loss_Etot_per_atom_val = loss_Etot_val/natom/natom
                loss_Ei_val = criterion(Ei_predict, Ei_label)
                if args.is_egroup is True:
                    loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
                if args.is_virial is True:
                    loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))
                    loss_Virial_per_atom_val = loss_Virial_val/natom/natom
                loss_val = loss_F_val + loss_Etot_val*natom

                # measure accuracy and record loss
                losses.update(loss_val.item(), batch_size)
                loss_Etot.update(loss_Etot_val.item(), batch_size)
                loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
                loss_Ei.update(loss_Ei_val.item(), batch_size)
                if args.is_egroup is True:
                    loss_Egroup.update(loss_Egroup_val.item(), batch_size)
                if args.is_virial is True:
                    loss_Virial.update(loss_Virial_val.item(), batch_size)
                    loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), batch_size)
                loss_Force.update(loss_F_val.item(), batch_size)
            # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
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
        len(val_loader)
        + (
            args.hvd
            and (len(val_loader.sampler) * hvd.size() < len(val_loader.dataset))
        ),
        [batch_time, losses, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup, loss_Virial, loss_Virial_per_atom],
        prefix="Test: ",    
    )
    
    # switch to evaluate mode
    model.eval()
    
    run_validate(val_loader)

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
        if args.is_virial is True:
            loss_Virial.all_reduce()
            loss_Virial_per_atom.all_reduce()

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
def predict(val_loader, model, criterion, device, args):
    train_lists = ["img_idx", "Etot_lab", "Etot_pre", "Ei_lab", "Ei_pre", "Force_lab", "Force_pre"]
    train_lists.extend(["RMSE_Etot", "RMSE_Etot_per_atom", "RMSE_Ei", "RMSE_F"])
    if args.is_egroup:
        train_lists.append("RMSE_Egroup")
    if args.is_virial:
        train_lists.append("RMSE_virial")
        train_lists.append("RMSE_virial_per_atom")

    res_pd = pd.DataFrame(columns=train_lists)
    force_label_list = []
    force_predict_list = []
    ei_label_list = []
    ei_predict_list = []
    model.eval()

    for i, sample_batches in enumerate(val_loader):
        # measure data loading time
        # load data to cpu
        if args.datatype == "float64":
            Ei_label_cpu = sample_batches["Ei"].double()
            Etot_label_cpu = sample_batches["Etot"].double()
            Force_label_cpu = sample_batches["Force"][:, :, :].double()

            if args.is_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].double()
                Divider_cpu = sample_batches["Divider"].double()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].double()

            if args.is_virial is True:
                Virial_label_cpu = sample_batches["Virial"].double()

            ImageDR_cpu = sample_batches["ImageDR"].double()
            Ri_cpu = sample_batches["Ri"].double()
            Ri_d_cpu = sample_batches["Ri_d"].double()

        elif args.datatype == "float32":
            Ei_label_cpu = sample_batches["Ei"].float()
            Etot_label_cpu = sample_batches["Etot"].float()
            Force_label_cpu = sample_batches["Force"][:, :, :].float()

            if args.is_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].float()
                Divider_cpu = sample_batches["Divider"].float()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].float()

            if args.is_virial is True:
                Virial_label_cpu = sample_batches["Virial"].float()

            ImageDR_cpu = sample_batches["ImageDR"].float()
            Ri_cpu = sample_batches["Ri"].float()
            Ri_d_cpu = sample_batches["Ri_d"].float()
        else:
            raise Exception("Error! Please specify floating point type: float32 or float64 by the parameter --datatype! ")
        
        dR_neigh_list_cpu = sample_batches["ListNeighbor"].int()
        natoms_img_cpu = sample_batches["ImageAtomNum"].int()
        atom_type_cpu = sample_batches["AtomType"].int()
        # classify batchs according to their atom type and atom nums
        batch_clusters = _classify_batchs(np.array(atom_type_cpu), np.array(natoms_img_cpu))

        for batch_indexs in batch_clusters:
            # transport data to GPU
            natoms_img = Variable(natoms_img_cpu[batch_indexs].int().to(device))
            natoms_img = torch.squeeze(natoms_img, 1)
            natoms = natoms_img[0,1:].sum()
            
            dR_neigh_list = Variable(dR_neigh_list_cpu[batch_indexs, :natoms].int().to(device))
            # atom list of image
            atom_type = Variable(atom_type_cpu[batch_indexs].to(device))
            natom = natoms_img[0,0]
            Ei_label = Variable(Ei_label_cpu[batch_indexs, :natoms].to(device))
            Etot_label = Variable(Etot_label_cpu[batch_indexs].to(device))
            Force_label = Variable(Force_label_cpu[batch_indexs, :natoms].to(device))  # [40,108,3]

            if args.is_egroup is True:
                Egroup_label = Variable(Egroup_label_cpu[batch_indexs, :natoms].to(device))
                Divider = Variable(Divider_cpu[batch_indexs, :natoms].to(device))
                Egroup_weight = Variable(Egroup_weight_cpu[batch_indexs, :natoms, :natoms].to(device))

            if args.is_virial is True:
                Virial_label = Variable(Virial_label_cpu[batch_indexs].to(device))
            
            ImageDR = Variable(ImageDR_cpu[batch_indexs, :natoms].to(device))
            Ri = Variable(Ri_cpu[batch_indexs, :natoms].to(device), requires_grad=True)
            Ri_d = Variable(Ri_d_cpu[batch_indexs, :natoms].to(device))

            batch_size = len(batch_indexs)

            if args.is_egroup is True:
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                    Ri, Ri_d, dR_neigh_list, natoms_img, atom_type, ImageDR, Egroup_weight, Divider)
            else:
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                    Ri, Ri_d, dR_neigh_list, natoms_img, atom_type, ImageDR, None, None 
                )
            # mse
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_F_val = criterion(Force_predict, Force_label)
            loss_Ei_val = criterion(Ei_predict, Ei_label)
            loss_val = loss_F_val + loss_Etot_val
            if args.is_egroup is True:
                loss_Egroup_val = criterion(Egroup_predict, Egroup_label)

            if args.is_virial is True:
                loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))
                loss_Virial_per_atom_val = loss_Virial_val/natom/natom
            # rmse
            Etot_rmse = loss_Etot_val ** 0.5
            etot_atom_rmse = Etot_rmse / natoms_img[0][0]
            Ei_rmse = loss_Ei_val ** 0.5
            F_rmse = loss_F_val ** 0.5

            res_list = [i, float(Etot_label), float(Etot_predict), \
                        float(Ei_label.abs().mean()), float(Ei_predict.abs().mean()), \
                        float(Force_label.abs().mean()), float(Force_predict.abs().mean()),\
                        float(Etot_rmse), float(etot_atom_rmse), float(Ei_rmse), float(F_rmse)]

            if args.is_egroup:
                train_lists.append(loss_Egroup_val)
            if args.is_virial:
                res_list.append(loss_Virial_val)
                res_list.append(loss_Virial_per_atom_val)
            
            force_label_list.append(Force_label.flatten().cpu().numpy())
            force_predict_list.append(Force_predict.flatten().detach().cpu().numpy())
            ei_label_list.append(Ei_label.flatten().cpu().numpy())
            ei_predict_list.append(Ei_predict.flatten().detach().cpu().numpy())
            res_pd.loc[res_pd.shape[0]] = res_list
    
    return res_pd, ei_label_list, ei_predict_list, force_label_list, force_predict_list

def save_checkpoint(state, is_best, filename, prefix):
    filename = os.path.join(prefix, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(prefix, "best.pth.tar"))

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
        total = hvd.allreduce(total, hvd.Sum)
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
        print("\t".join(entries))

    def display_summary(self, entries=[" *"]):
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
