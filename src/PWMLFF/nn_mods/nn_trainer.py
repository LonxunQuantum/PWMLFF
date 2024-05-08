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
# import horovod.torch as hvd
from torch.profiler import profile, record_function, ProfilerActivity
from src.aux.plot_nn_inference import plot
from src.user.input_param import InputParam
from utils.file_operation import write_line_to_file

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

        nr_batch_sample = sample_batches["input_feat"].shape[0]
        global_step = (epoch - 1) * len(train_loader) + i * nr_batch_sample
        real_lr = adjust_lr(global_step, start_lr, \
                            args.optimizer_param.stop_step, args.optimizer_param.decay_step, args.optimizer_param.stop_lr) #  stop_step, decay_step

        for param_group in optimizer.param_groups:
            param_group["lr"] = real_lr * (nr_batch_sample**0.5)

        if args.precision == "float64":
            Ei_label_cpu = sample_batches['output_energy'][:,:,:].double()
            Force_label_cpu = sample_batches['output_force'][:,:,:].double()
            input_data_cpu = sample_batches['input_feat'].double()
            dfeat_cpu = sample_batches['input_dfeat'].double()
            if args.file_paths.alive_atomic_energy and args.optimizer_param.train_egroup:
                Egroup_label_cpu = sample_batches['input_egroup'].double()
                egroup_weight_cpu = sample_batches['input_egroup_weight'].double()
                divider_cpu = sample_batches['input_divider'].double()
            if args.optimizer_param.train_egroup is True:
                pass
            if args.optimizer_param.train_virial is True:
                Virial_label_cpu = sample_batches["Virial"].double()
        else:
            Ei_label_cpu = sample_batches['output_energy'][:,:,:].float()
            Force_label_cpu = sample_batches['output_force'][:,:,:].float()
            input_data_cpu = sample_batches['input_feat'].float()
            dfeat_cpu = sample_batches['input_dfeat'].float()
            if args.file_paths.alive_atomic_energy and args.optimizer_param.train_egroup:
                Egroup_label = sample_batches['input_egroup'].float()
                egroup_weight = sample_batches['input_egroup_weight'].float()
                divider = sample_batches['input_divider'].float()
            if args.optimizer_param.train_egroup is True:
                pass
            if args.optimizer_param.train_virial is True:
                Virial_label_cpu = sample_batches["Virial"].double()

        neighbor_cpu = sample_batches['input_nblist'].int()
        natoms_img_cpu = sample_batches['natoms_img'].int()
        atom_type_cpu = sample_batches['atom_type'].int()

        Ei_label = Variable(Ei_label_cpu.to(device))
        Etot_label = torch.sum(Ei_label, dim=1)
        Force_label = Variable(Force_label_cpu.to(device))   #[40,108,3]
        input_data = Variable(input_data_cpu.to(device), requires_grad=True)
        dfeat = Variable(dfeat_cpu.to(device))
        if args.file_paths.alive_atomic_energy and args.optimizer_param.train_egroup:
            Egroup_label = Variable(Egroup_label_cpu.to(device))
            egroup_weight = Variable(egroup_weight_cpu.to(device))
            divider = Variable(divider_cpu.to(device))
        if args.optimizer_param.train_egroup is True:
            pass
        if args.optimizer_param.train_virial is True:
            Virial_label = Variable(Virial_label_cpu.to(device))

        neighbor = Variable(neighbor_cpu.to(device))  # [40,108,100]
        natoms_img = Variable(natoms_img_cpu.to(device))
        natom = natoms_img[0,1:].sum()
        atom_type = Variable(atom_type_cpu.to(device))
        batch_size = len(sample_batches)

        if args.optimizer_param.train_egroup:
            kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, egroup_weight, divider]
        else:
            kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, None, None]
        
        Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = \
            model(kalman_inputs[0], kalman_inputs[1], kalman_inputs[2], kalman_inputs[3], kalman_inputs[4], kalman_inputs[5], kalman_inputs[6])
                    
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
        if args.optimizer_param.train_egroup is True:
            loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
        if args.optimizer_param.train_virial is True:
            loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))
            loss_Virial_per_atom_val = loss_Virial_val/natom/natom

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
                natoms_img[0, 0].item(),
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
                natoms_img[0, 0].item(),
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
                natoms_img[0, 0].item(),
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
        if args.optimizer_param.train_egroup is True:
            loss_Egroup.update(loss_Egroup_val.item(), batch_size)
        if args.optimizer_param.train_virial is True:
            loss_Virial.update(loss_Virial_val.item(), batch_size)
            loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), batch_size)
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
    )

'''
description: 
    Virial not realized!
param {*} train_loader
param {*} model
param {*} criterion
param {*} optimizer
param {*} epoch
param {*} device
param {InputParam} args
return {*}
author: wuxingxing
'''
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
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup, loss_Virial, loss_Virial_per_atom],
        prefix="Epoch: [{}]".format(epoch),
    )

    KFOptWrapper = KFOptimizerWrapper(
        model, optimizer, args.optimizer_param.nselect, args.optimizer_param.groupsize
    )
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, sample_batches in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # load data to cpu
        if args.precision == "float64":
            Ei_label_cpu = sample_batches['output_energy'][:,:,:].double()
            Force_label_cpu = sample_batches['output_force'][:,:,:].double()
            input_data_cpu = sample_batches['input_feat'].double()
            dfeat_cpu = sample_batches['input_dfeat'].double()
            if args.file_paths.alive_atomic_energy and args.optimizer_param.train_egroup:
                Egroup_label_cpu = sample_batches['input_egroup'].double()
                egroup_weight_cpu = sample_batches['input_egroup_weight'].double()
                divider_cpu = sample_batches['input_divider'].double()
            if args.optimizer_param.train_egroup is True:
                pass
            if args.optimizer_param.train_virial is True:
                Virial_label_cpu = sample_batches["Virial"].double()
        else:
            Ei_label_cpu = sample_batches['output_energy'][:,:,:].float()
            Force_label_cpu = sample_batches['output_force'][:,:,:].float()
            input_data_cpu = sample_batches['input_feat'].float()
            dfeat_cpu = sample_batches['input_dfeat'].float()
            if args.file_paths.alive_atomic_energy and args.optimizer_param.train_egroup:
                Egroup_label = sample_batches['input_egroup'].float()
                egroup_weight = sample_batches['input_egroup_weight'].float()
                divider = sample_batches['input_divider'].float()
            if args.optimizer_param.train_egroup is True:
                pass
            if args.optimizer_param.train_virial is True:
                Virial_label_cpu = sample_batches["Virial"].double()

        neighbor_cpu = sample_batches['input_nblist'].int()
        natoms_img_cpu = sample_batches['natoms_img'].int()
        atom_type_cpu = sample_batches['atom_type'].int()

        Ei_label = Variable(Ei_label_cpu.to(device))
        Etot_label = torch.sum(Ei_label, dim=1)
        Force_label = Variable(Force_label_cpu.to(device))   #[40,108,3]
        input_data = Variable(input_data_cpu.to(device), requires_grad=True)
        dfeat = Variable(dfeat_cpu.to(device))
        if args.file_paths.alive_atomic_energy and args.optimizer_param.train_egroup:
            Egroup_label = Variable(Egroup_label_cpu.to(device))
            egroup_weight = Variable(egroup_weight_cpu.to(device))
            divider = Variable(divider_cpu.to(device))
        if args.optimizer_param.train_egroup is True:
            pass
        if args.optimizer_param.train_virial is True:
            Virial_label = Variable(Virial_label_cpu.to(device))

        neighbor = Variable(neighbor_cpu.to(device))  # [40,108,100]
        natoms_img = Variable(natoms_img_cpu.to(device))
        natom = natoms_img[0,1:].sum()
        atom_type = Variable(atom_type_cpu.to(device))
        batch_size = Etot_label.shape[0]
        if args.optimizer_param.train_egroup:
            kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, egroup_weight, divider]
        else:
            kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, None, None]

        if args.optimizer_param.train_energy: 
            # kalman.update_energy(kalman_inputs, Etot_label, update_prefactor = args.optimizer_param.pre_fac_etot)
            Etot_predict = KFOptWrapper.update_energy(kalman_inputs, Etot_label, args.optimizer_param.pre_fac_etot, train_type = "NN")
            
        if args.optimizer_param.train_ei:
            Ei_predict = KFOptWrapper.update_ei(kalman_inputs,Ei_label, update_prefactor = args.optimizer_param.pre_fac_ei, train_type = "NN")     

        if args.optimizer_param.train_egroup:
            # kalman.update_egroup(kalman_inputs, Egroup_label)
            Egroup_predict = KFOptWrapper.update_egroup(kalman_inputs, Egroup_label, args.optimizer_param.pre_fac_egroup, train_type = "NN")

        # if Egroup does not participate in training, the output of Egroup_predict will be None
        if args.optimizer_param.train_force:
            Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = KFOptWrapper.update_force(
                    kalman_inputs, Force_label, args.optimizer_param.pre_fac_force, train_type = "NN")

        Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(kalman_inputs[0], kalman_inputs[1], kalman_inputs[2], kalman_inputs[3], kalman_inputs[4], kalman_inputs[5], kalman_inputs[6])

        loss_F_val = criterion(Force_predict, Force_label)

        # divide by natom 
        loss_Etot_val = criterion(Etot_predict, Etot_label)
        loss_Etot_per_atom_val = loss_Etot_val/natom/natom
        
        loss_Ei_val = criterion(Ei_predict, Ei_label)   
        if args.optimizer_param.train_egroup is True:
            loss_Egroup_val = criterion(Egroup_predict, Egroup_label)

        if args.optimizer_param.train_virial is True:
            loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))
            loss_Virial_per_atom_val = loss_Virial_val/natom/natom
        loss_val = loss_F_val + loss_Etot_val*natom

        # measure accuracy and record loss
        losses.update(loss_val.item(), batch_size)

        loss_Etot.update(loss_Etot_val.item(), batch_size)
        loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
        loss_Ei.update(loss_Ei_val.item(), batch_size)
        if args.optimizer_param.train_egroup is True:
            loss_Egroup.update(loss_Egroup_val.item(), batch_size)
        if args.optimizer_param.train_virial is True:
            loss_Virial.update(loss_Virial_val.item(), batch_size)
            loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), batch_size)
        loss_Force.update(loss_F_val.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.optimizer_param.print_freq == 0:
            progress.display(i + 1)
 
    # if args.hvd:
    #     losses.all_reduce()
    #     loss_Etot.all_reduce()
    #     loss_Etot_per_atom.all_reduce()
    #     loss_Force.all_reduce()
    #     loss_Ei.all_reduce()
    #     if args.optimizer_param.train_egroup is True:
    #         loss_Egroup.all_reduce()
    #     if args.optimizer_param.train_virial is True:
    #         loss_Virial.all_reduce()
    #         loss_Virial_per_atom.all_reduce()
    #     batch_time.all_reduce()

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

def valid(val_loader, model, criterion, device, args:InputParam):
    def run_validate(loader, base_progress=0):
        end = time.time()
        for i, sample_batches in enumerate(loader):
            if args.precision == "float64":
                Ei_label_cpu = sample_batches['output_energy'][:,:,:].double()
                Force_label_cpu = sample_batches['output_force'][:,:,:].double()
                input_data_cpu = sample_batches['input_feat'].double()
                dfeat_cpu = sample_batches['input_dfeat'].double()
                if args.file_paths.alive_atomic_energy and args.optimizer_param.train_egroup:
                    Egroup_label_cpu = sample_batches['input_egroup'].double()
                    egroup_weight_cpu = sample_batches['input_egroup_weight'].double()
                    divider_cpu = sample_batches['input_divider'].double()
                if args.optimizer_param.train_egroup is True:
                    pass
                if args.optimizer_param.train_virial is True:
                    Virial_label_cpu = sample_batches["Virial"].double()
            else:
                Ei_label_cpu = sample_batches['output_energy'][:,:,:].float()
                Force_label_cpu = sample_batches['output_force'][:,:,:].float()
                input_data_cpu = sample_batches['input_feat'].float()
                dfeat_cpu = sample_batches['input_dfeat'].float()
                if args.file_paths.alive_atomic_energy and args.optimizer_param.train_egroup:
                    Egroup_label = sample_batches['input_egroup'].float()
                    egroup_weight = sample_batches['input_egroup_weight'].float()
                    divider = sample_batches['input_divider'].float()
                if args.optimizer_param.train_egroup is True:
                    pass
                if args.optimizer_param.train_virial is True:
                    Virial_label_cpu = sample_batches["Virial"].double()

            neighbor_cpu = sample_batches['input_nblist'].int()
            natoms_img_cpu = sample_batches['natoms_img'].int()
            atom_type_cpu = sample_batches['atom_type'].int()

            Ei_label = Variable(Ei_label_cpu.to(device))
            Etot_label = torch.sum(Ei_label, dim=1)
            Force_label = Variable(Force_label_cpu.to(device))   #[40,108,3]
            input_data = Variable(input_data_cpu.to(device), requires_grad=True)
            dfeat = Variable(dfeat_cpu.to(device))
            if args.file_paths.alive_atomic_energy and args.optimizer_param.train_egroup:
                Egroup_label = Variable(Egroup_label_cpu.to(device))
                egroup_weight = Variable(egroup_weight_cpu.to(device))
                divider = Variable(divider_cpu.to(device))
            if args.optimizer_param.train_egroup is True:
                pass
            if args.optimizer_param.train_virial is True:
                Virial_label = Variable(Virial_label_cpu.to(device))

            neighbor = Variable(neighbor_cpu.to(device))  # [40,108,100]
            natoms_img = Variable(natoms_img_cpu.to(device))
            natom = natoms_img[0,1:].sum()
            atom_type = Variable(atom_type_cpu.to(device))
            batch_size = len(sample_batches)

            if args.optimizer_param.train_egroup:
                kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, egroup_weight, divider]
            else:
                kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, None, None]
            Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = \
                model(kalman_inputs[0], kalman_inputs[1], kalman_inputs[2], kalman_inputs[3], kalman_inputs[4], kalman_inputs[5], kalman_inputs[6])
                    
            #return 
            loss_F_val = criterion(Force_predict, Force_label)
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_Etot_per_atom_val = loss_Etot_val/natom/natom
            loss_Ei_val = criterion(Ei_predict, Ei_label)
            if args.optimizer_param.train_egroup is True:
                loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
            if args.optimizer_param.train_virial is True:
                loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))
                loss_Virial_per_atom_val = loss_Virial_val/natom/natom
            loss_val = loss_F_val + loss_Etot_val*natom

            # measure accuracy and record loss
            losses.update(loss_val.item(), batch_size)
            loss_Etot.update(loss_Etot_val.item(), batch_size)
            loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
            loss_Ei.update(loss_Ei_val.item(), batch_size)
            if args.optimizer_param.train_egroup is True:
                loss_Egroup.update(loss_Egroup_val.item(), batch_size)
            if args.optimizer_param.train_virial is True:
                loss_Virial.update(loss_Virial_val.item(), batch_size)
                loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), batch_size)
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

    # if args.hvd and (len(val_loader.sampler) * hvd.size() < len(val_loader.dataset)):
    #     aux_val_dataset = Subset(
    #         val_loader.dataset,
    #         range(len(val_loader.sampler) * hvd.size(), len(val_loader.dataset)),
    #     )
    #     aux_val_loader = torch.utils.data.DataLoader(
    #         aux_val_dataset,
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         num_workers=args.workers,
    #         pin_memory=True,
    #     )
    #     run_validate(aux_val_loader, len(val_loader))

    # if args.hvd:
    #     losses.all_reduce()
    #     loss_Etot.all_reduce()
    #     loss_Etot_per_atom.all_reduce()
    #     loss_Force.all_reduce()
    #     loss_Ei.all_reduce()
    #     if args.optimizer_param.train_virial is True:
    #         loss_Virial.all_reduce()
    #         loss_Virial_per_atom.all_reduce()

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
def predict(val_loader, model, criterion, device, args:InputParam):
    train_lists = ["img_idx", "RMSE_Etot", "RMSE_Etot_per_atom", "RMSE_Ei", "RMSE_F"]
    if args.optimizer_param.train_egroup:
        train_lists.append("RMSE_Egroup")
    res_pd = pd.DataFrame(columns=train_lists)

    inf_dir = args.file_paths.test_dir
    if os.path.exists(inf_dir) is True:
        shutil.rmtree(inf_dir)
    os.mkdir(inf_dir)
    res_pd_save_path = os.path.join(inf_dir, "inference_loss.csv")
    inf_force_save_path = os.path.join(inf_dir,"inference_force.txt")
    lab_force_save_path = os.path.join(inf_dir,"dft_force.txt")
    inf_energy_save_path = os.path.join(inf_dir,"inference_total_energy.txt")
    lab_energy_save_path = os.path.join(inf_dir,"dft_total_energy.txt")
    if args.file_paths.alive_atomic_energy:
        inf_Ei_save_path = os.path.join(inf_dir,"inference_atomic_energy.txt")
        lab_Ei_save_path = os.path.join(inf_dir,"dft_atomic_energy.txt")
    inference_path = os.path.join(inf_dir,"inference_summary.txt") 

    for i, sample_batches in enumerate(val_loader):
        if args.precision == "float64":
            Ei_label_cpu = sample_batches['output_energy'][:,:,:].double()
            Force_label_cpu = sample_batches['output_force'][:,:,:].double()
            input_data_cpu = sample_batches['input_feat'].double()
            dfeat_cpu = sample_batches['input_dfeat'].double()
            if args.file_paths.alive_atomic_energy and args.optimizer_param.train_egroup:
                Egroup_label_cpu = sample_batches['input_egroup'].double()
                egroup_weight_cpu = sample_batches['input_egroup_weight'].double()
                divider_cpu = sample_batches['input_divider'].double()
            if args.optimizer_param.train_egroup is True:
                pass
            if args.optimizer_param.train_virial is True:
                Virial_label_cpu = sample_batches["Virial"].double()
        else:
            Ei_label_cpu = sample_batches['output_energy'][:,:,:].float()
            Force_label_cpu = sample_batches['output_force'][:,:,:].float()
            input_data_cpu = sample_batches['input_feat'].float()
            dfeat_cpu = sample_batches['input_dfeat'].float()
            if args.file_paths.alive_atomic_energy and args.optimizer_param.train_egroup:
                Egroup_label = sample_batches['input_egroup'].float()
                egroup_weight = sample_batches['input_egroup_weight'].float()
                divider = sample_batches['input_divider'].float()
            if args.optimizer_param.train_egroup is True:
                pass
            if args.optimizer_param.train_virial is True:
                Virial_label_cpu = sample_batches["Virial"].double()

        neighbor_cpu = sample_batches['input_nblist'].int()
        natoms_img_cpu = sample_batches['natoms_img'].int()
        atom_type_cpu = sample_batches['atom_type'].int()

        Ei_label = Variable(Ei_label_cpu.to(device))
        Etot_label = torch.sum(Ei_label, dim=1)
        Force_label = Variable(Force_label_cpu.to(device))   #[40,108,3]
        input_data = Variable(input_data_cpu.to(device), requires_grad=True)
        dfeat = Variable(dfeat_cpu.to(device))
        if args.file_paths.alive_atomic_energy and args.optimizer_param.train_egroup:
            Egroup_label = Variable(Egroup_label_cpu.to(device))
            egroup_weight = Variable(egroup_weight_cpu.to(device))
            divider = Variable(divider_cpu.to(device))
        if args.optimizer_param.train_egroup is True:
            pass
        if args.optimizer_param.train_virial is True:
            Virial_label = Variable(Virial_label_cpu.to(device))

        neighbor = Variable(neighbor_cpu.to(device))  # [40,108,100]
        natoms_img = Variable(natoms_img_cpu.to(device))
        natom = natoms_img[0,1:].sum()
        atom_type = Variable(atom_type_cpu.to(device))
        batch_size = len(sample_batches)

        if args.optimizer_param.train_egroup:
            kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, egroup_weight, divider]
        else:
            kalman_inputs = [input_data, dfeat, neighbor, natoms_img, atom_type, None, None]
        Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = \
            model(kalman_inputs[0], kalman_inputs[1], kalman_inputs[2], kalman_inputs[3], kalman_inputs[4], kalman_inputs[5], kalman_inputs[6])

        # mse
        loss_Etot_val = criterion(Etot_predict, Etot_label)
        loss_F_val = criterion(Force_predict, Force_label)
        loss_Ei_val = criterion(Ei_predict, Ei_label)
        if args.optimizer_param.train_egroup is True:
            loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
        # rmse
        Etot_rmse = loss_Etot_val ** 0.5
        etot_atom_rmse = Etot_rmse / natoms_img[0][0]
        Ei_rmse = loss_Ei_val ** 0.5
        F_rmse = loss_F_val ** 0.5

        res_list = [i, float(Etot_rmse), float(etot_atom_rmse), float(Ei_rmse), float(F_rmse)]
        if args.optimizer_param.train_egroup:
            res_list.append(float(loss_Egroup_val))
        res_pd.loc[res_pd.shape[0]] = res_list
        
        #''.join(map(str, list(np.array(Force_predict.flatten().cpu().data))))
        ''.join(map(str, list(np.array(Force_predict.flatten().cpu().data))))
        write_line_to_file(inf_force_save_path, \
                            ' '.join(np.array(Force_predict.flatten().cpu().data).astype('str')), "a")
        write_line_to_file(lab_force_save_path, \
                            ' '.join(np.array(Force_label.flatten().cpu().data).astype('str')), "a")
        if args.file_paths.alive_atomic_energy:
            write_line_to_file(inf_Ei_save_path, \
                            ' '.join(np.array(Ei_predict.flatten().cpu().data).astype('str')), "a")
            write_line_to_file(lab_Ei_save_path, \
                            ' '.join(np.array(Ei_label.flatten().cpu().data).astype('str')), "a")
        
        write_line_to_file(inf_energy_save_path, \
                            ' '.join(np.array(Etot_predict.flatten().cpu().data).astype('str')), "a")
        write_line_to_file(lab_energy_save_path, \
                            ' '.join(np.array(Etot_label.flatten().cpu().data).astype('str')), "a")
        
    res_pd.to_csv(res_pd_save_path)
    
    inference_cout = ""
    inference_cout += "For {} images: \n".format(res_pd.shape[0])
    inference_cout += "Avarage REMSE of Etot: {} \n".format(res_pd['RMSE_Etot'].mean())
    inference_cout += "Avarage REMSE of Etot per atom: {} \n".format(res_pd['RMSE_Etot_per_atom'].mean())
    if args.file_paths.alive_atomic_energy:
        inference_cout += "Avarage REMSE of Ei: {} \n".format(res_pd['RMSE_Ei'].mean())
    inference_cout += "Avarage REMSE of RMSE_F: {} \n".format(res_pd['RMSE_F'].mean())
    if args.optimizer_param.train_egroup:
        inference_cout += "Avarage REMSE of RMSE_Egroup: {} \n".format(res_pd['RMSE_Egroup'].mean())
    # if args.optimizer_param.train_virial:  #not realized
    #     inference_cout += "Avarage REMSE of RMSE_virial: {} \n".format(res_pd['RMSE_virial'].mean())
    #     inference_cout += "Avarage REMSE of RMSE_virial_per_atom: {} \n".format(res_pd['RMSE_virial_per_atom'].mean())

    inference_cout += "\nMore details can be found under the file directory:\n{}\n".format(inf_dir)
    print(inference_cout)

    if args.file_paths.alive_atomic_energy:
        if args.optimizer_param.train_ei or args.optimizer_param.train_egroup:
            plot_ei = True
        else:
            plot_ei = False
    else:
        plot_ei = False
        
    plot(inf_dir, plot_ei = plot_ei)

    with open(inference_path, 'w') as wf:
        wf.writelines(inference_cout)
    return
    
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
        print("\t".join(entries))

    def display_summary(self, entries=[" *"]):
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
