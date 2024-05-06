import os
import pandas as pd
import numpy as np
import time
from enum import Enum
import torch
from torch.utils.data import Subset
from torch.autograd import Variable
from src.loss.dploss import dp_loss, adjust_lr
from src.optimizer.SNES import SNESOptimizer
# import horovod.torch as hvd
from torch.profiler import profile, record_function, ProfilerActivity
from src.user.input_param import InputParam
from user.nep_param import NepParam
from src.model.nep_net import NEP

def train_snes(train_loader, model:NEP, criterion, optimizer:SNESOptimizer, epoch, device, args:InputParam):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    loss_l1 = AverageMeter("loss1", ":.4e", Summary.AVERAGE)
    loss_l2 = AverageMeter("loss2", ":.4e", Summary.AVERAGE)
    loss_Etot = AverageMeter("Etot", ":.4e", Summary.ROOT)
    loss_Etot_per_atom = AverageMeter("Etot_per_atom", ":.4e", Summary.ROOT)
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT)
    loss_Egroup = AverageMeter("Egroup", ":.4e", Summary.ROOT)
    loss_Virial = AverageMeter("Virial", ":.4e", Summary.ROOT)
    loss_Virial_per_atom = AverageMeter("Virial_per_atom", ":.4e", Summary.ROOT)
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup, loss_Virial, loss_Virial_per_atom, loss_l1, loss_l2],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    # Sij_max = 0.0   # max Rij before davg and dstd cacled
    for i, sample_batches in enumerate(train_loader):
        # measure data loading time
        # load data to cpu
        if args.precision == "float64":
            Ei_label_cpu    = sample_batches["Ei"].double()
            Etot_label_cpu  = sample_batches["Etot"].double()
            Force_label_cpu = sample_batches["Force"][:, :, :].double()
            if args.optimizer_param.train_egroup is True:
                Egroup_label_cpu  = sample_batches["Egroup"].double()
                Divider_cpu       = sample_batches["Divider"].double()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].double()

            if args.optimizer_param.train_virial is True:
                Virial_label_cpu  = sample_batches["Virial"].double()
            ImageDR_cpu = sample_batches["ImageDR"].double()

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
        else:
            raise Exception("Error! Please specify floating point type: float32 or float64 by the parameter --datatype! ")
        
        dR_neigh_list_cpu = sample_batches["ListNeighbor"].int()
        natoms_img_cpu    = sample_batches["ImageAtomNum"].int()
        atom_type_cpu     = sample_batches["AtomType"].int()
        atom_type_map_cpu = sample_batches["AtomTypeMap"].int()
        # classify batchs according to their atom type and atom nums
        batch_clusters    = _classify_batchs(np.array(atom_type_cpu), np.array(natoms_img_cpu))

        for batch_indexs in batch_clusters:
            Etot_label = None
            Force_label = None
            Ei_label = None
            Egroup_label = None
            Virial_label = None

            # transport data to GPU
            natoms_img = Variable(natoms_img_cpu[batch_indexs].int().to(device))
            natoms     = natoms_img[0]
            type_nums = len(atom_type_cpu[0])
            dR_neigh_list = Variable(dR_neigh_list_cpu[batch_indexs, :natoms*type_nums].int().to(device))
            # atom list of image
            atom_type     = Variable(atom_type_cpu[batch_indexs].to(device))
            atom_type_map = Variable(atom_type_map_cpu[batch_indexs, :natoms].to(device))
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
            ImageDR = Variable(ImageDR_cpu[batch_indexs, :natoms*type_nums].to(device))
            # Ri = Variable(Ri_cpu[batch_indexs, :natoms].to(device), requires_grad=True)
            # Ri_d = Variable(Ri_d_cpu[batch_indexs, :natoms].to(device))
            batch_size = dR_neigh_list.shape[0]
            if args.optimizer_param.train_egroup is True:
                kalman_inputs = [dR_neigh_list, atom_type_map[0], atom_type[0], ImageDR, Egroup_weight, Divider]
            else:
                # atom_type_map: we only need the first element, because it is same for each image of MOVEMENT
                kalman_inputs = [dR_neigh_list, atom_type_map[0], atom_type[0], ImageDR, None, None]
            # 调用SNES
            _low_loss, _low_mse_etot, _low_mse_ei, _low_mse_F, _low_mse_Egroup, _low_mse_Virial, _low_L1, _low_L2 = \
                optimizer.update_params(
                        kalman_inputs, 
                        Etot_label, 
                        Force_label, 
                        Ei_label,
                        Egroup_label, 
                        Virial_label, 
                        criterion, 
                        train_type="NEP"
                )
            losses.update(_low_loss.item(), batch_size)
            loss_l1.update(_low_L1.item(), batch_size)
            loss_l2.update(_low_L2.item(), batch_size)
            # loss_Etot.update(_low_mse_etot.item(), batch_size)
            loss_Etot.update(_low_mse_etot.item(), batch_size) if _low_mse_etot is not None else loss_Etot.update(0, batch_size)
            # loss_Etot_per_atom.update((_low_mse_etot/natoms/natoms).item(), batch_size)
            loss_Etot_per_atom.update((_low_mse_etot/natoms/natoms).item(), batch_size) if _low_mse_etot is not None else loss_Etot_per_atom.update(0, batch_size)
            # loss_Ei.update(_low_mse_ei.item(), batch_size) 
            loss_Ei.update(_low_mse_ei.item(), batch_size) if _low_mse_ei is not None else loss_Ei.update(0, batch_size)
            # loss_Egroup.update(_low_mse_Egroup.item(), batch_size)
            loss_Egroup.update(_low_mse_Egroup.item(), batch_size) if _low_mse_Egroup is not None else loss_Egroup.update(0, batch_size)
            # loss_Virial.update(_low_mse_Virial.item(), batch_size)
            loss_Virial.update(_low_mse_Virial.item(), batch_size) if _low_mse_Virial is not None else loss_Virial.update(0, batch_size)
            # loss_Virial_per_atom.update((_low_mse_Virial/natoms/natoms).item(), batch_size)
            loss_Virial_per_atom.update((_low_mse_Virial/natoms/natoms).item(), batch_size) if _low_mse_Virial is not None else loss_Virial_per_atom.update(0, batch_size)
            # loss_Force.update(_low_mse_F.item(), batch_size)
            loss_Force.update(_low_mse_F.item(), batch_size) if _low_mse_F is not None else loss_Force.update(0, batch_size)

        # loss = "{:18} {:18} {:18} {:18} {:18} {:18} {:18} {:18}".format(
        #         _low_loss, _low_rmse_etot, _low_rmse_ei, _low_rmse_F, _low_rmse_Egroup, _low_rmse_Virial, _low_L1, _low_L2)

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
    return losses.avg, loss_Etot.root, loss_Etot_per_atom.root, loss_Force.root, loss_Ei.root, loss_Egroup.root, loss_Virial.root, loss_Virial_per_atom.root, loss_l1.avg, loss_l2.avg

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
        key += "{}".format(img_natoms[i])
        if key in dicts.keys():
            dicts[key].append(i)
        else:
            dicts[key] = [i]
    return [dicts[_] for _ in dicts.keys()]


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
        print("\t".join(entries), flush=True)

    def display_summary(self, entries=[" *"]):
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
