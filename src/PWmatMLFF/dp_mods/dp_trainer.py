import os
import shutil
import time
from enum import Enum
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Subset
from torch.autograd import Variable
import sys
import pathlib

codepath = str(pathlib.Path(__file__).parent.resolve())
#for optimizer 
sys.path.append(codepath+'/../..')

from loss.dploss import dp_loss, adjust_lr
from optimizer.KFWrapper import KFOptimizerWrapper

# import parameters as pm

def train(train_loader, model, criterion, optimizer, epoch, start_lr, device, config):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    loss_Etot = AverageMeter("Etot", ":.4e", Summary.ROOT)
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_Etot, loss_Force, loss_Ei],
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
            param_group['lr'] = real_lr * (nr_batch_sample ** 0.5)
        
        if config.datatype == "float64":
            Ei_label = Variable(
                sample_batches["Ei"].double().to(device)
            )
            Force_label = Variable(
                sample_batches["Force"][:, :, :].double().to(device)
            )  # [40,108,3]
           
            Ri = Variable(
                sample_batches["Ri"].double().to(device), requires_grad=True
            )
            Ri_d = Variable(sample_batches["Ri_d"].to(device))
           
        elif config.datatype == "float32":
            Ei_label = Variable(
                sample_batches["Ei"].float().to(device)
            )
            Force_label = Variable(
                sample_batches["Force"][:, :, :].float().to(device)
            )  # [40,108,3]
             
            
            Ri = Variable(
                sample_batches["Ri"].float().to(device), requires_grad=True
            )
            Ri_d = Variable(sample_batches["Ri_d"].float().to(device))
            
        Etot_label = torch.sum(torch.unsqueeze(Ei_label, 2), dim=1)
        dR_neigh_list = Variable(
                sample_batches["ListNeighbor"].int().to(device)
            )
        natoms_img = Variable(
                sample_batches["ImageAtomNum"].int().to(device)
            )
        natoms_img = torch.squeeze(natoms_img, 1)
        batch_size = Ri.shape[0]
        Etot_predict, Ei_predict, Force_predict = model(
            Ri, Ri_d, dR_neigh_list, natoms_img, None, None
        )

        optimizer.zero_grad()

        loss_F_val = criterion(Force_predict, Force_label)
        loss_Etot_val = criterion(Etot_predict, Etot_label)
        # loss_Ei_val = criterion(Ei_predict, Ei_label)
        loss_Ei_val, loss_Egroup_val = 0, 0
        loss_val = loss_F_val + loss_Etot_val

        w_f, w_e, w_eg, w_ei = 1, 1, 0, 0
        loss, _, _ = dp_loss(
            0.001,
            real_lr,
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

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss_val.item(), batch_size)
        loss_Etot.update(loss_Etot_val.item(), batch_size)
        # loss_Ei.update(loss_Ei_val.item(), batch_size)
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
        loss_Force.root,
        loss_Ei.root,
        batch_time.sum,
        real_lr,
    )


def train_KF(train_loader, model, criterion, optimizer, epoch, device, config):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    loss_Etot = AverageMeter("Etot", ":.4e", Summary.ROOT)
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_Etot, loss_Force, loss_Ei],
        prefix="Epoch: [{}]".format(epoch),
    )

    KFOptWrapper = KFOptimizerWrapper(
        model, optimizer, config.nselect, config.groupsize, config.distributed, "torch"
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample_batches in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if config.datatype == "float64":
            Ei_label = Variable(
                sample_batches["Ei"].double().to(device)
            )
            Force_label = Variable(
                sample_batches["Force"][:, :, :].double().to(device)
            )  # [40,108,3]

            Ri = Variable(
                sample_batches["Ri"].double().to(device), requires_grad=True
            )
            Ri_d = Variable(sample_batches["Ri_d"].to(device))
    

        elif config.datatype == "float32":
            Ei_label = Variable(
                sample_batches["Ei"].float().to(device)
            )
            Force_label = Variable(
                sample_batches["Force"][:, :, :].float().to(device)
            )  # [40,108,3]
           
            Ri = Variable(
                sample_batches["Ri"].float().to(device), requires_grad=True
            )
            Ri_d = Variable(sample_batches["Ri_d"].float().to(device))
  
        Etot_label = torch.sum(Ei_label.unsqueeze(2), dim=1)
        dR_neigh_list = Variable(
                sample_batches["ListNeighbor"].int().to(device)
            )
        natoms_img = Variable(
                sample_batches["ImageAtomNum"].int().to(device)
            )
        natoms_img = torch.squeeze(natoms_img, 1)

        Etot_predict, Ei_predict, Force_predict = model(
            Ri, Ri_d, dR_neigh_list, natoms_img, None, None
        )

        loss_F_val = criterion(Force_predict, Force_label)
        loss_Etot_val = criterion(Etot_predict, Etot_label)
        loss_Ei_val = criterion(Ei_predict, Ei_label)
        loss_val = loss_F_val + loss_Etot_val

        batch_size = Ri.shape[0]
        kalman_inputs = [Ri, Ri_d, dR_neigh_list, natoms_img, None, None]
        
        KFOptWrapper.update_energy(kalman_inputs, Etot_label)
        KFOptWrapper.update_force(kalman_inputs, Force_label, 2)

        # measure accuracy and record loss
        losses.update(loss_val.item(), batch_size)
        loss_Etot.update(loss_Etot_val.item(), batch_size)
        loss_Ei.update(loss_Ei_val.item(), batch_size)
        loss_Force.update(loss_F_val.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i + 1)
    
    if config.distributed:
        losses.all_reduce()
        loss_Etot.all_reduce()
        loss_Force.all_reduce()
        loss_Ei.all_reduce()
        batch_time.all_reduce()

    progress.display_summary(["Training Set:"])
    return losses.avg, loss_Etot.root, loss_Force.root, loss_Ei.root, batch_time.sum


def valid(val_loader, model, criterion, device, args):
    def run_validate(loader, base_progress=0):
        end = time.time()
        for i, sample_batches in enumerate(loader):
            i = base_progress + i
            if args.datatype == "float64":
                Ei_label = Variable(
                    sample_batches["Ei"].double().to(device)
                )
                Force_label = Variable(
                    sample_batches["Force"][:, :, :].double().to(device)
                )  # [40,108,3]
                
                Ri = Variable(
                    sample_batches["Ri"].double().to(device),
                    requires_grad=True,
                )
                Ri_d = Variable(sample_batches["Ri_d"].to(device))
               

            elif args.datatype == "float32":
                Ei_label = Variable(
                    sample_batches["Ei"].float().to(device)
                )
                Force_label = Variable(
                    sample_batches["Force"][:, :, :].float().to(device)
                )  # [40,108,3]
               
                Ri = Variable(
                    sample_batches["Ri"].float().to(device),
                    requires_grad=True,
                )
                Ri_d = Variable(sample_batches["Ri_d"].float().to(device))
                
            Etot_label = torch.sum(torch.unsqueeze(Ei_label, 2), dim=1)
            dR_neigh_list = Variable(
                sample_batches["ListNeighbor"].int().to(device)
            )
            natoms_img = Variable(
                sample_batches["ImageAtomNum"].int().to(device)
            )
            natoms_img = torch.squeeze(natoms_img, 1)

            batch_size = Ri.shape[0]
            Etot_predict, Ei_predict, Force_predict = model(
                Ri, Ri_d, dR_neigh_list, natoms_img, None, None
            )

            loss_F_val = criterion(Force_predict, Force_label)
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_Ei_val = criterion(Ei_predict, Ei_label)
            loss_val = loss_F_val + loss_Etot_val

            # measure accuracy and record loss
            losses.update(loss_val.item(), batch_size)
            loss_Etot.update(loss_Etot_val.item(), batch_size)
            loss_Ei.update(loss_Ei_val.item(), batch_size)
            loss_Force.update(loss_F_val.item(), batch_size)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i + 1)

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    loss_Etot = AverageMeter("Etot", ":.4e", Summary.ROOT)
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT)
    progress = ProgressMeter(
        len(val_loader)
        + (
            args.distributed
            and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))
        ),
        [batch_time, losses, loss_Etot, loss_Force, loss_Ei],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        losses.all_reduce()
        loss_Etot.all_reduce()
        loss_Force.all_reduce()
        loss_Ei.all_reduce()

    if args.distributed and (
        len(val_loader.sampler) * args.world_size < len(val_loader.dataset)
    ):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary(["Test Set:"])

    return losses.avg, loss_Etot.root, loss_Force.root, loss_Ei.root


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
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
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
