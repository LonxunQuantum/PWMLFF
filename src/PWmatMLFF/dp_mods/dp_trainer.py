import os
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
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_Virial = AverageMeter("Virial", ":.4e", Summary.ROOT)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT)
    loss_Egroup = AverageMeter("Egroup", ":.4e", Summary.ROOT)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_Etot, loss_Force, loss_Ei, loss_Egroup, loss_Virial],
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
            Ei_label = Variable(sample_batches["Ei"].double().to(device))
            Egroup_label = Variable(sample_batches["Egroup"].double().to(device))
            Divider = Variable(sample_batches["Divider"].double().to(device))
            Egroup_weight = Variable(
                sample_batches["Egroup_weight"].double().to(device)
            )
            Force_label = Variable(
                sample_batches["Force"][:, :, :].double().to(device)
            )  # [40,108,3]
            Virial_label = Variable(
                sample_batches["Virial"].double().to(device)
            ).squeeze(1)

            ImageDR = Variable(sample_batches["ImageDR"].double().to(device))
            Ri = Variable(sample_batches["Ri"].double().to(device), requires_grad=True)
            Ri_d = Variable(sample_batches["Ri_d"].to(device))

        elif config.datatype == "float32":
            Ei_label = Variable(sample_batches["Ei"].float().to(device))
            Egroup_label = Variable(sample_batches["Egroup"].float().to(device))
            Divider = Variable(sample_batches["Divider"].float().to(device))
            Egroup_weight = Variable(sample_batches["Egroup_weight"].float().to(device))
            Force_label = Variable(
                sample_batches["Force"][:, :, :].float().to(device)
            )  # [40,108,3]
            Virial_label = Variable(
                sample_batches["Virial"].float().to(device))

            ImageDR = Variable(sample_batches["ImageDR"].float().to(device))
            Ri = Variable(sample_batches["Ri"].float().to(device), requires_grad=True)
            Ri_d = Variable(sample_batches["Ri_d"].float().to(device))

        Etot_label = torch.sum(torch.unsqueeze(Ei_label, 2), dim=1)
        dR_neigh_list = Variable(sample_batches["ListNeighbor"].int().to(device))
        natoms_img = Variable(sample_batches["ImageAtomNum"].int().to(device))
        natoms_img = torch.squeeze(natoms_img, 1)
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
            Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                ImageDR, Ri, Ri_d, dR_neigh_list, natoms_img, Egroup_weight, Divider
            )

        optimizer.zero_grad()

        loss_F_val = criterion(Force_predict, Force_label)
        loss_Etot_val = criterion(Etot_predict, Etot_label)
        loss_Ei_val = criterion(Ei_predict, Ei_label)
        loss_Ei_val = 0

        #print ("Etot_predict") 
        #print (Etot_predict)
        
        #print ("Force_predict")
        #print (Force_predict)
        
        #print ("Virial_predict")
        #print (Virial_predict)
        #Ei_predict, Force_predict, Egroup_predict, Virial_predict)
        #print("Egroup_predict",Egroup_predict)

        #print("Egroup_label",Egroup_label)
        loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
        
        loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))

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

        loss, _, _ = dp_loss(
            0.001,
            real_lr,
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
        # import ipdb;ipdb.set_trace()
        loss.backward()
        optimizer.step()

        
        # measure accuracy and record loss
        losses.update(loss_val.item(), batch_size)
        loss_Etot.update(loss_Etot_val.item(), batch_size)
        # loss_Ei.update(loss_Ei_val.item(), batch_size)
        loss_Egroup.update(loss_Egroup_val.item(), batch_size)
        loss_Virial.update(loss_Virial_val.item(), batch_size)
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
        loss_Egroup.root,
        loss_Virial.root,
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
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup, loss_Virial],
        prefix="Epoch: [{}]".format(epoch),
    )

    KFOptWrapper = KFOptimizerWrapper(
        model, optimizer, config.nselect, config.groupsize, config.hvd, "hvd"
    )
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, sample_batches in enumerate(train_loader):
        #if i < 960:
        #    continue

        # measure data loading time
        data_time.update(time.time() - end)

        if config.datatype == "float64":
            Ei_label = Variable(sample_batches["Ei"].double().to(device))
            Etot_label = Variable(sample_batches["Etot"].double().to(device))
            #Etot_per_atom_label = Variable(sample_batches["Etot_per_atom"].double().to(device))
            Egroup_label = Variable(sample_batches["Egroup"].double().to(device))
            Divider = Variable(sample_batches["Divider"].double().to(device))
            Egroup_weight = Variable(
                sample_batches["Egroup_weight"].double().to(device)
            )
            
            Force_label = Variable(
                sample_batches["Force"][:, :, :].double().to(device)
            )  # [40,108,3]
            Virial_label = Variable(
                sample_batches["Virial"].double().to(device)
            )
            
            ImageDR = Variable(sample_batches["ImageDR"].double().to(device))
            Ri = Variable(sample_batches["Ri"].double().to(device), requires_grad=True)
            Ri_d = Variable(sample_batches["Ri_d"].to(device))

        elif config.datatype == "float32":
            Ei_label = Variable(sample_batches["Ei"].float().to(device))
            Etot_label = Variable(sample_batches["Etot"].float().to(device))
            #Etot_per_atom_label = Variable(sample_batches["Etot_per_atom"].float().to(device))
            """
            Egroup_label = Variable(sample_batches["Egroup"].float().to(device))
            Divider = Variable(sample_batches["Divider"].float().to(device))
            Egroup_weight = Variable(sample_batches["Egroup_weight"].float().to(device))
            """
            Force_label = Variable(
                sample_batches["Force"][:, :, :].float().to(device)
            )  # [40,108,3]
            Virial_label = Variable(
                sample_batches["Virial"].float().to(device))

            ImageDR = Variable(sample_batches["ImageDR"].float().to(device))
            Ri = Variable(sample_batches["Ri"].float().to(device), requires_grad=True)
            Ri_d = Variable(sample_batches["Ri_d"].float().to(device))

        # Etot_label = torch.sum(Ei_label.unsqueeze(2), dim=1)
        dR_neigh_list = Variable(sample_batches["ListNeighbor"].int().to(device))
        natoms_img = Variable(sample_batches["ImageAtomNum"].int().to(device))
        natoms_img = torch.squeeze(natoms_img, 1)
        natom = natoms_img[0,0]
        #Egroup_weight = [] 
        #Divider = [] 

        batch_size = Ri.shape[0]
        kalman_inputs = [ImageDR, Ri, Ri_d, dR_neigh_list, natoms_img, Egroup_weight, Divider]

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
            
            if config.is_etot is True: 
                KFOptWrapper.update_energy(kalman_inputs, Etot_label, config.pre_fac_etot)
            
            if config.is_egroup is True:
                KFOptWrapper.update_egroup(kalman_inputs, Egroup_label, config.pre_fac_egroup)

            if config.is_force is True:
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = KFOptWrapper.update_force(
                    kalman_inputs, Force_label, config.pre_fac_force) 
                
            if config.is_virial is True:
                KFOptWrapper.update_virial(kalman_inputs, Virial_label, config.pre_fac_virial)
            

        loss_F_val = criterion(Force_predict, Force_label)

        # divide by natom 
        loss_Etot_val = criterion(Etot_predict, Etot_label)
        loss_Etot_per_atom_val = criterion(Etot_predict, Etot_label)/natom/natom
        
        loss_Ei_val = criterion(Ei_predict, Ei_label)   
        loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
        loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))
        loss_val = loss_F_val + loss_Etot_val*natom

        # measure accuracy and record loss
        losses.update(loss_val.item(), batch_size)

        loss_Etot.update(loss_Etot_val.item(), batch_size)
        loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
        loss_Ei.update(loss_Ei_val.item(), batch_size)
        loss_Egroup.update(loss_Egroup_val.item(), batch_size)
        loss_Virial.update(loss_Virial_val.item(), batch_size)
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
        loss_Egroup.all_reduce()
        loss_Virial.all_reduce()
        batch_time.all_reduce()

    progress.display_summary(["Training Set:"])
    return losses.avg, loss_Etot.root, loss_Etot_per_atom.root, loss_Force.root, loss_Ei.root, loss_Egroup.root, loss_Virial.root

"""
def get_type_num(q:list):
    
    type_num = [] 
    
    type_now = q[0]
    idx_now = 0 

    for type in q:
        if type == type_now:
            idx_now += 1
        else:
            type_num.append([type_now,idx_now])
            type_now = type
            idx_now += 1

    type_num.append([type_now,idx_now]) 
    
    return type_num

def get_etot_type(Ei_predict, ): 
    # return a list of Etot_i
    return 
"""

def valid(val_loader, model, criterion, device, args):
    def run_validate(loader, base_progress=0):
        end = time.time()
        for i, sample_batches in enumerate(loader):

            i = base_progress + i
            
            # list of atom type in the image
            #atom_type_num = sample_batches["AtomType"] 
            #print(atom_type_num[0]) 
            #print(get_type_num(atom_type_num[0].tolist()))
            
            if args.datatype == "float64":
                Ei_label = Variable(sample_batches["Ei"].double().to(device))
                Etot_label = Variable(sample_batches["Etot"].double().to(device))
                #Etot_per_atom_label = Variable(sample_batches["Etot_per_atom"].double().to(device))
                Egroup_label = Variable(sample_batches["Egroup"].double().to(device))
                Divider = Variable(sample_batches["Divider"].double().to(device))
                Egroup_weight = Variable(
                    sample_batches["Egroup_weight"].double().to(device)
                )
                
                Force_label = Variable(
                    sample_batches["Force"][:, :, :].double().to(device)
                )
                Virial_label = Variable(
                sample_batches["Virial"].double().to(device))

                ImageDR = Variable(sample_batches["ImageDR"].double().to(device))
                Ri = Variable(
                    sample_batches["Ri"].double().to(device),
                    requires_grad=True,
                )
                Ri_d = Variable(sample_batches["Ri_d"].to(device))

            elif args.datatype == "float32":
                Ei_label = Variable(sample_batches["Ei"].float().to(device))
                Etot_label = Variable(sample_batches["Etot"].float().to(device))
                #Etot_per_atom_label = Variable(sample_batches["Etot_per_atom"].float().to(device))
                """
                Egroup_label = Variable(sample_batches["Egroup"].float().to(device))
                Divider = Variable(sample_batches["Divider"].float().to(device))
                
                Egroup_weight = Variable(
                    sample_batches["Egroup_weight"].float().to(device)
                )
                """
                Force_label = Variable(sample_batches["Force"][:, :, :].float().to(device))
                
                Virial_label = Variable(sample_batches["Virial"].float().to(device))

                ImageDR = Variable(sample_batches["ImageDR"].float().to(device))
                Ri = Variable(
                    sample_batches["Ri"].float().to(device),
                    requires_grad=True,
                )
                Ri_d = Variable(sample_batches["Ri_d"].float().to(device))

            # Etot_label = torch.sum(torch.unsqueeze(Ei_label, 2), dim=1)
            dR_neigh_list = Variable(sample_batches["ListNeighbor"].int().to(device))
            natoms_img = Variable(sample_batches["ImageAtomNum"].int().to(device))

            natoms_img = torch.squeeze(natoms_img, 1)
            
            natom = natoms_img[0,0]
            #print("num atoms")
            #print (natoms_img[0,0])
            # for division in RMSE Etot
            
            batch_size = Ri.shape[0]
            """
                Dim of Ri [bs, natom, ntype*max_neigh_num, 4] 
            """ 
            Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                ImageDR, Ri, Ri_d, dR_neigh_list, natoms_img, Egroup_weight, Divider
            )

            #print("Etot_predict")
            #print(Etot_predict)

            #print("Force_predict")
            #print(Force_predict)

            #print("Virial_predict")
            #print(Virial_predict)
            
            """
            idx = 2 

            model.embedding_net[0].weights['weight0'][0][idx].data -= 0.001
            
            print ("Wij before forward propagation")
            print (model.embedding_net[0].weights['weight0'][0])

            print ("the chosen Wij")
            print (model.embedding_net[0].weights['weight0'][0][idx])

            Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                ImageDR, Ri, Ri_d, dR_neigh_list, natoms_img, Egroup_weight, Divider
            )
            
            error = Virial_label.squeeze(1) - Virial_predict
            mask = error < 0
            Virial_predict[mask] = -1.0 * Virial_predict[mask]

            print("virial")
            print(Virial_predict)

            print("virial sum")
            print(Virial_predict.sum())
            
            Virial_predict.sum().backward()

            print("grad after backward")
            print(model.embedding_net[0].weights['weight0'].grad)
            
            print("the grad w.r.t chosen Wij")
            print(model.embedding_net[0].weights['weight0'].grad[0][idx])
            print("***********************************\n")
            """
            #return 
            loss_F_val = criterion(Force_predict, Force_label)
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_Etot_per_atom_val = criterion(Etot_predict, Etot_label)/natom/natom
            loss_Ei_val = criterion(Ei_predict, Ei_label)
            loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
            loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))
            loss_val = loss_F_val + loss_Etot_val*natom

            # measure accuracy and record loss
            losses.update(loss_val.item(), batch_size)
            loss_Etot.update(loss_Etot_val.item(), batch_size)
            loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
            loss_Ei.update(loss_Ei_val.item(), batch_size)
            loss_Egroup.update(loss_Egroup_val.item(), batch_size)
            loss_Virial.update(loss_Virial_val.item(), batch_size)
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

    progress = ProgressMeter(
        len(val_loader)
        + (
            args.hvd
            and (len(val_loader.sampler) * hvd.size() < len(val_loader.dataset))
        ),
        [batch_time, losses, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup],
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
        loss_Force.all_reduce()
        loss_Ei.all_reduce()

    progress.display_summary(["Test Set:"])

    return losses.avg, loss_Etot.root, loss_Etot_per_atom.root, loss_Force.root, loss_Ei.root, loss_Egroup.root, loss_Virial.root

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
