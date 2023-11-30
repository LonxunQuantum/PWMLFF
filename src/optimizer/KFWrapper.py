import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import time
import numpy as np
import torch.distributed as dist
import math


class KFOptimizerWrapper:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        atoms_selected: int,
        atoms_per_group: int,
        is_distributed: bool = False,
        distributed_backend: str = "torch",  # torch or horovod
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.atoms_selected = atoms_selected  # 24
        self.atoms_per_group = atoms_per_group  # 6
        self.is_distributed = is_distributed
        self.distributed_backend = distributed_backend

    def update_energy(
        self, inputs: list, Etot_label: torch.Tensor, update_prefactor: float = 1, train_type = "DP"
    ) -> None:
        if train_type == "DP":
            Etot_predict, _, _, _, _ = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                0,
                inputs[4],
                inputs[5],
                is_calc_f=False
            )
        elif train_type == "NN": # nn training
            Etot_predict, _, _, _, _ = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                is_calc_f=False,
            )
        else:
            raise Exception("Error! the train type {} is not realized!".format(train_type))
        # natoms_sum = inputs[2][0, 0]
        natoms_sum = len(inputs[1])
        self.optimizer.set_grad_prefactor(natoms_sum)

        self.optimizer.zero_grad()
        bs = Etot_label.shape[0]
        error = Etot_label - Etot_predict
        error = error / natoms_sum
        mask = error < 0

        error = error * update_prefactor
        error[mask] = -1 * error[mask]
        error = error.mean()

        # if self.is_distributed:
        #     if self.distributed_backend == "horovod":
        #         import horovod as hvd

        #         error = hvd.torch.allreduce(error)
        #     elif self.distributed_backend == "torch":
        #         dist.all_reduce(error)
        #         error /= dist.get_world_size()
        
        Etot_predict = update_prefactor * Etot_predict
        Etot_predict[mask] = -1.0 * Etot_predict[mask]

        Etot_predict.sum().backward()
        error = error * math.sqrt(bs)
        #print("Etot steping")
        self.optimizer.step(error)
        return Etot_predict

    def update_egroup(
        self, inputs: list, Egroup_label: torch.Tensor, update_prefactor: float = 1, train_type = "DP"
    ) -> None:
        if train_type == "DP":
            _, _, _, Egroup_predict, _ = self.model( #dp inputs has 7 para
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                inputs[7],
                inputs[8],
                is_calc_f=False,
            )
        elif train_type == "NN": # nn training
            _, _, _, Egroup_predict, _ = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                is_calc_f=False,
            )
        else:
            raise Exception("Error! the train type {} is not realized!".format(train_type))
        natoms_sum = inputs[3][0, 0]
        self.optimizer.set_grad_prefactor(1.0)

        self.optimizer.zero_grad()
        bs = Egroup_label.shape[0]
        error = Egroup_label - Egroup_predict
        #TODO: as the comment from RuNNer, the error why scaler by atom_num in Etot is because of
        #Etot is the sum of Ei, so maybe we don't need to scaler the egroup error. NEED CHECK!
        error = error 
        mask = error < 0

        error = error * update_prefactor
        error[mask] = -1 * error[mask]
        error = error.mean()

        # if self.is_distributed:
        #     if self.distributed_backend == "horovod":
        #         import horovod as hvd

        #         error = hvd.torch.allreduce(error)
        #     elif self.distributed_backend == "torch":
        #         dist.all_reduce(error)
        #         error /= dist.get_world_size()
        
        Egroup_predict = update_prefactor * Egroup_predict
        Egroup_predict[mask] = -1.0 * Egroup_predict[mask]

        Egroup_predict.sum().backward()
        error = error * math.sqrt(bs)
        self.optimizer.step(error)
        return Egroup_predict
    
    def update_virial(
        self, inputs: list, Virial_label: torch.Tensor, update_prefactor: float = 1, train_type = "DP"
    ) -> None:
        if train_type == "DP":
            Etot_predict, _, _, _, Virial_predict = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                inputs[7],
                inputs[8]
            )
        elif train_type == "NN":
            Etot_predict, _, _, Virial_predict = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6]
            )
        else:
            raise Exception("Error! the train type {} is not realized!".format(train_type))

        natoms_sum = inputs[3][0, 0]
        self.optimizer.set_grad_prefactor(natoms_sum)
        
        self.optimizer.zero_grad()

        bs = Virial_label.shape[0]  
        
        error = Virial_label.squeeze(1) - Virial_predict
        error = error / natoms_sum
        mask = error < 0

        # essentially a step length for weight update 
        error = error * update_prefactor
        error[mask] = -1 * error[mask]
        
        error = error.mean()

        # if self.is_distributed:
        #     if self.distributed_backend == "horovod":
        #         import horovod as hvd

        #         error = hvd.torch.allreduce(error)
        #     elif self.distributed_backend == "torch":
        #         dist.all_reduce(error)
        #         error /= dist.get_world_size()

        Virial_predict = update_prefactor * Virial_predict
        Virial_predict[mask] = -1.0 * Virial_predict[mask]
        
        Virial_predict.sum().backward()

        error = error * math.sqrt(bs) 
        
        self.optimizer.step(error)
        return Virial_predict

    def update_egroup_select(
        self, inputs: list, Egroup_label: torch.Tensor, update_prefactor: float = 1
    ) -> None:
        '''
        A select atoms version for egroup update.
        Base the simply test, it seems like update_egroup is a better choise.
        NEED CHECK!
        '''
        natoms_sum = inputs[3][0, 0]
        #print ("natoms_sum",natoms_sum)
        bs = Egroup_label.shape[0]
        self.optimizer.set_grad_prefactor(self.atoms_per_group)

        index = self.__sample(self.atoms_selected, self.atoms_per_group, natoms_sum)

        for i in range(index.shape[0]):
            self.optimizer.zero_grad()
            _, _, _, Egroup_predict, _ = self.model(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]
            )
            error_tmp = Egroup_label[:, index[i]] - Egroup_predict[:, index[i]]
            error_tmp = update_prefactor * error_tmp
            mask = error_tmp < 0
            error_tmp[mask] = -1 * error_tmp[mask]
            error = error_tmp.mean()

            # if self.is_distributed:
            #     if self.distributed_backend == "horovod":
            #         import horovod as hvd

            #         error = hvd.torch.allreduce(error)
            #     elif self.distributed_backend == "torch":
            #         dist.all_reduce(error)
            #         error /= dist.get_world_size()

            tmp_egroup_predict = Egroup_predict[:, index[i]] * update_prefactor
            tmp_egroup_predict[mask] = -1.0 * tmp_egroup_predict[mask]

            tmp_egroup_predict.sum().backward()
            error = error * math.sqrt(bs)
            self.optimizer.step(error)
        return Egroup_predict

    def update_force(
        self, inputs: list, Force_label: torch.Tensor, update_prefactor: float = 1, train_type = "DP"
    ) -> None:
        natoms_sum = len(inputs[1])
        bs = Force_label.shape[0]
        self.optimizer.set_grad_prefactor(natoms_sum * self.atoms_per_group * 3)

        index = self.__sample(self.atoms_selected, self.atoms_per_group, natoms_sum)

        for i in range(index.shape[0]):
            self.optimizer.zero_grad()
            if train_type == "DP":
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = self.model(
                    inputs[0], inputs[1], inputs[2], inputs[3], 0, inputs[4], inputs[5]
                )
            elif train_type == "NN":  # nn training
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = self.model(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]
                )
            else:
                raise Exception("Error! the train type {} is not realized!".format(train_type))

            error_tmp = Force_label[:, index[i]] - Force_predict[:, index[i]]
            error_tmp = update_prefactor * error_tmp
            mask = error_tmp < 0
            error_tmp[mask] = -1 * error_tmp[mask]
            error = error_tmp.mean() / natoms_sum

            # if self.is_distributed:
            #     if self.distributed_backend == "horovod":
            #         import horovod as hvd

            #         error = hvd.torch.allreduce(error)
            #     elif self.distributed_backend == "torch":
            #         dist.all_reduce(error)
            #         error /= dist.get_world_size()

            tmp_force_predict = Force_predict[:, index[i]] * update_prefactor
            tmp_force_predict[mask] = -1.0 * tmp_force_predict[mask]

            # In order to solve a pytorch bug, reference: https://github.com/pytorch/pytorch/issues/43259
            (tmp_force_predict.sum() + Etot_predict.sum() * 0).backward()
            error = error * math.sqrt(bs)
            #print("force steping")
            self.optimizer.step(error)
        return Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict

    '''
    description: 
            1. pick out 20% images 
            2. cut it into a several groups 
    return {*}
    author: wuxingxing
    '''
    def update_ei(
        self, inputs: list, Ei_label: torch.Tensor, update_prefactor: float = 1, train_type = "DP"
    ) -> None:
        if train_type == "DP":
            _, Ei_predict, _, _, _ = self.model( #dp inputs has 7 para
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                inputs[7],
                inputs[8],
                is_calc_f=False,
            )
        elif train_type == "NN": # nn training
            _, Ei_predict, _, _, _ = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                is_calc_f=False,
            )
        else:
            raise Exception("Error! the train type {} is not realized!".format(train_type))

        natoms_sum = inputs[3][0, 0]
        #print ("natoms_sum",natoms_sum)
        bs = Ei_label.shape[0]
        self.optimizer.set_grad_prefactor(1.0)
        self.optimizer.zero_grad()
        error = Ei_label - Ei_predict
        #TODO: as the comment from RuNNer, the error why scaler by atom_num in Etot is because of
        #Etot is the sum of Ei, so maybe we don't need to scaler the egroup error. NEED CHECK!
        error = error 
        mask = error < 0

        error = error * update_prefactor
        error[mask] = -1 * error[mask]
        error = error.mean()

        # if self.is_distributed:
        #     if self.distributed_backend == "horovod":
        #         import horovod as hvd

        #         error = hvd.torch.allreduce(error)
        #     elif self.distributed_backend == "torch":
        #         dist.all_reduce(error)
        #         error /= dist.get_world_size()
        
        Ei_predict = update_prefactor * Ei_predict
        Ei_predict[mask] = -1.0 * Ei_predict[mask]

        Ei_predict.sum().backward()
        error = error * math.sqrt(bs)
        self.optimizer.step(error)
        return Ei_predict
    
        # # |<---group1--->|<---group2--->|<-- ... -->|<---groupN--->| 
        # #     randomly choose atomidx for udpate 
        # selectNum = int(natoms_sum * 0.2) # choose 20%
        # updateNum = 5
        # selectNum = int(selectNum / updateNum ) * updateNum
        # weights = torch.ones(natoms_sum)  
        # selectedIdx = torch.multinomial(weights, selectNum, replacement=False)      # randomly choose atomidx for udpate
        
        # for i in range(int(selectNum / updateNum) + 1):
        #     self.optimizer.zero_grad()
        #     if train_type == "DP":
        #         Etot_predict, Ei_predict, _, _, _ = self.model(
        #             inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], is_calc_f=False
        #         )
        #     elif train_type == "NN":  # nn training
        #         Etot_predict, Ei_predict, _, _, _ = self.model(
        #             inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
        #         )
        #     else:
        #         raise Exception("Error! the train type {} is not realized!".format(train_type))     
        #     select_indexs = selectedIdx[i * updateNum : (i+1) * updateNum]     
        #     error_tmp = Ei_label[:, select_indexs] - Ei_predict[:, select_indexs]
        #     error_tmp = update_prefactor * error_tmp
        #     mask = error_tmp < 0
        #     error_tmp[mask] = -1 * error_tmp[mask]
        #     error = error_tmp.mean()

        #     if self.is_distributed:
        #         if self.distributed_backend == "horovod":
        #             import horovod as hvd

        #             error = hvd.torch.allreduce(error)
        #         elif self.distributed_backend == "torch":
        #             dist.all_reduce(error)
        #             error /= dist.get_world_size()

        #     mask = Ei_predict[:, select_indexs] < 0
        #     tmp_ei_predict = Ei_predict[:, select_indexs] * update_prefactor
        #     tmp_ei_predict[mask] = -1.0 * tmp_ei_predict[mask]

        #     # In order to solve a pytorch bug, reference: https://github.com/pytorch/pytorch/issues/43259
        #     (tmp_ei_predict.sum() + Etot_predict.sum() * 0).backward()
        #     error = error * math.sqrt(bs)
        #     #print("force steping")
        #     self.optimizer.step(error)
        # return Ei_predict
    
    def __sample(
        self, atoms_selected: int, atoms_per_group: int, natoms: int
    ) -> np.ndarray:
        # natoms can be smaller than n_select !
        # dbg : fix chosen atoms 
        # np.random.seed(0)
        if atoms_selected % atoms_per_group:
            raise Exception("divider")
        index = range(natoms)
        res = np.random.choice(index, atoms_selected).reshape(-1, atoms_per_group)
        return res
        