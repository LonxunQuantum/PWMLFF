import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import time
import numpy as np
import torch.distributed as dist

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
        self, inputs: list, Etot_label: torch.Tensor, update_prefactor: float = 1
    ) -> None:
        time_start = time.time()
        Etot_predict, _, _ = self.model(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            is_calc_f=False,
        )
        natoms_sum = inputs[3][0, 0]
        self.optimizer.set_grad_prefactor(natoms_sum)

        self.optimizer.zero_grad()

        error = Etot_label - Etot_predict
        error = error / natoms_sum
        mask = error < 0

        error = error * update_prefactor
        error[mask] = -1 * error[mask]
        error = error.mean()
        
        if self.is_distributed:
            if self.distributed_backend == "horovod":
                import horovod as hvd
                error = hvd.torch.allreduce(error)
            elif self.distributed_backend == "torch":
                dist.all_reduce(error)
                error /= dist.get_world_size()

        Etot_predict = update_prefactor * Etot_predict
        Etot_predict[mask] = -update_prefactor * Etot_predict[mask]

        Etot_predict.mean().backward()
        self.optimizer.step(error)
        time_end = time.time()
        print("KF update Energy time:", time_end - time_start, "s")

    def update_force(
        self, inputs: list, Force_label: torch.Tensor, update_prefactor: float = 1
    ) -> None:
        time_start = time.time()
        natoms_sum = inputs[3][0, 0]

        self.optimizer.set_grad_prefactor(natoms_sum * self.atoms_per_group * 3)

        index = self.__sample(self.atoms_selected, self.atoms_per_group, natoms_sum)

        for i in range(index.shape[0]):
            self.optimizer.zero_grad()
            Etot_predict, _, force_predict = self.model(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
            )
            error_tmp = Force_label[:, index[i]] - force_predict[:, index[i]]
            error_tmp = update_prefactor * error_tmp
            mask = error_tmp < 0
            error_tmp[mask] = -1 * error_tmp[mask]
            error = error_tmp.mean() / natoms_sum

            if self.is_distributed:
                if self.distributed_backend == "horovod":
                    import horovod as hvd
                    error = hvd.torch.allreduce(error)
                elif self.distributed_backend == "torch":
                    dist.all_reduce(error)
                    error /= dist.get_world_size()

            tmp_force_predict = force_predict[:, index[i]] * update_prefactor
            tmp_force_predict[mask] = -update_prefactor * tmp_force_predict[mask]

            # In order to solve a pytorch bug, reference: https://github.com/pytorch/pytorch/issues/43259
            (tmp_force_predict.sum() + Etot_predict.sum() * 0).backward()

            self.optimizer.step(error)
        time_end = time.time()
        print("KF update Energy time:", time_end - time_start, "s")

    def __sample(
        self, atoms_selected: int, atoms_per_group: int, natoms: int
    ) -> np.ndarray:
        if atoms_selected % atoms_per_group:
            raise Exception("divider")
        index = range(natoms)
        res = np.random.choice(index, atoms_selected).reshape(-1, atoms_per_group)
        return res
