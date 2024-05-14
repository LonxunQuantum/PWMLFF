import torch
import torch.nn as nn
import math
from user.input_param import InputParam
from user.nep_param import NepParam
from torch.optim.optimizer import Optimizer
from utils.debug_operation import check_cuda_memory

class SNESOptimizer(Optimizer):
    def __init__(
        self,
        model: nn.Module,
        input_param: InputParam,
        is_distributed: bool = False,
        distributed_backend: str = "torch",  # torch or horovod
    ) -> None:
        default_dict = {
            "lr" : 0.1,
            "s" : 2
        }
        super(SNESOptimizer, self).__init__(model.parameters(), default_dict)

        self.input_param = input_param
        self.model = model
        self.population = input_param.optimizer_param.population
        self.generation = input_param.optimizer_param.generation
        self.is_distributed = is_distributed
        self.distributed_backend = distributed_backend
        
        self.dtype  = next(self.model.parameters()).dtype
        self.device = next(self.model.parameters()).device

        self.u_k = torch.tensor(self.__init_utility(), dtype=self.dtype, device=self.device)
        self.opt_param_num_list = []
        self.opt_param_index = [0]
        for param_group in self.param_groups:
            params = param_group["params"]
            for param in params:
                self.opt_param_num_list.append(param.data.nelement())
                self.opt_param_index.append(sum(self.opt_param_num_list))
        self.opt_param_nums = sum(self.opt_param_num_list)
        self.recip_npart = 1 / self.opt_param_nums

        self.m = torch.rand(self.opt_param_nums, dtype=self.dtype, device=self.device) - 0.5
        self.s = torch.full_like(self.m, 0.1)

        self.lambda_1 = self.opt_param_nums**0.5 / 1000 / len(self.input_param.atom_type) if self.input_param.optimizer_param.lambda_1 == -1 else\
            self.input_param.optimizer_param.lambda_1
        self.lambda_2 = self.opt_param_nums**0.5 / 1000 / len(self.input_param.atom_type) if self.input_param.optimizer_param.lambda_2 == -1 else\
            self.input_param.optimizer_param.lambda_2

        self.eta_m = 1 if self.input_param.optimizer_param.eta_m is None \
            else self.input_param.optimizer_param.eta_m
        self.eta_s = (3 + math.log(self.opt_param_nums)) / (5 * self.opt_param_nums**0.5) * 0.5 if self.input_param.optimizer_param.eta_s is None \
            else self.input_param.optimizer_param.eta_s

    def __init_utility(self):
        utility = []
        _utility = []
        for n in range(0, self.population):
            _utility.append(max(0.0, math.log(self.population*0.5 + 1.0) - math.log(n + 1.0)))
        for n in range(0, self.population):
            utility.append(_utility[n] / sum(_utility) - 1.0/self.population)
        return utility

    def load_m_s(self, m, s):
        self.m = torch.tensor(m, dtype=self.dtype, device=self.device) - 0.5
        self.s = torch.tensor(s, dtype=self.dtype, device=self.device) - 0.5
    
    def get_m(self):
        return self.m.flatten().cpu().detach().numpy()
    
    def get_s(self):
        return self.s.flatten().cpu().detach().numpy()

    def set_model_param(self, z:torch.Tensor) -> None:
        for param_group in self.param_groups:
            params = param_group["params"]
            for i, param in enumerate(params):
                param.data = (
                    # z[param_num1:param_num2].reshape(param.data.T.shape).T
                    z[self.opt_param_index[i]:self.opt_param_index[i+1]].reshape(param.data.T.shape).T
                )

    def update_params(
        self,
        inputs: list, 
        Etot_label: torch.Tensor, 
        Force_label: torch.Tensor, 
        Ei_label: torch.Tensor,
        Egroup_label: torch.Tensor, 
        Virial_label: torch.Tensor, 
        criterion, 
        train_type : str ="NEP"
    ) -> None:
        # generation = self.generation
        population = self.population
        # batch_size = inputs[0].shape[0]
        # natoms_sum = inputs[0].shape[1]
        # max_neighbor_type = inputs[0].shape[2]
        # c_num = self.input_param.nep_param.c2_param
        loss_list = []
        mse_etot, mse_ei, mse_F, mse_Egroup, mse_Virial = None, None, None, None, None
        low_loss, low_mse_etot, low_mse_ei, low_mse_F, low_mse_Egroup, low_mse_Virial, low_L1, low_L2 = None, None, None, None, None, None, None, None
        low_loss = float('inf')
        r_k = torch.normal(mean=0, std=1, size=(population, self.opt_param_nums), dtype=self.dtype, device=self.device)
        z = self.m.unsqueeze(0).repeat(population, 1) + \
            self.s.unsqueeze(0).repeat(population, 1) * r_k
        check_cuda_memory(-1, -1, "before train")
        for k in range(0, population):
            # self.zero_grad()
            # check_cuda_memory(k, population, "start k {}".format(k))
            # self.set_model_param(z[k, :])
            if train_type == "DP" or train_type == "NEP":
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = self.model(
                inputs[0], inputs[1], inputs[2], inputs[3], 0, inputs[4], inputs[5], is_calc_f = True
                )
            elif train_type == "NN":  # nn training
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = self.model(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]
                )
            with torch.no_grad():#会让autograd in calculate force 失效，因为没有计算图
                # check_cuda_memory(k, population, "end k {}".format(k))
                # calculate loss
                loss = 0
                if self.input_param.optimizer_param.train_energy is True:
                    mse_etot = criterion(Etot_predict, Etot_label)
                    loss += self.input_param.optimizer_param.pre_fac_etot * mse_etot**0.5
                if self.input_param.optimizer_param.train_ei is True:
                    mse_ei = criterion(Ei_predict, Ei_label)
                    loss += self.input_param.optimizer_param.pre_fac_ei * mse_ei**0.5
                if self.input_param.optimizer_param.train_force is True:
                    mse_F = criterion(Force_predict, Force_label)
                    loss += self.input_param.optimizer_param.pre_fac_force * mse_F**0.5
                if self.input_param.optimizer_param.train_egroup is True:
                    mse_Egroup = criterion(Egroup_predict, Egroup_label)
                    loss += self.input_param.optimizer_param.pre_fac_egroup * mse_Egroup**0.5
                if self.input_param.optimizer_param.train_virial is True:
                    mse_Virial = criterion(Virial_predict, Virial_label.squeeze(1))
                    loss += self.input_param.optimizer_param.pre_fac_virial * mse_Virial**0.5
                L1 = self.lambda_1 * self.recip_npart * torch.sum(z[k, :].abs())
                L2 = self.lambda_2 * (self.recip_npart * torch.sum(z[k:]**2))**0.5
                loss += L1 + L2
                if loss < low_loss:
                    low_loss = loss
                    low_mse_etot = mse_etot
                    low_mse_ei = mse_ei
                    low_mse_F = mse_F
                    low_mse_Egroup = mse_Egroup
                    low_mse_Virial = mse_Virial
                    low_L1 = L1
                    low_L2 = L2
                if loss >= float('inf'):
                    print(loss, low_loss)
                # check_cuda_memory(k, population, "end k {} cal loss".format(k))
                loss_list.append(loss)
            # loss_Virial_per_atom_val = loss_Virial_val/natoms_sum/natoms_sum
        # 更新 m s
        sorted_indices = sorted(range(len(loss_list)), key=lambda i: loss_list[i])
        del_m_J = torch.sum(self.u_k[sorted_indices].unsqueeze(-1) * r_k, dim=0)
        del_s_J = torch.sum(self.u_k[sorted_indices].unsqueeze(-1) * (r_k*r_k -1), dim=0)
        self.m = self.m + self.eta_m * (self.s * del_m_J)
        self.s = self.s * torch.exp(self.eta_s * del_s_J)
        return  low_loss, low_mse_etot, low_mse_ei, low_mse_F, low_mse_Egroup, low_mse_Virial, low_L1, low_L2
