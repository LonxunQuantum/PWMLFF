import sys, os
import time
from math import pi as PI
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_ as normal

from typing import List, Tuple, Optional
from src.user.input_param import InputParam
from src.user.nep_param import NepParam
from utils.debug_operation import check_cuda_memory
sys.path.append(os.getcwd())
from src.model.nep_fitting import FittingNet
# from src.model.calculate_force import CalculateCompress, CalculateForce, CalculateVirialForce
if torch.cuda.is_available():
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "op/build/lib/libCalcOps_bind.so")
    torch.ops.load_library(lib_path)
    CalcOps = torch.ops.CalcOps_cuda
else:
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "op/build/lib/libCalcOps_bind_cpu.so")
    torch.ops.load_library(lib_path)    # load the custom op, no use for cpu version
    CalcOps = torch.ops.CalcOps_cpu     # only for compile while no cuda device
    
class NEP(nn.Module):
    def __init__(self, input_param:InputParam, energy_shift):
        super(NEP, self).__init__()
        self.Pi = PI
        self.half_Pi = self.Pi/2
        self.model_type = input_param.model_type.upper()
        self.input_param = input_param
        self.set_init_nep_param(input_param)
        if self.input_param.precision == "float64":
            self.dtype = torch.double
        elif self.input_param.precision == "float32":
            self.dtype = torch.float32
        else:
            raise RuntimeError("train(): unsupported training data type")
        self.set_cparam()
        self.maxNeighborNum = input_param.max_neigh_num
        self.fitting_net = nn.ModuleList()
        self.update_scaler = True
        for i in range(self.ntypes):
            nep_txt_param = None
            if input_param.nep_param.c2_param is not None:
                nep_txt_param = [input_param.nep_param.model_wb[i*3+0], input_param.nep_param.model_wb[i*3+1], input_param.nep_param.model_wb[i*3+2], input_param.nep_param.bias_lastlayer]
            self.fitting_net.append(FittingNet(network_size   = self.neuron, #[50, 1]
                                                    bias      = True,
                                                    resnet_dt = False,
                                                    activation= "tanh",
                                                    input_dim = self.feature_nums,
                                                    ener_shift= energy_shift[i],
                                                    magic     = False,
                                                    nep_txt_param = nep_txt_param
                                                    #    self.nep_param["net_cfg"]["fitting_net"]["resnet_dt"],
                                                    #    self.nep_param["net_cfg"]["fitting_net"]["activation"], 
                                                    ))
        self.max_neigh_num = self.input_param.max_neigh_num

    '''
    description: 
        for nep.txt 
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def get_nn_params(self):
        nn_params = []
        type_bias = []
        for i in range(self.ntypes):
            params = self.fitting_net[i].get_param_list()
            nn_params.extend(params[:-1])
            type_bias.append(params[-1])
        nn_params.append(np.mean(type_bias))
        nn_params.extend(list(self.c_param_2.permute(2, 3, 0, 1).flatten().cpu().detach().numpy()))
        nn_params.extend(list(self.c_param_3.permute(2, 3, 0, 1).flatten().cpu().detach().numpy()))
        nn_params.extend(list(self.q_scaler.flatten().cpu().detach().numpy()))
        return nn_params
        
    '''
    description: 
    maybe these params could be get from model, descriptor and optimizor object
    param {*} self
    param {InputParam} input_param
    return {*}
    author: wuxingxing
    '''
    def set_init_nep_param(self, input_param:InputParam):
        nep_param = input_param.nep_param
        self.atom_type = input_param.atom_type
        self.ntypes = len(input_param.atom_type)
        self.ntypes_sq = self.ntypes * self.ntypes

        self.cutoff_radial  = float(nep_param.cutoff[0])
        self.cutoff_angular = float(nep_param.cutoff[1])
        self.rcinv_radial   = 1.0/self.cutoff_radial
        self.rcinv_angular  = 1.0/self.cutoff_angular
        self.neuron         = nep_param.neuron
        
        self.n_max_radial  = nep_param.n_max[0]
        self.n_max_angular = nep_param.n_max[1]
        
        self.n_base_radial = nep_param.basis_size[0]
        self.n_base_angular= nep_param.basis_size[1]

        self.l_max_3b = nep_param.l_max[0]
        self.l_max_4b = nep_param.l_max[1]
        self.l_max_5b = nep_param.l_max[2]
        # feature nums
        self.two_feat_num   = self.n_max_radial + 1
        self.three_feat_num = (self.n_max_angular + 1) * self.l_max_3b
        self.four_feat_num  = (self.n_max_angular + 1) if self.l_max_4b > 0 else 0
        self.five_feat_num  = (self.n_max_angular + 1) if self.l_max_5b > 0 else 0
        self.feature_nums   = self.two_feat_num + self.three_feat_num + self.four_feat_num + self.five_feat_num
        # c param nums, the 4-body and 5-body use the same c param of 3-body, their N_base_a the same
        self.two_c_num   = self.ntypes_sq * (self.n_max_radial+1)  * (self.n_base_radial+1)
        self.three_c_num = self.ntypes_sq * (self.n_max_angular+1) * (self.n_base_angular+1)
        self.c_num       = self.two_c_num + self.three_c_num

    def get_q_scaler(self):
        return self.q_scaler.cpu().detach().numpy()

    '''
    description: 
        c_params is init from randly or c_params if init from checkpoint
        or c_params is init from nep.txt
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def set_cparam(self):
        if self.input_param.nep_param.c2_param is not None:
            self.c_param_2 = torch.nn.Parameter(torch.tensor(self.input_param.nep_param.c2_param), requires_grad=True)
            self.c_param_3 = torch.nn.Parameter(torch.tensor(self.input_param.nep_param.c3_param), requires_grad=True)
        else: # init by randly (for first training) or checkpoint
            r_k = torch.normal(mean=0, std=1, size=(self.c_num,), dtype=self.dtype)
            m = torch.rand(self.c_num, dtype=self.dtype) - 0.5
            s = torch.full_like(m, 0.1)
            c_param = m + s*r_k
            self.c_param_2 = torch.nn.Parameter(c_param[:self.two_c_num].reshape(self.ntypes, self.ntypes, (self.n_max_radial+1), (self.n_base_radial+1)), requires_grad=True)
            self.c_param_3 = torch.nn.Parameter(c_param[self.two_c_num : ].reshape(self.ntypes, self.ntypes, (self.n_max_angular+1), (self.n_base_angular+1)), requires_grad=True)

            # self.c_param_2 = torch.nn.Parameter(torch.ones([self.ntypes, self.ntypes, (self.n_max_radial+1), (self.n_base_radial+1)]), requires_grad=False)
            # self.c_param_3 = torch.nn.Parameter(torch.ones([self.ntypes, self.ntypes, (self.n_max_radial+1), (self.n_base_radial+1)]), requires_grad=False)

    '''
    description: 
        init q_scaler params
        if load from init
            q_scaler set to None and calculate at first interation in first epoch
            c_params init randly
        if load from nep.txt

        if load from checkpoint
    param {*} self
    param {*} q_scaler
    return {*}
    author: wuxingxing
    '''    
    def set_nep_param_device(self, q_scaler = None):
        # init param c
        dtype  = next(self.parameters()).dtype
        device = next(self.parameters()).device
        # load from nep.txt
        self.q_min = None
        self.q_max = None
        if self.input_param.nep_param.q_scaler is not None:
            self.q_scaler = torch.tensor(self.input_param.nep_param.q_scaler, dtype=dtype, device=device)
        else:
            if q_scaler is None:
                self.q_scaler = None
            else:
                self.q_scaler = torch.tensor(q_scaler, dtype=dtype, device=device)

        self.C3B = torch.tensor([0.238732414637843, 0.238732414637843, 0.238732414637843, #c10, c11, 12
                0.099471839432435, 1.1936620731892151, 1.1936620731892151, 0.2984155182973038, 0.2984155182973038, #c20,c21=c22,c23=c24
                0.139260575205408, 0.20889086280811264, 0.20889086280811264, 2.088908628081126, 2.088908628081126, 0.34815143801352105, 0.34815143801352105, #c30, c31=c32,c33=c34
                0.01119058193614889, 0.44762327744595565, 0.44762327744595565, 0.22381163872297782, 0.22381163872297782, # c40, c41=c42, c43=c44
                3.1333629421216895, 3.1333629421216895, 0.3916703677652112, 0.3916703677652112 #c45=c46, c47=c48
                ], dtype=dtype, device=device)

        self.C4B = torch.tensor([-0.007499480826664, -0.134990654879954, 0.067495327439977, 0.404971964639861, -0.809943929279723], 
                                dtype=dtype, device=device)
        
        self.C5B = torch.tensor([0.026596810706114, 0.053193621412227, 0.026596810706114], 
                                dtype=dtype, device=device)

        # self.c_param_2 = torch.normal(mean=0, std=1, size=(self.ntypes, self.ntypes, (self.n_max_radial+1), (self.n_base_radial+1)), dtype=self.dtype, device=device)
        # self.c_param_3 = torch.normal(mean=0, std=1, size=(self.ntypes, self.ntypes, (self.n_max_angular+1), (self.n_base_angular+1)), dtype=self.dtype, device=device)

        # self.c_param_2 = torch.ones([self.ntypes, self.ntypes, (self.n_max_radial+1), (self.n_base_radial+1)], dtype=self.dtype, device=device)
        # self.c_param_3 = torch.ones([self.ntypes, self.ntypes, (self.n_max_angular+1), (self.n_base_angular+1)], dtype=self.dtype, device=device)

        # print(self.c_param_2)
        # print(self.c_param_3)

    def get_egroup(self,
                   Ei: torch.Tensor,
                   Egroup_weight: Optional[torch.Tensor] = None,
                   divider: Optional[torch.Tensor] = None)-> Optional[torch.Tensor]:
        # commit by wuxing and replace by the under line code
        # batch_size = Ei.shape[0]
        # Egroup = torch.zeros_like(Ei)

        # for i in range(batch_size):
        #     Etot1 = Ei[i]
        #     weight_inner = Egroup_weight[i]
        #     E_inner = torch.matmul(weight_inner, Etot1)
        #     Egroup[i] = E_inner
        if Egroup_weight is not None and divider is not None:       # Egroup_out is not defined in the false branch:
            Egroup = torch.matmul(Egroup_weight, Ei)
            Egroup_out = torch.divide(Egroup.squeeze(-1), divider)
        else:
            Egroup_out = None
        
        return Egroup_out

    '''
    description: 
    return the embeding net index list and type nums of the image
    for example: 
        when the user input atom_type is [3, 14]:
            the atom_type_data is [14, 3], the index of user atom_type is [2, 1], then return:
                [[[1, 1], [1, 0]], [[0, 1], [0, 0]]], 2

            the atom_type_data is [14, 0], the index of user atom_type is [2, 1], then return:
                [[[1, 1]]], 1
            
        attention: 1. '0' is used in hybrid multi-batch training for completing tensor dimensions
                    2. in this user atom_type [3, 14]: the [1, 1] is the Si-Si embeding net index, [0, 1] is the Li-Si embeding net index
    
    param {*} self
    param {*} atom_type_data: the atom type list of image from dataloader
    return {*}
    author: wuxingxing
    '''
    def get_index(self, user_input_order: List[int], key:torch.Tensor):
        for i, v in enumerate(user_input_order):
            if v == key:
                return i
        return -1

    def get_fitnet_index(self, atom_type: torch.Tensor) -> List[int]:
        fitnet_index: List[int] = []
        for i, atom in enumerate(atom_type):
            if atom == 0: # for hybrid training, 0 means no atom
                continue
            index = self.get_index(self.atom_type, atom)
            fitnet_index.append(index)
        return fitnet_index
   
    def forward(self, 
                list_neigh: torch.Tensor,   # int32
                Imagetype_map: torch.Tensor,    # int32
                atom_type: torch.Tensor,    # int32
                ImageDR: torch.Tensor,      # float64
                list_neigh_type: torch.Tensor,
                nghost: int, 
                Egroup_weight: Optional[torch.Tensor] = None, 
                divider: Optional[torch.Tensor] = None, 
                is_calc_f: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            list_neigh (torch.Tensor): Tensor representing the neighbor list. Shape: (batch_size, natoms_sum, max_neighbor * ntypes).
            Imagetype_map (torch.Tensor): The tensor mapping atom types to image types.. Shape: (natoms_sum).
            atom_type (torch.Tensor): Tensor representing the image's atom types. Shape: (ntypes).
            ImageDR (torch.Tensor): Tensor representing the image DRneigh. Shape: (batch_size, natoms_sum, max_neighbor * ntypes, 4).
            nghost (int): Number of ghost atoms.
            Egroup_weight (Optional[torch.Tensor], optional): Tensor representing the Egroup weight. Defaults to None.
            divider (Optional[torch.Tensor], optional): Tensor representing the divider. Defaults to None.
            is_calc_f (Optional[bool], optional): Flag indicating whether to calculate forces and virial. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: Tuple containing the total energy (Etot), atomic energies (Ei), forces (Force), energy group (Egroup), and virial (Virial).
        """
        # check_cuda_memory(-1, -1, "=====FORWAR START=====") self.fitting_net[0].layers[0].weight
        # t0 = time.time()
        device = ImageDR.device
        dtype = ImageDR.dtype
        # t10 = time.time()
        batch_size = list_neigh.shape[0]
        natoms_sum = list_neigh.shape[1]#no use
        doub_natoms_sum = list_neigh.shape[1]
        max_neighbor_type = list_neigh.shape[2]  # ntype * max_neighbor_num
        # t11 = time.time()
        fitnet_index = self.get_fitnet_index(atom_type)
        # t1 = time.time()
        # check_cuda_memory(-1, -1, "FORWAR get_fitnet_index")
        # print("t12 {} t11 {} t10 {}".format(t1-t11, t11-t10, t10-t0))
        Ri, Ri_d = self.calculate_Ri(doub_natoms_sum, batch_size, max_neighbor_type, Imagetype_map, ImageDR, device, dtype)
        Ri.requires_grad_()
        # j_type_map = torch.tensor(list_neigh_type, dtype=list_neigh_type.dtype, device=list_neigh_type.device, requires_grad=False)
        j_type_map = list_neigh_type.clone().detach().requires_grad_(False)#可以在调用前做复制，从forward移出去，只有训练需要
        # t2 = time.time()
        # j_type = list_neigh - 1
        # neigh_j_types = torch.where(j_type != -1, Imagetype_map[j_type], -1).requires_grad_(False)
        # check_cuda_memory(-1, -1, "FORWAR calculate_Ri")
        feats = self.calculate_qn(doub_natoms_sum, batch_size, max_neighbor_type, Imagetype_map, j_type_map, Ri, device, dtype)

        if self.q_scaler is None:
            q_max = torch.max(feats.reshape([-1, feats.shape[-1]]), dim=-2)[0]
            q_min = torch.min(feats.reshape([-1, feats.shape[-1]]), dim=-2)[0]
            self.q_scaler = (1/(q_max-q_min)).detach()
        elif is_calc_f is False and self.update_scaler:

            q_max = torch.max(feats.reshape([-1, feats.shape[-1]]), dim=-2)[0]
            q_min = torch.min(feats.reshape([-1, feats.shape[-1]]), dim=-2)[0]
            self.q_scaler = torch.min(self.q_scaler, 1/(q_max-q_min)).detach()
            
        # t3 = time.time()
        # if self.q_scaler is None:
        #     # before_scaler = feats.reshape([-1, feats.shape[-1]])
        #     self.q_scaler = (1 / (torch.max(feats.reshape([-1, feats.shape[-1]]), dim=-2)[0] - torch.min(feats.reshape([-1, feats.shape[-1]]), dim=-2)[0])).detach()
        #     # self.q_scaler = torch.tensor([0.279121883555, 0.279121883555, 0.279121883555, 0.279121883555,
        #     #                                 0.279121883555, 1.033094276106, 1.033094276106, 1.033094276106,
        #     #                                 1.033094276106, 1.033094276106, 0.631036108470, 0.631036108470,
        #     #                                 0.631036108470, 0.631036108470, 0.631036108470, 0.252691679979,
        #     #                                 0.252691679979, 0.252691679979, 0.252691679979, 0.252691679979,
        #     #                                 0.064832683827, 0.064832683827, 0.064832683827, 0.064832683827,
        #     #                                 0.064832683827, 1.533943480748, 1.533943480748, 1.533943480748,
        #     #                                 1.533943480748, 1.533943480748, 2.254929491959, 2.254929491959,
        #     #                                 2.254929491959, 2.254929491959, 2.254929491959],
        #     #                 device=device, dtype=dtype)

        feats_in = self.q_scaler * feats
        # feats_in = (feats-self.q_min)/(self.q_max-self.q_min)
        # print(feats_in[0,:10,:])
        # check_cuda_memory(-1, -1, "FORWAR set q_scaler")
        # t4 = time.time()
        Ei = self.calculate_Ei(Imagetype_map, batch_size, feats_in, fitnet_index, device)
        # t5 = time.time()
        # check_cuda_memory(-1, -1, "FORWAR calculate_Ei")
        assert Ei is not None
        Etot = torch.sum(Ei, 1)
        
        # mask: List[Optional[torch.Tensor]] = [torch.ones_like(Etot)]
        # dE = torch.autograd.grad([Etot], [feats_in], grad_outputs=mask, retain_graph=True, create_graph=True)[0]

        Egroup = self.get_egroup(Ei, Egroup_weight, divider) if Egroup_weight is not None else None
        Ei = torch.squeeze(Ei, 2)
        # t6 = time.time()
        # check_cuda_memory(-1, -1, "FORWAR calculate_Egroup")
        if is_calc_f is False:
            Force, Virial = None, None
            # print("==single time: t1 {} t2 {} t3 {} t4 {} t5 {} t6 {}".format(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5))
        else:
            # t4 = time.time()
            Force, Virial = self.calculate_force_virial(Ri, Ri_d, Etot, natoms_sum, batch_size, list_neigh, ImageDR, nghost, device, dtype)
            # t7 = time.time()
            # ==single time: t1 0.0015997886657714844 t2 0.0016467571258544922 t3 0.03717923164367676 t4 2.8371810913085938e-05 t5 0.0011038780212402344 t6 4.267692565917969e-05 t7 0.08994221687316895
            # print("==single time: t1 {} t2 {} t3 {} t4 {} t5 {} t6 {} t7 {}".format(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6))
            # check_cuda_memory(-1, -1, "FORWAR calculate_force")
        # del Ri, Ri_d, feats
        # torch.cuda.empty_cache()
        return Etot, Ei, Force, Egroup, Virial

    def calculate_Ri(self,
                     natoms_sum: int,
                     batch_size: int,
                     max_neighbor_type: int,
                     Imagetype_map: torch.Tensor,
                     ImagedR: torch.Tensor, 
                     device: torch.device,
                     dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # R = ImagedR[:, :, :, 0]
        # Ri = torch.zeros(batch_size, natoms_sum, max_neighbor_type, 4, dtype=dtype, device=device)
        # Ri[:, :, :, 0] = R
        # Ri[:, :, :, 1] = ImagedR[:, :, :, 1]
        # Ri[:, :, :, 2] = ImagedR[:, :, :, 2]
        # Ri[:, :, :, 3] = ImagedR[:, :, :, 3]
        mask = ImagedR[:, :, :, 0].abs() > 1e-5
        Ri_d = torch.zeros(batch_size, natoms_sum, max_neighbor_type, 4, 3, dtype=dtype, device=device)
        # xij = ImagedR[:, :, :, 1][mask]
        # yij = ImagedR[:, :, :, 2][mask]
        # zij = ImagedR[:, :, :, 3][mask]
        Ri_d[:, :, :, 0, 0][mask] = ImagedR[:, :, :, 1][mask] / ImagedR[:, :, :, 0][mask]
        Ri_d[:, :, :, 1, 0][mask] = 1
        # dy
        Ri_d[:, :, :, 0, 1][mask] = ImagedR[:, :, :, 2][mask] / ImagedR[:, :, :, 0][mask]
        Ri_d[:, :, :, 2, 1][mask] = 1
        # dz
        Ri_d[:, :, :, 0, 2][mask] = ImagedR[:, :, :, 3][mask] / ImagedR[:, :, :, 0][mask]
        Ri_d[:, :, :, 3, 2][mask] = 1
        return ImagedR, Ri_d

    def calculate_Ei(self, 
                     Imagetype_map: torch.Tensor,
                     batch_size: int,
                     feats: torch.Tensor,
                     fit_index: List[int],
                     device: torch.device) -> Optional[torch.Tensor]:
        """
        Calculate the energy Ei for each type of atom in the system.

        Args:
            Imagetype_map (torch.Tensor): The tensor mapping atom types to image types.
            Ri (torch.Tensor): A tensor representing the atomic descriptors.
            batch_size (int): The size of the batch.
            emb_list (List[List[List[int]]]): A list of embedded atom types.
            type_nums (int): The number of atom types.
            device (torch.device): The device to perform the calculations on.

        Returns:
            Optional[torch.Tensor]: The calculated energy Ei for each type of atom, or None if the calculation fails.
        """
        Ei : Optional[torch.Tensor] = None
        fit_net_dict = {idx: fit_net for idx, fit_net in enumerate(self.fitting_net)}
        for nn_i in fit_index:
            fit_net = fit_net_dict.get(nn_i)
            assert fit_net is not None
            # S_Rij = Ri[:, indices, ntype_1 * self.maxNeighborNum:(ntype_1+1) * self.maxNeighborNum, 0].unsqueeze(-1)
            mask = (Imagetype_map == nn_i).flatten()
            if not mask.any():
                continue
            indices = torch.arange(len(Imagetype_map.flatten()),device=device)[mask]  
            feat = feats[:, indices, :]
            Ei_ntype = fit_net.forward(feat)
            Ei = Ei_ntype if Ei is None else torch.concat((Ei, Ei_ntype), dim=1)
            # print()
        # Etot = torch.sum(Ei, 1)
        # mask2: List[Optional[torch.Tensor]] = [torch.ones_like(Etot)]
        # dE = torch.autograd.grad([Etot], [feats], grad_outputs=mask2, retain_graph=True, create_graph=True)[0]

        return Ei
     
    def calculate_force_virial(self, 
                               Ri: torch.Tensor,
                               Ri_d: torch.Tensor,
                               Etot: torch.Tensor,
                               natoms_sum: int,
                               batch_size: int,
                               list_neigh: torch.Tensor,
                               ImageDR: torch.Tensor, 
                               nghost: int,
                               device: torch.device,
                               dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # t7 = time.time()
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(Etot)]
        dE = torch.autograd.grad([Etot], [Ri], grad_outputs=mask, retain_graph=True, create_graph=True)[0]
        # t8 = time.time()
        '''
        # this result is same as the above code
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(Ei)]
        dE = torch.autograd.grad([Ei], [Ri], grad_outputs=mask, retain_graph=True, create_graph=True)[0]
        '''
        assert dE is not None
        if device.type == "cpu":
            dE = torch.unsqueeze(dE, dim=-1)
            # dE * Ri_d [batch, natom, max_neighbor * len(atom_type),4,1] * [batch, natom, max_neighbor * len(atom_type), 4, 3]
            # dE_Rid [batch, natom, max_neighbor * len(atom_type), 3]
            dE_Rid = torch.mul(dE, Ri_d).sum(dim=-2)
            Force = torch.zeros((batch_size, natoms_sum + nghost + 1, 3), device=device, dtype=dtype)
            Force[:, 1:natoms_sum + 1, :] = -1 * dE_Rid.sum(dim=-2)
            Virial = torch.zeros((batch_size, 9), device=device, dtype=dtype)
            for batch_idx in range(batch_size):
                indice = list_neigh[batch_idx].flatten().unsqueeze(-1).expand(-1, 3).to(torch.int64) # list_neigh's index start from 1, so the Force's dimension should be natoms_sum + 1
                values = dE_Rid[batch_idx].view(-1, 3)
                Force[batch_idx].scatter_add_(0, indice, values).view(natoms_sum + nghost + 1, 3)
                Virial[batch_idx, 0] = (ImageDR[batch_idx, :, :, 1] * dE_Rid[batch_idx, :, :, 0]).flatten().sum(dim=0) # xx
                Virial[batch_idx, 1] = (ImageDR[batch_idx, :, :, 1] * dE_Rid[batch_idx, :, :, 1]).flatten().sum(dim=0) # xy
                Virial[batch_idx, 2] = (ImageDR[batch_idx, :, :, 1] * dE_Rid[batch_idx, :, :, 2]).flatten().sum(dim=0) # xz
                Virial[batch_idx, 4] = (ImageDR[batch_idx, :, :, 2] * dE_Rid[batch_idx, :, :, 1]).flatten().sum(dim=0) # yy
                Virial[batch_idx, 5] = (ImageDR[batch_idx, :, :, 2] * dE_Rid[batch_idx, :, :, 2]).flatten().sum(dim=0) # yz
                Virial[batch_idx, 8] = (ImageDR[batch_idx, :, :, 3] * dE_Rid[batch_idx, :, :, 2]).flatten().sum(dim=0) # zz
                # testxx = (ImageDR[batch_idx, :, :, 1] * dE_Rid[batch_idx, :, :, 0]).sum(dim=1)
                # testxy = (ImageDR[batch_idx, :, :, 1] * dE_Rid[batch_idx, :, :, 1]).sum(dim=1)
                # testxz = (ImageDR[batch_idx, :, :, 1] * dE_Rid[batch_idx, :, :, 2]).sum(dim=1)
                # testyy = (ImageDR[batch_idx, :, :, 2] * dE_Rid[batch_idx, :, :, 1]).sum(dim=1)
                Virial[batch_idx, [3, 6, 7]] = Virial[batch_idx, [1, 2, 5]]
            Force = Force[:, 1:, :]
        else:
            Ri_d = Ri_d.view(batch_size, natoms_sum, -1, 3)
            dE = dE.view(batch_size, natoms_sum, 1, -1)
            Force = -1 * torch.matmul(dE, Ri_d).squeeze(-2)
            ImageDR = Ri[:,:,:,1:].clone()
            nghost_tensor = torch.tensor(nghost, device=device, dtype=torch.int64)
            list_neigh = torch.unsqueeze(list_neigh,2)
            list_neigh = (list_neigh - 1).type(torch.int)
            Force = CalcOps.calculateForce(list_neigh, dE, Ri_d, Force, nghost_tensor)[0]
            Virial = CalcOps.calculateVirial(list_neigh, dE, ImageDR, Ri_d, nghost_tensor)[0]
        # t9 = time.time()
        # print("t8 {} t9 {}".format(t8-t7, t9-t8))
        del dE
        # print(-Force)
        return -Force, -Virial

    def calculate_qn(self,
                     natoms_sum: int,#no use
                     batch_size: int,
                     max_neighbor_type: int,
                     Imagetype_map: torch.Tensor,
                     j_type_map: torch.Tensor,
                     Ri: torch.Tensor, 
                     device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn start")
        atom_nums = Imagetype_map.shape[0]
        R = Ri[:, :, :, 0]
        xyz = Ri[:, :, :, 1:]
        # feat_2b = self.cal_feat_2body(R[:, :atom_nums, :], Imagetype_map, j_type_map[:, :atom_nums, :], self.n_max_radial, self.n_base_radial, self.cutoff_radial, self.rcinv_radial)
        # # check_cuda_memory(-1, -1, "FORWAR calculate_qn 2b end")
        # feat_3b, feat_4b, feat_5b = self.cal_feat_multi_body(R[:,atom_nums:, :], xyz[:,atom_nums:, :], Imagetype_map, j_type_map[:, atom_nums:, :], self.n_max_angular, self.n_base_angular, self.cutoff_angular, self.rcinv_angular, self.l_max_3b)
        
        c2, c3 = self.get_c(self.c_param_2, self.c_param_3, self.n_max_radial, self.n_base_radial, self.n_max_angular, self.n_base_angular, Imagetype_map, j_type_map)
        feat_2b = self.cal_feat_2body(R, Imagetype_map, 
                                        # j_type_map, 
                                        c2,
                                        self.n_max_radial, self.n_base_radial, self.cutoff_radial, self.rcinv_radial)
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 2b end")
        multi_feat = self.cal_feat_multi_body(R, xyz, Imagetype_map, 
                                        # j_type_map, 
                                        c3,
                                        self.n_max_angular, self.n_base_angular, self.cutoff_angular, self.rcinv_angular, self.l_max_3b)

        return torch.concat([feat_2b, multi_feat], dim=-1)

    def get_c(self,
            c_2b : torch.Tensor,
            c_3b : torch.Tensor,
            n_max_r : int,
            n_base_r:int,
            n_max_a : int,
            n_base_a :int,
            Imagetype_map : torch.Tensor,
            j_type_map : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: #get c params from c[n_type,n_type, n_max, n_base] 
        batch_size = j_type_map.shape[0]
        atom_nums = j_type_map.shape[1]
        j_list_nums = j_type_map.shape[2]

        # j_type_map = j_type_map.clone()
        mask = j_type_map > -1
        # j_type_map = j_type_map -1
        j_type_map[mask] = (Imagetype_map*self.ntypes).unsqueeze(-1).unsqueeze(0).repeat(batch_size, 1, self.ntypes*self.maxNeighborNum)[mask]+j_type_map[mask]
        j_type_map3 = j_type_map.flatten()
        mask2 = j_type_map3 > -1
        
        c_list = torch.zeros([j_type_map3.shape[0], n_max_r+1, n_base_r+1], dtype=c_2b.dtype, device=c_2b.device)
        c2 = c_2b.reshape(self.ntypes_sq, c_2b.shape[-2],c_2b.shape[-1])
        c_list[mask2] = c2[j_type_map3[mask2]]
        # c1 = c[Imagetype_map, :, :, :] # search by i
        c2 = c_list.view(batch_size, atom_nums, j_list_nums, n_max_r+1, n_base_r+1)

        c_3b_list = torch.zeros([j_type_map3.shape[0], n_max_a+1, n_base_a+1], dtype=c_3b.dtype, device=c_3b.device)
        c3 = c_3b.reshape(self.ntypes_sq, c_3b.shape[-2],c_3b.shape[-1])
        c_3b_list[mask2] = c3[j_type_map3[mask2]]
        # c3 = c[Imagetype_map, :, :, :] # search by i
        c3 = c_3b_list.view(batch_size, atom_nums, j_list_nums, n_max_a +1, n_base_a +1)

        return c2.transpose(3, 2), c3.transpose(3, 2)

    # def get_c(self,
    #         c : torch.Tensor,
    #         n_max : int,
    #         n_base:int,
    #         Imagetype_map : torch.Tensor,
    #         j_type_map : torch.Tensor) -> Tuple[torch.Tensor]: #get c params from c[n_type,n_type, n_max, n_base] 
    #     batch_size = j_type_map.shape[0]
    #     atom_nums = j_type_map.shape[1]
    #     j_list_nums = j_type_map.shape[2]

    #     # j_type_map = j_type_map.clone()
    #     mask = j_type_map > -1
    #     # j_type_map = j_type_map -1
    #     j_type_map[mask] = (Imagetype_map*self.ntypes).unsqueeze(-1).unsqueeze(0).repeat(1, 1, 200)[mask]+j_type_map[mask]
    #     j_type_map3 = j_type_map.flatten()
    #     mask2 = j_type_map3 > -1
    #     c_list = torch.zeros([j_type_map3.shape[0], n_max+1, n_base+1], dtype=c.dtype, device=c.device)
    #     c1 = c.reshape(self.ntypes_sq, c.shape[-2],c.shape[-1])
    #     c_list[mask2] = c1[j_type_map3[mask2]]
    #     # c1 = c[Imagetype_map, :, :, :] # search by i
    #     c1 = c_list.view(batch_size, atom_nums, j_list_nums, n_max+1, n_base+1)
    #     return c1.transpose(3, 2)

    def cal_fk(self,
                rij: torch.Tensor,
                n_base: int,
                rcut: float,
                rcinv: float) -> torch.Tensor:
        mask = (rij.abs() > 1e-5) & (rij <= rcut) # 超过截断半径的rij, fk(rij) 为0，那么 c*t*fc = 0,导数也为0，因为fk=0, dfk=0
        fc = torch.zeros_like(rij)
        fc[mask]  = 0.5 + 0.5 * torch.cos(self.Pi * rij[mask] * rcinv)

        tk  = torch.zeros([rij.shape[0], rij.shape[1], rij.shape[2], n_base+1], dtype=rij.dtype).to(rij.device)# [b,i,j,M]
        fk  = torch.zeros([rij.shape[0], rij.shape[1], rij.shape[2], n_base+1], dtype=rij.dtype).to(rij.device)# [b,i,j,M]
        
        x = torch.zeros_like(rij)
        x[mask] = 2 * (rij[mask] * rcinv - 1)**2 - 1

        # 先不要考虑n_max_r计算完之后做扩展,再与c做乘法, fc也要最后再乘
        tk[:, :, :, 0][mask]   = 1.0 # t0
        tk[:, :, :, 1][mask]   = x[mask] # t1

        fk[:, :, :, 0][mask]   = fc[mask]   # 0.5 *( t0(x) + 1) * fc(rij), t0(x) = 1
        fk[:, :, :, 1][mask]   = 0.5 * (x[mask] + 1) * fc[mask]   # 0.5 *( t1(x) + 1 ) * fc(rij), t1(x) = x
        # fk[:,:,:,1] = torch.tensor(x.data).unsqueeze(2).repeat(1,1,fk.shape[2],1)
        for n in range(2, n_base + 1):## 参考nep-cpu
            tk[:,:,:,n][mask]      = 2 * x[mask] * tk[:,:,:,n - 1][mask] - tk[:,:,:,n - 2][mask]
            fk[:,:,:,n][mask]      = 0.5 * (tk[:,:,:,n][mask] +1) * fc[mask]                  # [b,i,N,j,M]
            
        return fk

    def cal_feat_2body(self,
                        rij: torch.Tensor,
                        Imagetype_map: torch.Tensor,
                        # j_type_map: torch.Tensor,
                        c2:torch.Tensor,
                        n_max: int,
                        n_base: int,
                        rcut: float,
                        rcinv: float) -> torch.Tensor:
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 2b start")
        # c2 = self.get_c(self.c_param_2, n_max, n_base, Imagetype_map, j_type_map)
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 2b c2")
        fk = self.cal_fk(rij, n_base, rcut, rcinv)
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 2b fk")
        fk_res = fk.unsqueeze(2).repeat(1, 1, n_max+1, 1, 1)    # n_max_r+1 个feature区别是在c系数上，fk是一样的
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 2b fk_res")
        feat_2b = (c2 * fk_res).sum(-1).sum(-1) # sum n_base_r and sum j
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 2b feat_2b")
        return feat_2b # 检查下dp 的feature 和dfeature 维度,本应该对每一个feature，都有对应的rij的导数

    '''
    description: 
        for nep_cpu, the qn of 3b, 4b, 5b orders are :
                n=0       n=1       n=2       n=3       n=4 (n to max_angular+1)
        L=1 q_3b_01   q_3b_11   q_3b_21   q_3b_31   q_3b_41
        L=2 q_3b_02   q_3b_12   q_3b_22   q_3b_32   q_3b_42
        L=3 q_3b_03   q_3b_13   q_3b_23   q_3b_33   q_3b_43
        L=4 q_3b_04   q_3b_14   q_3b_24   q_3b_34   q_3b_44    
        L=4 q_4b_022  q_4b_122  q_4b_222  q_4b_322  q_4b_422
        L=4 q_5b_0111 q_5b_1111 q_5b_2111 q_5b_3111 q_5b_4111
    return {*}
    author: wuxingxing
    '''    
    def cal_feat_multi_body(
                    self,
                    rij: torch.Tensor,
                    xyz: torch.Tensor,
                    Imagetype_map: torch.Tensor,
                    # j_type_map: torch.Tensor,
                    c3:torch.Tensor,
                    n_max: int,
                    n_base: int,
                    rcut: float,
                    rcinv: float,
                    l_max_3b: int) -> torch.Tensor:
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 3b start")
        # c3 = self.get_c(self.c_param_3, rij, n_max, n_base, Imagetype_map)
        # c3 = self.get_c(self.c_param_3, n_max, n_base, Imagetype_map, j_type_map)
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn ck start")
        fk = self.cal_fk(rij, n_base, rcut, rcinv)
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn fk start")
        # c * tk # now fk 是对的 c3 不对，rij 对，
        gn1 = (c3 * (fk.unsqueeze(2).repeat(1, 1, n_max+1, 1, 1))).sum(-1)   # n_max_r 个feature区别是在c系数上，fk是一样的 # sum n_base
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn gn1 start")
        gn2 = gn1.unsqueeze(-1).repeat(1, 1, 1, 1, 24) # lmax_3body = 4 [1, 96, 5, 200, 24]
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn gn2 start")
        blm = self.cal_blm_ij(rij, xyz, rcut) #[1, 96, 200, 24]
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn blm start")
        blm2 = blm.unsqueeze(2).repeat(1, 1, n_max + 1, 1, 1)# [1, 96, 5, 200, 24] 这里blm = blm(xij,yij,zij) / rij^l
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn blm2 start")
        snlm = (gn2 * blm2).sum(3) #gn * blm, then sum j : [1, 96, 5, 200, 24] -> [1, 96, 5, 24]
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn snlm start")
        snlm_sq = snlm**2
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn snlm_sq start")
        # 常系数C
        c_lm = self.C3B.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(snlm_sq.shape[0],snlm_sq.shape[1],snlm_sq.shape[2],1)
        qnlm = c_lm * snlm_sq
        qnl = torch.zeros([snlm_sq.shape[0], snlm_sq.shape[1], snlm_sq.shape[2], l_max_3b], dtype=qnlm.dtype, device=qnlm.device)
        qnl[:, :, :, 0] = qnlm[:, :, :, 0:3].sum(-1)
        qnl[:, :, :, 1] = qnlm[:, :, :, 3:8].sum(-1)
        qnl[:, :, :, 2] = qnlm[:, :, :, 8:15].sum(-1)
        qnl[:, :, :, 3] = qnlm[:, :, :, 15:24].sum(-1)
        # feat_3b = qnl.view(qnl.shape[0], qnl.shape[1], -1) # 3体feature
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 3b end")

        # feature 4
        # feat_4b = None
        if self.l_max_4b != 0:
            sn20_sq = snlm_sq[:, :, :, 3]
            sn21_sq = snlm_sq[:, :, :, 4]
            sn22_sq = snlm_sq[:, :, :, 5]
            sn23_sq = snlm_sq[:, :, :, 6]
            sn24_sq = snlm_sq[:, :, :, 7]
            feat_4b = self.C4B[0] * snlm[:, :, :, 3] * sn20_sq + \
                    self.C4B[1] * snlm[:, :, :, 3] * (sn21_sq + sn22_sq) +\
                    self.C4B[2] * snlm[:, :, :, 3] * (sn23_sq + sn24_sq) + \
                    self.C4B[3] * snlm[:, :, :, 6] * (sn22_sq - sn21_sq) +\
                    self.C4B[4] * snlm[:, :, :, 4] * snlm[:, :, :, 5] * snlm[:, :, :, 7]
        else:
            feat_4b = None
        # feature 5
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 4b end")
        # feat_5b = None
        if self.l_max_5b != 0:
            sn10_sq = snlm_sq[:, :, :, 0]
            sn11_sq = snlm_sq[:, :, :, 1]
            sn12_sq = snlm_sq[:, :, :, 2]
            feat_5b = self.C5B[0] * sn10_sq * sn10_sq + self.C5B[1] * sn10_sq * (sn11_sq + sn12_sq) + self.C5B[2] * (sn11_sq + sn12_sq)**2
        else:
            feat_5b = None
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 5b end")
        
        # if feat_5b is not None:
        #     return torch.concat([qnl.transpose(3,2), feat_4b.unsqueeze(-2), feat_5b.unsqueeze(-2)], dim = -2).view(qnl.shape[0], qnl.shape[1], -1)
        # elif feat_4b is not None:
        #     return torch.concat([qnl.transpose(3,2), feat_4b.unsqueeze(-2)], dim = -2).view(qnl.shape[0], qnl.shape[1], -1)
        # else:
        #     return qnl.transpose(3,2).view(qnl.shape[0], qnl.shape[1], -1) # return 3b_feature
 
        if feat_5b is not None:
            feat_5b = feat_5b.unsqueeze(-2)
        if feat_4b is not None:
            feat_4b = feat_4b.unsqueeze(-2)

        if feat_5b is not None and feat_4b is not None:
            return torch.concat([qnl.transpose(3,2), feat_4b, feat_5b], dim=-2).view(qnl.shape[0], qnl.shape[1], -1)
        elif feat_5b is not None:
            return torch.concat([qnl.transpose(3,2), feat_5b], dim=-2).view(qnl.shape[0], qnl.shape[1], -1)
        elif feat_4b is not None:
            return torch.concat([qnl.transpose(3,2), feat_4b], dim=-2).view(qnl.shape[0], qnl.shape[1], -1)
        else:
            return qnl.transpose(3,2).reshape(qnl.shape[0], qnl.shape[1], -1)
            
    def cal_blm_ij(self,
            rij: torch.Tensor,
            xyz: torch.Tensor,
            rcut: float,
            ) -> torch.Tensor:
        mask = (rij.abs() > 1e-5) & (rij <= rcut)
        d12inv = torch.zeros_like(rij)
        d12inv[mask] = 1/rij[mask]
        x12 = d12inv[mask] * xyz[:, :, :, 0][mask]
        y12 = d12inv[mask] * xyz[:, :, :, 1][mask]
        z12 = d12inv[mask] * xyz[:, :, :, 2][mask]
        
        x12sq = x12 ** 2
        y12sq = y12 ** 2
        z12sq = z12 ** 2
        x12sq_minus_y12sq = x12sq - y12sq

        blm = torch.zeros([xyz.shape[0], xyz.shape[1], xyz.shape[2], 24], dtype=xyz.dtype, device=xyz.device)
        blm[:, :, :, 0][mask] = z12                                                            # Y10       b10 / r^1 
        blm[:, :, :, 1][mask] = x12                                                            # Y11_real  b11 / r^1
        blm[:, :, :, 2][mask] = y12                                                            # Y11_imag  b12 / r^1
        blm[:, :, :, 3][mask] = (3.0 * z12sq - 1.0)                                            # Y20       b20 / r^2
        blm[:, :, :, 4][mask] = x12 * z12                                                      # Y21_real  b21 / r^2
        blm[:, :, :, 5][mask] = y12 * z12                                                      # Y21_imag  b22 / r^2
        blm[:, :, :, 6][mask] = x12sq_minus_y12sq                                              # Y22_real  b23 / r^2
        blm[:, :, :, 7][mask] = 2.0 * x12 * y12                                                # Y22_imag  b24 / r^2
        blm[:, :, :, 8][mask] = (5.0 * z12sq - 3.0) * z12                                      # Y30       b30 / r^3       
        blm[:, :, :, 9][mask] = (5.0 * z12sq - 1.0) * x12                                      # Y31_real  b31 / r^3
        blm[:, :, :, 10][mask] = (5.0 * z12sq - 1.0) * y12                                      # Y31_imag  b32 / r^3
        blm[:, :, :, 11][mask] = x12sq_minus_y12sq * z12                                        # Y32_real  b33 / r^3
        blm[:, :, :, 12][mask] = 2.0 * x12 * y12 * z12                                          # Y32_imag  b34 / r^3
        blm[:, :, :, 13][mask] = (x12 * x12 - 3.0 * y12 * y12) * x12                            # Y33_real  b35 / r^3
        blm[:, :, :, 14][mask] = (3.0 * x12 * x12 - y12 * y12) * y12                            # Y33_imag  b36 / r^3
        blm[:, :, :, 15][mask] = ((35.0 * z12sq - 30.0) * z12sq + 3.0)                          # Y40       b40 / r^4
        blm[:, :, :, 16][mask] = (7.0 * z12sq - 3.0) * x12 * z12                                # Y41_real  b41 / r^4
        blm[:, :, :, 17][mask] = (7.0 * z12sq - 3.0) * y12 * z12                                # Y41_iamg  b42 / r^4
        blm[:, :, :, 18][mask] = (7.0 * z12sq - 1.0) * x12sq_minus_y12sq                        # Y42_real  b43 / r^4
        blm[:, :, :, 19][mask] = (7.0 * z12sq - 1.0) * x12 * y12 * 2.0                          # Y42_imag  b44 / r^4
        blm[:, :, :, 20][mask] = (x12sq - 3.0 * y12sq) * x12 * z12                              # Y43_real  b45 / r^4
        blm[:, :, :, 21][mask] = (3.0 * x12sq - y12sq) * y12 * z12                              # Y43_imag  b46 / r^4
        blm[:, :, :, 22][mask] = (x12sq_minus_y12sq * x12sq_minus_y12sq - 4.0 * x12sq * y12sq)  # Y44_real  b47 / r^4
        blm[:, :, :, 23][mask] = (4.0 * x12 * y12 * x12sq_minus_y12sq)                          # Y44_imag  b48 / r^4

        return blm
