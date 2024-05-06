"""
NEP 手动求导
"""
import sys, os
import time
from math import pi as PI
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_ as normal

from typing import List, Tuple, Optional
from src.user.input_param import InputParam
from src.user.nep_param import NepParam

sys.path.append(os.getcwd())
from src.model.dp_embedding import FittingNet
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

        self.input_param = input_param
        self.set_init_nep_param(input_param)
        if self.input_param.precision == "float64":
            self.dtype = torch.double
        elif self.input_param.precision == "float32":
            self.dtype = torch.float32
        else:
            raise RuntimeError("train(): unsupported training data type")

        self.fitting_net = nn.ModuleList()
        
        for i in range(self.ntypes):
            self.fitting_net.append(FittingNet(network_size   = [self.neuron, 1], #[50, 1]
                                                    bias      = True,
                                                    resnet_dt = False,
                                                    activation= "tanh",
                                                    input_dim = self.feature_nums,
                                                    ener_shift= energy_shift[i],
                                                    magic     = False
                                                    #    self.nep_param["net_cfg"]["fitting_net"]["resnet_dt"],
                                                    #    self.nep_param["net_cfg"]["fitting_net"]["activation"], 
                                                    ))
        self.device = None # init after optimizer created
        self.max_neigh_num = self.input_param.max_neigh_num

    def set_init_nep_param(self, input_param:InputParam):
        nep_param = input_param.nep_param
        self.atom_type = input_param.atom_type
        self.ntypes = len(input_param.atom_type)
        self.ntypes_sq = self.ntypes * self.ntypes

        self.cutoff_radial  = nep_param.cutoff[0]
        self.cutoff_angular = nep_param.cutoff[1]
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
        # init param c
        r_k = torch.normal(mean=0, std=1, size=(self.c_num,))
        m = torch.rand(self.c_num) - 0.5
        s = torch.full_like(m, 0.1)
        c_param = m + s*r_k
        self.c_param_2 = c_param[:self.two_c_num].reshape(self.ntypes, self.ntypes, (self.n_max_radial+1), (self.n_base_radial+1))
        self.c_param_3 = c_param[self.two_c_num : ].reshape(self.ntypes, self.ntypes, (self.n_max_angular+1), (self.n_base_angular+1))
        self.j_list = [k for k, _ in enumerate(self.atom_type)]




    def set_nep_cparam_device(self, device):
        self.c_param_2 = self.c_param_2.to(device)
        self.c_param_3 = self.c_param_3.to(device)
        # for 3-body C_lm 常系数C
        self.C3B = torch.tensor([0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435, 0.596831036594608,
                    0.596831036594608, 0.149207759148652, 0.149207759148652, 0.139260575205408, 0.104445431404056,
                    0.104445431404056, 1.044454314040563, 1.044454314040563, 0.174075719006761, 0.174075719006761,
                    0.011190581936149, 0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
                    1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606], 
                    dtype=torch.double, device=device)
        self.C4B = torch.tensor([-0.007499480826664, -0.134990654879954, 0.067495327439977, 0.404971964639861, -0.809943929279723], 
                                dtype=torch.double, device=device)
        
        self.C5B = torch.tensor([0.026596810706114, 0.053193621412227, 0.026596810706114], 
                                dtype=torch.double, device=device)
        self.K_C_SP = 14.399645 # 1/(4*PI*epsilon_0)

        j_list = []
        for k, _ in enumerate(self.atom_type):
            for i in range(0, self.max_neigh_num):
                j_list.append(k)
        self.j_list = torch.tensor(j_list)
        self.j_list = self.j_list.to(device)

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
    def get_train_2body_type(self, atom_type: torch.Tensor) -> Tuple[List[List[List[int]]], int]:
        type_2body_list: List[List[List[int]]] = []         
        type_2body_index: List[int] = []
        for i, atom in enumerate(atom_type):
            # if self.atom_type.eq(atom).any():
            #     type_2body_index.append(i)
            index = self.get_index(self.atom_type, atom)
            if index != -1:
                type_2body_index.append(self.get_index(self.atom_type, atom))
        for atom in type_2body_index:
            type_2body: List[List[int]] = []        # 整数列表的列表
            for atom2 in type_2body_index:
                type_2body.append([atom, atom2])        # 整数列表
            type_2body_list.append(type_2body)
        pair_indices = len(type_2body_index)
        return type_2body_list, pair_indices
   
    def forward(self, 
                list_neigh: torch.Tensor,   # int32
                Imagetype_map: torch.Tensor,    # int32
                atom_type: torch.Tensor,    # int32
                ImageDR: torch.Tensor,      # float64
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
        device = list_neigh.device
        dtype = list_neigh.dtype
        batch_size = list_neigh.shape[0]
        natoms_sum = list_neigh.shape[1]
        max_neighbor_type = list_neigh.shape[2]  # ntype * max_neighbor_num
        emb_list, type_nums = self.get_train_2body_type(atom_type)
        # t1 = time.time()
        self.calculate_qn(natoms_sum, batch_size, max_neighbor_type, Imagetype_map, ImageDR, device, dtype)
        Ri.requires_grad_()
        # t2 = time.time()
        Ei = self.calculate_Ei(Imagetype_map, Ri, batch_size, emb_list, type_nums, device)
        # t3 = time.time()
        assert Ei is not None
        Etot = torch.sum(Ei, 1)
        Egroup = self.get_egroup(Ei, Egroup_weight, divider) if Egroup_weight is not None else None
        Ei = torch.squeeze(Ei, 2)
        if is_calc_f is False:
            Force, Virial = None, None
        else:
            # t4 = time.time()
            Force, Virial = self.calculate_force_virial(Ri, Ri_d, Etot, natoms_sum, batch_size, list_neigh, ImageDR, nghost, device, dtype)
            # t5 = time.time()
            # print("\ncalculate_Ri:", t2-t1, "\ncalculate_Ei:", t3-t2, "\ncalculate_force_virial:", t5-t4, "\n********************")
        return Etot, Ei, Force, Egroup, Virial

    def calculate_Ei(self, 
                     Imagetype_map: torch.Tensor,
                     Ri: torch.Tensor,
                     batch_size: int,
                     emb_list: List[List[List[int]]],
                     type_nums: int, 
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
        for type_emb in emb_list:
            # t1 = time.time()
            xyz_scater_a, xyz_scater_b = self.calculate_xyz_scater(Imagetype_map, Ri, type_emb, type_nums, device)
            if xyz_scater_a.any() == 0:
                continue
            DR_ntype = torch.matmul(xyz_scater_a.transpose(-2, -1), xyz_scater_b)
            DR_ntype = DR_ntype.reshape(batch_size, xyz_scater_a.shape[1], -1)
            # t2 = time.time()
            fit_net = fit_net_dict.get(type_emb[0][0]) 
            assert fit_net is not None
            Ei_ntype = fit_net.forward(DR_ntype)
            # t3 = time.time()
            Ei = Ei_ntype if Ei is None else torch.concat((Ei, Ei_ntype), dim=1)
            # t4 = time.time()
            # print("calculate_xyz_scater:", t2-t1, "\nfitting_net:", t3-t2, "\nconcat:", t4-t3, "\n********************")
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
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(Etot)]
        dE = torch.autograd.grad([Etot], [Ri], grad_outputs=mask, retain_graph=True, create_graph=True)[0]
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
            ImageDR = ImageDR[:,:,:,1:].clone()
            nghost_tensor = torch.tensor(nghost, device=device, dtype=torch.int64)
            list_neigh = torch.unsqueeze(list_neigh,2)
            list_neigh = (list_neigh - 1).type(torch.int)
            Force = CalcOps.calculateForce(list_neigh, dE, Ri_d, Force, nghost_tensor)[0]
            Virial = CalcOps.calculateVirial(list_neigh, dE, ImageDR, Ri_d, nghost_tensor)[0]
        
        return Force, Virial 

    def calculate_qn(self,
                     natoms_sum: int,
                     batch_size: int,
                     max_neighbor_type: int,
                     Imagetype_map: torch.Tensor,
                     ImagedR: torch.Tensor, 
                     device: torch.device,
                     dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        R = ImagedR[:, :, :, 0]
        xyz = ImagedR[:, :, :, 1:]
        c2 = self.get_c(R, self.n_max_radial, self.n_base_radial, Imagetype_map)
        fk_2body, d_fk_2body = self.cal_feat_2body(R, c2, self.n_max_radial, self.n_base_radial, self.cutoff_radial, self.rcinv_radial)
        c3 = self.get_c(R, self.n_max_angular, self.n_base_angular, Imagetype_map)
        self.cal_feat_3body(R, xyz, c3, self.n_max_angular, self.n_base_angular, self.cutoff_angular, self.rcinv_angular, self.l_max_3b)
        # fk_3body, d_fy_4body = self.cal_feat_3body(R, c2)
    
    def get_c(self,
            R : torch.Tensor,
            n_max : int,
            n_base:int,
            Imagetype_map : torch.Tensor) -> Tuple[torch.Tensor]: #get c params from c[n_type,n_type, n_max, n_base] 
        c1 = self.c_param_2[Imagetype_map, :, :, :] #search by i
        c2 = c1[:, self.j_list, :, :]   # search by j
        c3 = c2.transpose(1,2)
        c4 = c3.unsqueeze(0).repeat(R.shape[0], 1, 1, 1 ,1)
        return c4

    def cal_fk(self,
                rij: torch.Tensor,
                c : torch.Tensor, #no used
                n_max: int, # no used
                n_base: int,
                rcut: float,
                rcinv: float) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = (rij.abs() > 1e-5) & (rij <= rcut)
        fc = torch.zeros_like(rij)
        d_fc = torch.zeros_like(rij)
        fc[mask]  = 0.5 + 0.5 * torch.cos(PI * rij[mask] * rcinv)
        d_fc[mask]= -0.5 * PI * rcinv * torch.sin(PI * rij[mask] * rcinv)

        # fk = torch.zeros(R.shape[0], R.shape[1], self.n_max_radial, R.shape[2], self.n_base_radial)#[b,i,N,j,M]
        mask2 = (rij.abs() > 1e-6) & (rij <= rcut) # 超过截断半径的rij, fk(rij) 为0，那么 c*t*fc = 0,导数也为0，因为fk=0, dfk=0
        tk  = torch.zeros([rij.shape[0], rij.shape[1], rij.shape[2], n_base+1], dtype=rij.dtype).to(rij.device)# [b,i,j,M]
        d_tk= torch.zeros([rij.shape[0], rij.shape[1], rij.shape[2], n_base+1], dtype=rij.dtype).to(rij.device)
        
        fk  = torch.zeros([rij.shape[0], rij.shape[1], rij.shape[2], n_base+1], dtype=rij.dtype).to(rij.device)# [b,i,j,M]
        d_fk= torch.zeros([rij.shape[0], rij.shape[1], rij.shape[2], n_base+1], dtype=rij.dtype).to(rij.device)
        uk  = torch.zeros_like(fk)
        
        x = torch.zeros_like(rij)
        dx = torch.zeros_like(rij)
        x[mask2]  = 2 * (rij[mask2] * rcinv - 1)**2 - 1
        dx[mask2] = 4 * (rij[mask2] * rcinv - 1) * rcinv

        # 先不要考虑n_max_r计算完之后做扩展,再与c做乘法, fc也要最后再乘
        tk[:, :, :, 0][mask2]   = 1.0 # t0
        tk[:, :, :, 1][mask2]   = x[mask2] # t1

        uk[:, :, :, 0][mask2]   = 1.0 #可以只要两维，做更新即可
        uk[:, :, :, 1][mask2]   = 2*x[mask2]

        d_tk[:, :, :, 0][mask2] = 0 # dt0: 0 * dx = 0
        d_tk[:, :, :, 1][mask2] = dx[mask2]# dt1: 1 * dx = dx

        fk[:, :, :, 0][mask2]   = fc[mask2]   # 0.5 *( t0(x) + 1) * fc(rij), t0(x) = 1
        fk[:, :, :, 1][mask2]   = 0.5 * (x[mask2] + 1) * fc[mask2]   # 0.5 *( t1(x) + 1 ) * fc(rij), t1(x) = x
        d_fk[:, :, :, 0][mask2] = d_fc[mask2] # 0.5 * dT0 * dx * fc + 0.5 * (T0 + 1) * dfc, dT0 = 0, T0 = 1, ==> dfc
        d_fk[:, :, :, 1][mask2] = 0.5 * dx[mask2] * fc[mask2] + 0.5 * d_fc[mask2] * (x[mask2] + 1)   # 0.5 * dT1 * dx + 0.5 * (T1 + 1) * dfc, T1 = x, Dt1 = U0 = 1 
        # fc2 = fc.unsqueeze(3).repeat(1,1,1,fk.shape[3])
        # fk[:,:,:,1] = torch.tensor(x.data).unsqueeze(2).repeat(1,1,fk.shape[2],1)
        for n in range(2, n_base + 1):## 参考nep-cpu
            tk[:,:,:,n][mask2]      = 2 * x[mask2] * tk[:,:,:,n - 1][mask2] - tk[:,:,:,n - 2][mask2]
            fk[:,:,:,n][mask2]      = 0.5 * (tk[:,:,:,n][mask2] +1) * fc[mask2]                  # [b,i,N,j,M]
            d_tk[:, :, :, n][mask2] = 0.5 * n * uk[:, :, :, n-1][mask2] * dx[mask2] * fc[mask2] + 0.5 * (tk[:,:,:,n][mask2] + 1) * d_fc[mask2]                    # d_tn = n*Un-1(x)
            uk[:, :, :, n][mask2]   = 2 * x[mask2] * uk[:, :, :, n-1][mask2] - uk[:, :, :, n-2][mask2]  # Un = 2x*Un-1 - Un-2
        return fk, d_tk 

    def cal_feat_2body(self,
                        rij: torch.Tensor,
                        c : torch.Tensor, 
                        n_max: int,
                        n_base: int,
                        rcut: float,
                        rcinv: float) -> Tuple[torch.Tensor, torch.Tensor]:
        fk, d_fk = self.cal_fk(rij, c, n_max, n_base, rcut, rcinv)
        fk_res = fk.unsqueeze(2).repeat(1, 1, n_max+1, 1, 1)    # n_max_r 个feature区别是在c系数上，fk是一样的
        fk_res = (c * fk_res).sum(-1).sum(-1) # sum n_base_r and sum j
        d_fk_res = d_fk.unsqueeze(2).repeat(1, 1, n_max+1, 1, 1)
        d_fk_res = (c * d_fk_res).sum(-1) # sum n_base_r [1, 96, 5, 200, 13] -> [1, 96, 5, 200]
        return fk_res, d_fk_res # 检查下dp 的feature 和dfeature 维度,本应该对每一个feature，都有对应的rij的导数

    def cal_feat_3body(
                    self,
                    rij: torch.Tensor,
                    xyz: torch.Tensor,
                    c : torch.Tensor, 
                    n_max: int,
                    n_base: int,
                    rcut: float,
                    rcinv: float,
                    l_max_3b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fk, d_fk = self.cal_fk(rij, c, n_max, n_base, rcut, rcinv)
        # c * tk
        gn = fk.unsqueeze(2).repeat(1, 1, n_max+1, 1, 1)    # n_max_r 个feature区别是在c系数上，fk是一样的
        gn1 = gn.sum(-1)
        gn2 = gn1.unsqueeze(-1).repeat(1, 1, 1, 1, 24) # lmax_3body = 4 [1, 96, 5, 200, 24]
        blm, dblm = self.cal_blm_ij(rij, xyz) #[1, 96, 200, 24]
        blm2 = blm.unsqueeze(2).repeat(1, 1, n_max + 1, 1, 1)# [1, 96, 5, 200, 24] 这里blm = blm(xij,yij,zij) / rij^l
        sn21 = gn2 * blm2 #[1, 96, 5, 200, 24]
        sn22 = sn21.sum(3)# sum j -> [1, 96, 5, 24]
        # 常系数C 
        c_lm = self.C3B.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(sn22.shape[0],sn22.shape[1],sn22.shape[2],1)
        qnlm = c_lm * sn22
        qnl = torch.zeros([sn22.shape[0], sn22.shape[1], sn22.shape[2], l_max_3b], dtype=qnlm.dtype, device=qnlm.device)
        qnl[:, :, :, 0] = qnlm[:, :, :, 0:3].sum(-1)
        qnl[:, :, :, 1] = qnlm[:, :, :, 3:8].sum(-1)
        qnl[:, :, :, 2] = qnlm[:, :, :, 8:15].sum(-1)
        qnl[:, :, :, 3] = qnlm[:, :, :, 15:24].sum(-1)
        feat_3b = qnl.view(qnl.shape[0], qnl.shape[1], -1) # 3体feature 
        print()


        
        # fk_res = (c * fk_res).sum(-1).sum(-1)
        # d_fk_res = d_fk.unsqueeze(2).repeat(1, 1, n_max+1, 1, 1)
        # d_fk_res = (c * d_fk_res).sum(-1).sum(-1)



        return None, None

    def cal_blm_ij(self,
            rij: torch.Tensor,
            xyz: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        d12inv = 1/rij
        x12 = d12inv * xyz[:, :, :, 0]
        y12 = d12inv * xyz[:, :, :, 1]
        z12 = d12inv * xyz[:, :, :, 2]
        
        x12sq = x12 ** 2
        y12sq = y12 ** 2
        z12sq = z12 ** 2
        x12sq_minus_y12sq = x12sq - y12sq

        blm = torch.zeros([xyz.shape[0], xyz.shape[1],xyz.shape[2],24]).to(xyz.device)
        blm[:, :, :, 0] = z12                                                            # Y10       b10 / r^1 
        blm[:, :, :, 1] = x12                                                            # Y11_real  b11 / r^1
        blm[:, :, :, 2] = y12                                                            # Y11_imag  b12 / r^1
        blm[:, :, :, 3] = (3.0 * z12sq - 1.0)                                            # Y20       b20 / r^2
        blm[:, :, :, 4] = x12 * z12                                                      # Y21_real  b21 / r^2
        blm[:, :, :, 5] = y12 * z12                                                      # Y21_imag  b22 / r^2
        blm[:, :, :, 6] = x12sq_minus_y12sq                                              # Y22_real  b23 / r^2
        blm[:, :, :, 7] = 2.0 * x12 * y12                                                # Y22_imag  b24 / r^2
        blm[:, :, :, 8] = (5.0 * z12sq - 3.0) * z12                                      # Y30       b30 / r^3       
        blm[:, :, :, 9] = (5.0 * z12sq - 1.0) * x12                                      # Y31_real  b31 / r^3
        blm[:, :, :, 10] = (5.0 * z12sq - 1.0) * y12                                      # Y31_imag  b32 / r^3
        blm[:, :, :, 11] = x12sq_minus_y12sq * z12                                        # Y32_real  b33 / r^3
        blm[:, :, :, 12] = 2.0 * x12 * y12 * z12                                          # Y32_imag  b34 / r^3
        blm[:, :, :, 13] = (x12 * x12 - 3.0 * y12 * y12) * x12                            # Y33_real  b35 / r^3
        blm[:, :, :, 14] = (3.0 * x12 * x12 - y12 * y12) * y12                            # Y33_imag  b36 / r^3
        blm[:, :, :, 15] = ((35.0 * z12sq - 30.0) * z12sq + 3.0)                          # Y40       b40 / r^4
        blm[:, :, :, 16] = (7.0 * z12sq - 3.0) * x12 * z12                                # Y41_real  b41 / r^4
        blm[:, :, :, 17] = (7.0 * z12sq - 3.0) * y12 * z12                                # Y41_iamg  b42 / r^4
        blm[:, :, :, 18] = (7.0 * z12sq - 1.0) * x12sq_minus_y12sq                        # Y42_real  b43 / r^4
        blm[:, :, :, 19] = (7.0 * z12sq - 1.0) * x12 * y12 * 2.0                          # Y42_imag  b44 / r^4
        blm[:, :, :, 20] = (x12sq - 3.0 * y12sq) * x12 * z12                              # Y43_real  b45 / r^4
        blm[:, :, :, 21] = (3.0 * x12sq - y12sq) * y12 * z12                              # Y43_imag  b46 / r^4
        blm[:, :, :, 22] = (x12sq_minus_y12sq * x12sq_minus_y12sq - 4.0 * x12sq * y12sq)  # Y44_real  b47 / r^4
        blm[:, :, :, 23] = (4.0 * x12 * y12 * x12sq_minus_y12sq)                          # Y44_imag  b48 / r^4

        return blm, None  
# mask = rij <= rcut
# fc = torch.zeros_like(rij)
# d_fc = torch.zeros_like(rij)
# fc[mask]  = 0.5 + 0.5 * torch.cos(PI * rij[mask] * rcinv)
# d_fc[mask]= -0.5 * PI * rcinv * torch.sin(PI * rij[mask] * rcinv)

# # fk = torch.zeros(R.shape[0], R.shape[1], self.n_max_radial, R.shape[2], self.n_base_radial)#[b,i,N,j,M]
# fk  = torch.zeros(rij.shape[0], rij.shape[1], rij.shape[2], n_base+1).to(rij.device)# [b,i,j,M]
# d_fk= torch.zeros(rij.shape[0], rij.shape[1], rij.shape[2], n_base+1).to(rij.device)
# x  = 2 * (rij * rcinv - 1)**2 - 1
# dx = 4 * (rij * rcinv - 1) * rcinv

# # 先不要考虑n_max_r计算完之后做扩展,再与c做乘法, fc也要最后再乘
# fk[:, :, :, 0]   = 1 * fc
# fk[:, :, :, 1]   = x * fc #?
# d_fk[:, :, :, 0] = 1 * d_fc
# d_fk[:, :, :, 1] = x * d_fc + dx * fc

# # fc2 = fc.unsqueeze(3).repeat(1,1,1,fk.shape[3])
# # fk[:,:,:,1] = torch.tensor(x.data).unsqueeze(2).repeat(1,1,fk.shape[2],1)
# for n in range(2, n_base + 1):
#     fk[:,:,:,n] = 2 * x * fk[:,:,:,n - 1] + fk[:,:,:,n - 2]
#     fk[:,:,:,n] = 0.5 *  (fk[:,:,:,n] +1) * fc # [b,i,N,j,M]
# # c * fk
# fk1 = fk.unsqueeze(2).repeat(1, 1, n_max+1, 1, 1)
# fk2 = (c * fk1).sum(-1).sum(-1)