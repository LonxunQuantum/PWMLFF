import sys, os
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from src.user.input_param import InputParam
import time

sys.path.append(os.getcwd())
from src.model.dp_embedding import FittingNet

from feature.chebyshev.descriptor_cheby import Descriptor
from feature.chebyshev.build.lib import descriptor_pybind

if torch.cuda.is_available():
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "op/build/lib/libCalcOps_bind.so")
    torch.ops.load_library(lib_path)
    CalcOps = torch.ops.CalcOps_cuda
else:
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "op/build/lib/libCalcOps_bind_cpu.so")
    torch.ops.load_library(lib_path)    # load the custom op, no use for cpu version
    CalcOps = torch.ops.CalcOps_cpu     # only for compile while no cuda device

class ChebyNet(nn.Module):
    def __init__(self, input_param: InputParam, energy_shift):
        super(ChebyNet, self).__init__()
        self.input_param = input_param
        self.atom_types = input_param.atom_type
        self.ntypes = len(self.atom_types)
        self.m_neigh = input_param.max_neigh_num
        self.Rc_M = input_param.descriptor.Rmax
        self.rcut_smooth = input_param.descriptor.Rmin
        self.beta = input_param.descriptor.cheby_order
        self.m1 = input_param.descriptor.radial_num1
        self.m2 = input_param.descriptor.radial_num2
        self.nfeat = self.m1 * self.m2
        if self.input_param.precision == "float64":
            self.dtype = torch.double
        elif self.input_param.precision == "float32":
            self.dtype = torch.float32
        else:
            raise RuntimeError("train(): unsupported training data type")
        
        self.fitting_net = nn.ModuleList()
        for i in range(self.ntypes):
            self.fitting_net.append(FittingNet(network_size = input_param.model_param.fitting_net.network_size,
                                               bias         = input_param.model_param.fitting_net.bias,
                                               resnet_dt    = input_param.model_param.fitting_net.resnet_dt,
                                               activation   = input_param.model_param.fitting_net.activation,
                                               input_dim    = self.nfeat,
                                               ener_shift   = energy_shift[i],
                                               magic        = False))
        self.set_cparam()

    def set_cparam(self):
        size = self.ntypes * self.ntypes * self.m1 * self.beta
        r_k = torch.normal(mean=0, std=1, size=(size,), dtype=self.dtype)
        m = torch.rand(size, dtype=self.dtype) - 0.5
        s = torch.full_like(m, 0.1)
        c_param = m + s*r_k

        self.c_param = nn.Parameter(c_param.reshape(self.ntypes, self.ntypes, self.m1, self.beta), requires_grad=True)
                
    def get_egroup(self,
                   Ei: torch.Tensor,
                   Egroup_weight: Optional[torch.Tensor] = None,
                   divider: Optional[torch.Tensor] = None)-> Optional[torch.Tensor]:
        if Egroup_weight is not None and divider is not None:       # Egroup_out is not defined in the false branch:
            Egroup = torch.matmul(Egroup_weight, Ei)
            Egroup_out = torch.divide(Egroup.squeeze(-1), divider)
        else:
            Egroup_out = None
        
        return Egroup_out
    
    def get_index(self, user_input_order: List[int], key:torch.Tensor):
        for i, v in enumerate(user_input_order):
            if v == key:
                return i
        return -1

    def get_fitnet_index(self, atom_type: torch.Tensor) -> Tuple[List[int]]:
        fitnet_index: List[int] = []
        for i, atom in enumerate(atom_type):
            index = self.get_index(self.atom_types, atom)
            fitnet_index.append(index)
        return fitnet_index
    
    def forward(self,
                list_neigh: torch.Tensor,   # int32
                Imagetype_map: torch.Tensor,    # int32
                atom_type: torch.Tensor,    # int32
                ImageDR: torch.Tensor,      # float64
                num_neigh: torch.Tensor,    # int32
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
            num_neigh (torch.Tensor): Tensor representing the number of neighbors for each atom. Shape: (batch_size, natoms_sum, ntypes).
            ImageDR (torch.Tensor): Tensor representing the image DRneigh. Shape: (batch_size, natoms_sum, max_neighbor * ntypes, 4).
            nghost (int): Number of ghost atoms.
            Egroup_weight (Optional[torch.Tensor], optional): Tensor representing the Egroup weight. Defaults to None.
            divider (Optional[torch.Tensor], optional): Tensor representing the divider. Defaults to None.
            is_calc_f (Optional[bool], optional): Flag indicating whether to calculate forces and virial. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: Tuple containing the total energy (Etot), atomic energies (Ei), forces (Force), energy group (Egroup), and virial (Virial).
        """
        device = ImageDR.device
        dtype = ImageDR.dtype
        batch_size = list_neigh.shape[0]
        natoms_sum = list_neigh.shape[1]
        ntypes = list_neigh.shape[2]
        m_neigh = list_neigh.shape[3]
        fitnet_index = self.get_fitnet_index(atom_type)
        # t1 = time.time()
        feat, dfeat, list_neigh_alltype, feat_c, dfeat_c, dfeat2c_c, ddfeat2c_c = self.calculate_feat(batch_size, natoms_sum, ntypes, Imagetype_map, num_neigh, list_neigh, ImageDR, device, dtype)
        # feat.requires_grad_(True)
        # feat_c.requires_grad_()
        # t2 = time.time()
        Ei, Ei_c = self.calculate_Ei(Imagetype_map, feat, feat_c, batch_size, fitnet_index, device)
        # t3 = time.time()
        assert Ei is not None
        Etot = torch.sum(Ei, 1)
        Egroup = self.get_egroup(Ei, Egroup_weight, divider) if Egroup_weight is not None else None
        Ei = torch.squeeze(Ei, 2)

        if is_calc_f is False:
            Force, Virial, dF_dc = None, None, None
        else:
            # t4 = time.time()
            Force, Virial = self.calculate_force_virial(Ei, feat, dfeat, natoms_sum, m_neigh, batch_size, list_neigh_alltype, ImageDR, nghost, device, dtype)
            # Force2, dF_dc = self.calculate_force_virial_c(dE_c, feat_c, dfeat_c, dfeat2c_c, ddfeat2c_c, natoms_sum, ntypes, Imagetype_map, m_neigh, batch_size, list_neigh_alltype, ImageDR, nghost, device, dtype)
            # print("Force, Etot \n", Force, Etot)
            # t5 = time.time()
        return Etot, Ei, Force, Egroup, Virial, None, None
    
    def calculate_feat(self,
                     batch_size: int,
                     natoms_sum: int,
                     ntypes: int,
                     Imagetype_map: torch.Tensor,
                     num_neigh: torch.Tensor,
                     list_neigh: torch.Tensor,
                     ImagedR: torch.Tensor, 
                     device: torch.device,
                     dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the feat and dfeat tensors.

        Args:
            natoms_sum (int): The total number of atoms.
            batch_size (int): The batch size.
            num_neigh (torch.Tensor): The tensor representing the number of neighbors for each atom.
            list_neigh (torch.Tensor): The tensor representing the neighbor list.
            ImagedR (torch.Tensor): The tensor containing the atom's distances and Δ(position vectors).
            device (torch.device): The device to perform the calculations on.
            dtype (torch.dtype): The data type of the tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The feat and dfeat tensors.
        """
        descriptor = Descriptor(batch_size, self.beta, self.m1, self.m2, self.Rc_M, self.rcut_smooth, natoms_sum, ntypes, self.m_neigh, Imagetype_map, num_neigh, list_neigh, ImagedR, self.c_param, device, dtype)
        feat = descriptor.feats                                       # (batch_size, natoms_sum, nfeat)
        dfeat = descriptor.dfeats                                     # (batch_size, natoms_sum, nfeat, m_neigh, 3) 
        dfeat = dfeat.transpose(2, 3)                                 # (batch_size, natoms_sum, m_neigh, nfeat, 3)
        list_neigh_alltype = descriptor.neighbor_list_alltypes        # (batch_size, natoms_sum, m_neigh)
        feat_c, dfeat_c, dfeat2c_c, ddfeat2c_c = None, None, None, None
        # c = self.c_param.cpu().detach().numpy()
        # # c[0,1,1,0] = c[0,1,1,0] - 0.0001
        # descriptor_c = descriptor_pybind.MultiDescriptor(batch_size, self.beta, self.m1, self.m2, self.Rc_M, self.rcut_smooth, natoms_sum, ntypes, self.m_neigh, Imagetype_map.cpu().numpy(), num_neigh.cpu().numpy(), list_neigh.cpu().numpy(), ImagedR.cpu().numpy(), c)
        # feat_c = descriptor_c.get_feat()                            # (batch_size, natoms_sum, nfeat)
        # dfeat_c = descriptor_c.get_dfeat()                          # (batch_size, natoms_sum, nfeat, m_neigh, 3)
        # dfeat2c_c = descriptor_c.get_dfeat2c()                      # (batch_size, natoms_sum, nfeat, ntypes, m1, beta)
        # ddfeat2c_c = descriptor_c.get_ddfeat2c()                    # (batch_size, natoms_sum, nfeat, ntypes, m1, beta, m_neigh, 3)
        # list_neigh_alltype_c = descriptor_c.get_neighbor_list()     # (batch_size, natoms_sum, m_neigh)
        # feat_c = torch.tensor(feat_c, dtype=dtype, device=device)
        # dfeat_c = torch.tensor(dfeat_c, dtype=dtype, device=device).transpose(2, 3)
        # dfeat2c_c = torch.tensor(dfeat2c_c, dtype=dtype, device=device)
        # ddfeat2c_c = torch.tensor(ddfeat2c_c, dtype=dtype, device=device).transpose(2, 6)
        # list_neigh_alltype_c = torch.tensor(list_neigh_alltype_c, dtype=torch.int32, device=device)
        
        # Calculate the scaler
        scaler_means = torch.empty(ntypes, dtype=dtype, device=device)
        scaler_stds = torch.empty(ntypes, dtype=dtype, device=device)
        for type in range(ntypes):
            indices = torch.where(Imagetype_map == type)[0]
            feat_type = feat[:, indices].flatten()
            scaler = StandardScaler()
            # scaler = MinMaxScaler()
            scaler.fit_transform(feat_type)
            scaler_means[type] = scaler.mean
            scaler_stds[type] = scaler.std
        
        mean_types = scaler_means[Imagetype_map]
        std_types = scaler_stds[Imagetype_map]
        mean_types = mean_types.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, self.nfeat).detach()
        std_types = std_types.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, self.nfeat).detach()
        feat = (feat - mean_types) / std_types
        # feat_c = (feat_c - mean_types) / std_types
        # std_types = std_types.unsqueeze(2).unsqueeze(-1).repeat(1, 1, self.m_neigh, 1, 3)
        dfeat = dfeat / std_types.unsqueeze(2).unsqueeze(-1).repeat(1, 1, self.m_neigh, 1, 3)
        # dfeat_c = dfeat_c / std_types.unsqueeze(2).unsqueeze(-1).repeat(1, 1, self.m_neigh, 1, 3)
        # dfeat2c_c = dfeat2c_c / std_types.unsqueeze(3).unsqueeze(4).unsqueeze(5).repeat(1, 1, 1, ntypes, self.m1, self.beta)
        # ddfeat2c_c = ddfeat2c_c / std_types.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(7).repeat(1, 1, self.m_neigh, ntypes, self.m1, self.beta, 1, 3)
        return feat, dfeat, list_neigh_alltype, feat_c, dfeat_c, dfeat2c_c, ddfeat2c_c
        
    def calculate_Ei(self, 
                     Imagetype_map: torch.Tensor,
                     feat: torch.Tensor,
                     feat_c: torch.Tensor,
                     batch_size: int,
                     fit_index: List[int],
                     device: torch.device) -> Optional[torch.Tensor]:
        """
        Calculate the energy Ei for each type of atom in the system.

        Args:
            Imagetype_map (torch.Tensor): The tensor mapping atom types to image types.
            feat (torch.Tensor): A tensor representing the atomic descriptors.
            batch_size (int): The size of the batch.
            fit_index (List[int]): The index of the fitting network to use for each atom type.
            type_nums (int): The number of atom types.
            device (torch.device): The device to perform the calculations on.

        Returns:
            Optional[torch.Tensor]: The calculated energy Ei for each type of atom, or None if the calculation fails.
        """
        Ei : Optional[torch.Tensor] = None
        Ei_c : Optional[torch.Tensor] = None
        fit_net_dict = {idx: fit_net for idx, fit_net in enumerate(self.fitting_net)}
        for net in fit_index:
            fit_net = fit_net_dict.get(net) 
            assert fit_net is not None
            mask = (Imagetype_map == net).flatten()
            if not mask.any():
                continue
            indices = torch.arange(len(Imagetype_map.flatten()),device=device)[mask]
            Ei_ntype = fit_net.forward(feat[:, indices])
            # Ei_ntype_c = fit_net.forward(feat_c[:, indices])
            # 打印权重和偏置
            # for name, param in fit_net.named_parameters():
            #     if 'weight' in name:
            #         print('Weight:', name, param.data)
            #     if 'bias' in name:
            #         print('Bias:', name, param.data)
            Ei = Ei_ntype if Ei is None else torch.concat((Ei, Ei_ntype), dim=1)
            # Ei_c = Ei_ntype_c if Ei_c is None else torch.concat((Ei_c, Ei_ntype_c), dim=1)
        return Ei, Ei_c

    def calculate_force_virial(self, 
                               Ei: torch.Tensor,
                               feat: torch.Tensor,
                               dfeat: torch.Tensor,
                               natoms_sum: int,
                               m_neigh: int,
                               batch_size: int,
                               list_neigh_alltype: torch.Tensor,
                               ImageDR: torch.Tensor, 
                               nghost: int,
                               device: torch.device,
                               dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Step 1: Calculate dE / dfeat
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(Ei)]
        dE = torch.autograd.grad([Ei], [feat], grad_outputs=mask, retain_graph=True, create_graph=True)[0]
        assert dE is not None
        # Step 2: Calculate dE / dx (Force calculation)
        # torch.autograd.set_detect_anomaly(True)
        dE_tmp = dE.repeat(1, 1, m_neigh).reshape(batch_size, natoms_sum, m_neigh, -1).unsqueeze(-2)
        dE_dxyz = torch.matmul(dE_tmp, dfeat).sum(-2)  # partial E / partial delta(x,y,z) = (partial E / partial feat) * (partial feat / partial delta(x,y,z) )

        # Step 3: Calculate Force
        list_index = list_neigh_alltype + 1
        Force = torch.zeros((batch_size, natoms_sum + nghost + 1, 3), device=device, dtype=dtype)
        Force[:, 1:natoms_sum + 1, :] = dE_dxyz.sum(-2)
        Virial = torch.zeros((batch_size, 9), device=device, dtype=dtype)
        for batch_idx in range(batch_size):
            indices = list_index[batch_idx].flatten().unsqueeze(-1).expand(-1, 3).to(torch.int64)
            values = - dE_dxyz[batch_idx].view(-1, 3)
            Force[batch_idx].scatter_add_(0, indices, values).view(natoms_sum + nghost + 1, 3)
            # Virial[batch_idx, 0] = (ImageDR[batch_idx, :, :, 1] * dE_dxyz[batch_idx, :, :, 0]).flatten().sum(0)
        Force = Force[:, 1:, :]
        return Force, Virial
    
    def calculate_force_virial_c(self, 
                                dE: torch.Tensor,
                                feat: torch.Tensor,
                                dfeat: torch.Tensor,
                                dfeat2c: torch.Tensor,
                                ddfeat2c: torch.Tensor,
                                natoms_sum: int,
                                ntypes: int,
                                Imagetype_map: torch.Tensor,
                                m_neigh: int,
                                batch_size: int,
                                list_neigh_alltype: torch.Tensor,
                                ImageDR: torch.Tensor, 
                                nghost: int,
                                device: torch.device,
                                dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: Calculate d dE / d feat
        ddE_f2c = None
        for j in range(dE.shape[2]):
            mask: List[Optional[torch.Tensor]] = [torch.ones_like(dE[:, :, j])]
            # partial dE[:, :, j] / partial feat_j' :  j --> feat_j
            grad_dE_j = torch.autograd.grad(dE[:, :, j], feat, grad_outputs=mask, retain_graph=True, create_graph=True)[0]
            ddE = grad_dE_j.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            tmp = (ddE * dfeat2c).sum(2).unsqueeze(2)
            ddE_f2c = tmp if ddE_f2c is None else torch.cat((ddE_f2c, tmp), dim=2)  # partial (partial E / partial feat) / partial c
        assert ddE_f2c is not None
        # Step 2: Calculate dE / dx (Force calculation)
        dE_tmp = dE.repeat(1, 1, m_neigh).reshape(batch_size, natoms_sum, m_neigh, -1).unsqueeze(-2)
        dE_dxyz = torch.matmul(dE_tmp, dfeat).sum(-2)  # partial E / partial delta(x,y,z) = (partial E / partial feat) * (partial feat / partial delta(x,y,z) )
        dE_tmp.unsqueeze_(-2).unsqueeze_(-2).unsqueeze_(-2)
        # Step 3: Calculate dF / dc
        with torch.no_grad():
            # 's' -> batch, 'a' -> natom, 'n' -> neigh, 'f' -> nfeat, 't' -> ntype, 'm' -> mu, 'b' -> beta, 'c' -> 3
            dE_df2c_2 = torch.einsum('sanfc, saftmb -> santmbc', dfeat, ddE_f2c)
            dE_df2c = torch.matmul(dE_tmp, ddfeat2c).sum(-2) + dE_df2c_2   
            # partial F / partial c = (partial E / partial feat) * partial (partial feat / partial delta(x,y,z) ) / partial c
            #                       + (partial (partial E / partial feat) / partial c ) * (partial feat / partial delta(x,y,z) )

        # Step 4: Calculate Force
        # list_neigh_map = torch.zeros_like(list_neigh_alltype)
        # mask_index = list_neigh_alltype == -1
        list_index = list_neigh_alltype + 1
        # list_neigh_alltype[mask_index] = 0.0
        # list_neigh_map.scatter_add_(2, list_neigh_alltype.type(torch.int64), list_index)
        Force = torch.zeros((batch_size, natoms_sum + nghost + 1, 3), device=device, dtype=dtype)
        dF_dc = torch.zeros((batch_size, natoms_sum + nghost, ntypes, ntypes, self.m1, self.beta, 3), device=device, dtype=dtype)
        Force[:, 1:natoms_sum + 1, :] = dE_dxyz.sum(-2)
        # Step 5: Calculate ∂Force / ∂c
        dF_dc_ = dE_df2c.sum(2)
        itype_indices = Imagetype_map.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, ntypes)
        itype_range = torch.arange(ntypes, device=device).view(1, 1, -1)
        itype_mask = (itype_indices == itype_range)
        dF_dc = torch.where(itype_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 
                            dF_dc_.unsqueeze(2).expand(-1, -1, ntypes, -1, -1, -1, -1),
                            dF_dc)
        Virial = torch.zeros((batch_size, 9), device=device, dtype=dtype)
        for batch_idx in range(batch_size):
            indices = list_index[batch_idx].flatten().unsqueeze(-1).expand(-1, 3).to(torch.int64)
            values = - dE_dxyz[batch_idx].view(-1, 3)
            Force[batch_idx].scatter_add_(0, indices, values).view(natoms_sum + nghost + 1, 3)
            # Virial[batch_idx, 0] = (ImageDR[batch_idx, :, :, 1] * dE_dxyz[batch_idx, :, :, 0]).flatten().sum(0)
            # for itype in range(ntypes):
            #     itype_indices = torch.where(Imagetype_map == itype)[0]
            #     dF_dc[batch_idx, itype_indices, itype] = dF_dc_[batch_idx][itype_indices]
            # Create a mask for valid neighbors (not -1)
            neighbor_mask = list_neigh_alltype[batch_idx] != -1
            
            # Get the itypes and jtypes for all atoms and their neighbors
            itypes = Imagetype_map.unsqueeze(1).expand(-1, list_neigh_alltype.size(2))
            # jtypes = Imagetype_map[list_neigh_alltype[batch_idx]]
            
            # Create indices for the result tensor
            atom_indices = torch.arange(natoms_sum, device=device).unsqueeze(1).expand_as(itypes)
            
            scatter_neighs = torch.arange(len(atom_indices[neighbor_mask]), device=device)
            scatter_values = -dE_df2c[batch_idx][neighbor_mask]
            scatter_values_ = torch.zeros_like(scatter_values.unsqueeze(1).expand(-1, ntypes, -1, -1, -1, -1))
            scatter_values_.index_put_((scatter_neighs, itypes[neighbor_mask]), scatter_values)
            dF_dc[batch_idx].index_add_(0, list_neigh_alltype[batch_idx][neighbor_mask], scatter_values_)
            """
            # Compute the values to be added
            _, counts = torch.unique(atom_indices[neighbor_mask], return_counts=True)
            end_indices = counts.cumsum(dim=0)
            start_indices = end_indices - counts

            scatter_indices = torch.stack([
                atom_indices[neighbor_mask],
                jtypes[neighbor_mask],
                list_neigh_alltype[batch_idx][neighbor_mask]
            ], dim=-1)

            for i, (iatom, itype, jatom) in enumerate(scatter_indices):
                mask_c = (scatter_indices[start_indices[jatom]:end_indices[jatom], -1] == iatom)
                dF_dc[batch_idx, iatom, itype] += scatter_values[start_indices[jatom]:end_indices[jatom]][mask_c].sum(0)
            """
        Force = Force[:, 1:, :]
        return Force, dF_dc

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise Exception("You need to call `fit` before `transform`.")
        return (data - self.mean) / self.std

    def fit_transform(self, data):
        self.fit(data)