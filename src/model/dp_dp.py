import numpy as np
import torch
from torch import embedding
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import sys, os
from typing import List, Tuple, Optional

sys.path.append(os.getcwd())
# import parameters as pm    
# import prepare as pp
# pp.readFeatnum()
from model.dp_embedding import EmbeddingNet, FittingNet
from model.calculate_force import CalculateForce, CalculateVirialForce
# logging and our extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# setup logging module
logger = logging.getLogger('train.DPFF')

def dump(msg, *args, **kwargs):
    logger.log(logging_level_DUMP, msg, *args, **kwargs)
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)
def summary(msg, *args, **kwargs):
    logger.log(logging_level_SUMMARY, msg, *args, **kwargs)
def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)
def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)
def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs, exc_info=True)

class DP(nn.Module):
    def __init__(self, config, device, stat, magic=False):
        super(DP, self).__init__()
        self.config = config
        self.ntypes = len(config['atomType'])
        self.atom_type = [_['type'] for _ in config['atomType']] #this value in used in forward for hybrid Training
        self.device = device
        self.stat = stat
        self.M2 = config["M2"]
        self.maxNeighborNum = config["maxNeighborNum"]
        if self.config["training_type"] == "float64":
            self.dtype = torch.double
        elif self.config["training_type"] == "float32":
            self.dtype = torch.float32
        else:
            raise RuntimeError("train(): unsupported training data type")

        self.embedding_net = nn.ModuleList()
        self.fitting_net = nn.ModuleList()
        
        # initial bias for fitting net? 
        for i in range(self.ntypes):
            for j in range(self.ntypes):
                self.embedding_net.append(EmbeddingNet(self.config["net_cfg"]["embedding_net"]["network_size"], 
                                                       self.config["net_cfg"]["embedding_net"]["bias"], 
                                                       self.config["net_cfg"]["embedding_net"]["resnet_dt"], 
                                                       self.config["net_cfg"]["embedding_net"]["activation"], 
                                                       self.device, 
                                                       magic))
                # self.embedding_net.append(EmbeddingNet(self.config["net_cfg"]["embedding_net"], magic))
            fitting_net_input_dim = self.config["net_cfg"]["embedding_net"]["network_size"][-1]
            self.fitting_net.append(FittingNet(self.config["net_cfg"]["fitting_net"]["network_size"], 
                                               self.config["net_cfg"]["fitting_net"]["bias"],
                                               self.config["net_cfg"]["fitting_net"]["resnet_dt"],
                                               self.config["net_cfg"]["fitting_net"]["activation"], 
                                               self.device, 
                                               self.M2 * fitting_net_input_dim, self.stat[2][i], magic))

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
    def get_train_2body_type(self, atom_type_data: List[int]) -> Tuple[List[List[List[int]]], int]:
        type_2body_list: List[List[List[int]]] = []         # 整数列表的列表
        type_2body_index: List[int] = []
        for atom_type in atom_type_data:
            if atom_type != 0:
                # type_2body_index.append(self.atom_type.index(atom_type))
                for i, known_type in enumerate(self.atom_type):
                    if atom_type == known_type:
                        type_2body_index.append(i)
                        break

        for atom in type_2body_index:
            type_2body: List[List[int]] = []        # 整数列表的列表
            for atom2 in type_2body_index:
                type_2body.append([atom, atom2])        # 整数列表
            type_2body_list.append(type_2body)
        return type_2body_list, len(type_2body_index)

    '''
    description: 
        when we do forward, we should adjust the data input to adapt the model
    param {*} self
    param {*} Ri
    param {*} dfeat
    param {*} list_neigh
    param {*} natoms_img
    param {*} atom_type
    param {*} ImageDR
    param {array} is_egroup
    param {*} is_calc_f
    return {*}
    author: wuxingxing
    '''
    def forward0(self, Ri, dfeat, list_neigh, natoms_img, atom_type, ImageDR, Egroup_weight = None, divider = None, is_calc_f = True):

        #torch.autograd.set_detect_anomaly(True)
        Ri_d = dfeat
        # dim of natoms_img: batch size, natom_sum & natom_types ([9, 6, 2, 1])
        natoms = natoms_img[0, 1:]
        natoms_sum = Ri.shape[1]
        batch_size = Ri.shape[0]
        atom_sum = 0
        ntype = 0  # 因为for循环外没有定义ntype，但是DR_ntype中却直接用？
        # emb_list, type_nums =  self.get_train_2body_type(list(np.array(atom_type.cpu())[0]))
        atom_type_list: List[int] = atom_type.cpu().tolist()
        emb_list, type_nums = self.get_train_2body_type(atom_type_list[0])
        Ei = torch.tensor([])
        for type_emb in emb_list:
            xyz_scater_a = torch.tensor([])
            for emb in type_emb:
                ntype, ntype_1 = emb
                # print(ntype, "\t\t", ntype_1)
                # dim of Ri: batch size, natom_sum, ntype*max_neigh_num, local environment matrix , ([10,9,300,4])
                S_Rij = Ri[:, atom_sum:atom_sum+natoms[ntype], ntype_1 * self.maxNeighborNum:(ntype_1+1) * self.maxNeighborNum, 0].unsqueeze(-1)
                # determines which embedding net
                embedding_index = ntype * self.ntypes + ntype_1
                # itermediate output of embedding net 
                G = torch.tensor([])
                found = False
                for idx, emb_net in enumerate(self.embedding_net):
                    if idx == embedding_index and not found:
                        G = emb_net(S_Rij)
                        found = True
                # dim of G: batch size, natom of ntype, max_neigh_num, final layer dim
                # G = self.embedding_net[embedding_index](S_Rij)
                # symmetry conserving 
                tmp_a = Ri[:, atom_sum:atom_sum+natoms[ntype], ntype_1 * self.maxNeighborNum:(ntype_1+1) * self.maxNeighborNum].transpose(-2, -1)
                tmp_b = torch.matmul(tmp_a, G)
                # xyz_scater_a = tmp_b if xyz_scater_a is None else xyz_scater_a + tmp_b
                if xyz_scater_a.numel() == 0:  # 检查 tensor 是否为空
                    xyz_scater_a = tmp_b
                else:
                    xyz_scater_a = xyz_scater_a + tmp_b
            
            # attention: for hybrid training, the division should be done based on \
            #   the number of element types in the current image, because the images may from different systems.
            xyz_scater_a = xyz_scater_a / (self.maxNeighborNum * type_nums)
            xyz_scater_b = xyz_scater_a[:, :, :, :self.M2]
            DR_ntype = torch.matmul(xyz_scater_a.transpose(-2, -1), xyz_scater_b)
            DR_ntype = DR_ntype.reshape(batch_size, natoms[ntype], -1)
            
            Ei_ntype = torch.tensor([])
            found = False
            for idx, fit_net in enumerate(self.fitting_net):
                if idx == ntype and not found:
                    Ei_ntype = fit_net(DR_ntype)
                    found = True
            # Ei_ntype = self.fitting_net[ntype](DR_ntype)
            # Ei = Ei_ntype if Ei is None else torch.concat((Ei, Ei_ntype), dim=1)
            if Ei.numel() == 0:
                Ei = Ei_ntype
            else:
                Ei = torch.concat((Ei, Ei_ntype), dim=1)
            atom_sum = atom_sum + natoms[ntype]
        
        Etot = torch.sum(Ei, 1)   

        if Egroup_weight is not None:
            Egroup = self.get_egroup(Ei, Egroup_weight, divider)
        else:
            Egroup = None
        #Egroup = 0 
        # F = torch.zeros((batch_size, atom_sum, 3), device=self.device)
        # Virial = torch.zeros((batch_size, 9), device=self.device)
        Ei = torch.squeeze(Ei, 2)
        Force, Virial = None, None
        if is_calc_f == False:
            return Etot, Ei, Force, Egroup, Virial
        # start_autograd = time.time()
        # print("fitting time:", start_autograd - start_fitting, 's')
        
        # mask = torch.ones_like(Ei)
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(Ei)]
        dE = torch.autograd.grad([Ei], [Ri], grad_outputs=mask, retain_graph=True, create_graph=True)
        valid_grads = []
        for g in dE:
            if g is not None:
                valid_grads.append(g)
        dE = torch.stack(valid_grads, dim=0).squeeze(0) 
        # dE = torch.cat([g.unsqueeze(0) for g in dE if g is not None], dim=0).squeeze(0)
        # dE = torch.stack(list(dE), dim=0).squeeze(0)  #[:,:,:,:-1] #[2,108,100,4]-->[2,108,100,3]
        Ri_d = Ri_d.reshape(batch_size, natoms_sum, -1, 3)
        dE = dE.reshape(batch_size, natoms_sum, 1, -1)

        # start_force = time.time()
        # print("autograd time:", start_force - start_autograd, 's')
        F = torch.matmul(dE, Ri_d).squeeze(-2) # batch natom 3
        F = F * (-1)
        
        list_neigh = torch.unsqueeze(list_neigh,2)
        list_neigh = (list_neigh - 1).type(torch.int)
        F = CalculateForce.apply(list_neigh, dE, Ri_d, F)
        
        #print ("Force")
        #print (F)
        # virial = CalculateVirialForce.apply(list_neigh, dE, Ri[:,:,:,:3], Ri_d)
        virial = CalculateVirialForce.apply(list_neigh, dE, ImageDR, Ri_d)
        
        # no need to switch sign here 
        #virial = virial * (-1)

        return Etot, Ei, F, Egroup, virial  #F is Force
    
    def forward(self, 
                Ri: torch.Tensor, 
                dfeat: torch.Tensor, 
                list_neigh: torch.Tensor, 
                natoms_img: torch.Tensor, 
                atom_type: torch.Tensor, 
                ImageDR: torch.Tensor, 
                Egroup_weight: Optional[torch.Tensor] = None, 
                divider: Optional[torch.Tensor] = None, 
                is_calc_f: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        Ri_d = dfeat
        natoms = natoms_img[0, 1:]
        natoms_sum = Ri.shape[1]
        batch_size = Ri.shape[0]
        atom_type_list: List[int] = atom_type.cpu().tolist()
        emb_list, type_nums = self.get_train_2body_type(atom_type_list[0])

        Ei = self.calculate_Ei(Ri, natoms, batch_size, emb_list, type_nums)

        Etot = torch.sum(Ei, 1)
        Egroup = self.get_egroup(Ei, Egroup_weight, divider) if Egroup_weight is not None else None
        Ei = torch.squeeze(Ei, 2)

        if is_calc_f is False:
            Force, Virial = None, None
        else:
            Force, Virial = self.calculate_force_virial(Ri, Ri_d, Ei, natoms_sum, batch_size, list_neigh, ImageDR)

        return Etot, Ei, Force, Egroup, Virial
        
    def calculate_Ei(self, 
                     Ri: torch.Tensor,
                     natoms: torch.Tensor,
                     batch_size: int,
                     emb_list: List[List[List[int]]],
                     type_nums: int):
        Ei = torch.tensor([])
        atom_sum = 0
        for type_emb in emb_list:
            xyz_scater_a, xyz_scater_b, ntype = self.calculate_xyz_scater(Ri, atom_sum, natoms, type_emb, type_nums)
            DR_ntype = torch.matmul(xyz_scater_a.transpose(-2, -1), xyz_scater_b)
            DR_ntype = DR_ntype.reshape(batch_size, natoms[ntype], -1)

            Ei_ntype = torch.tensor([])
            found = False
            for idx, fit_net in enumerate(self.fitting_net):
                if idx == ntype and not found:
                    Ei_ntype = fit_net(DR_ntype)
                    found = True
            Ei = Ei_ntype if Ei.numel() == 0 else torch.concat((Ei, Ei_ntype), dim=1)
            atom_sum = atom_sum + natoms[ntype]
        return Ei
            
    def calculate_xyz_scater(self, 
                             Ri: torch.Tensor,
                             atom_sum: int,
                             natoms: torch.Tensor,
                             type_emb: List[List[int]],
                             type_nums: int):
        xyz_scater_a = torch.tensor([])
        ntype = 0
        for emb in type_emb:
            ntype, ntype_1 = emb
            S_Rij = Ri[:, atom_sum:atom_sum+natoms[ntype], ntype_1 * self.maxNeighborNum:(ntype_1+1) * self.maxNeighborNum, 0].unsqueeze(-1)
            embedding_index = ntype * self.ntypes + ntype_1
            G = torch.tensor([])
            found = False
            for idx, emb_net in enumerate(self.embedding_net):
                if idx == embedding_index and not found:
                    G = emb_net(S_Rij)
                    found = True
            tmp_a = Ri[:, atom_sum:atom_sum+natoms[ntype], ntype_1 * self.maxNeighborNum:(ntype_1+1) * self.maxNeighborNum].transpose(-2, -1)
            tmp_b = torch.matmul(tmp_a, G)
            xyz_scater_a = tmp_b if xyz_scater_a.numel() == 0 else xyz_scater_a + tmp_b
        return xyz_scater_a / (self.maxNeighborNum * type_nums), xyz_scater_a[:, :, :, :self.M2], ntype

    def calculate_force_virial(self, 
                               Ri: torch.Tensor,
                               Ri_d: torch.Tensor,
                               Ei: torch.Tensor,
                               natoms_sum: int,
                               batch_size: int,
                               list_neigh: torch.Tensor,
                               ImageDR: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(Ei)]
        dE = torch.autograd.grad([Ei], [Ri], grad_outputs=mask, retain_graph=True, create_graph=True)
        dE_list = []
        for g in dE:
            if g is not None:
                dE_list.append(g.unsqueeze(0))
        dE = torch.cat(dE_list, dim=0).squeeze(0)
        # dE = torch.cat([g.unsqueeze(0) for g in dE if g is not None], dim=0).squeeze(0)
        Ri_d = Ri_d.reshape(batch_size, natoms_sum, -1, 3)
        dE = dE.reshape(batch_size, natoms_sum, 1, -1)
        F = torch.matmul(dE, Ri_d).squeeze(-2) * (-1)
        '''this part for cuda version, not support for torchscript
        list_neigh = torch.unsqueeze(list_neigh,2)
        list_neigh = (list_neigh - 1).type(torch.int)
        F = CalculateForce.apply(list_neigh, dE, Ri_d, F)
        Virial = CalculateVirialForce.apply(list_neigh, dE, ImageDR, Ri_d)
        '''
        # 1. Get atom_idx & neighbor_idx
        # non_zero_indices = list_neigh.nonzero(as_tuple=True)
        non_zero_indices = list_neigh.nonzero().unbind(1)
        batch_indices, atom_indices, neighbor_indices = non_zero_indices
        atom_idx = list_neigh[non_zero_indices].long() - 1
        # 2. Calculate Force using indexing
        expanded_dims_dE = [len(batch_indices), 1, 1]
        expanded_dims_Ri_d = [len(batch_indices), 1, 3]
        start_indices = neighbor_indices.view(-1, 1) * 4
        index_range_dE = torch.arange(4, device=start_indices.device).view(1, 1, -1).repeat(expanded_dims_dE)
        index_range_Ri_d = torch.arange(4, device=start_indices.device).view(1, -1, 1).repeat(expanded_dims_Ri_d)
        gather_indices_dE = (start_indices.view(-1, 1, 1) + index_range_dE).long()
        gather_indices_Ri_d = (start_indices.view(-1, 1, 1) + index_range_Ri_d).long()
        dE_selected = torch.gather(dE[batch_indices, atom_indices], -1, gather_indices_dE)
        Ri_d_selected = torch.gather(Ri_d[batch_indices, atom_indices], -2, gather_indices_Ri_d)
        dE_dx = torch.matmul(dE_selected, Ri_d_selected).squeeze(-2)
        # F[0].index_add_(0, atom_idx, dE_dx)          # batch_size = 1
        # for i in range(len(atom_idx)):
        #     F[batch_indices[i], atom_idx[i]] += dE_dx[i]
        for batch_idx in range(batch_size):
            mask_accumulation = batch_indices == batch_idx
            F[batch_idx].index_add_(0, atom_idx[mask_accumulation], dE_dx[mask_accumulation])

        # 3. Calculate Virial
        Virial = torch.zeros((batch_size, 9), device="cuda:2", dtype=self.dtype)
        virial_components = torch.zeros((len(batch_indices), 6), device=Virial.device, dtype=self.dtype)
        virial_components[:, 0] = ImageDR[batch_indices, atom_indices, neighbor_indices][:, 0] * dE_dx[:, 0] # xx
        virial_components[:, 1] = ImageDR[batch_indices, atom_indices, neighbor_indices][:, 0] * dE_dx[:, 1] # xy
        virial_components[:, 2] = ImageDR[batch_indices, atom_indices, neighbor_indices][:, 0] * dE_dx[:, 2] # xz
        virial_components[:, 3] = ImageDR[batch_indices, atom_indices, neighbor_indices][:, 1] * dE_dx[:, 1] # yy
        virial_components[:, 4] = ImageDR[batch_indices, atom_indices, neighbor_indices][:, 1] * dE_dx[:, 2] # yz 
        virial_components[:, 5] = ImageDR[batch_indices, atom_indices, neighbor_indices][:, 2] * dE_dx[:, 2] # zz 
        Virial[:, [0, 1, 2, 4, 5, 8]] = virial_components.sum(dim=0)
        Virial[:, [3, 6, 7]] = Virial[:, [1, 2, 5]]

        return F, Virial    