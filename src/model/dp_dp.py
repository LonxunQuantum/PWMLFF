import sys, os
import time
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
from src.model.dp_embedding import EmbeddingNet, FittingNet
from src.model.calculate_force import CalculateCompress, CalculateForce, CalculateVirialForce
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
    def __init__(self, config, device, davg, dstd, energy_shift, magic=False):
        super(DP, self).__init__()
        self.config = config
        self.ntypes = len(config['atomType'])
        self.atom_type = [_['type'] for _ in config['atomType']] #this value in used in forward for hybrid Training
        self.device = device
        self.M2 = config["M2"]
        self.Rmax = config["Rc_M"]
        self.Rmin = config['atomType'][0]['Rm']
        self.maxNeighborNum = config["maxNeighborNum"]
        if self.config["training_type"] == "float64":
            self.dtype = torch.double
        elif self.config["training_type"] == "float32":
            self.dtype = torch.float32
        else:
            raise RuntimeError("train(): unsupported training data type")
        self.davg = torch.tensor(davg, device=device, dtype=self.dtype)
        self.dstd = torch.tensor(dstd, device=device, dtype=self.dtype)
        self.energy_shift = torch.tensor(energy_shift, device=device, dtype=self.dtype)

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
                                               self.M2 * fitting_net_input_dim, energy_shift[i], magic))
            # self.fitting_net.append(FittingNet(config["net_cfg"]["fitting_net"], self.M2 * fitting_net_input_dim, energy_shift[i], magic))
        
        self.compress_tab = None #for dp compress

    def set_comp_tab(self, compress_dict:dict):
        self.compress_tab = torch.tensor(compress_dict["table"], dtype=self.dtype, device=self.device)
        self.dx = compress_dict["dx"]
        # self.davg = compress_dict["davg"]
        # self.dstd = compress_dict["dstd"]
        self.sij_min = compress_dict["sij_min"]
        self.sij_max = compress_dict["sij_max"]
        self.sij_out_max = compress_dict["sij_out_max"]
        self.sij_len = compress_dict["sij_len"]
        self.sij_out_len = compress_dict["sij_out_len"]
        self.order = compress_dict["order"] if "order" in compress_dict.keys() else 5 #the default compress order is 5

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
        type_2body_list: List[List[List[int]]] = []         
        type_2body_index: List[int] = []
        for i, atom_type in enumerate(atom_type_data):
            if atom_type != 0 and atom_type in self.atom_type:
                type_2body_index.append(i)
        # for atom in type_2body_index:
        #     type_2body: List[List[int]] = []        # 整数列表的列表
        #     for atom2 in type_2body_index:
        #         type_2body.append([atom, atom2])        # 整数列表
        #     type_2body_list.append(type_2body)
        pair_indices = len(type_2body_index)
        num_pairs = int(pair_indices ** 2)
        type_2body: List[List[int]] = []
        for n in range(num_pairs):
            i = n // pair_indices
            j = n % pair_indices
            # atom = type_2body_index[i]
            # atom2 = type_2body_index[j]
            if j == 0:
                type_2body = []
                type_2body_list.append(type_2body)
            type_2body.append([i, j])
        return type_2body_list, pair_indices

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
        # from torchviz import make_dot
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
                tmp_a = Ri[:, atom_sum:atom_sum+natoms[ntype], ntype_1 * self.maxNeighborNum:(ntype_1+1) * self.maxNeighborNum].transpose(-2, -1)
                # start_emb = time.time()
                if self.compress_tab is None:
                    G = self.embedding_net[embedding_index](S_Rij)
                    # symmetry conserving
                else:
                    if self.order == 2:
                        G = self.calc_compress(S_Rij, embedding_index)
                    elif self.order == 5:
                        G = self.calc_compress_5order(S_Rij, embedding_index)
                    elif self.order == 3:
                        G = self.calc_compress_3order(S_Rij, embedding_index)
                # end_emb = time.time()
                # print("embedding time:", end_emb - start_emb, 's')
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
        
        # error code
        # dE1 = dE.squeeze(2).reshape(batch_size, atom_sum, self.maxNeighborNum*self.ntypes,4).unsqueeze(-1) #[5, 76, 1, 800] -> [5, 76, 800] -> [5, 76, 200, 4] -> [5, 76, 200, 4, 1]
        # Ri_d1 = Ri_d.reshape(batch_size, atom_sum, self.maxNeighborNum*self.ntypes, 4, 3)#[5, 76, 800, 3] -> [5, 76, 200, 4, 3]
        # temp_dE_dx = torch.sum(dE1*Ri_d1, dim=3) #[5, 76, 200, 4, 3]->[5, 76, 200, 3]
        # F_res = F + torch.sum(temp_dE_dx, dim=2)

        # for cpu device
        if self.device.type == 'cpu':
            Virial = torch.zeros((batch_size, 9), device=self.device, dtype=self.dtype)
            for batch_idx in range(batch_size):   
                for i in range(natoms_sum):
                    # get atom_idx & neighbor_idx
                    i_neighbor = list_neigh[batch_idx, i]  #[100]
                    neighbor_idx = i_neighbor.nonzero().squeeze().type(torch.int64)  #[78]
                    atom_idx = i_neighbor[neighbor_idx].type(torch.int64) - 1
                    # calculate Force
                    for neigh_tmp, neighbor_id in zip(atom_idx, neighbor_idx):
                        tmpA = dE[batch_idx, i, :, neighbor_id*4:neighbor_id*4+4]
                        tmpB = Ri_d[batch_idx, i, neighbor_id*4:neighbor_id*4+4]
                        dE_dx = torch.matmul(tmpA, tmpB).squeeze(0)
                        F[batch_idx, neigh_tmp] += dE_dx

                        Virial[batch_idx][0] += ImageDR[batch_idx, i, neighbor_id][0]*dE_dx[0] #xx
                        Virial[batch_idx][4] += ImageDR[batch_idx, i, neighbor_id][1]*dE_dx[1] #yy
                        Virial[batch_idx][8] += ImageDR[batch_idx, i, neighbor_id][2]*dE_dx[2] #zz

                        Virial[batch_idx][1] += ImageDR[batch_idx, i, neighbor_id][0]*dE_dx[1] 
                        Virial[batch_idx][2] += ImageDR[batch_idx, i, neighbor_id][0]*dE_dx[2]
                        Virial[batch_idx][5] += ImageDR[batch_idx, i, neighbor_id][1]*dE_dx[2]

                Virial[batch_idx][3] = Virial[batch_idx][1]
                Virial[batch_idx][6] = Virial[batch_idx][2]
                Virial[batch_idx][7] = Virial[batch_idx][5]
        else:
            list_neigh = torch.unsqueeze(list_neigh,2)
            list_neigh = (list_neigh - 1).type(torch.int)
            F = CalculateForce.apply(list_neigh, dE, Ri_d, F)
            # virial = CalculateVirialForce.apply(list_neigh, dE, Ri[:,:,:,:3], Ri_d)
            Virial = CalculateVirialForce.apply(list_neigh, dE, ImageDR, Ri_d)
        
        # end_forward = time.time()
        # print("forward time:", end_forward - start_forward, 's')
        return Etot, Ei, F, Egroup, Virial  #F is Force
                      
   
    def forward(self, 
                list_neigh: torch.Tensor,   # int32
                Imagetype_map: torch.Tensor,    # int32
                Imagetype: torch.Tensor,    # int32
                ImageDR: torch.Tensor,      # float64
                nghost: int, 
                Egroup_weight: Optional[torch.Tensor] = None, 
                divider: Optional[torch.Tensor] = None, 
                is_calc_f: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        device = ImageDR.device
        dtype = ImageDR.dtype
        # atom_num_per_image = torch.unique(Imagetype_map, sorted=True, return_counts=True)[1]
        # atom_num_per_image = torch.zeros(len(self.atom_type), dtype=torch.int32, device=device)
        # for i, atom_type in enumerate(self.atom_type):
        #     atom_num_per_image[i] = (Imagetype == atom_type).sum()
        batch_size = list_neigh.shape[0]
        natoms_sum = list_neigh.shape[1]
        max_neighbor_type = list_neigh.shape[2]  # ntype * max_neighbor_num
        emb_list, type_nums = self.get_train_2body_type(self.atom_type)
        # t1 = time.time()
        Ri, Ri_d = self.calculate_Ri(natoms_sum, batch_size, max_neighbor_type, Imagetype_map, ImageDR, device, dtype)
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
    
    def calculate_Ri(self,
                     natoms_sum: int,
                     batch_size: int,
                     max_neighbor_type: int,
                     Imagetype_map: torch.Tensor,
                     ImagedR: torch.Tensor, 
                     device: torch.device,
                     dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # t1 = time.time()
        R = ImagedR[:, :, :, 0]
        R.requires_grad_()

        Srij = torch.zeros(batch_size, natoms_sum, max_neighbor_type, dtype=dtype, device=device)
        Srij[(R > 0) & (R < self.Rmin)] = 1 / R[(R > 0) & (R < self.Rmin)]
        r = R[(R > self.Rmin) & (R < self.Rmax)]
        Srij[(R > self.Rmin) & (R < self.Rmax)] = 1 / r * (((r - self.Rmin) / (self.Rmax - self.Rmin)) ** 3 * (
                -6 * ((r - self.Rmin) / (self.Rmax - self.Rmin)) ** 2 + 15 * (
                   (r - self.Rmin) / (self.Rmax - self.Rmin)) - 10) + 1)
        mask = R.abs() > 1e-5

        # Ri[batch_size, natoms_sum, max_neighbor * ntypes, 4] 4-->[Srij, Srij * xij / rij, Srij * yij / rij, Srij * yij / rij]
        Ri = torch.zeros(batch_size, natoms_sum, max_neighbor_type, 4, dtype=dtype, device=device)
        Ri[:, :, :, 0] = Srij
        Ri[:, :, :, 1][mask] = Srij[mask] * ImagedR[:, :, :, 1][mask] / ImagedR[:, :, :, 0][mask]
        Ri[:, :, :, 2][mask] = Srij[mask] * ImagedR[:, :, :, 2][mask] / ImagedR[:, :, :, 0][mask]
        Ri[:, :, :, 3][mask] = Srij[mask] * ImagedR[:, :, :, 3][mask] / ImagedR[:, :, :, 0][mask]
        # t2 = time.time()
        # d_Srij[batch_size, natoms_sum, max_neighbor * ntypes]  d_Srij = dSrij/drij
        mask_Srij: List[Optional[torch.Tensor]] = [torch.ones_like(R)]
        d_Srij = torch.autograd.grad([Srij], [R], grad_outputs=mask_Srij, retain_graph=True, create_graph=True)[0]
        assert d_Srij is not None
        # feat [batch_size, natoms_sum, max_neighbor * ntypes, 4]
        # dfeat [batch_size, natoms_sum, max_neighbor * ntypes, 4, 3] 3-->[dx, dy, dz]
        # feat = torch.zeros(batch_size, natoms_sum, max_neighbor_type, 4, dtype=dtype, device=device)
        dfeat = torch.zeros(batch_size, natoms_sum, max_neighbor_type, 4, 3, dtype=dtype, device=device)
        rij = ImagedR[:, :, :, 0][mask]
        xij = ImagedR[:, :, :, 1][mask]
        yij = ImagedR[:, :, :, 2][mask]
        zij = ImagedR[:, :, :, 3][mask]
        Srij_temp = Srij[mask]
        common = (d_Srij[mask] - Srij_temp / rij) / rij ** 2
        # dSrij / dxij = d_Srij * xij / rij (x-->x,y,z)
        # d(Srij * xij / rij) / dxij = xij * xij * (d_Srij - Srij / rij) / rij ** 2 + Srij / rij * ( dxij / dxij)
        dfeat[:, :, :, 0, 0][mask] = d_Srij[mask] * xij / rij
        dfeat[:, :, :, 1, 0][mask] = common * xij * xij + Srij_temp / rij
        dfeat[:, :, :, 2, 0][mask] = common * yij * xij
        dfeat[:, :, :, 3, 0][mask] = common * zij * xij
        # dy
        dfeat[:, :, :, 0, 1][mask] = d_Srij[mask] * yij / rij
        dfeat[:, :, :, 1, 1][mask] = common * xij * yij
        dfeat[:, :, :, 2, 1][mask] = common * yij * yij + Srij_temp / rij
        dfeat[:, :, :, 3, 1][mask] = common * zij * yij
        # dz
        dfeat[:, :, :, 0, 2][mask] = d_Srij[mask] * zij / rij
        dfeat[:, :, :, 1, 2][mask] = common * xij * zij
        dfeat[:, :, :, 2, 2][mask] = common * yij * zij
        dfeat[:, :, :, 3, 2][mask] = common * zij * zij + Srij_temp / rij

        # t3 = time.time()
        # 0 is that the atom nums is zero, for example, CH4 system in CHO system hybrid training, O atom nums is zero.\
        # beacuse the dstd or davg does not contain O atom, therefore, special treatment is needed here for atoms with 0 elements
        davg_res = self.davg[Imagetype_map].to(dtype)
        dstd_res = self.dstd[Imagetype_map].to(dtype)
        davg_res = davg_res.reshape(-1, natoms_sum, max_neighbor_type, 4).repeat(batch_size, 1, 1, 1)
        dstd_res = dstd_res.reshape(-1, natoms_sum, max_neighbor_type, 4).repeat(batch_size, 1, 1, 1) 
        Ri = (Ri - davg_res) / dstd_res
        dstd_res = dstd_res.unsqueeze(-1).repeat(1, 1, 1, 1, 3)
        dfeat = - dfeat / dstd_res
        print("Ri.max", Ri.max())
        # t4 = time.time()
        # print("\nRi:", t2-t1, "\ndfeat:", t3-t2, "\ndavg_res:", t4-t3, "\n********************")
        return Ri, dfeat
        
    def calculate_Ei(self, 
                     Imagetype_map: torch.Tensor,
                     Ri: torch.Tensor,
                     batch_size: int,
                     emb_list: List[List[List[int]]],
                     type_nums: int, 
                     device: torch.device) -> Optional[torch.Tensor]:
        Ei : Optional[torch.Tensor] = None
        fit_net_dict = {idx: fit_net for idx, fit_net in enumerate(self.fitting_net)}
        for type_emb in emb_list:
            # t1 = time.time()
            xyz_scater_a, xyz_scater_b, ntype = self.calculate_xyz_scater(Imagetype_map, Ri, type_emb, type_nums, device)
            DR_ntype = torch.matmul(xyz_scater_a.transpose(-2, -1), xyz_scater_b)
            DR_ntype = DR_ntype.reshape(batch_size, xyz_scater_a.shape[1], -1)
            # t2 = time.time()
            Ei_ntype: Optional[torch.Tensor] = None
            fit_net = fit_net_dict.get(ntype) 
            assert fit_net is not None
            Ei_ntype = fit_net.forward(DR_ntype)
            # found = False
            # for idx, fit_net in enumerate(self.fitting_net):
            #     if idx == ntype and not found:
            #         Ei_ntype = fit_net(DR_ntype)
            #         found = True 
            # t3 = time.time()
            assert Ei_ntype is not None
            Ei = Ei_ntype if Ei is None else torch.concat((Ei, Ei_ntype), dim=1)
            # t4 = time.time()
            # print("calculate_xyz_scater:", t2-t1, "\nfitting_net:", t3-t2, "\nconcat:", t4-t3, "\n********************")
        return Ei
            
    def calculate_xyz_scater(self, 
                             Imagetype_map: torch.Tensor,
                             Ri: torch.Tensor,
                             type_emb: List[List[int]],
                             type_nums: int, 
                             device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
        xyz_scater_a : Optional[torch.Tensor] = None
        ntype = 0
        emb_net_dict = {idx: emb_net for idx, emb_net in enumerate(self.embedding_net)}
        for emb in type_emb:
            # t1 = time.time()
            ntype, ntype_1 = emb
            mask = (Imagetype_map == ntype).flatten()
            indices = torch.arange(len(Imagetype_map.flatten()),device=device)[mask]      
            S_Rij = Ri[:, indices, ntype_1 * self.maxNeighborNum:(ntype_1+1) * self.maxNeighborNum, 0].unsqueeze(-1)
            # t2 = time.time()
            embedding_index = ntype * self.ntypes + ntype_1
            G: Optional[torch.Tensor] = None
            if self.compress_tab is None:
                emb_net = emb_net_dict.get(embedding_index)
                assert emb_net is not None
                G = emb_net.forward(S_Rij)
                # found = False
                # for idx, emb_net in enumerate(self.embedding_net):
                #     if idx == embedding_index and not found:
                #         G = emb_net(S_Rij)
                #         found = True
            else:
                if self.order == 2:
                    G = self.calc_compress(S_Rij, embedding_index)
                elif self.order == 5:
                    G = self.calc_compress_5order(S_Rij, embedding_index)
                elif self.order == 3:
                    G = self.calc_compress_3order(S_Rij, embedding_index)
            # t3 = time.time()
            tmp_a = Ri[:, indices, ntype_1 * self.maxNeighborNum:(ntype_1+1) * self.maxNeighborNum].transpose(-2, -1)

            assert G is not None
            tmp_b = torch.matmul(tmp_a, G)
            xyz_scater_a = tmp_b if xyz_scater_a is None else xyz_scater_a + tmp_b
            # t4 = time.time()
            # print("calculate_Rij:", t2-t1, "\nembedding_net:", t3-t2, "\nconcat:", t4-t3, "\n********************")
        assert xyz_scater_a is not None
        xyz_scater_a = xyz_scater_a / (self.maxNeighborNum * type_nums)
        xyz_scater_b = xyz_scater_a[:, :, :, :self.M2]
        return xyz_scater_a, xyz_scater_b, ntype

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
        # t1 = time.time()          
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(Etot)]
        dE = torch.autograd.grad([Etot], [Ri], grad_outputs=mask, retain_graph=True, create_graph=True)[0]
        assert dE is not None
        # t2 = time.time()
        Ri_d = Ri_d.view(batch_size, natoms_sum, -1, 3)
        dE = dE.view(batch_size, natoms_sum, 1, -1)
        Force = torch.zeros((batch_size, natoms_sum + nghost, 3), device=device, dtype=dtype)
        Force[:, :natoms_sum, :] = -1 * torch.matmul(dE, Ri_d).squeeze(-2)
        # t3 = time.time()
        '''this part for cuda version, not support for torchscript
        list_neigh = torch.unsqueeze(list_neigh,2)
        list_neigh = (list_neigh - 1).type(torch.int)
        F = CalculateForce.apply(list_neigh, dE, Ri_d, F)
        Virial = CalculateVirialForce.apply(list_neigh, dE, ImageDR, Ri_d)
        '''
        # t4 = time.time()
        # 1. Get atom_idx & neighbor_idx
        non_zero_indices = list_neigh.nonzero()
        batch_indices, atom_indices, neighbor_indices = non_zero_indices.unbind(1)
        atom_idx = list_neigh[batch_indices, atom_indices, neighbor_indices] - 1
        # 2. Calculate Force using indexing
        start_indices = neighbor_indices.unsqueeze(1) * 4
        gather_indices = start_indices + torch.arange(4, device=device).unsqueeze(0)
        dE_selected = torch.gather(dE[batch_indices, atom_indices], -1, gather_indices.unsqueeze(1))
        Ri_d_selected = torch.gather(Ri_d[batch_indices, atom_indices], -2, gather_indices.unsqueeze(-1).expand(-1,-1,3))
        dE_dx = torch.matmul(dE_selected, Ri_d_selected).squeeze(-2)
        # t5 = time.time()
        for batch_idx in range(batch_size):
            mask_accumulation = batch_indices == batch_idx
            Force[batch_idx].index_add_(0, atom_idx[mask_accumulation], dE_dx[mask_accumulation]).view(natoms_sum + nghost, 3)
        # t6 = time.time()
        # print("autograd:", t2-t1, "\nmatmul:", t3-t2, "\nindexing:", t5-t4, "\naccumulation:", t6-t5, "\n********************")
        # 3. Calculate Virial
        Virial = torch.zeros((batch_size, 9), device=device, dtype=dtype)
        # virial_components = torch.zeros((len(batch_indices), 6), device=Virial.device, dtype=self.dtype)
        # virial_components[:, 0] = ImageDR[batch_indices, atom_indices, neighbor_indices][:, 1] * dE_dx[:, 0] # xx
        # virial_components[:, 1] = ImageDR[batch_indices, atom_indices, neighbor_indices][:, 1] * dE_dx[:, 1] # xy
        # virial_components[:, 2] = ImageDR[batch_indices, atom_indices, neighbor_indices][:, 1] * dE_dx[:, 2] # xz
        # virial_components[:, 3] = ImageDR[batch_indices, atom_indices, neighbor_indices][:, 2] * dE_dx[:, 1] # yy
        # virial_components[:, 4] = ImageDR[batch_indices, atom_indices, neighbor_indices][:, 2] * dE_dx[:, 2] # yz 
        # virial_components[:, 5] = ImageDR[batch_indices, atom_indices, neighbor_indices][:, 3] * dE_dx[:, 2] # zz 
        # Virial[:, [0, 1, 2, 4, 5, 8]] = virial_components.sum(dim=0)
        # Virial[:, [3, 6, 7]] = Virial[:, [1, 2, 5]]
        return Force, Virial 
       
    '''
    description: 
        F(x) = f2*(F(k+1)-F(k))+F(k)
        f1 = k+1-x
        f2 = 1-f1 = x-k
        
        dG/df2 = F(k+1)-F(k), hear the self.compress_tab is constant
        df2/dx = 1 / (self.dstd[itype]*10**self.dx)
        dx/d_sij = self.dstd[itype]*(10**self.dx)
        df2/d_sij = 1

        return Etot, Ei, F, Egroup, virial  #F is Force
        dG/ds_ij = F(k+1)-F(k) = sum_l(F_l(k+1)-F_l(k)), l = last layer node

    param {*} self
    param {torch} S_Rij
    param {int} embedding_index
    param {int} itype
    return {*}
    author: wuxingxing
    '''    
    def calc_compress(self, S_Rij:torch.Tensor, embedding_index:int):
        sij = S_Rij.flatten()
        out_len = int(self.compress_tab.shape[-1]/2)
        x = (sij-self.sij_min)/self.dx
        index_k1 = x.type(torch.long) # get floor
        index_k2 = index_k1 + 1
        xk = self.sij_min + index_k1*self.dx
        f2 = (sij - xk).flatten().unsqueeze(-1)
        # f2 = ((x-index_k1)/(self.dstd[itype]*10**self.dx)).unsqueeze(-1)
        # f2 = ((((x - index_k1)/(10**self.dx)))/self.dstd[itype]).unsqueeze(-1)
        # f2 = (S_Rij.flatten() - ((index_k1*(1/10**self.dx)-self.davg[itype])/self.dstd[itype])).unsqueeze(-1)
        # G = f2*(self.compress_tab[embedding_index][index_k2.flatten()]-self.compress_tab[embedding_index][index_k1.flatten()]) + self.compress_tab[embedding_index][index_k1.flatten()]
        deriv_sij = (self.compress_tab[embedding_index][index_k2][:, out_len:] + self.compress_tab[embedding_index][index_k1][:, out_len:])/2
        G = self.compress_tab[embedding_index][index_k1][:, :out_len] + f2*deriv_sij
        G = G.reshape(S_Rij.shape[0], S_Rij.shape[1], S_Rij.shape[2], G.shape[1])
        return G
    
    def calc_compress_5order(self, S_Rij:torch.Tensor, embedding_index:int):
        sij = S_Rij.flatten()

        x = (sij-self.sij_min)/self.dx
        index_k1 = x.type(torch.long) # get floor
        xk = self.sij_min + index_k1*self.dx
        f2 = (sij - xk).flatten().unsqueeze(-1)

        coefficient = self.compress_tab[embedding_index, index_k1, :]

        # G = CalculateCompress.apply(f2, coefficient)

        G = f2**5 *coefficient[:, :, 0] + f2**4 * coefficient[:, :, 1] + \
            f2**3 * coefficient[:, :, 2] + f2**2 * coefficient[:, :, 3] + \
            f2 * coefficient[:, :, 4] + coefficient[:, :, 5]
        
        G = G.reshape(S_Rij.shape[0], S_Rij.shape[1], S_Rij.shape[2], G.shape[1])
        return G

    def calc_compress_3order(self, S_Rij:torch.Tensor, embedding_index:int):
        sij = S_Rij.flatten()
        mask = sij < self.sij_max
        print("self.sij_max", self.sij_max)
        x = torch.zeros_like(sij)
        x[mask] = (sij[mask]-self.sij_min)/self.dx
        x[~mask] = (sij[~mask]-self.sij_max)/(10*self.dx)
        index_k1 = x.type(torch.long) # get floor

        xk = torch.zeros_like(sij, dtype=torch.float32) # the index * dx + sij_min is a float type data
        xk[mask] = index_k1[mask]*self.dx + self.sij_min
        xk[~mask] = self.sij_max + index_k1[~mask]*self.dx*10
        f2 = (sij - xk).flatten().unsqueeze(-1)
        print("max index_k1", index_k1.max())
        print("self.compress_tab.shape", self.compress_tab.shape)
        coefficient = self.compress_tab[embedding_index, index_k1, :]
        # G = CalculateCompress.apply(f2, coefficient)
        G = f2**3 *coefficient[:, :, 0] + f2**2 * coefficient[:, :, 1] + \
            f2 * coefficient[:, :, 2] + coefficient[:, :, 3]
        G = G.reshape(S_Rij.shape[0], S_Rij.shape[1], S_Rij.shape[2], G.shape[1])
        return G
    