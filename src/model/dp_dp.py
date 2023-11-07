import sys, os
import time
import numpy as np
import torch
from torch import embedding
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
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
    def __init__(self, config, device, energy_shift, magic=False):
        super(DP, self).__init__()
        self.config = config
        self.ntypes = len(config['atomType'])
        self.atom_type = [_['type'] for _ in config['atomType']] #this value in used in forward for hybrid Training
        self.device = device
        self.energy_shift = energy_shift
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
                self.embedding_net.append(EmbeddingNet(self.config["net_cfg"]["embedding_net"], magic))
            fitting_net_input_dim = self.config["net_cfg"]["embedding_net"]["network_size"][-1]
            self.fitting_net.append(FittingNet(config["net_cfg"]["fitting_net"], self.M2 * fitting_net_input_dim, energy_shift[i], magic))
        
        self.compress_tab = None #for dp compress

    def set_comp_tab(self, compress_dict:dict):
        self.compress_tab = torch.tensor(compress_dict["table"], dtype=self.dtype, device=self.device)
        self.dx = compress_dict["dx"]
        self.davg = compress_dict["davg"]
        self.dstd = compress_dict["dstd"]
        self.sij_min = compress_dict["sij_min"]
        self.sij_max = compress_dict["sij_max"]
        self.sij_out_max = compress_dict["sij_out_max"]
        self.sij_len = compress_dict["sij_len"]
        self.sij_out_len = compress_dict["sij_out_len"]
        self.order = compress_dict["order"] if "order" in compress_dict.keys() else 5 #the default compress order is 5

    def get_egroup(self, Ei, Egroup_weight, divider):
        # commit by wuxing and replace by the under line code
        # batch_size = Ei.shape[0]
        # Egroup = torch.zeros_like(Ei)

        # for i in range(batch_size):
        #     Etot1 = Ei[i]
        #     weight_inner = Egroup_weight[i]
        #     E_inner = torch.matmul(weight_inner, Etot1)
        #     Egroup[i] = E_inner
        Egroup = torch.matmul(Egroup_weight, Ei)
        Egroup_out = torch.divide(Egroup.squeeze(-1), divider)
        
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
    def get_train_2body_type(self, atom_type_data):
        type_2body_list = []
        type_2body_index = []
        for _ in atom_type_data:
            if _ != 0:
                type_2body_index.append(self.atom_type.index(_))

        for atom in type_2body_index:
            type_2body = []
            for atom2 in type_2body_index:
                type_2body.append([atom, atom2])
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
    def forward(self, Ri, dfeat, list_neigh, natoms_img, atom_type, ImageDR, Egroup_weight = None, divider = None, is_calc_f=True):
        #torch.autograd.set_detect_anomaly(True)
        # from torchviz import make_dot
        Ri_d = dfeat
        # dim of natoms_img: batch size, natom_sum & natom_types ([9, 6, 2, 1])
        natoms = natoms_img[0, 1:]
        natoms_sum = Ri.shape[1]
        batch_size = Ri.shape[0]
        atom_sum = 0
        emb_list, type_nums =  self.get_train_2body_type(list(np.array(atom_type.cpu())[0]))
        Ei = None
        # start_forward = time.time()
        for type_emb in emb_list:
            xyz_scater_a = None
            for emb in type_emb:
                ntype, ntype_1 = emb
                # print(ntype, "\t\t", ntype_1)
                # dim of Ri: batch size, natom_sum, ntype*max_neigh_num, local environment matrix , ([10,9,300,4])
                S_Rij = Ri[:, atom_sum:atom_sum+natoms[ntype], ntype_1 * self.maxNeighborNum:(ntype_1+1) * self.maxNeighborNum, 0].unsqueeze(-1)
                # determines which embedding net
                embedding_index = ntype * self.ntypes + ntype_1
                # itermediate output of embedding net 
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
                xyz_scater_a = tmp_b if xyz_scater_a is None else xyz_scater_a + tmp_b
            
            # attention: for hybrid training, the division should be done based on \
            #   the number of element types in the current image, because the images may from different systems.
            xyz_scater_a = xyz_scater_a / (self.maxNeighborNum * type_nums)
            xyz_scater_b = xyz_scater_a[:, :, :, :self.M2]
            DR_ntype = torch.matmul(xyz_scater_a.transpose(-2, -1), xyz_scater_b)
            DR_ntype = DR_ntype.reshape(batch_size, natoms[ntype], -1)

            Ei_ntype = self.fitting_net[ntype](DR_ntype)
            Ei = Ei_ntype if Ei is None else torch.concat((Ei, Ei_ntype), dim=1)
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
        
        mask = torch.ones_like(Ei)
        # start_autograd = time.time()
        dE = torch.autograd.grad(Ei, Ri, grad_outputs=mask, retain_graph=True, create_graph=True)
        # end_autograd = time.time()
        # print("autograd time:", end_autograd - start_autograd, 's')
        # dot = make_dot(Ei, params={"x":Ri})
        # dot.render("compute_graph.png", format="png")
        dE = torch.stack(list(dE), dim=0).squeeze(0)  #[:,:,:,:-1] #[2,108,100,4]-->[2,108,100,3]

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
                      
    '''
    description: 
        F(x) = f2*(F(k+1)-F(k))+F(k)
        f1 = k+1-x
        f2 = 1-f1 = x-k
        
        dG/df2 = F(k+1)-F(k), hear the self.compress_tab is constant
        df2/dx = 1 / (self.dstd[itype]*10**self.dx)
        dx/d_sij = self.dstd[itype]*(10**self.dx)
        df2/d_sij = 1

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

        G = CalculateCompress.apply(f2, coefficient)

        # G = f2**5 *coefficient[:, :, 0] + f2**4 * coefficient[:, :, 1] + \
        #     f2**3 * coefficient[:, :, 2] + f2**2 * coefficient[:, :, 3] + \
        #     f2 * coefficient[:, :, 4] + coefficient[:, :, 5]
        
        G = G.reshape(S_Rij.shape[0], S_Rij.shape[1], S_Rij.shape[2], G.shape[1])
        return G

    def calc_compress_3order(self, S_Rij:torch.Tensor, embedding_index:int):
        sij = S_Rij.flatten()
        mask = sij < self.sij_max

        x = torch.zeros_like(sij)
        x[mask] = (sij[mask]-self.sij_min)/self.dx
        x[~mask] = (sij[~mask]-self.sij_max)/(10*self.dx)
        index_k1 = x.type(torch.long) # get floor

        xk = torch.zeros_like(sij, dtype=torch.float32) # the index * dx + sij_min is a float type data
        xk[mask] = index_k1[mask]*self.dx + self.sij_min
        xk[~mask] = self.sij_max + index_k1[~mask]*self.dx*10
        f2 = (sij - xk).flatten().unsqueeze(-1)
        
        coefficient = self.compress_tab[embedding_index, index_k1, :]
        G = CalculateCompress.apply(f2, coefficient)
        # G = f2**3 *coefficient[:, :, 0] + f2**2 * coefficient[:, :, 1] + \
        #     f2 * coefficient[:, :, 2] + coefficient[:, :, 3]
        G = G.reshape(S_Rij.shape[0], S_Rij.shape[1], S_Rij.shape[2], G.shape[1])
        return G


    