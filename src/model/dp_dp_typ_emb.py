import numpy as np
import torch
from torch import embedding
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import sys, os
sys.path.append(os.getcwd())
# import parameters as pm    
# import prepare as pp
# pp.readFeatnum()
from src.model.dp_embedding_typ_emb import EmbeddingNet, FittingNet
from src.model.calculate_force import CalculateForce, CalculateVirialForce
from utils.atom_type_emb_dict import get_normalized_data_list

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

class TypeDP(nn.Module):
    def __init__(self, config, device, stat, magic=False):
        super(TypeDP, self).__init__()
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
        # initial type embedding net
        if len(self.config["net_cfg"]["type_embedding_net"]["network_size"]) > 0:
            self.embedding_net.append(EmbeddingNet(self.config["net_cfg"]["type_embedding_net"], type_feat_num=None, is_type_emb=True))
            type_feat_num = self.config["net_cfg"]["type_embedding_net"]["network_size"][-1]
        else:
            type_feat_num = len(self.config["net_cfg"]["type_embedding_net"]["physical_property"])
        # initial embedding net
        self.embedding_net.append(EmbeddingNet(self.config["net_cfg"]["embedding_net"], 
                                               type_feat_num= type_feat_num,# if type_emb_net exists, type_feat_num is last layer of net work, otherwise, is type num of physical_property
                                               is_type_emb=False))
        # initial fitting net
        fitting_net_input_dim = self.config["net_cfg"]["embedding_net"]["network_size"][-1]
        for i in range(self.ntypes):
            self.fitting_net.append(FittingNet(config["net_cfg"]["fitting_net"], self.M2 * fitting_net_input_dim, self.stat[2][i], magic))

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
    def forward(self, Ri, dfeat, list_neigh, natoms_img, atom_type, ImageDR, Egroup_weight = None, divider = None, is_calc_f=None):

        #torch.autograd.set_detect_anomaly(True)
        Ri_d = dfeat
        # dim of natoms_img: batch size, natom_sum & natom_types ([9, 6, 2, 1])
        natoms = natoms_img[0, 1:]
        natoms_sum = Ri.shape[1]
        batch_size = Ri.shape[0]
        atom_sum = 0
        atom_type_cpu = list(np.array(atom_type.cpu())[0])
        emb_list, type_nums =  self.get_train_2body_type(atom_type_cpu)
        # get type_embedding_vector
        physical_property = self.config["net_cfg"]["type_embedding_net"]["physical_property"]
        type_vector = get_normalized_data_list(atom_type_cpu, physical_property)
        Ei = None
        for type_emb in emb_list:
            S_Rij = None
            tmp_a = None
            type_emb_feat = None
            for emb in type_emb:
                ntype, ntype_1 = emb # Compatible with hybrid training
                #        Ri[images, atom_type_list            ,  neighbor_list_of_different_atom_type,  SRij_value]
                S_Rij_ = Ri[:, atom_sum:atom_sum+natoms[ntype], ntype_1 * self.maxNeighborNum:(ntype_1+1) * self.maxNeighborNum, 0].unsqueeze(-1)
                S_Rij = S_Rij_ if S_Rij is None else torch.concat((S_Rij, S_Rij_), dim=2)

                tmp_a_ = Ri[:, atom_sum:atom_sum+natoms[ntype], ntype_1 * self.maxNeighborNum:(ntype_1+1) * self.maxNeighborNum]
                tmp_a = tmp_a_ if tmp_a is None else torch.concat((tmp_a, tmp_a_), dim=2)

                type_emb_ = torch.tensor(type_vector[atom_type_cpu[ntype_1]], dtype=self.dtype, device=self.device).repeat(self.maxNeighborNum,1)
                type_emb_feat = type_emb_ if type_emb_feat is None else torch.concat((type_emb_feat, type_emb_), dim=0)
            tmp_a = tmp_a.transpose(-2, -1)
            # For each neighbor of the central atom 'ntype', obtain the type code by passing it through the type embedding net
            
            if len(self.embedding_net) > 1:
                type_emb_encoded = self.embedding_net[0](type_emb_feat)
                S_Rij_type = torch.concat((S_Rij, type_emb_encoded.unsqueeze(0).unsqueeze(0).expand(S_Rij.shape[0], S_Rij.shape[1], -1, -1)), dim=3)
            else:
                S_Rij_type = torch.concat((S_Rij, type_emb_feat.unsqueeze(0).unsqueeze(0).expand(S_Rij.shape[0], S_Rij.shape[1], -1, -1)), dim=3)
            G = self.embedding_net[-1](S_Rij_type) #[4, 60, 200, 25] li-si S_Rij_type
            xyz_scater_a = torch.matmul(tmp_a, G)
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
        dE = torch.autograd.grad(Ei, Ri, grad_outputs=mask, retain_graph=True, create_graph=True)
        dE = torch.stack(list(dE), dim=0).squeeze(0)  #[:,:,:,:-1] #[2,108,100,4]-->[2,108,100,3]

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
            
        