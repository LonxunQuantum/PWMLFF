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
                self.embedding_net.append(EmbeddingNet(self.config["net_cfg"]["embedding_net"], magic))
            fitting_net_input_dim = self.config["net_cfg"]["embedding_net"]["network_size"][-1]
            self.fitting_net.append(FittingNet(config["net_cfg"]["fitting_net"], self.M2 * fitting_net_input_dim, self.stat[2][i], magic))

    def get_egroup(self, Ei, Egroup_weight, divider):
        batch_size = Ei.shape[0]
        Egroup = torch.zeros_like(Ei)

        for i in range(batch_size):
            Etot1 = Ei[i]
            weight_inner = Egroup_weight[i]
            E_inner = torch.matmul(weight_inner, Etot1)
            Egroup[i] = E_inner
        Egroup_out = torch.divide(Egroup.squeeze(-1), divider)
        
        return Egroup_out

    def forward(self, ImageDR, Ri, dfeat, list_neigh, natoms_img, Egroup_weight, divider, is_calc_f=None):

        torch.autograd.set_detect_anomaly(True)
        
        Ri_d = dfeat
        natoms = natoms_img[0, 1:]
        natoms_sum = Ri.shape[1]
        batch_size = Ri.shape[0]
        atom_sum = 0

        for ntype in range(self.ntypes):
            for ntype_1 in range(self.ntypes):
                S_Rij = Ri[:, atom_sum:atom_sum+natoms[ntype], ntype_1 * self.maxNeighborNum:(ntype_1+1) * self.maxNeighborNum, 0].unsqueeze(-1)
                
                # determines which embedding net
                embedding_index = ntype * self.ntypes + ntype_1
                
                # itermediate output of embedding net 
                # dim of G: batch size, natom of ntype, max_neigh_num, final layer dim
                #  
                G = self.embedding_net[embedding_index](S_Rij) 
                #if ntype == 0 and ntype_1==0:
                #print (ntype, ntype_1 )
                #print ("dim of G")
                #print (G.size())
                #if ntype == 0 and ntype_1==0: 
                #    print(G[0,0,1,:])
                # symmetry conserving 
                tmp_a = Ri[:, atom_sum:atom_sum+natoms[ntype], ntype_1 * self.maxNeighborNum:(ntype_1+1) * self.maxNeighborNum].transpose(-2, -1)
                tmp_b = torch.matmul(tmp_a, G)
                
                if ntype_1 == 0:
                    xyz_scater_a = tmp_b
                else:
                    xyz_scater_a = xyz_scater_a + tmp_b
            xyz_scater_a = xyz_scater_a * 4.0 / (self.maxNeighborNum * self.ntypes * 4)
            xyz_scater_b = xyz_scater_a[:, :, :, :self.M2]
            DR_ntype = torch.matmul(xyz_scater_a.transpose(-2, -1), xyz_scater_b)
            DR_ntype = DR_ntype.reshape(batch_size, natoms[ntype], -1)
            
            if ntype == 0:
                DR = DR_ntype
            else:
                DR = torch.concat((DR, DR_ntype), dim=1)
            # dim of DR_ntype: bs, natom in type, 400
            #print ("input for fitting net of type:",ntype)
            #print ("DR_ntype[0,:,0]")
            #print (DR_ntype[0,0,:50])
            #print(ntype, DR_ntype.size())

            Ei_ntype = self.fitting_net[ntype](DR_ntype)
            
            #print ("type:",ntype)
            #print (Ei_ntype)

            if ntype == 0:
                Ei = Ei_ntype
            else:
                Ei = torch.concat((Ei, Ei_ntype), dim=1)
            
            
            atom_sum = atom_sum + natoms[ntype]
        
        Etot = torch.sum(Ei, 1)   
        Egroup = self.get_egroup(Ei, Egroup_weight, divider)
        #Egroup = 0 
        F = torch.zeros((batch_size, atom_sum, 3), device=self.device)
        Virial = torch.zeros((batch_size, 9), device=self.device)
        Ei = torch.squeeze(Ei, 2)

        if is_calc_f == False:
            return Etot, Ei, F, Egroup, Virial
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

        return Etot, Ei, F, Egroup, virial
        