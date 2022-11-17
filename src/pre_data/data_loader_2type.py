#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
codepath = os.path.abspath(sys.path[0])
sys.path.append(codepath+'/../lib')
sys.path.append(os.getcwd())
import prepare as pp
#import parameters as pm
import default_para as pm 
"""
import use_para as pm
import parse_input
parse_input.parse_input()
"""

class MovementDataset(Dataset):

    def __init__(self, feat_path, dfeat_path,
                 egroup_path, egroup_weight_path, divider_path,
                 itype_path, nblist_path, weight_all_path,
                 energy_path, force_path, ind_img_path, natoms_img_path,
                 dR_neigh_path=None, Ri_path=None, Ri_d_path=None):  # , natoms_path


        """
            pm.is_dfeat_sparse is True, self.dfeat will not be generated
        """

        super(MovementDataset, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.feat = np.load(feat_path)

        if pm.is_dfeat_sparse==False:
            self.dfeat = np.load(dfeat_path)
        
        self.egroup = np.load(egroup_path)
        self.egroup_weight = np.load(egroup_weight_path)
        self.divider = np.load(divider_path)

        # self.natoms_sum = natoms
        # self.natoms = pd.read_csv(natoms_path)   #/fread_dfeat/NN_output/natoms_train.csv
        self.itype = np.load(itype_path)
        self.nblist = np.load(nblist_path)
        self.weight_all = np.load(weight_all_path)
        self.ind_img = np.load(ind_img_path)

        self.energy = np.load(energy_path)
        self.force = np.load(force_path)
        self.use_dR_neigh = False
        
        self.ntypes = pm.ntypes
        self.natoms_img = np.load(natoms_img_path)
        
        if dR_neigh_path:
            self.use_dR_neigh = True
            tmp = np.load(dR_neigh_path)
            self.dR = tmp[:, :, :, :3]
            self.dR_neigh_list = np.squeeze(tmp[:, :, :, 3:], axis=-1).astype(int)
            self.force = -1 * self.force

            if (not os.path.exists(Ri_path)) or (not os.path.exists(Ri_d_path)):
                self.get_stat()
                self.prepare(Ri_path, Ri_d_path)

            self.Ri_all = np.load(Ri_path) #(12, 108, 100, 4)
            self.Ri_d_all = np.load(Ri_d_path)
        
    # for dR
    def prepare(self, Ri_path, Ri_d_path):
        image_dR = self.dR
        list_neigh = self.dR_neigh_list

        natoms_sum = self.natoms_img[0, 0]
        natoms_per_type = self.natoms_img[0, 1:]

        image_dR = np.reshape(image_dR, (-1, natoms_sum, self.ntypes * pm.maxNeighborNum, 3))
        list_neigh = np.reshape(list_neigh, (-1, natoms_sum, self.ntypes * pm.maxNeighborNum))

        image_dR = torch.tensor(image_dR, device=self.device, dtype=torch.double)
        list_neigh = torch.tensor(list_neigh, device=self.device, dtype=torch.int)

        # deepmd neighbor id 从 0 开始，MLFF从1开始
        mask = list_neigh > 0

        dR2 = torch.zeros_like(list_neigh, dtype=torch.double)
        Rij = torch.zeros_like(list_neigh, dtype=torch.double)
        dR2[mask] = torch.sum(image_dR[mask] * image_dR[mask], -1)
        Rij[mask] = torch.sqrt(dR2[mask])

        nr = torch.zeros_like(dR2)
        inr = torch.zeros_like(dR2)
        
        dR2_copy = dR2.unsqueeze(-1).repeat(1, 1, 1, 3)
        Ri_xyz = torch.zeros_like(dR2_copy)

        nr[mask] = dR2[mask] / Rij[mask]
        Ri_xyz[mask] = image_dR[mask] / dR2_copy[mask]
        inr[mask] = 1 / Rij[mask]

        davg = torch.tensor(self.davg, device=self.device, dtype=torch.float64)
        dstd = torch.tensor(self.dstd, device=self.device, dtype=torch.float64)

        Ri, Ri_d = self.__smooth(image_dR, nr, Ri_xyz, mask, inr, davg, dstd, natoms_per_type)

        Ri = Ri.detach().cpu().numpy()
        Ri_d = Ri_d.detach().cpu().numpy()

        np.save(Ri_path, Ri)
        np.save(Ri_d_path, Ri_d)


    def __getitem__(self, index):
        # index = index + 10
        ind_image = np.zeros(2)
        ind_image[0] = self.ind_img[index]
        ind_image[1] = self.ind_img[index+1]

        dic= {} 

        if pm.is_dfeat_sparse==False:

            dic = {
                'input_feat': self.feat[self.ind_img[index]:self.ind_img[index+1]],

                'input_dfeat': self.dfeat[self.ind_img[index]:self.ind_img[index+1]],
                
                'input_egroup': self.egroup[self.ind_img[index]:self.ind_img[index+1]],
                'input_egroup_weight': self.egroup_weight[self.ind_img[index]:self.ind_img[index+1]],
                'input_divider': self.divider[self.ind_img[index]:self.ind_img[index+1]],
                'input_itype': self.itype[self.ind_img[index]:self.ind_img[index+1]],
                'input_nblist': self.nblist[self.ind_img[index]:self.ind_img[index+1]],
                'input_weight_all': self.weight_all[self.ind_img[index]:self.ind_img[index+1]],

                'output_energy': self.energy[self.ind_img[index]:self.ind_img[index+1]],
                'output_force': self.force[self.ind_img[index]:self.ind_img[index+1]],
                'ind_image': ind_image,
                'natoms_img': self.natoms_img[index]
            }

        else:
            # not __getitem__ functionality for dfeat 
            dic = {
                'input_feat': self.feat[self.ind_img[index]:self.ind_img[index+1]],

                # place holder 
                'input_dfeat': [], 

                'input_egroup': self.egroup[self.ind_img[index]:self.ind_img[index+1]],
                'input_egroup_weight': self.egroup_weight[self.ind_img[index]:self.ind_img[index+1]],
                'input_divider': self.divider[self.ind_img[index]:self.ind_img[index+1]],
                'input_itype': self.itype[self.ind_img[index]:self.ind_img[index+1]],
                'input_nblist': self.nblist[self.ind_img[index]:self.ind_img[index+1]],
                'input_weight_all': self.weight_all[self.ind_img[index]:self.ind_img[index+1]],

                'output_energy': self.energy[self.ind_img[index]:self.ind_img[index+1]],
                'output_force': self.force[self.ind_img[index]:self.ind_img[index+1]],
                'ind_image': ind_image,
                'natoms_img': self.natoms_img[index]
            }   

            
        if self.use_dR_neigh:
            dic['input_dR'] = self.dR[self.ind_img[index]:self.ind_img[index+1]]
            dic['input_dR_neigh_list'] = self.dR_neigh_list[self.ind_img[index]:self.ind_img[index+1]]
            dic['input_Ri'] = self.Ri_all[index]  
            dic['input_Ri_d'] = self.Ri_d_all[index]
        
        return dic
    
    def __len__(self):
        image_number = self.ind_img.shape[0] - 1
        return image_number

    def __compute_std(self, sum2, sum, sumn):
        if sumn == 0:
            return 1e-2
        val = np.sqrt(sum2/sumn - np.multiply(sum/sumn, sum/sumn))
        if np.abs(val) < 1e-2:
            val = 1e-2
        return val
    
    def __smooth(self, image_dR, x, Ri_xyz, mask, inr, davg, dstd, natoms):

        inr2 = torch.zeros_like(inr)
        inr3 = torch.zeros_like(inr)
        inr4 = torch.zeros_like(inr)
        
        inr2[mask] = inr[mask] * inr[mask]
        inr4[mask] = inr2[mask] * inr2[mask]
        inr3[mask] = inr4[mask] * x[mask]
        
        uu = torch.zeros_like(x)
        vv = torch.zeros_like(x)
        dvv = torch.zeros_like(x)
        
        res = torch.zeros_like(x)

        # x < rcut_min vv = 1
        mask_min = x < 5.8   #set rcut=25, 10  min=0,max=30
        mask_1 = mask & mask_min  #[2,108,100]
        vv[mask_1] = 1
        dvv[mask_1] = 0

        # rcut_min< x < rcut_max
        mask_max = x < 6.0
        mask_2 = ~mask_min & mask_max & mask
        # uu = (xx - rmin) / (rmax - rmin) ;
        uu[mask_2] = (x[mask_2] - 5.8)/(6.0 -5.8)
        vv[mask_2] = uu[mask_2] * uu[mask_2] * uu[mask_2] * (-6 * uu[mask_2] * uu[mask_2] + 15 * uu[mask_2] - 10) + 1
        du = 1.0 / ( 6.0 - 5.8)
        # dd = ( 3 * uu*uu * (-6 * uu*uu + 15 * uu - 10) + uu*uu*uu * (-12 * uu + 15) ) * du;
        dvv[mask_2] = (3 * uu[mask_2] * uu[mask_2] * (-6 * uu[mask_2] * uu[mask_2] + 15 * uu[mask_2] -10) + uu[mask_2] * uu[mask_2] * uu[mask_2] * (-12 * uu[mask_2] + 15)) * du
 
        mask_3 = ~mask_max & mask
        vv[mask_3] = 0
        dvv[mask_3] = 0

        res[mask] = 1.0 / x[mask]
        Ri = torch.cat((res.unsqueeze(-1), Ri_xyz), dim=-1)
        Ri_d = torch.zeros_like(Ri).unsqueeze(-1).repeat(1, 1, 1, 1, 3) # 2 108 100 4 3
        tmp = torch.zeros_like(x)

        # deriv of component 1/r
        tmp[mask] = image_dR[:, :, :, 0][mask] * inr3[mask] * vv[mask] - Ri[:, :, :, 0][mask] * dvv[mask] * image_dR[:, :, :, 0][mask] * inr[mask]
        Ri_d[:, :, :, 0, 0][mask] = tmp[mask]
        tmp[mask] = image_dR[:, :, :, 1][mask] * inr3[mask] * vv[mask] - Ri[:, :, :, 0][mask] * dvv[mask] * image_dR[:, :, :, 1][mask] * inr[mask]
        Ri_d[:, :, :, 0, 1][mask] = tmp[mask]
        tmp[mask] = image_dR[:, :, :, 2][mask] * inr3[mask] * vv[mask] - Ri[:, :, :, 0][mask] * dvv[mask] * image_dR[:, :, :, 2][mask] * inr[mask]
        Ri_d[:, :, :, 0, 2][mask] = tmp[mask]

        # deriv of component x/r
        tmp[mask] = (2 * image_dR[:, :, :, 0][mask] * image_dR[:, :, :, 0][mask] * inr4[mask] - inr2[mask]) * vv[mask] - Ri[:, :, :, 1][mask] * dvv[mask] * image_dR[:, :, :, 0][mask] * inr[mask]
        Ri_d[:, :, :, 1, 0][mask] = tmp[mask]
        tmp[mask] = (2 * image_dR[:, :, :, 0][mask] * image_dR[:, :, :, 1][mask] * inr4[mask]) * vv[mask] - Ri[:, :, :, 1][mask] * dvv[mask] * image_dR[:, :, :, 1][mask] * inr[mask]
        Ri_d[:, :, :, 1, 1][mask] = tmp[mask]
        tmp[mask] = (2 * image_dR[:, :, :, 0][mask] * image_dR[:, :, :, 2][mask] * inr4[mask]) * vv[mask] - Ri[:, :, :, 1][mask] * dvv[mask] * image_dR[:, :, :, 2][mask] * inr[mask]
        Ri_d[:, :, :, 1, 2][mask] = tmp[mask]
       
        # deriv of component y/r
        tmp[mask] = (2 * image_dR[:, :, :, 1][mask] * image_dR[:, :, :, 0][mask] * inr4[mask]) * vv[mask] - Ri[:, :, :, 2][mask] * dvv[mask] * image_dR[:, :, :, 0][mask] * inr[mask]
        Ri_d[:, :, :, 2, 0][mask] = tmp[mask]
        tmp[mask] = (2 * image_dR[:, :, :, 1][mask] * image_dR[:, :, :, 1][mask] * inr4[mask] - inr2[mask]) * vv[mask] - Ri[:, :, :, 2][mask] * dvv[mask] * image_dR[:, :, :, 1][mask] * inr[mask]
        Ri_d[:, :, :, 2, 1][mask] = tmp[mask]
        tmp[mask] = (2 * image_dR[:, :, :, 1][mask] * image_dR[:, :, :, 2][mask] * inr4[mask]) * vv[mask] - Ri[:, :, :, 2][mask] * dvv[mask] * image_dR[:, :, :, 2][mask] * inr[mask]
        Ri_d[:, :, :, 2, 2][mask] = tmp[mask]
    
        # deriv of component z/r
        tmp[mask] = (2 * image_dR[:, :, :, 2][mask] * image_dR[:, :, :, 0][mask] * inr4[mask]) * vv[mask] - Ri[:, :, :, 3][mask] * dvv[mask] * image_dR[:, :, :, 0][mask] * inr[mask]
        Ri_d[:, :, :, 3, 0][mask] = tmp[mask]
        tmp[mask] = (2 * image_dR[:, :, :, 2][mask] * image_dR[:, :, :, 1][mask] * inr4[mask]) * vv[mask] - Ri[:, :, :, 3][mask] * dvv[mask] * image_dR[:, :, :, 1][mask] * inr[mask]
        Ri_d[:, :, :, 3, 1][mask] = tmp[mask]
        tmp[mask] = (2 * image_dR[:, :, :, 2][mask] * image_dR[:, :, :, 2][mask] * inr4[mask] - inr2[mask]) * vv[mask] - Ri[:, :, :, 3][mask] * dvv[mask] * image_dR[:, :, :, 2][mask] * inr[mask]
        Ri_d[:, :, :, 3, 2][mask] = tmp[mask]

        vv_copy = vv.unsqueeze(-1).repeat(1, 1, 1, 4)
        Ri[mask] *= vv_copy[mask]

        for ntype in range(self.ntypes):
            atom_num_ntype = natoms[ntype]
            davg_ntype = davg[ntype].reshape(-1, 4).squeeze().repeat(atom_num_ntype, 1, 1) #[32,100,4]
            dstd_ntype = dstd[ntype].reshape(-1, 4).squeeze().repeat(atom_num_ntype, 1, 1) #[32,100,4]
            if ntype == 0:
                davg_res = davg_ntype
                dstd_res = dstd_ntype
            else:
                davg_res = torch.concat((davg_res, davg_ntype), dim=0)
                dstd_res = torch.concat((dstd_res, dstd_ntype), dim=0)
        Ri = (Ri - davg_res) / dstd_res  #[1,64,200,4]
        dstd_res = dstd_res.unsqueeze(-1).repeat(1, 1, 1, 3)
        Ri_d = Ri_d / dstd_res 
        
        # res[mask_2] = 0.5 * torch.cos(math.pi * (x[mask_2]-10)/(25-10)) + 0.5 * torch.ones_like(x[mask_2])
        return Ri, Ri_d

    def __compute_stat(self, image_num=10):

        self.davg = []
        self.dstd = []
        # self.natoms = sum(self.natoms)

        # only for one atom type
        if image_num > self.__len__():
            image_num = self.__len__()
        image_dR = self.dR[self.ind_img[0]:self.ind_img[image_num]]
        list_neigh = self.dR_neigh_list[self.ind_img[0]:self.ind_img[image_num]]

        natoms_sum = self.natoms_img[0, 0]
        natoms_per_type = self.natoms_img[0, 1:]

        image_dR = np.reshape(image_dR, (-1, natoms_sum, self.ntypes * pm.maxNeighborNum, 3))
        list_neigh = np.reshape(list_neigh, (-1, natoms_sum, self.ntypes * pm.maxNeighborNum))

        image_dR = torch.tensor(image_dR, device=self.device, dtype=torch.double)
        list_neigh = torch.tensor(list_neigh, device=self.device, dtype=torch.int)

        # deepmd neighbor id 从 0 开始，MLFF从1开始
        mask = list_neigh > 0

        dR2 = torch.zeros_like(list_neigh, dtype=torch.double)
        Rij = torch.zeros_like(list_neigh, dtype=torch.double)
        dR2[mask] = torch.sum(image_dR[mask] * image_dR[mask], -1)
        Rij[mask] = torch.sqrt(dR2[mask])

        nr = torch.zeros_like(dR2)
        inr = torch.zeros_like(dR2)

        dR2_copy = dR2.unsqueeze(-1).repeat(1, 1, 1, 3)
        Ri_xyz = torch.zeros_like(dR2_copy)

        nr[mask] = dR2[mask] / Rij[mask]
        Ri_xyz[mask] = image_dR[mask] / dR2_copy[mask]
        inr[mask] = 1 / Rij[mask]

        davg = torch.zeros((pm.maxNeighborNum * self.ntypes, 4), dtype=torch.float64, device=self.device)
        dstd = torch.ones((pm.maxNeighborNum * self.ntypes, 4), dtype=torch.float64, device=self.device)
        Ri, _ = self.__smooth(image_dR, nr, Ri_xyz, mask, inr, davg, dstd, natoms_per_type)
        Ri2 = Ri * Ri

        atom_sum = 0

        for i in range(self.ntypes):
            Ri_ntype = Ri[:, atom_sum:atom_sum+natoms_per_type[i]].reshape(-1, 4)
            Ri2_ntype = Ri2[:, atom_sum:atom_sum+natoms_per_type[i]].reshape(-1, 4)
            sum_Ri = Ri_ntype.sum(axis=0).tolist()
            sum_Ri_r = sum_Ri[0]
            sum_Ri_a = np.average(sum_Ri[1:])
            sum_Ri2 = Ri2_ntype.sum(axis=0).tolist()
            sum_Ri2_r = sum_Ri2[0]
            sum_Ri2_a = np.average(sum_Ri2[1:])
            sum_n = Ri_ntype.shape[0]


            davg_unit = [sum_Ri[0] / (sum_n + 1e-15), 0, 0, 0]
            dstd_unit = [
                self.__compute_std(sum_Ri2_r, sum_Ri_r, sum_n),
                self.__compute_std(sum_Ri2_a, sum_Ri_a, sum_n),
                self.__compute_std(sum_Ri2_a, sum_Ri_a, sum_n),
                self.__compute_std(sum_Ri2_a, sum_Ri_a, sum_n)
            ]

            self.davg.append(np.tile(davg_unit, pm.maxNeighborNum * self.ntypes).reshape(-1, 4))
            self.dstd.append(np.tile(dstd_unit, pm.maxNeighborNum * self.ntypes).reshape(-1, 4))
            atom_sum = atom_sum + natoms_per_type[i]
        
        self.davg = np.array(self.davg).reshape(self.ntypes, -1)
        self.dstd = np.array(self.dstd).reshape(self.ntypes, -1)      
        # import ipdb;ipdb.set_trace()  


    def __compute_stat_output(self, image_num=10,  rcond=1e-3):
        self.ener_shift = []
        natoms_sum = self.natoms_img[0, 0]
        natoms_per_type = self.natoms_img[0, 1:]
        # only for one atom type
        if image_num > self.__len__():
            image_num = self.__len__()
        energy = self.energy[self.ind_img[0]:self.ind_img[image_num]]
        energy = np.reshape(energy, (-1, natoms_sum, 1))
        # natoms_sum = 0
        # for ntype in range(self.ntypes):
        #     energy_ntype = energy[:, natoms_sum:natoms_sum+natoms_per_type[ntype]]
        #     natoms_sum += natoms_per_type[ntype]
        #     energy_sum = energy_ntype.sum(axis=1)
        #     energy_one = np.ones_like(energy_sum) * natoms_per_type[ntype]
        #     ener_shift, _, _, _ = np.linalg.lstsq(energy_one, energy_sum, rcond=rcond)
        #     self.ener_shift.append(ener_shift[0, 0])
        # energy_ntype = energy[:, natoms_sum:natoms_sum+natoms_per_type[ntype]]
        # natoms_sum += natoms_per_type[ntype]
        energy_sum = energy.sum(axis=1)
        energy_avg = np.average(energy_sum)
        # energy_one = np.ones_like(energy_sum) * natoms_per_type[ntype]
        ener_shift, _, _, _ = np.linalg.lstsq([natoms_per_type], [energy_avg], rcond=rcond)
        self.ener_shift = ener_shift.tolist()

    def get_stat(self, image_num=20, rcond=1e-3):
        # image_num = batch_size * batch_stat_num
        self.__compute_stat(image_num)
        self.__compute_stat_output(image_num, rcond)
        return self.davg, self.dstd, self.ener_shift


def get_torch_data(examplespath):
    '''
    input para:
    examplespath : npy_file_dir
    data_file_frompwmat : read train_data.csv or test_data.csv
    '''
    # examplespath='./train_data/final_train'   # for example
    f_feat = os.path.join(examplespath+'/feat_scaled.npy')
    if pm.dR_neigh:
        f_dR_neigh = os.path.join(examplespath+'/dR_neigh.npy')
        f_Ri = os.path.join(examplespath+'/Ri_all.npy')
        f_Ri_d = os.path.join(examplespath+'/Ri_d_all.npy')
    else:
        f_dR_neigh = None
        f_Ri = None
        f_Ri_d = None

    f_dfeat = os.path.join(examplespath+'/dfeat_scaled.npy')

    f_egroup = os.path.join(examplespath+'/egroup.npy')
    f_egroup_weight = os.path.join(examplespath+'/egroup_weight.npy')
    f_divider = os.path.join(examplespath+'/divider.npy')

    f_itype = os.path.join(examplespath+'/itypes.npy')
    f_nblist = os.path.join(examplespath+'/nblist.npy')
    f_weight_all = os.path.join(examplespath+'/weight_all.npy')
    ind_img = os.path.join(examplespath+'/ind_img.npy')
    natoms_img = os.path.join(examplespath+'/natoms_img.npy')

    f_energy = os.path.join(examplespath+'/engy_scaled.npy')
    f_force = os.path.join(examplespath+'/fors_scaled.npy')
    # f_force = os.path.join(examplespath+'/force.npy')

    torch_data = MovementDataset(f_feat, f_dfeat,
                                 f_egroup, f_egroup_weight, f_divider,
                                 f_itype, f_nblist, f_weight_all,
                                 f_energy, f_force, ind_img, natoms_img, f_dR_neigh, f_Ri, f_Ri_d)
    return torch_data
