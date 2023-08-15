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

class MovementHybridDataset(Dataset):

    def __init__(self, feat_path, dfeat_path,
                 egroup_path, egroup_weight_path, divider_path,
                 itype_path, nblist_path, weight_all_path,
                 energy_path, force_path, ind_img_path, natoms_img_path, data_num, atom_type, is_dfeat_sparse=False
                 ):  # , natoms_path

        """
            pm.is_dfeat_sparse is True, self.dfeat will not be generated
        """
        self.data_num = data_num
        self.atom_type_input =atom_type  
        self.ntypes = len(atom_type)
        self.is_dfeat_sparse = is_dfeat_sparse
        self.max_atom_nums = 0
        super(MovementHybridDataset, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.feat, self.dfeat= [], []
        self.egroup, self.egroup_weight, self.divider =[], [], []
        self.itype, self.nblist, self.weight_all, self.ind_img = [], [], [], []
        self.energy, self.force, self.natoms_img = [], [], []
        self.atom_type = []
        self.energy_shift = []
        for i in range(data_num):
            self.feat.append(np.load(feat_path[i]))

            if self.is_dfeat_sparse==False:
                self.dfeat.append(np.load(dfeat_path[i]))
        
            self.egroup.append(np.load(egroup_path[i]))
            self.egroup_weight.append(np.load(egroup_weight_path[i]))
            self.divider.append(np.load(divider_path[i]))

            # self.natoms_sum = natoms
            # self.natoms = pd.read_csv(natoms_path)   #/fread_dfeat/NN_output/natoms_train.csv
            self.itype.append(np.load(itype_path[i]))
            self.nblist.append(np.load(nblist_path[i]))
            self.weight_all.append(np.load(weight_all_path[i]))
            self.ind_img.append(np.load(ind_img_path[i]))

            self.energy.append(np.load(energy_path[i]))
            self.force.append(np.load(force_path[i]))

            self.natoms_img.append(np.load(natoms_img_path[i]))
            self.atom_type.append(np.array(self.get_atom_type(list(self.itype[i][0:self.ind_img[i][1]]))))

            if self.atom_type[i].size == len(self.atom_type_input):
                self.energy_shift = self._get_energy_shift(self.ind_img[i], self.itype[i], self.energy[i])
            
        self.image_num_list = []
        atom_nums_list = []
        feature_type_list =[]
        for i in range(self.data_num):
            image_number = self.ind_img[i].shape[0] - 1
            self.image_num_list.append(image_number)
            feature_type_list.append(self.dfeat[i].shape[2])
            atom_nums_list.append(self.natoms_img[0][0][0])
        self.max_feature_nums = max(feature_type_list)
        self.max_atom_nums = max(atom_nums_list)

    def __get_data_index(self, index, image_num_list):
        t = 0
        for i, image_num in enumerate(image_num_list):
            t += image_num
            if t > index:
                break
        image_index = index - sum(image_num_list[:i])
        return i, image_index
    
    def __getitem__(self, index):
        # index = index + 10
        data_index, index = self.__get_data_index(index, self.image_num_list)
        ind_image = np.zeros(2)
        ind_image[0] = self.ind_img[data_index][index]
        ind_image[1] = self.ind_img[data_index][index+1]
        dic= {}
    
        start_index = self.ind_img[data_index][index]
        end_index = self.ind_img[data_index][index+1]

        dic = {
            'input_feat': self.feat[data_index][start_index:end_index],
            'input_egroup': self.egroup[data_index][start_index:end_index],
            'input_egroup_weight': self.egroup_weight[data_index][start_index:end_index],
            'input_divider': self.divider[data_index][start_index:end_index],
            'input_itype': self.itype[data_index][start_index:end_index],
            'input_nblist': self.nblist[data_index][start_index:end_index],
            'atom_type': self.atom_type[data_index],
            'input_weight_all': self.weight_all[data_index][start_index:end_index],

            'output_energy': self.energy[data_index][start_index:end_index],
            'output_force': self.force[data_index][start_index:end_index],
            'ind_image': ind_image,
            'natoms_img': self.natoms_img[data_index][index]
        }
        if self.is_dfeat_sparse==False:
            dic['input_dfeat'] = self.dfeat[data_index][start_index:end_index]
        else:
            dic['input_dfeat'] = []
        return dic
    
    def get_atom_type(self, atom_type_list):
        atom_type_data_input = sorted(set(atom_type_list), key=atom_type_list.index) 
        return atom_type_data_input

    def __len__(self):
        return sum(self.image_num_list)

    '''
    description: 
    get energy shift from first image of input
    param {*} ind_img
    param {*} itype
    param {*} energy
    param {*} atom_type_input
    return {*}
    author: wuxingxing
    '''
    def _get_energy_shift(self, ind_img, itype, energy):
        type_dict = {}
        result = []

        num_atom = ind_img[1] # get atom nums in the first image
        # atom type list of a image
        type_list = itype[0:num_atom] # get atom types of the atoms
        atomic_energy_list = energy[0:num_atom] # get Ei of the atoms
        
        for atom, energy in zip(type_list, atomic_energy_list):
            if atom not in type_dict:
                type_dict[atom] = [energy]
            else:
                type_dict[atom].append(energy)
        
        for atom in self.atom_type_input:
            if atom in type_dict.keys():
                result.append(np.mean(type_dict[atom]))

        return result
    
    '''
    input para:
    examplespath : npy_file_dir
    data_file_frompwmat : read train_data.csv or test_data.csv
    '''
def get_torch_data_hybrid(data_root_path, sub_data_dir, data_type = "final_train", atom_type = None, is_dfeat_sparse=False):
    # examplespath='./train_data/final_train'   # for example

    f_feat, f_dfeat, f_egroup, f_egroup_weight, f_divider, \
        f_itype, f_nblist, f_weight_all, ind_img, natoms_img, f_energy, f_force\
        =[], [], [], [], [], [], [], [], [], [], [], []
    
    for dir in sub_data_dir:
            
        f_feat.append(os.path.join(data_root_path, dir, data_type, 'feat_scaled.npy'))
        f_dfeat.append(os.path.join(data_root_path, dir, data_type, 'dfeat_scaled.npy'))

        f_egroup.append(os.path.join(data_root_path, dir, data_type, 'egroup.npy'))
        f_egroup_weight.append(os.path.join(data_root_path, dir, data_type, 'egroup_weight.npy'))
        f_divider.append(os.path.join(data_root_path, dir, data_type, 'divider.npy'))

        f_itype.append(os.path.join(data_root_path, dir, data_type, 'itypes.npy'))
        f_nblist.append(os.path.join(data_root_path, dir, data_type, 'nblist.npy'))
        f_weight_all.append(os.path.join(data_root_path, dir, data_type, 'weight_all.npy'))
        ind_img.append(os.path.join(data_root_path, dir, data_type, 'ind_img.npy'))
        natoms_img.append(os.path.join(data_root_path, dir, data_type, 'natoms_img.npy'))

        f_energy.append(os.path.join(data_root_path, dir, data_type, 'engy_scaled.npy'))
        f_force.append(os.path.join(data_root_path, dir, data_type, 'fors_scaled.npy'))
        # f_force = os.path.join(examplespath+'/force.npy')

    torch_data = MovementHybridDataset(f_feat, f_dfeat,
                                 f_egroup, f_egroup_weight, f_divider,
                                 f_itype, f_nblist, f_weight_all,
                                 f_energy, f_force, ind_img, natoms_img, len(sub_data_dir), atom_type, is_dfeat_sparse)
    return torch_data
