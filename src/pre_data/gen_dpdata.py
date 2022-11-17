#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.lib.type_check import real
import os
import sys

import default_para as pm
import prepare

import numpy as np
import pandas as pd

from read_all import read_allnn

import torch


#
# check for feature_dtype, that determins
# the dtype (and precision) of feature data files


def process_data(f_train_feat, f_train_dfeat, f_train_dR_neigh,
                 f_train_natoms, f_train_egroup, nn_data_path): # f_train_ep):
    
    if (pm.feature_dtype == 'float64'):
        from convert_dfeat64 import convert_dfeat
    elif (pm.feature_dtype == 'float32'):
        from convert_dfeat import convert_dfeat
        print("using single precision  is not recommended")
    else:
        raise RuntimeError(
        "unsupported feature_dtype: %s, check feature_dtype in your parameters.py"
        %pm.feature_dtype)

    if not os.path.exists(nn_data_path):
        os.makedirs(nn_data_path)
    
    # natoms contain all atomnum of each image, format: totnatom, type1n, type2 n

    natoms = np.loadtxt(f_train_natoms, dtype=int)
    natoms = np.atleast_2d(natoms)  
    nImg = natoms.shape[0]
    indImg = np.zeros((nImg+1,), dtype=int)
    indImg[0] = 0
    
    
    for i in range(nImg):
        indImg[i+1] = indImg[i] + natoms[i, 0]
    
    # 设置打印时显示方式：输出数组的时候完全输出
    np.set_printoptions(threshold=np.inf)
    # pd.set_option('display.float_format',lambda x : '%.15f' % x)
    pd.options.display.float_format = '${:,.15f}'.format
    itypes, feat, engy = prepare.r_feat_csv(f_train_feat)
    natoms_img = np.zeros((nImg, pm.ntypes + 1), dtype=np.int32)

    for i in range(nImg):
        natoms_img[i][0] = indImg[i+1] - indImg[i]
        tmp = itypes[indImg[i]:indImg[i+1]]
        mask, ind = np.unique(tmp, return_index=True)
        mask = mask[np.argsort(ind)]
        type_id = 1
        for type in mask:
            natoms_img[i][type_id] = np.sum(tmp == type)
            type_id += 1

    feat_scaled = feat
    engy_scaled = engy

    egroup, divider, egroup_weight = prepare.r_egroup_csv(f_train_egroup)
    if os.path.exists(os.path.join(pm.dir_work, 'weight_for_cases')):
        weight_all = pd.read_csv(os.path.join(pm.dir_work, 'weight_for_cases'),
                                 header=None, encoding= 'unicode_escape').values[:, 0].astype(pm.feature_dtype).reshape(-1, 1)
    else:
        weight_all = np.ones((engy_scaled.shape[0], 1))
    nfeat0m = feat_scaled.shape[1]  # 每个原子特征的维度
    itype_atom = np.asfortranarray(np.array(pm.atomType).transpose())  # 原子类型

    feat_scale_a=np.ones((nfeat0m,pm.ntypes))
    feat_scale_a = np.asfortranarray(feat_scale_a)
    
    init = pm.use_Ftype[0]
    

    dfeatdirs = {}
    energy_all = {}
    force_all = {}
    num_neigh_all = {}
    list_neigh_all = {}
    iatom_all = {}
    dfeat_tmp_all = {}
    num_tmp_all = {}
    iat_tmp_all = {}
    jneigh_tmp_all = {}
    ifeat_tmp_all = {}
    nfeat = {}
    nfeat[0] = 0
    flag = 0
    
    # 读取 dfeat file
    for m in pm.use_Ftype:
        dfeatdirs[m] = np.unique(pd.read_csv(
            f_train_dfeat+str(m), header=None, encoding= 'unicode_escape').values[:, 0])
        for k in dfeatdirs[m]:
            read_allnn.read_dfeat(k, itype_atom, feat_scale_a, nfeat[flag])
            if flag == 0:
                force_all[k] = np.array(read_allnn.force_all).transpose(1, 0, 2).astype(pm.feature_dtype)
                list_neigh_all[k] = np.array(read_allnn.list_neigh_all).transpose(1, 0, 2).astype(int)

            nfeat[flag+1] = np.array(read_allnn.feat_all).shape[0]
            read_allnn.deallo()
        flag = flag+1
    
    with open(os.path.join(pm.fitModelDir, "feat.info"), 'w') as f:
        print(os.path.join(pm.fitModelDir, "feat.info"))
        f.writelines(str(pm.iflag_PCA)+'\n')
        f.writelines(str(len(pm.use_Ftype))+'\n')
        for m in range(len(pm.use_Ftype)):
            f.writelines(str(pm.use_Ftype[m])+'\n')

        f.writelines(str(pm.ntypes)+', '+str(pm.maxNeighborNum)+'\n')
        for i in range(pm.ntypes):
            f.writelines(str(pm.atomType[i])+'  ' +
                         str(nfeat0m)+'  '+str(nfeat0m)+'\n')
        for i in range(pm.ntypes):
            for m in range(len(pm.use_Ftype)):
                f.writelines(str(nfeat[m+1])+'  ')
            f.writelines('\n')

    dfeat_names = {}
    image_nums = {}
    pos_nums = {}
    for m in pm.use_Ftype:
        values = pd.read_csv(f_train_dfeat+str(m), header=None, encoding= 'unicode_escape').values
        dfeat_names[m] = values[:, 0]
        image_nums[m] = values[:, 1].astype(int)
        pos_nums[m] = values[:, 2].astype(int)
        nImg = image_nums[m].shape[0]

    fors_scaled = []
    nblist = []
    for ll in range(len(image_nums[init])):
        fors_scaled.append(force_all[dfeat_names[init][ll]][:, :, image_nums[init][ll]-1])
        nblist.append(list_neigh_all[dfeat_names[init][ll]][:, :, image_nums[init][ll]-1])
    fors_scaled = np.concatenate(fors_scaled, axis=0)
    nblist = np.concatenate(nblist, axis=0)
    
# ========================================================================
    img_num = indImg.shape[0] - 1
    
    print("fors_scaled shape" + str(fors_scaled.shape))
    print("nblist shape" + str(nblist.shape))
    print("engy_scaled shape" + str(engy_scaled.shape))
    print("itypes shape" + str(itypes.shape))
    print("natoms_img shape" + str(natoms_img.shape))
    
    # neighbor 不排序
    if (pm.dR_neigh):
        dR_neigh = pd.read_csv(f_train_dR_neigh, header=None).values.reshape(indImg[-1], len(pm.atomType), pm.maxNeighborNum, 4) # 1 是 ntype
        print("dR neigh shape" + str(dR_neigh.shape))
        np.save(nn_data_path + "/dR_neigh.npy", dR_neigh)

    # neighbor 升序排列
    """
    if (pm.dR_neigh):
        # dR_neigh = pd.read_csv(f_train_dR_neigh, header=None).values.reshape(indImg[-1], pm.maxNeighborNum, 4) # 1 是 ntype
        names = ['dx', 'dy', 'dz', 'neigh_id']
        tmp = pd.read_csv(f_train_dR_neigh, header=None, names=names)
        tmp['dist'] = (tmp['dx'] ** 2 + tmp['dy'] ** 2 + tmp['dz'] ** 2) ** 0.5
        for i in range(indImg[-1] * len(pm.atomType)):
            if i == 0:
                res = tmp[i*pm.maxNeighborNum:(i+1)*pm.maxNeighborNum].sort_values(by=['dist'], ascending=True)
                zero_count = np.sum(res["neigh_id"].values == 0)
                res = pd.concat([res[zero_count:], res[:zero_count]])
                # import ipdb; ipdb.set_trace()
            else:
                second = tmp[i*pm.maxNeighborNum:(i+1)*pm.maxNeighborNum].sort_values(by=['dist'], ascending=True)
                zero_count = np.sum(second["neigh_id"].values == 0)
                res = pd.concat([res, second[zero_count:], second[:zero_count]])
        res = res[names]
        dR_neigh = res.values.reshape(indImg[-1], len(pm.atomType), pm.maxNeighborNum, 4)
        print("dR neigh shape" + str(dR_neigh.shape))
        np.save(nn_data_path + "/dR_neigh.npy", dR_neigh)
    """
    # if (pm.dR_neigh):
    #     tmp = pd.read_csv(pm.f_train_force, header=None)
    #     force = tmp.values
    #     np.save(nn_data_path + "/force.npy", force)
    

    # np.save(nn_data_path + "/feat_scaled.npy", feat_scaled)
    np.save(nn_data_path + "/fors_scaled.npy", fors_scaled)
    np.save(nn_data_path + "/nblist.npy", nblist)
    np.save(nn_data_path + "/engy_scaled.npy", engy_scaled)
    np.save(nn_data_path + "/itypes.npy", itypes)
    
    # np.save(nn_data_path + "/egroup_weight.npy", egroup_weight)
    # np.save(nn_data_path + "/weight_all.npy", weight_all)
    # np.save(nn_data_path + "/egroup.npy", egroup)
    # np.save(nn_data_path + "/divider.npy", divider)
    # np.save(nn_data_path + "/dfeat_scaled.npy", dfeat_scaled)
    np.save(nn_data_path + "/ind_img.npy", np.array(indImg).reshape(-1))
    np.save(nn_data_path + "/natoms_img.npy", natoms_img)
    # np.save(nn_data_path + "/ep.npy", ep)

def color_print(string, fg=31, bg=49):
    print("\33[0m\33[%d;%dm%s\33[0m" %(fg, bg, string))
    return 0

def main():

    print("")
    print("<================ Start of feature data file generation ================>")
    print("")
    read_allnn.read_wp(pm.fitModelDir, pm.ntypes)
    # 计算scale变换的参数
    # data_scalers = DataScalers(f_ds=pm.f_data_scaler,
                                #    f_feat=pm.f_train_feat)
    # scalers_train = get_scalers(pm.f_train_feat, pm.f_data_scaler, True)
    if pm.test_ratio != 1:
        print(read_allnn.wp_atom)
        process_data(pm.f_train_feat,
                    pm.f_train_dfeat,
                    pm.f_train_dR_neigh,
                    pm.f_train_natoms,
                    pm.f_train_egroup,
                    pm.train_data_path)        
    # scalers_test = get_scalers(pm.f_test_feat, pm.f_data_scaler, False)
    # data_scalers = DataScalers(f_ds=pm.f_data_scaler,
    #                                f_feat=pm.f_test_feat)
    process_data(pm.f_test_feat,
                 pm.f_test_dfeat,
                 pm.f_test_dR_neigh,
                 pm.f_test_natoms,
                 pm.f_test_egroup,
                 pm.test_data_path)

    print("")
    print("<=============== Summary of feature data file generation  ===============>")
    print("")
    if (pm.feature_dtype == 'float64'):
        print("feature_dtype = float64, feature data are stored as float64 in files,")
        print("check above detailed logs to make sure all files are stored in correct dtype")
        print("")
        print("you can run high-or-low precision training/inference with these feature data")
        print("by specify training_dtype and inference_dtype in you parameters.py")
    elif (pm.feature_dtype == 'float32'):
        print("feature_dtype = float32, feature data are stored as float32 in files,")
        print("check above detailed logs to make sure all files are stored in correct dtype")
        print("")
        color_print("WARNING: you are generating low-precision feature data file")
        print("")
        print("we suggest you generate high-precision feature data file, and specify the")
        print("precision of training/inference separately, check feature_dtype / ")
        print("training_dtype / inference_dtype / in your parameters.py")
    print("")
    print("<=============== The end of feature data file generation  ===============>")
    print("")
        

if __name__ == '__main__':
    main()
