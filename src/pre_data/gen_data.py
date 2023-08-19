#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from ipdb.__main__ import set_trace
import os
import sys

import default_para as pm 

import prepare
import numpy as np
import pandas as pd

from read_all import read_allnn

def process_data(f_train_feat, f_train_dfeat, f_train_dR_neigh,
                 f_train_natoms, f_train_egroup, nn_data_path): # f_train_ep):
    
    """
        Lines below only involves converting DENSE dfeat. 
        Not used when is_dfeat_sparse = True 
    """ 
    if (pm.feature_dtype == 'float64'):
        from convert_dfeat64 import convert_dfeat
    elif (pm.feature_dtype == 'float32'):
        from convert_dfeat import convert_dfeat
    else:
        raise RuntimeError("unsupported feature_dtype: %s, check feature_dtype in your parameters.py" %pm.feature_dtype)

    if not os.path.exists(nn_data_path):
        os.makedirs(nn_data_path)
    
    natoms = np.loadtxt(f_train_natoms, dtype=int)
    natoms = np.atleast_2d(natoms)
    
    nImg = natoms.shape[0]
    
    indImg = np.zeros((nImg+1,), dtype=int)
    indImg[0] = 0
    
    for i in range(nImg):
        indImg[i+1] = indImg[i] + natoms[i, 0]
    
    np.set_printoptions(threshold=np.inf)
    # pd.set_option('display.float_format',lambda x : '%.15f' % x)

    pd.options.display.float_format = '${:,.15f}'.format
    
    # Note: 
    itypes, feat, engy = prepare.r_feat_csv(f_train_feat)
    
    # pm.ntypes is number of atom types 
    natoms_img = np.zeros((nImg, pm.ntypes + 1), dtype=np.int32)

    for i in range(nImg):
        # total atom number in this image
        natoms_img[i][0] = indImg[i+1] - indImg[i]

        tmp = itypes[indImg[i]:indImg[i+1]]
        mask, ind = np.unique(tmp, return_index=True)
        
        mask = mask[np.argsort(ind)]

        type_id = 1

        for type in mask:
            natoms_img[i][type_id] = np.sum(tmp == type)
            type_id += 1
        
    # 进行scale
    # feat_scaled = scalers.pre_feat(feat, itypes)
    # engy_scaled = scalers.pre_engy(engy, itypes)
    # 不scale
    
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
    
    # feat_scale_a = np.zeros((nfeat0m, pm.ntypes))
    # for i in range(pm.ntypes):
    #     itype = pm.atomType[i]
    #     feat_scale_a[:, i] = scalers.scalers[itype].feat_scaler.a
    # feat_scale_a = np.asfortranarray(feat_scale_a)  # scaler 的 a参数

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
    
    """ 
        In fortran code, dfeat_tmp is the compact storage of dfeats. Only nonzero values are saved. 
    """

    # 读取 dfeat file
    for m in pm.use_Ftype:

        """
            get the path for dfeat
            
            compatible for multiple input movement file such as 

            PWdata/system1/dfeat.fbin.Ftype1
            PWdata/system2/dfeat.fbin.Ftype1
            ... 
            
            dfeat.fbin.Ftype1 records: (for feature 1)
            path to this dfeat, image index , absolute index 

            In the fortran code, dfeat_tmp is the sparse array that stores the dfeat values
        """
        
        # movement file location 
        # f_train_dfeat:  fread_dfeat/NN_output/dfeatname_train.csv

        dfeatdirs[m] = np.unique(pd.read_csv(
            f_train_dfeat+str(m), header=None, encoding= 'unicode_escape').values[:, 0])
        """
            looping over all dfeat binary files in different directory
        """
        for dfeatBinIdx in dfeatdirs[m]:
            read_allnn.read_dfeat(dfeatBinIdx, itype_atom, feat_scale_a, nfeat[flag])
            
            if flag == 0:
                energy_all[dfeatBinIdx] = np.array(read_allnn.energy_all).astype(pm.feature_dtype)
                force_all[dfeatBinIdx] = np.array(read_allnn.force_all).transpose(1, 0, 2).astype(pm.feature_dtype)
                list_neigh_all[dfeatBinIdx] = np.array(read_allnn.list_neigh_all).transpose(1, 0, 2).astype(int)
                iatom_all[dfeatBinIdx] = np.array(read_allnn.iatom)

            # dfeat here is already sparse
            nfeat[flag+1] = np.array(read_allnn.feat_all).shape[0]

            # dfeat_sparse[k] = np.array(read_allnn.dfeat_sparse).astype(pm.feature_dtype)
            # print("dfeat sparse shape:",dfeat_sparse[k].shape)
            # dfeat_tmp_all contains all images 
            dfeat_tmp_all[dfeatBinIdx] = np.array(read_allnn.dfeat_tmp_all).astype(pm.feature_dtype)
            
            #print("dfeat_tmp_all shape:",dfeat_tmp_all[dfeatBinIdx].shape)
            #print("dfeat nnz:", len(np.where(dfeat_tmp_all[dfeatBinIdx]!=0)[0]) ) 
            num_tmp_all[dfeatBinIdx] = np.array(read_allnn.num_tmp_all).astype(int)

            #below are the auxiliary arrays for dfeat_tmp_all
            iat_tmp_all[dfeatBinIdx] = np.array(read_allnn.iat_tmp_all).astype(int)
            jneigh_tmp_all[dfeatBinIdx] = np.array(read_allnn.jneigh_tmp_all).astype(int)
            ifeat_tmp_all[dfeatBinIdx] = np.array(read_allnn.ifeat_tmp_all).astype(int)
            
            read_allnn.deallo()
            
            #print("dfeat from Fort:\n",type(read_allnn.dfeat_tmp_all))

        flag = flag+1

    #pm.fitModelDir=./fread_dfeat  
    with open(os.path.join(pm.fitModelDir, "feat.info"), 'w') as f:
        print(os.path.join(pm.fitModelDir, "feat.info"))
        f.writelines(str(pm.iflag_PCA)+'\n')
        f.writelines(str(len(pm.use_Ftype))+'\n')
        for m in range(len(pm.use_Ftype)):
            f.writelines(str(pm.use_Ftype[m])+'\n')

        f.writelines(str(pm.ntypes)+' '+str(pm.maxNeighborNum)+'\n')
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
    
    for featureIdx in pm.use_Ftype:
        values = pd.read_csv(f_train_dfeat+str(featureIdx), header=None, encoding= 'unicode_escape').values
        # dfeat_names[m] 
        dfeat_names[featureIdx] = values[:, 0]
        image_nums[featureIdx] = values[:, 1].astype(int)
        pos_nums[featureIdx] = values[:, 2].astype(int)
        
        nImg = image_nums[featureIdx].shape[0]

    fors_scaled = []
    nblist = []

    for ll in range(len(image_nums[init])):
        fors_scaled.append(force_all[dfeat_names[init][ll]][:, :, image_nums[init][ll]-1])
        nblist.append(list_neigh_all[dfeat_names[init][ll]][:, :, image_nums[init][ll]-1])

    fors_scaled = np.concatenate(fors_scaled, axis=0)
    nblist = np.concatenate(nblist, axis=0)
    
    # ========================================================================
    
    """
        indImg : atom index for each image 
        [0,64,128,...]

        nfeat: 
        feature number 
    """

    """
        this label enforces the use of sparse dfeat.
        defualt = True, meaning that this block will not be executed
    """ 
    if pm.is_dfeat_sparse==False:
    
        img_num = indImg.shape[0] - 1
        #print ("indImg",indImg) #number of image 
        #print ("image num", img_num) 
        #print ("nfeat", nfeat) # feature number of each type 
        convert_dfeat.allo(nfeat0m, indImg[-1], pm.maxNeighborNum)
        
        for i in range(img_num):
            dfeat_name={}
            image_num={}
            
            for featureKey in pm.use_Ftype:
                """
                    dfeat_name: path to the dfeat file 
                    image_num : image index selected 
                """
                dfeat_name[featureKey] = dfeat_names[featureKey][i]
                image_num[featureKey] = image_nums[featureKey][i]
                
            featureIdx = 0
            
            for mm in pm.use_Ftype:
                # feature value array 
                """
                    index order of dfeat_tmp_all:
                    spatial dimension, non-zero element index, image index 
                """ 
                dfeat_tmp=np.asfortranarray(dfeat_tmp_all[dfeat_name[mm]][:,:,image_num[mm]-1])
                
                #neighbor index array
                jneigh_tmp=np.asfortranarray(jneigh_tmp_all[dfeat_name[mm]][:,image_num[mm]-1])
                
                # feature index array
                ifeat_tmp=np.asfortranarray(ifeat_tmp_all[dfeat_name[mm]][:,image_num[mm]-1])
                
                # atom index array 
                iat_tmp=np.asfortranarray(iat_tmp_all[dfeat_name[mm]][:,image_num[mm]-1])

                convert_dfeat.conv_dfeat(image_num[mm],nfeat[featureIdx],indImg[i],num_tmp_all[dfeat_name[mm]][image_num[mm]-1],dfeat_tmp,jneigh_tmp,ifeat_tmp,iat_tmp)

                featureIdx += 1         

                """
                    fortran convert_dfeat

                    conv_dfeat(image_Num,ipos,natom_p,num_tmp,dfeat_tmp,jneigh_tmp,ifeat_tmp,iat_tmp)

                    do jj=1,num_tmp
                        dfeat(ifeat_tmp(jj)+ipos,iat_tmp(jj)+natom_p,jneigh_tmp(jj),1)=dfeat_tmp(1,jj)
                        dfeat(ifeat_tmp(jj)+ipos,iat_tmp(jj)+natom_p,jneigh_tmp(jj),2)=dfeat_tmp(2,jj)
                        dfeat(ifeat_tmp(jj)+ipos,iat_tmp(jj)+natom_p,jneigh_tmp(jj),3)=dfeat_tmp(3,jj)
                    enddo
                    
                    natom_p: starting absolute atom index of this image
                """
        """ 
            index order of dense dfeat_scaled: 
            absolute atom index, neighbor index, feature index, spatial dimension
        """         
        
        dfeat_scaled = np.array(convert_dfeat.dfeat).transpose(1,2,0,3).astype(pm.feature_dtype)
        
        convert_dfeat.deallo()      

    print("feat_scaled shape" + str(feat_scaled.shape))
    print("fors_scaled shape" + str(fors_scaled.shape))
    print("nblist shape" + str(nblist.shape))
    print("engy_scaled shape" + str(engy_scaled.shape))
    print("itypes shape" + str(itypes.shape))
    print("egroup_weight shape" + str(egroup_weight.shape))
    print("weight_all shape" + str(weight_all.shape))
    print("egroup shape" + str(egroup.shape))
    print("divider shape" + str(egroup.shape))

    if pm.is_dfeat_sparse==False:
        print("dfeat_scaled shape" + str(dfeat_scaled.shape))
    
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
    
    # converting 
    np.save(nn_data_path + "/feat_scaled.npy", feat_scaled.astype(pm.feature_dtype))
    np.save(nn_data_path + "/fors_scaled.npy", fors_scaled.astype(pm.feature_dtype))
    np.save(nn_data_path + "/nblist.npy", nblist)
    np.save(nn_data_path + "/engy_scaled.npy", engy_scaled.astype(pm.feature_dtype))
    np.save(nn_data_path + "/itypes.npy", itypes)
    np.save(nn_data_path + "/egroup_weight.npy", egroup_weight.astype(pm.feature_dtype))
    np.save(nn_data_path + "/weight_all.npy", weight_all.astype(pm.feature_dtype))
    np.save(nn_data_path + "/egroup.npy", egroup.astype(pm.feature_dtype))
    np.save(nn_data_path + "/divider.npy", divider)
    
    if pm.is_dfeat_sparse==False:
        np.save(nn_data_path + "/dfeat_scaled.npy", dfeat_scaled.astype(pm.feature_dtype))
    
    np.save(nn_data_path + "/ind_img.npy", np.array(indImg).reshape(-1))
    np.save(nn_data_path + "/natoms_img.npy", natoms_img)
    # np.save(nn_data_path + "/ep.npy", ep)
    
def write_data():

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
        print("WARNING: you are generating low-precision feature data file")
        print("")
        print("we suggest you generate high-precision feature data file, and specify the")
        print("precision of training/inference separately, check feature_dtype / ")
        print("training_dtype / inference_dtype / in your parameters.py")
    
    print("")
    print("<=============== The end of feature data file generation  ===============>")
    print("")


if __name__ == '__main__':
    write_data()
