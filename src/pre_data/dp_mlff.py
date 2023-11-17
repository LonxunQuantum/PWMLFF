#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import yaml
import numpy as np
import math
import torch
from collections import Counter
import subprocess as sp
import random
import time
from utils.random_utils import random_index
from utils.extract_movement import MOVEMENT
'''
description: 
 get all movement files under the dir 'workDir'
 author: wuxingxing
'''
def collect_all_sourcefiles(workDir, sourceFileName="MOVEMENT"):
    movement_dir = []
    if not os.path.exists(workDir):
        raise FileNotFoundError(workDir + "  does not exist")
    
    # added followlinks = True to support softlinks
    # get dir of movements
    for path, dirList, fileList in os.walk(workDir, followlinks=True):
        if sourceFileName in fileList:
            movement_dir.append(os.path.abspath(path))
    return movement_dir
    
def gen_config_inputfile(config):
    output_path = os.path.join(config["dRFeatureInputDir"], "gen_dR_feature.in")
    with open(output_path, "w") as GenFeatInput:
        GenFeatInput.write(
            str(config["Rc_M"])
            + ", "
            + str(config["maxNeighborNum"])
            + "             !  Rc_M, m_neigh \n"
        )
        
        atomTypeNum = len(config["atomType"])
        GenFeatInput.write(str(atomTypeNum) + "               ! ntype \n")
        for i in range(atomTypeNum):
            idAtomTypeConfig = config["atomType"][i]
            GenFeatInput.write(
                str(idAtomTypeConfig["type"]) + "              ! iat-type \n"
            )

            GenFeatInput.write(
                str(idAtomTypeConfig["Rc"])
                + ","
                + str(idAtomTypeConfig["Rm"])
                + ","
                + str(idAtomTypeConfig["iflag_grid"])
                + ","
                + str(idAtomTypeConfig["fact_base"])
                + ","
                + str(idAtomTypeConfig["dR1"])
                + "      !Rc,Rm,iflag_grid,fact_base,dR1 \n"
            )

        GenFeatInput.write(str(config["E_tolerance"]) + "    ! E_tolerance  \n")
        GenFeatInput.write(str(config["gen_egroup_input"]) + "    ! calculate Egroup if 1  \n")

    # output_path = os.path.join(config["dRFeatureInputDir"], "egroup.in")
    # with open(output_path, "w") as f:
    #     f.writelines(str(config["dwidth"]) + "\n")
    #     f.writelines(str(len(config["atomType"])) + "\n")
    #     for i in range(len(config["atomType"])):
    #         f.writelines(str(config["atomType"][i]["b_init"]) + "\n")

'''
description: 
write Ei.dat, the Ei is calculated by Ep
param {*} movement_files
param {*} train_set_dir "PWdata/"
return {*}
author: wuxingxing
'''
def set_Ei_dat_by_Ep(movement_files,train_set_dir):
    # make Ei label the Ep (one with minus sign)
    for mvm_index, movement_file in enumerate(movement_files):
        mvm_file = os.path.join(movement_file, "MOVEMENT")
        mvm = MOVEMENT(mvm_file)
        for k in range(0, mvm.image_nums):
            Ep = mvm.image_list[k].Ep
            atom_type_nums_list = mvm.image_list[k].atom_type_num
            print(atom_type_nums_list, Ep)
            tmp_Ep_shift, _, _, _ = np.linalg.lstsq([atom_type_nums_list], np.array([Ep]), rcond=1e-3)
            with open(os.path.join(train_set_dir, "Ei.dat"), "a") as Ei_out:
                for i in range(len(atom_type_nums_list)):
                    for j in range(atom_type_nums_list[i]):
                        Ei_out.write(str(tmp_Ep_shift[i]) + "\n")

def gen_train_data(config, is_egroup = True, is_virial = True, alive_atomic_energy = True):

    trainset_dir = config["trainSetDir"]
    dRFeatureInputDir = config["dRFeatureInputDir"]
    dRFeatureOutputDir = config["dRFeatureOutputDir"]

    # directories that contain MOVEMENT 
    movement_files = collect_all_sourcefiles(trainset_dir, "MOVEMENT")

    #os.system("clean_data.sh")
    cmd_clear = "rm "+trainset_dir+"/*.dat"
    sp.run([cmd_clear],shell=True)
    
    if not os.path.exists(dRFeatureInputDir):
        os.mkdir(dRFeatureInputDir)

    if not os.path.exists(dRFeatureOutputDir):
        os.mkdir(dRFeatureOutputDir)

    gen_config_inputfile(config)

    for movement_file in movement_files:
        with open(os.path.join(movement_file, "MOVEMENT"), "r") as mov:
            lines = mov.readlines()
            etot_tmp = []
            for idx, line in enumerate(lines):
                
                if "Lattice vector" in line and "stress" in lines[idx + 1]:
                    # Virial.dat
                    if is_virial:
                        print (line)

                        tmp_v = []
                        cell = []
                        for dd in range(3):
                            tmp_l = lines[idx + 1 + dd]
                            cell.append([float(ss) for ss in tmp_l.split()[0:3]])
                            tmp_v.append([float(stress) for stress in tmp_l.split()[5:8]])

                        tmp_virial = np.zeros([3, 3])
                        tmp_virial[0][0] = tmp_v[0][0]
                        tmp_virial[0][1] = tmp_v[0][1]
                        tmp_virial[0][2] = tmp_v[0][2]
                        tmp_virial[1][0] = tmp_v[1][0]
                        tmp_virial[1][1] = tmp_v[1][1]
                        tmp_virial[1][2] = tmp_v[1][2]
                        tmp_virial[2][0] = tmp_v[2][0]
                        tmp_virial[2][1] = tmp_v[2][1]
                        tmp_virial[2][2] = tmp_v[2][2]
                        volume = np.linalg.det(np.array(cell))
                        print(volume)
                        print("====================================")
                        # import ipdb;ipdb.set_trace()
                        # tmp_virial = tmp_virial * 160.2 * 10.0 / volume
                        with open(
                            os.path.join(trainset_dir, "Virial.dat"), "a"
                        ) as virial_file:
                            virial_file.write(
                                str(tmp_virial[0, 0])
                                + " "
                                + str(tmp_virial[0, 1])
                                + " "
                                + str(tmp_virial[0, 2])
                                + " "
                                + str(tmp_virial[1, 0])
                                + " "
                                + str(tmp_virial[1, 1])
                                + " "
                                + str(tmp_virial[1, 2])
                                + " "
                                + str(tmp_virial[2, 0])
                                + " "
                                + str(tmp_virial[2, 1])
                                + " "
                                + str(tmp_virial[2, 2])
                                + "\n"
                            )
                
                elif "Lattice vector" in line and "stress" not in lines[idx + 1]:
                    if is_virial:
                        raise ValueError("Invalid input file: 'stress' is not present in the line.")
                    else:
                        Virial = None

                # Etot.dat
                if "Etot,Ep,Ek" in line:
                    etot_tmp.append(line.split()[9])

        with open(os.path.join(trainset_dir, "Etot.dat"), "a") as etot_file:
            for etot in etot_tmp:
                etot_file.write(etot + "\n")
    
    # ImgPerMVT  
    if alive_atomic_energy is False:
        set_Ei_dat_by_Ep(movement_files, config["trainSetDir"]) # set Ei.dat by Ep

    for movement_file in movement_files:
        tgt = os.path.join(movement_file, "MOVEMENT") 
        res = sp.check_output(["grep", "Iteration", tgt ,"-c"]) 
        
        with open(os.path.join(trainset_dir, "ImgPerMVT.dat"), "a") as ImgPerMVT:
            ImgPerMVT.write(str(int(res))+"\n")     

    location_path = os.path.join(config["dRFeatureInputDir"], "location")
    with open(location_path, "w") as location_writer:
        location_writer.write(str(len(movement_files)) + "\n")
        location_writer.write(os.path.abspath(trainset_dir) + "\n")

        for movement_path in movement_files:
            location_writer.write(movement_path + "\n")

    # if is_real_Ep is True:
    #     command = "gen_dR_nonEi.x | tee ./output/out"
    # else:
    command = "gen_dR.x | tee ./output/out"
    print("==============Start generating data==============")
    os.system(command)
    # command = "gen_egroup.x | tee ./output/out_write_egroup"
    # if is_egroup is True:
        # print("==============Start generating egroup==============")
        # os.system(command)
    
    print("==============Success==============")
    

def save_npy_files(data_path, data_set):
    print("Saving to ", data_path)
    print("    AtomType.npy", data_set["AtomType"].shape)
    np.save(os.path.join(data_path, "AtomType.npy"), data_set["AtomType"])
    print("    ImageDR.npy", data_set["ImageDR"].shape)
    np.save(os.path.join(data_path, "ImageDR.npy"), data_set["ImageDR"])
    print("    ListNeighbor.npy", data_set["ListNeighbor"].shape)
    np.save(os.path.join(data_path, "ListNeighbor.npy"), data_set["ListNeighbor"])
    print("    Ei.npy", data_set["Ei"].shape)
    np.save(os.path.join(data_path, "Ei.npy"), data_set["Ei"])
    
    if "Egroup" in data_set.keys():
        print("    Egroup.npy", data_set["Egroup"].shape)
        np.save(os.path.join(data_path, "Egroup.npy"), data_set["Egroup"])
    if "Divider" in data_set.keys():
        print("    Divider.npy", data_set["Divider"].shape)
        np.save(os.path.join(data_path, "Divider.npy"), data_set["Divider"])
    if "Egroup_weight" in data_set.keys():
        print("    Egroup_weight.npy", data_set["Egroup_weight"].shape)
        np.save(os.path.join(data_path, "Egroup_weight.npy"), data_set["Egroup_weight"])

    print("    Ri.npy", data_set["Ri"].shape)
    np.save(os.path.join(data_path, "Ri.npy"), data_set["Ri"])
    print("    Ri_d.npy", data_set["Ri_d"].shape)
    np.save(os.path.join(data_path, "Ri_d.npy"), data_set["Ri_d"])
    print("    Force.npy", data_set["Force"].shape)
    np.save(os.path.join(data_path, "Force.npy"), data_set["Force"])
    if "Virial" in data_set.keys():
        print("    Virial.npy", data_set["Virial"].shape)
        np.save(os.path.join(data_path, "Virial.npy"), data_set["Virial"])
    print("    Etot.npy", data_set["Etot"].shape)
    np.save(os.path.join(data_path, "Etot.npy"), data_set["Etot"])
    print("    ImageAtomNum.npy", data_set["ImageAtomNum"].shape)
    np.save(os.path.join(data_path, "ImageAtomNum.npy"), data_set["ImageAtomNum"])


'''
description: 
claculate davg and dstd, the atom type order is the same as movement  
param {*} config
param {*} image_dR
param {*} list_neigh
param {*} natoms_img
return {*}
author: wuxingxing
'''
def calc_stat(config, image_dR, list_neigh, natoms_img):

    davg = []
    dstd = []

    natoms_sum = natoms_img[0, 0]
    natoms_per_type = natoms_img[0, 1:]
    ntypes = len(natoms_per_type)

    image_dR = np.reshape(
        image_dR, (-1, natoms_sum, ntypes * config["maxNeighborNum"], 3)
    )
    list_neigh = np.reshape(
        list_neigh, (-1, natoms_sum, ntypes * config["maxNeighborNum"])
    )

    image_dR = torch.tensor(image_dR, dtype=torch.float64)
    list_neigh = torch.tensor(list_neigh, dtype=torch.int)

    # deepmd neighbor id 从 0 开始，MLFF从1开始
    mask = list_neigh > 0

    dR2 = torch.zeros_like(list_neigh, dtype=torch.float64)
    Rij = torch.zeros_like(list_neigh, dtype=torch.float64)
    dR2[mask] = torch.sum(image_dR[mask] * image_dR[mask], -1)
    Rij[mask] = torch.sqrt(dR2[mask])

    nr = torch.zeros_like(dR2)
    inr = torch.zeros_like(dR2)

    dR2_copy = dR2.unsqueeze(-1).repeat(1, 1, 1, 3)
    Ri_xyz = torch.zeros_like(dR2_copy)

    nr[mask] = dR2[mask] / Rij[mask]
    Ri_xyz[mask] = image_dR[mask] / dR2_copy[mask]
    inr[mask] = 1 / Rij[mask]

    davg_tensor = torch.zeros(
        (ntypes, config["maxNeighborNum"] * ntypes, 4), dtype=torch.float64
    )
    dstd_tensor = torch.ones(
        (ntypes, config["maxNeighborNum"] * ntypes, 4), dtype=torch.float64
    )
    Ri, _, _ = smooth(
        config,
        image_dR,
        nr,
        Ri_xyz,
        mask,
        inr,
        davg_tensor,
        dstd_tensor,
        natoms_per_type,
    )
    Ri2 = Ri * Ri

    atom_sum = 0

    for i in range(ntypes):
        Ri_ntype = Ri[:, atom_sum : atom_sum + natoms_per_type[i]].reshape(-1, 4)
        Ri2_ntype = Ri2[:, atom_sum : atom_sum + natoms_per_type[i]].reshape(-1, 4)
        sum_Ri = Ri_ntype.sum(axis=0).tolist()
        sum_Ri_r = sum_Ri[0]
        sum_Ri_a = np.average(sum_Ri[1:])
        sum_Ri2 = Ri2_ntype.sum(axis=0).tolist()
        sum_Ri2_r = sum_Ri2[0]
        sum_Ri2_a = np.average(sum_Ri2[1:])
        sum_n = Ri_ntype.shape[0]

        davg_unit = [sum_Ri[0] / (sum_n + 1e-15), 0, 0, 0]
        dstd_unit = [
            compute_std(sum_Ri2_r, sum_Ri_r, sum_n),
            compute_std(sum_Ri2_a, sum_Ri_a, sum_n),
            compute_std(sum_Ri2_a, sum_Ri_a, sum_n),
            compute_std(sum_Ri2_a, sum_Ri_a, sum_n),
        ]
            
        davg.append(
            np.tile(davg_unit, config["maxNeighborNum"] * ntypes).reshape(-1, 4)
        )
        dstd.append(
            np.tile(dstd_unit, config["maxNeighborNum"] * ntypes).reshape(-1, 4)
        )
        atom_sum = atom_sum + natoms_per_type[i]

    davg = np.array(davg).reshape(ntypes, -1)
    dstd = np.array(dstd).reshape(ntypes, -1)
    return davg, dstd

def compute_std(sum2, sum, sumn):

    if sumn == 0:
        return 1e-2
    val = np.sqrt(sum2 / sumn - np.multiply(sum / sumn, sum / sumn))
    if np.abs(val) < 1e-2:
        val = 1e-2
    return val


def smooth(config, image_dR, x, Ri_xyz, mask, inr, davg, dstd, natoms):

    batch_size = image_dR.shape[0]
    ntypes = len(natoms)

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
    mask_min = x < config["atomType"][0]["Rm"]
    mask_1 = mask & mask_min  # [2,108,100]
    vv[mask_1] = 1
    dvv[mask_1] = 0

    # rcut_min< x < rcut_max
    mask_max = x < config["atomType"][0]["Rc"]
    mask_2 = ~mask_min & mask_max & mask
    # uu = (xx - rmin) / (rmax - rmin) ;
    uu[mask_2] = (x[mask_2] - config["atomType"][0]["Rm"]) / (
        config["atomType"][0]["Rc"] - config["atomType"][0]["Rm"]
    )
    vv[mask_2] = (
        uu[mask_2]
        * uu[mask_2]
        * uu[mask_2]
        * (-6 * uu[mask_2] * uu[mask_2] + 15 * uu[mask_2] - 10)
        + 1
    )
    du = 1.0 / (config["atomType"][0]["Rc"] - config["atomType"][0]["Rm"])
    # dd = ( 3 * uu*uu * (-6 * uu*uu + 15 * uu - 10) + uu*uu*uu * (-12 * uu + 15) ) * du;
    dvv[mask_2] = (
        3
        * uu[mask_2]
        * uu[mask_2]
        * (-6 * uu[mask_2] * uu[mask_2] + 15 * uu[mask_2] - 10)
        + uu[mask_2] * uu[mask_2] * uu[mask_2] * (-12 * uu[mask_2] + 15)
    ) * du

    mask_3 = ~mask_max & mask
    vv[mask_3] = 0
    dvv[mask_3] = 0

    res[mask] = 1.0 / x[mask]
    Ri = torch.cat((res.unsqueeze(-1), Ri_xyz), dim=-1)
    Ri_d = torch.zeros_like(Ri).unsqueeze(-1).repeat(1, 1, 1, 1, 3)  # 2 108 100 4 3
    tmp = torch.zeros_like(x)

    # deriv of component 1/r
    tmp[mask] = (
        image_dR[:, :, :, 0][mask] * inr3[mask] * vv[mask]
        - Ri[:, :, :, 0][mask] * dvv[mask] * image_dR[:, :, :, 0][mask] * inr[mask]
    )
    Ri_d[:, :, :, 0, 0][mask] = tmp[mask]
    tmp[mask] = (
        image_dR[:, :, :, 1][mask] * inr3[mask] * vv[mask]
        - Ri[:, :, :, 0][mask] * dvv[mask] * image_dR[:, :, :, 1][mask] * inr[mask]
    )
    Ri_d[:, :, :, 0, 1][mask] = tmp[mask]
    tmp[mask] = (
        image_dR[:, :, :, 2][mask] * inr3[mask] * vv[mask]
        - Ri[:, :, :, 0][mask] * dvv[mask] * image_dR[:, :, :, 2][mask] * inr[mask]
    )
    Ri_d[:, :, :, 0, 2][mask] = tmp[mask]

    # deriv of component x/r
    tmp[mask] = (
        2 * image_dR[:, :, :, 0][mask] * image_dR[:, :, :, 0][mask] * inr4[mask]
        - inr2[mask]
    ) * vv[mask] - Ri[:, :, :, 1][mask] * dvv[mask] * image_dR[:, :, :, 0][mask] * inr[
        mask
    ]
    Ri_d[:, :, :, 1, 0][mask] = tmp[mask]
    tmp[mask] = (
        2 * image_dR[:, :, :, 0][mask] * image_dR[:, :, :, 1][mask] * inr4[mask]
    ) * vv[mask] - Ri[:, :, :, 1][mask] * dvv[mask] * image_dR[:, :, :, 1][mask] * inr[
        mask
    ]
    Ri_d[:, :, :, 1, 1][mask] = tmp[mask]
    tmp[mask] = (
        2 * image_dR[:, :, :, 0][mask] * image_dR[:, :, :, 2][mask] * inr4[mask]
    ) * vv[mask] - Ri[:, :, :, 1][mask] * dvv[mask] * image_dR[:, :, :, 2][mask] * inr[
        mask
    ]
    Ri_d[:, :, :, 1, 2][mask] = tmp[mask]

    # deriv of component y/r
    tmp[mask] = (
        2 * image_dR[:, :, :, 1][mask] * image_dR[:, :, :, 0][mask] * inr4[mask]
    ) * vv[mask] - Ri[:, :, :, 2][mask] * dvv[mask] * image_dR[:, :, :, 0][mask] * inr[
        mask
    ]
    Ri_d[:, :, :, 2, 0][mask] = tmp[mask]
    tmp[mask] = (
        2 * image_dR[:, :, :, 1][mask] * image_dR[:, :, :, 1][mask] * inr4[mask]
        - inr2[mask]
    ) * vv[mask] - Ri[:, :, :, 2][mask] * dvv[mask] * image_dR[:, :, :, 1][mask] * inr[
        mask
    ]
    Ri_d[:, :, :, 2, 1][mask] = tmp[mask]
    tmp[mask] = (
        2 * image_dR[:, :, :, 1][mask] * image_dR[:, :, :, 2][mask] * inr4[mask]
    ) * vv[mask] - Ri[:, :, :, 2][mask] * dvv[mask] * image_dR[:, :, :, 2][mask] * inr[
        mask
    ]
    Ri_d[:, :, :, 2, 2][mask] = tmp[mask]

    # deriv of component z/r
    tmp[mask] = (
        2 * image_dR[:, :, :, 2][mask] * image_dR[:, :, :, 0][mask] * inr4[mask]
    ) * vv[mask] - Ri[:, :, :, 3][mask] * dvv[mask] * image_dR[:, :, :, 0][mask] * inr[
        mask
    ]
    Ri_d[:, :, :, 3, 0][mask] = tmp[mask]
    tmp[mask] = (
        2 * image_dR[:, :, :, 2][mask] * image_dR[:, :, :, 1][mask] * inr4[mask]
    ) * vv[mask] - Ri[:, :, :, 3][mask] * dvv[mask] * image_dR[:, :, :, 1][mask] * inr[
        mask
    ]
    Ri_d[:, :, :, 3, 1][mask] = tmp[mask]
    tmp[mask] = (
        2 * image_dR[:, :, :, 2][mask] * image_dR[:, :, :, 2][mask] * inr4[mask]
        - inr2[mask]
    ) * vv[mask] - Ri[:, :, :, 3][mask] * dvv[mask] * image_dR[:, :, :, 2][mask] * inr[
        mask
    ]
    Ri_d[:, :, :, 3, 2][mask] = tmp[mask]

    vv_copy = vv.unsqueeze(-1).repeat(1, 1, 1, 4)
    Ri[mask] *= vv_copy[mask]

    davg_res, dstd_res = None, None
    # 0 is that the atom nums is zero, for example, CH4 system in CHO system hybrid training, O atom nums is zero.\
    # beacuse the dstd or davg does not contain O atom, therefore, special treatment is needed here for atoms with 0 elements
    natoms = [_ for _ in natoms if _ != 0]
    ntypes = len(natoms)
    for ntype in range(ntypes):
        atom_num_ntype = natoms[ntype]
        davg_ntype = (
            davg[ntype].reshape(-1, 4).repeat(batch_size, atom_num_ntype, 1, 1)
        )  # [32,100,4]
        dstd_ntype = (
            dstd[ntype].reshape(-1, 4).repeat(batch_size, atom_num_ntype, 1, 1)
        )  # [32,100,4]
        davg_res = davg_ntype if davg_res is None else torch.concat((davg_res, davg_ntype), dim=1)
        dstd_res = dstd_ntype if dstd_res is None else torch.concat((dstd_res, dstd_ntype), dim=1)
        # if ntype == 0:
        #     davg_res = davg_ntype
        #     dstd_res = dstd_ntype
        # else:
        #     davg_res = torch.concat((davg_res, davg_ntype), dim=1)
        #     dstd_res = torch.concat((dstd_res, dstd_ntype), dim=1)
    max_ri = torch.max(Ri[:,:,:,0])
    Ri = (Ri - davg_res) / dstd_res
    dstd_res = dstd_res.unsqueeze(-1).repeat(1, 1, 1, 1, 3)
    Ri_d = Ri_d / dstd_res
    return Ri, Ri_d, max_ri


def compute_Ri(config, image_dR, list_neigh, natoms_img, ind_img, davg, dstd):
    natoms_sum = natoms_img[0, 0]
    natoms_per_type = natoms_img[0, 1:]
    ntypes = len(natoms_per_type)
    max_ri_list = [] # max Rij before davg and dstd cacled
    #if torch.cuda.is_available():
    #    device = torch.device("cuda")
    #else:
    
    device = torch.device("cpu")

    davg = torch.tensor(davg, device=device, dtype=torch.float64)
    dstd = torch.tensor(dstd, device=device, dtype=torch.float64)

    image_num = natoms_img.shape[0]

    img_seq = [0]
    seq_len = 0
    tmp_img = natoms_img[0]
    for i in range(image_num):
        if (natoms_img[i] != tmp_img).sum() > 0 or seq_len >= 500:
            img_seq.append(i)
            seq_len = 1
            tmp_img = natoms_img[i]
        else:
            seq_len += 1

    if img_seq[-1] != image_num:
        img_seq.append(image_num)

    for i in range(len(img_seq) - 1):
        start_index = img_seq[i]
        end_index = img_seq[i + 1]

        natoms_sum = natoms_img[start_index, 0]
        natoms_per_type = natoms_img[start_index, 1:]
       
        image_dR_i = image_dR[
            ind_img[start_index]
            * config["maxNeighborNum"]
            * ntypes : ind_img[end_index]
            * config["maxNeighborNum"]
            * ntypes
        ]
        list_neigh_i = list_neigh[
            ind_img[start_index]
            * config["maxNeighborNum"]
            * ntypes : ind_img[end_index]
            * config["maxNeighborNum"]
            * ntypes
        ]

        image_dR_i = np.reshape(
            image_dR_i, (-1, natoms_sum, ntypes * config["maxNeighborNum"], 3)
        )
        list_neigh_i = np.reshape(
            list_neigh_i, (-1, natoms_sum, ntypes * config["maxNeighborNum"])
        )

        image_dR_i = torch.tensor(image_dR_i, device=device, dtype=torch.float64)
        list_neigh_i = torch.tensor(list_neigh_i, device=device, dtype=torch.int)

        # deepmd neighbor id 从 0 开始，MLFF从1开始
        mask = list_neigh_i > 0 # 0 means the centor atom i does not have neighor

        dR2 = torch.zeros_like(list_neigh_i, dtype=torch.float64)
        Rij = torch.zeros_like(list_neigh_i, dtype=torch.float64)
        dR2[mask] = torch.sum(image_dR_i[mask] * image_dR_i[mask], -1)
        Rij[mask] = torch.sqrt(dR2[mask])

        nr = torch.zeros_like(dR2)
        inr = torch.zeros_like(dR2)

        dR2_copy = dR2.unsqueeze(-1).repeat(1, 1, 1, 3)
        Ri_xyz = torch.zeros_like(dR2_copy)

        nr[mask] = dR2[mask] / Rij[mask]
        Ri_xyz[mask] = image_dR_i[mask] / dR2_copy[mask]
        inr[mask] = 1 / Rij[mask]

        Ri_i, Ri_d_i, max_ri = smooth(
            config, image_dR_i, nr, Ri_xyz, mask, inr, davg, dstd, natoms_per_type
        )

        Ri_i = Ri_i.reshape(-1, ntypes * config["maxNeighborNum"], 4)
        Ri_d_i = Ri_d_i.reshape(-1, ntypes * config["maxNeighborNum"], 4, 3)

        if i == 0:
            Ri = Ri_i.detach().cpu().numpy()
            Ri_d = Ri_d_i.detach().cpu().numpy()
        else:
            Ri_i = Ri_i.detach().cpu().numpy()
            Ri_d_i = Ri_d_i.detach().cpu().numpy()
            Ri = np.concatenate((Ri, Ri_i), 0)
            Ri_d = np.concatenate((Ri_d, Ri_d_i), 0)
        max_ri_list.append(max_ri)

    if config['gen_egroup_input'] == 1:
        dwidth = np.sqrt(-config['atomType'][0]['Rc']**2 / np.log(0.01))
        egroup_weight_neigh = torch.exp(-dR2[mask] / dwidth / dwidth).to(device)
        egroup_weight_neigh = torch.reshape(egroup_weight_neigh, (-1, natoms_sum, natoms_sum - 1))
        egroup_weight_expanded = torch.zeros(size=(egroup_weight_neigh.shape[0], natoms_sum, natoms_sum), dtype=torch.float64)
        egroup_weight_low_diag = torch.tril(egroup_weight_neigh, diagonal=-1)
        egroup_weight_expanded[:, :natoms_sum, :egroup_weight_neigh.shape[2]] = egroup_weight_low_diag
        egroup_weight_all = egroup_weight_expanded + egroup_weight_expanded.transpose(-1,-2)
        for image in range(egroup_weight_neigh.shape[0]):
            egroup_weight_all[image].diagonal().fill_(1)   
                
        divider = egroup_weight_all.sum(-1)

        egroup_weight_all = egroup_weight_all.detach().cpu().numpy().reshape(-1, natoms_sum)
        divider = divider.detach().cpu().numpy().reshape(-1)
    else:
        egroup_weight_all = None
        divider = None

    return Ri, Ri_d, egroup_weight_all, divider, max(max_ri_list)

'''
description:
    classify movements according to thier atomtypes
    example: for systems: movement1[C,H,O], movement2[C,H], movement3[H,C,O], movement4[C], movement5[C,H,O]
                after classify:
                    movement1[C,H,O], movement5[C,H,O] belong to the same system.
                    movement2[C,H], movement3[H,C,O],movement4[C] are different system.
                    The atomic order of movement3[H,C,O] and movement5[C,H,O] is different, so they are different system.
param {*} img_per_mvmt
param {*} atom_num_per_image
param {*} atom_types
param {*} max_neighbor_num
param {*} ntypes the input of user
return {*}
author: wuxingxing
'''
def _classify_systems(img_per_mvmt, atom_num_per_image, atom_types, max_neighbor_num, ntypes):
    # read loactions of movement ?
    # for each movement, get the first image and count its atom type info
    movement_info = {}
    dr_start = 0
    for idx_mvmt, imgs_mvmt in enumerate(img_per_mvmt):
        idx_img = sum(img_per_mvmt[:idx_mvmt])
        img_atom_nums = atom_num_per_image[idx_img]
        # atom start index in AtomType.dat
        idx_atom_start_img = sum(atom_num_per_image[:sum(img_per_mvmt[:idx_mvmt])])
        atom_list = atom_types[idx_atom_start_img: idx_atom_start_img + img_atom_nums]
        types, type_nums, key = _get_type_info(atom_list)
        # Key consists of atomic type (atomic order is in the order of movement) and atomic number.
        movement_info[idx_mvmt] = {"image_nums":imgs_mvmt, 
            "atom_nums":sum(atom_num_per_image[idx_img:idx_img + imgs_mvmt]),
                "types":types, "type_nums":type_nums, "key":key}
        
        # get dRneigh indexs of the movement in the dRneigh.dat file
        # num of dRneigh = img_nums * atom_nums * all_atom_types * max nerghbor nums
        # all_atom_types and max nerghbor nums are global variables
        dr_rows = imgs_mvmt * sum(type_nums) * ntypes * max_neighbor_num 
        movement_info[idx_mvmt]["drneigh_rows"] = [dr_start, dr_start + dr_rows]
        dr_start += dr_rows
        # get Etot indexs of the movement in the Etot.dat file
        movement_info[idx_mvmt]["etot_rows"] = [idx_img, idx_img+imgs_mvmt]
        # get Ei index of the movement in the Ei.dat file
        movement_info[idx_mvmt]["ei_rows"] = [idx_atom_start_img, idx_atom_start_img + imgs_mvmt*img_atom_nums]
        # get Force index of the movement in the Force.dat file
        movement_info[idx_mvmt]["force_rows"] = movement_info[idx_mvmt]["ei_rows"]
        # the index of movement in input/local file
        movement_info[idx_mvmt]["mvm_index"] = idx_mvmt
        # set egroup and viral
        
    # The movement is sorted according to the number of atomic types. \
    # After that, the first MOVEMENT in movement_info dict has all atomic types, and will be used to calculate davg, dstd and energy_shift
    movement_info = sorted(movement_info.items(), key = lambda x: len(x[1]['types']), reverse=True)

    # assert len(movement_info[0][1]['types'])== ntypes, "Error: At least one input movement should contain all atomic types!"
    # classfiy movement_info by key
    classify = {}
    for mvm in movement_info:
        mvm = mvm[1] # mvm format is:(0, {'image_nums': 100, 'atom_nums': 1000, 'types': [...], 'type_nums': [...], 'key': '8_6_1_10', ...})
        if mvm['key'] not in classify.keys():
            classify[mvm['key']] = [mvm]
        else:
            classify[mvm['key']].append(mvm)      
    return classify

'''
description: 
    count atom type and atom numbers of each type, \
        key consists of atomic type (atomic order is in the order of movement) and atomic number.
param {*} atom_list the atom list of one image
    such as a Li-Si image: [3,3,3,3,3,14,14,14,14,14,14,14]
return {*}  types, the nums of type, and key
    such as [3, 14], [5, 7], "3_14_5_7"
author: wuxingxing
'''
def _get_type_info(atom_list: list):
    types = {}
    for atom in atom_list:
        if atom in types.keys():
            types[atom] += 1
        else:
            types[atom] = 1
    key = ""
    for k in types.keys():
        key += "{}_".format(k)
    key += "{}".format(len(atom_list))
    
    type_list = list(types.keys())
    type_list_nums = list(types.values())
    key1 = "_".join(str(item) for item in type_list)
    key2 = '_'.join(str(item) for item in type_list_nums)
    key = "{}_{}".format(key1, key2)
    return type_list, type_list_nums, key

'''
description:
    sepper .dat data to npy format:
    For hybrid data, there are many different systems, which contain different atomic numbers and different atomic types.
        1. classify movements according to thier atomtypes and atom numbers.
        2. calculate davg, dstd and energy_shift from system which contain all atom types.
        3. for movements in the same category, call function sepper_data.
        4. the last, save davg, dstd and energy_shift.
    
    Srij_max is the max S(rij) before doing scaled by dstd and davg, this value is used for model compress
param {*} config
param {*} is_egroup
param {*} is_load_stat
param {*} stat_add
return {*}
author: wuxingxing
'''
def sepper_data_main(config, is_egroup = True, stat_add = None, valid_random=False, seed=None): 
    trainset_dir = config["trainSetDir"]
    train_data_path = config["trainDataPath"] 
    valid_data_path = config["validDataPath"]
    max_neighbor_num = config["maxNeighborNum"]
    # directories that contain MOVEMENT 
    # _movement_files = np.loadtxt(os.path.join(config["dRFeatureInputDir"], "location"), dtype=str)[2:].tolist()
    ntypes = len(config["atomType"])
    atom_type_list = [int(_['type']) for _ in config["atomType"]] # get atom types,the order is consistent with user input order
    # image number in each movement 
    img_per_mvmt = np.loadtxt(os.path.join(trainset_dir, "ImgPerMVT.dat"), dtype=int)
    # when there is only one movement, convert img_per_mvmt type: array(num) -> [array(num)]
    if img_per_mvmt.size == 1:
        img_per_mvmt = [img_per_mvmt]
    # atom nums in each image
    atom_num_per_image = np.loadtxt(os.path.join(trainset_dir, "ImageAtomNum.dat"), dtype=int)    
    # atom type of each atom in the image
    atom_types = np.loadtxt(os.path.join(trainset_dir, "AtomType.dat"), dtype=int) 
    movement_classify = _classify_systems(img_per_mvmt, atom_num_per_image, atom_types, max_neighbor_num, ntypes)
       
    dR_neigh = np.loadtxt(os.path.join(trainset_dir, "dRneigh.dat"))
    Etot = np.loadtxt(os.path.join(trainset_dir, "Etot.dat"))
    Ei = np.loadtxt(os.path.join(trainset_dir, "Ei.dat"))
    Force = np.loadtxt(os.path.join(trainset_dir, "Force.dat"))
    if is_egroup:
        Egroup  = np.loadtxt(os.path.join(trainset_dir, "Egroup.dat"), delimiter=",", usecols=0)   
        # divider = np.loadtxt(os.path.join(trainset_dir, "Egroup_weight.dat"), delimiter=",", usecols=1)   
        # take care of weights
        # fp = open(os.path.join(trainset_dir, "Egroup_weight.dat"),"r")
        # raw_egroup = fp.readlines()
        # fp.close()
        # form a list to contain 1-d np arrays 
        # egroup_single_arr = []  
        # for line in raw_egroup:
        #     tmp  = [float(item) for item in line.split(",")]
        #     tmp  = tmp[2:]
        #     egroup_single_arr.append(np.array(tmp))
    else:
        # Egroup, divider, egroup_single_arr = None, None, None
        Egroup = None

    if os.path.exists(os.path.join(trainset_dir, "Virial.dat")):
        Virial = np.loadtxt(os.path.join(trainset_dir, "Virial.dat"), delimiter=" ")
    else:
        Virial = None

    if stat_add is not None:
        # load from prescribed path
        print("davg and dstd are from model checkpoint")
        davg, dstd, atom_type_order, energy_shift = stat_add
        # if energy_shift.size == 1: #
        #     energy_shift = [energy_shift]
    else:
        # calculate davg and dstd from first category of movement_classify
        davg, dstd = None, None
    Srij_max = 0.0
    img_start = [0, 0] # the index of images saved (train set and valid set)
    for mvm_type_key in movement_classify.keys():

        # _Egroup, _divider, _egroup_single_arr, _Virial = None, None, None, None
        #construct data
        for mvm in movement_classify[mvm_type_key]:
            _Etot, _Ei, _Force, _dR = None, None, None, None
            _atom_num_per_image, _atom_types, _img_per_mvmt = None, None, None,
            _Egroup, _Virial = None, None

            _Etot = Etot[mvm["etot_rows"][0]:mvm["etot_rows"][1]]
            _Ei = Ei[mvm["ei_rows"][0]:mvm["ei_rows"][1]]
            _Force = Force[mvm["force_rows"][0]:mvm["force_rows"][1]]
            _dR = dR_neigh[mvm["drneigh_rows"][0]:mvm["drneigh_rows"][1]]
            # egroup
            if Egroup is not None:
                _Egroup = Egroup[mvm["ei_rows"][0]:mvm["ei_rows"][1]]
                # _divider = divider[mvm["ei_rows"][0]:mvm["ei_rows"][1]] if _divider is None \
                #     else np.concatenate([_divider, divider[mvm["ei_rows"][0]:mvm["ei_rows"][1]]],axis=0)
                # _egroup_single_arr = egroup_single_arr[mvm["ei_rows"][0]:mvm["ei_rows"][1]] if _egroup_single_arr is None \
                #     else np.concatenate([_egroup_single_arr, egroup_single_arr[mvm["ei_rows"][0]:mvm["ei_rows"][1]]],axis=0)
            # Virial not realized
            if Virial is not None:
                _Virial = Virial[mvm["etot_rows"][0]:mvm["etot_rows"][1]]

            _atom_num_per_image = atom_num_per_image[mvm["etot_rows"][0]:mvm["etot_rows"][1]]
            _atom_types = atom_types[mvm["ei_rows"][0]:mvm["ei_rows"][1]]
            _img_per_mvmt = [img_per_mvmt[mvm["mvm_index"]]]

            if davg is None:
                # the davg and dstd only need calculate one time
                # the davg, dstd and energy_shift atom order are the same --> atom_type_order 
                davg, dstd = _calculate_davg_dstd(config, _dR, _atom_types, _atom_num_per_image)
                energy_shift, atom_type_order = _calculate_energy_shift(_Ei, _atom_types, _atom_num_per_image)
                davg, dstd, energy_shift, atom_type_order = adjust_order_same_as_user_input(davg, dstd, energy_shift,atom_type_order, atom_type_list)
            # reorder davg and dstd to consistent with atom type order of current movement
            _davg, _dstd = _reorder_davg_dstd(davg, dstd, list(atom_type_order), mvm['types'])

            accum_train_num, accum_valid_num, _Srij_max = sepper_data(config, _Etot, _Ei, _Force, _dR, \
                                                      _atom_num_per_image, _atom_types, _img_per_mvmt, \
                                                      _Egroup, _Virial, \
                                                      _davg, _dstd,\
                                                      stat_add, img_start, valid_random, seed)
            Srij_max = max(_Srij_max, Srij_max)
            img_start = [accum_train_num, accum_valid_num]

    if os.path.exists(os.path.join(train_data_path, "davg.npy")) is False:
        np.save(os.path.join(train_data_path, "davg.npy"), davg)
        np.save(os.path.join(valid_data_path, "davg.npy"), davg)
        np.save(os.path.join(train_data_path, "dstd.npy"), dstd)
        np.save(os.path.join(valid_data_path, "dstd.npy"), dstd)
        np.savetxt(os.path.join(train_data_path, "atom_map.raw"), atom_type_order, fmt="%d")
        np.savetxt(os.path.join(valid_data_path, "atom_map.raw"), atom_type_order, fmt="%d")
        np.savetxt(os.path.join(train_data_path, "energy_shift.raw"), energy_shift)
        np.savetxt(os.path.join(valid_data_path, "energy_shift.raw"), energy_shift)
        np.savetxt(os.path.join(train_data_path, "sij_max.raw"), [Srij_max], fmt="%.6f")
        np.savetxt(os.path.join(valid_data_path, "sij_max.raw"), [Srij_max], fmt="%.6f")
                

'''
description: 
According to the atom type and order of tar_order output corresponding davg and dstd
    example
         for source order O-C-H system: davg[0] is O atom, davg[1] is C atom, and davg[2] is H atom:
                for a target order H-C-O system: davg[0] is H atom, davg[1] is C atom, and davg[2] is O atom
                for a target order H-C system: davg[0] is H atom, davg[1] is C atom.
        dstd is same as above.
param {*} davg
param {*} dstd
param {*} sor_order the input order of davg and dstd
param {*} tar_order the output order of davg and dstd
return {*}
author: wuxingxing
'''
def _reorder_davg_dstd(davg, dstd, sor_order, tar_order):
    tar_davg, tar_dstd = [], []
    for i in tar_order:
        tar_davg.append(davg[sor_order.index(i)])
        tar_dstd.append(dstd[sor_order.index(i)])
    return tar_davg, tar_dstd

'''
description: 
    calculate energy shift, this value is used in DP model when doning the fitting net bias_init
    energy shift is the avrage value of atom's Ei. 
param {*} Ei
param {*} atom_type
param {*} atom_num_per_image 
param {*} chunk_size: the image nums for energy shift calculating
return: two list: energy shift list and atom type list
        example: for a li-si system: [-191.614604541, -116.03510427249998], [3, 14]
author: wuxingxing
'''
def _calculate_energy_shift(Ei, atom_type, atom_num_per_image,  chunk_size=10):
    if chunk_size > len(atom_num_per_image):
        chunk_size = len(atom_num_per_image)
    Ei = Ei[: sum(atom_num_per_image[:chunk_size])]
    atom_type = atom_type[: sum(atom_num_per_image[:chunk_size])]
    type_dict = {}
    for i in range(sum(atom_num_per_image[:chunk_size])):
        if atom_type[i] not in type_dict.keys():
            type_dict[atom_type[i]] = [Ei[i]]
        else:
            type_dict[atom_type[i]].append(Ei[i])
    res = []
    for t in type_dict.keys():
        res.append(np.mean(type_dict[t]))
    return res, list(type_dict.keys())

'''
description: 
adjust atom ordor of davg, dstd, energy_shift to same as user input order
param {list} davg
param {list} dstd
param {list} energy_shift
param {list} atom_type_order: the input davg, dstd atom order
param {list} atom_type_list: the user input order 
return {*}
author: wuxingxing
'''
def adjust_order_same_as_user_input(davg:list, dstd:list, energy_shift:list, atom_type_order:list, atom_type_list:list):
    davg_res, dstd_res, energy_shift_res = [], [], []
    for i, atom in enumerate(atom_type_list):
        davg_res.append(davg[atom_type_order.index(atom)])
        dstd_res.append(dstd[atom_type_order.index(atom)])
        energy_shift_res.append(energy_shift[atom_type_order.index(atom)])
    return davg_res, dstd_res, energy_shift_res, atom_type_list
        
'''
description: 
    calculate davg and dstd, the atom type order of davg and dstd is same as input paramter atom_type 
param {*} config
param {*} dR_neigh
param {*} atom_type, atom list in image, such as a li-si system: atom_type = [3,3,3,3,3,3,3,14,14,14,14,14] 
param {*} chunk_size, chose 10(default) images.
return {*}
author: wuxingxing
'''
def _calculate_davg_dstd(config, dR_neigh, atom_type, atom_num_per_image, chunk_size=10):
    ntypes = len(config["atomType"])
    max_neighbor_num = config["maxNeighborNum"]
    
    image_dR = dR_neigh[:, :3]
    list_neigh = dR_neigh[:, 3]
    
    image_index = np.insert(
        atom_num_per_image, 0, 0
    ).cumsum()  # array([  0, 108, 216, 324, 432, 540, 648, 756, 864, 972])
    
    image_num = atom_num_per_image.shape[0]
    
    diff_atom_types_num = []
    for i in range(image_num):
        atom_type_per_image = atom_type[image_index[i] : image_index[i + 1]]
        ######## mask need to flexibly change according to atom_type
        # unique_values, indices = np.unique(atom_type_per_image, return_index=True)
        # mask = unique_values[np.argsort(indices)]
        mask = np.array([atom_type['type'] for atom_type in config["atomType"]])
        #######
        diff_atom_types_num.append(
            [Counter(atom_type_per_image)[mask[type]] for type in range(mask.shape[0])]
        )
    narray_diff_atom_types_num = np.array(diff_atom_types_num)
    atom_num_per_image = np.concatenate(
        (atom_num_per_image.reshape(-1, 1), narray_diff_atom_types_num), axis=1
    )
    if len(image_index)-1 < chunk_size:
        chunk_size = len(image_index)-1
    davg, dstd = calc_stat(
            config,
            image_dR[0 : image_index[chunk_size] * max_neighbor_num * ntypes],
            list_neigh[0 : image_index[chunk_size] * max_neighbor_num * ntypes],
            atom_num_per_image[0:chunk_size],
        )

    return davg, dstd
    
def sepper_data(config, Etot, Ei, Force, dR_neigh,\
                atom_num_per_image, atom_type, img_per_mvmt, \
                Egroup=None, Virial=None, \
                davg=None, dstd=None,\
                stat_add = "./", img_start=[0, 0], valid_random=False, seed=None):

    train_data_path = config["trainDataPath"]
    valid_data_path = config["validDataPath"]
    max_neighbor_num = config["maxNeighborNum"]
    ntypes = len(config["atomType"])
    
    image_dR = dR_neigh[:, :3]
    list_neigh = dR_neigh[:, 3]
    
    image_index = np.insert(
        atom_num_per_image, 0, 0
    ).cumsum()  # array([  0, 108, 216, 324, 432, 540, 648, 756, 864, 972])
    
    image_num = atom_num_per_image.shape[0]
    
    diff_atom_types_num = []
    for i in range(image_num):
        atom_type_per_image = atom_type[image_index[i] : image_index[i + 1]]
        ######## mask need to flexibly change according to atom_type
        # unique_values, indices = np.unique(atom_type_per_image, return_index=True)
        # mask = unique_values[np.argsort(indices)]
        mask = np.array([atom_type['type'] for atom_type in config["atomType"]])
        #######
        diff_atom_types_num.append(
            [Counter(atom_type_per_image)[mask[type]] for type in range(mask.shape[0])]
        )
    narray_diff_atom_types_num = np.array(diff_atom_types_num)
    atom_num_per_image = np.concatenate(
        (atom_num_per_image.reshape(-1, 1), narray_diff_atom_types_num), axis=1
    )
    
    Ri, Ri_d, Egroup_weight, Divider, max_ri = compute_Ri( 
            config, image_dR, list_neigh, atom_num_per_image, image_index, davg, dstd
        )
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    if not os.path.exists(valid_data_path):
        os.makedirs(valid_data_path)
    
    list_neigh = list_neigh.reshape(-1, max_neighbor_num * ntypes)
    image_dR = image_dR.reshape(-1, max_neighbor_num * ntypes, 3)

    accum_train_num = img_start[0] 
    accum_valid_num = img_start[1]
    # width = len(str(accum_train_num))

    train_indexs, valid_indexs = random_index(image_num, config["ratio"], valid_random, seed)

    # index = 0
    for index in train_indexs:
        start_index = index
        end_index = index+1
        # end_index = min(end_index, len(train_indexs))
        train_set = {
                "AtomType": atom_type[image_index[start_index] : image_index[end_index]],
                "ImageDR": image_dR[image_index[start_index] : image_index[end_index]],
                "ListNeighbor": list_neigh[
                    image_index[start_index] : image_index[end_index]
                ],
                "Ei": Ei[image_index[start_index] : image_index[end_index]],
                "Ri": Ri[image_index[start_index] : image_index[end_index]],
                "Ri_d": Ri_d[image_index[start_index] : image_index[end_index]],
                "Force": Force[image_index[start_index] : image_index[end_index]],
                "Etot": Etot[start_index:end_index],
                "ImageAtomNum": atom_num_per_image[start_index:end_index],
            }
        
        if Egroup is not None:
            train_set["Egroup"] = Egroup[image_index[start_index] : image_index[end_index]]
            # train_set["Divider"] = divider[image_index[start_index] : image_index[end_index]]
            train_set["Divider"] = Divider[image_index[start_index] : image_index[end_index]]
            # train_set["Egroup_weight"] = np.vstack(tuple(egroup_single_arr[image_index[start_index] : image_index[end_index]]))
            train_set["Egroup_weight"] = Egroup_weight[image_index[start_index] : image_index[end_index]]
        if Virial is not None:
            train_set["Virial"] = Virial[start_index:end_index]

        save_path = os.path.join(train_data_path, "image_" + str(accum_train_num))

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_npy_files(save_path, train_set)

        accum_train_num += 1 

        # index = end_index

    # width = len(str(image_num - train_image_num))

    for index in valid_indexs:
        start_index = index
        end_index = index + 1

        # end_index = min(end_index, image_num)

        valid_set = {
                "AtomType": atom_type[image_index[start_index] : image_index[end_index]],
                "ImageDR": image_dR[image_index[start_index] : image_index[end_index]],
                "ListNeighbor": list_neigh[
                    image_index[start_index] : image_index[end_index]
                ],
                "Ei": Ei[image_index[start_index] : image_index[end_index]],
                "Ri": Ri[image_index[start_index] : image_index[end_index]],
                "Ri_d": Ri_d[image_index[start_index] : image_index[end_index]],
                "Force": Force[image_index[start_index] : image_index[end_index]],
                "Etot": Etot[start_index:end_index],
                "ImageAtomNum": atom_num_per_image[start_index:end_index],
            }
        
        if Egroup is not None:
            valid_set["Egroup"] = Egroup[image_index[start_index] : image_index[end_index]]
            # valid_set["Divider"] = divider[image_index[start_index] : image_index[end_index]]
            valid_set["Divider"] = Divider[image_index[start_index] : image_index[end_index]]
            # valid_set["Egroup_weight"] = np.vstack(tuple(egroup_single_arr[image_index[start_index] : image_index[end_index]]))
            valid_set["Egroup_weight"] = Egroup_weight[image_index[start_index] : image_index[end_index]]
        if Virial is not None:
            valid_set["Virial"] = Virial[start_index:end_index]

        save_path = os.path.join(valid_data_path, "image_" + str(accum_valid_num))

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_npy_files(save_path, valid_set)
        # index = end_index
        accum_valid_num += 1

    print("Saving npy file done")

    Rij_max = max_ri # for model compress
    return accum_train_num, accum_valid_num, Rij_max

