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

def collect_all_sourcefiles(workDir, sourceFileName="MOVEMENT"):

    res = []
    if not os.path.exists(workDir):
        raise FileNotFoundError(workDir + "  does not exist")
    for path, dirList, fileList in os.walk(workDir):
        if sourceFileName in fileList:
            res.append(os.path.abspath(path))
    return res


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

    output_path = os.path.join(config["dRFeatureInputDir"], "egroup.in")
    with open(output_path, "w") as f:
        f.writelines(str(config["dwidth"]) + "\n")
        f.writelines(str(len(config["atomType"])) + "\n")
        for i in range(len(config["atomType"])):
            f.writelines(str(config["atomType"][i]["b_init"]) + "\n")


def gen_train_data(config):
    trainset_dir = config["trainSetDir"]
    dRFeatureInputDir = config["dRFeatureInputDir"]
    dRFeatureOutputDir = config["dRFeatureOutputDir"]

    os.system("clean_data.sh")

    if not os.path.exists(dRFeatureInputDir):
        os.mkdir(dRFeatureInputDir)

    if not os.path.exists(dRFeatureOutputDir):
        os.mkdir(dRFeatureOutputDir)

    gen_config_inputfile(config)

    # directories that contain MOVEMENT 
    movement_files = collect_all_sourcefiles(trainset_dir, "MOVEMENT")
    
    # Virial.dat
    for movement_file in movement_files:
        with open(os.path.join(movement_file, "MOVEMENT"), "r") as mov:
            lines = mov.readlines()
            for idx, line in enumerate(lines):
                
                if "Lattice vector" in line and "stress" in lines[idx + 1]:
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

    # ImgPerMVT 
     
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

    command = "gen_dR.x | tee ./output/out"
    print("==============Start generating data==============")
    os.system(command)
    command = "gen_egroup.x | tee ./output/out_write_egroup"
    print("==============Start generating egroup==============")
    os.system(command)
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
    """
    print("    Egroup.npy", data_set["Egroup"].shape)
    np.save(os.path.join(data_path, "Egroup.npy"), data_set["Egroup"])
    print("    Divider.npy", data_set["Divider"].shape)
    np.save(os.path.join(data_path, "Divider.npy"), data_set["Divider"])
    print("    Egroup_weight.npy", data_set["Egroup_weight"].shape)
    np.save(os.path.join(data_path, "Egroup_weight.npy"), data_set["Egroup_weight"])
    """
    print("    Ri.npy", data_set["Ri"].shape)
    np.save(os.path.join(data_path, "Ri.npy"), data_set["Ri"])
    print("    Ri_d.npy", data_set["Ri_d"].shape)
    np.save(os.path.join(data_path, "Ri_d.npy"), data_set["Ri_d"])
    print("    Force.npy", data_set["Force"].shape)
    np.save(os.path.join(data_path, "Force.npy"), data_set["Force"])
    print("    Virial.npy", data_set["Virial"].shape)
    np.save(os.path.join(data_path, "Virial.npy"), data_set["Virial"])
    print("    ImageAtomNum.npy", data_set["ImageAtomNum"].shape)
    np.save(os.path.join(data_path, "ImageAtomNum.npy"), data_set["ImageAtomNum"])


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
    Ri, _ = smooth(
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

    for ntype in range(ntypes):
        atom_num_ntype = natoms[ntype]
        davg_ntype = (
            davg[ntype].reshape(-1, 4).repeat(batch_size, atom_num_ntype, 1, 1)
        )  # [32,100,4]
        dstd_ntype = (
            dstd[ntype].reshape(-1, 4).repeat(batch_size, atom_num_ntype, 1, 1)
        )  # [32,100,4]
        if ntype == 0:
            davg_res = davg_ntype
            dstd_res = dstd_ntype
        else:
            davg_res = torch.concat((davg_res, davg_ntype), dim=1)
            dstd_res = torch.concat((dstd_res, dstd_ntype), dim=1)
    Ri = (Ri - davg_res) / dstd_res
    dstd_res = dstd_res.unsqueeze(-1).repeat(1, 1, 1, 1, 3)
    Ri_d = Ri_d / dstd_res
    return Ri, Ri_d


def compute_Ri(config, image_dR, list_neigh, natoms_img, ind_img, davg, dstd):
    natoms_sum = natoms_img[0, 0]
    natoms_per_type = natoms_img[0, 1:]
    ntypes = len(natoms_per_type)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
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
        mask = list_neigh_i > 0

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

        Ri_i, Ri_d_i = smooth(
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

    return Ri, Ri_d


def sepper_data(config):
    """
        do shuffling here 
        Shuffle for each MOVEMENT  
    """
    trainset_dir = config["trainSetDir"]

    train_data_path = config["trainDataPath"] 
    valid_data_path = config["validDataPath"]

    max_neighbor_num = config["maxNeighborNum"]
    ntypes = len(config["atomType"])

    # image number in each movement 
    img_per_mvmt = np.loadtxt(os.path.join(trainset_dir, "ImgPerMVT.dat"), dtype=int)
    
    atom_type = np.loadtxt(os.path.join(trainset_dir, "AtomType.dat"), dtype=int)
    dR_neigh = np.loadtxt(os.path.join(trainset_dir, "dRneigh.dat"))
    image_dR = dR_neigh[:, :3]
    list_neigh = dR_neigh[:, 3]
    Ei = np.loadtxt(os.path.join(trainset_dir, "Ei.dat"))

    """
    Egroup_file = np.loadtxt(
        os.path.join(trainset_dir, "Egroup_weight.dat"), delimiter=","
    )
    Egroup = Egroup_file[:, 0]
    divider = Egroup_file[:, 1]
    Egroup_weight = Egroup_file[:, 2:]
    """
    Force = np.loadtxt(os.path.join(trainset_dir, "Force.dat"))
    Virial = np.loadtxt(os.path.join(trainset_dir, "Virial.dat"), delimiter=" ")
    atom_num_per_image = np.loadtxt(
        os.path.join(trainset_dir, "ImageAtomNum.dat"), dtype=int
    )
    image_index = np.insert(
        atom_num_per_image, 0, 0
    ).cumsum()  # array([  0, 108, 216, 324, 432, 540, 648, 756, 864, 972])
    
    image_num = atom_num_per_image.shape[0]

    diff_atom_types_num = []
    for i in range(image_num):
        atom_type_per_image = atom_type[image_index[i] : image_index[i + 1]]
        mask = np.unique(atom_type_per_image)
        diff_atom_types_num.append(
            [Counter(atom_type_per_image)[mask[type]] for type in range(mask.shape[0])]
        )
    narray_diff_atom_types_num = np.array(diff_atom_types_num)
    atom_num_per_image = np.concatenate(
        (atom_num_per_image.reshape(-1, 1), narray_diff_atom_types_num), axis=1
    )

    davg, dstd = calc_stat(
        config,
        image_dR[0 : image_index[10] * max_neighbor_num * ntypes],
        list_neigh[0 : image_index[10] * max_neighbor_num * ntypes],
        atom_num_per_image[0:10],
    )

    Ri, Ri_d = compute_Ri(
        config, image_dR, list_neigh, atom_num_per_image, image_index, davg, dstd
    )

    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    if not os.path.exists(valid_data_path):
        os.makedirs(valid_data_path)

    print("Saving npy file")

    np.save(os.path.join(train_data_path, "davg.npy"), davg)
    np.save(os.path.join(valid_data_path, "davg.npy"), davg)
    np.save(os.path.join(train_data_path, "dstd.npy"), dstd)
    np.save(os.path.join(valid_data_path, "dstd.npy"), dstd)

    train_image_num = math.ceil(image_num * config["ratio"])

    list_neigh = list_neigh.reshape(-1, max_neighbor_num * ntypes)
    image_dR = image_dR.reshape(-1, max_neighbor_num * ntypes, 3)

    width_train = len(str(train_image_num))
    width_valid = len(str(image_num - train_image_num))

    index = 0

    """
        Note: the seperation is done linearly. Should not be. 
    """ 
    accum_train_num = 0 
    accum_valid_num = 0 

    
    range_mvmt = [0 for i in range(img_per_mvmt.size)]
        
    if img_per_mvmt.size == 1:
        range_mvmt[0] = img_per_mvmt
    else:

        for idx in range(img_per_mvmt.size):
            range_mvmt[idx] = sum(img_per_mvmt[:idx+1])
    
    range_mvmt = [0] + range_mvmt
    
    
    for i in range(len(range_mvmt)-1):
        """
            (0,100) (100,200) ... 
        """
        
        # shuffled image in a single movement 
        local_img_idx = [i for i in range(range_mvmt[i],range_mvmt[i+1])]
        random.shuffle(local_img_idx)
        
        local_img_num = range_mvmt[i+1] - range_mvmt[i]
        local_train_num = math.ceil(local_img_num * config["ratio"])

        local_train_idx = local_img_idx[:local_train_num] 
        local_valid_idx = local_img_idx[local_train_num:] 
        
        #training   
        #while index < train_image_num:
        for index in local_train_idx:
            start_index = index
            end_index = index + 1

            train_set = {
                "AtomType": atom_type[image_index[start_index] : image_index[end_index]],
                "ImageDR": image_dR[image_index[start_index] : image_index[end_index]],
                "ListNeighbor": list_neigh[
                    image_index[start_index] : image_index[end_index]
                ],
                "Ei": Ei[image_index[start_index] : image_index[end_index]],
                #"Egroup": Egroup[image_index[start_index] : image_index[end_index]],
                #"Divider": divider[image_index[start_index] : image_index[end_index]],
                #"Egroup_weight": Egroup_weight[image_index[start_index] : image_index[end_index]],
                "Ri": Ri[image_index[start_index] : image_index[end_index]],
                "Ri_d": Ri_d[image_index[start_index] : image_index[end_index]],
                "Force": Force[image_index[start_index] : image_index[end_index]],
                "Virial": Virial[start_index:end_index],
                "ImageAtomNum": atom_num_per_image[start_index:end_index],
            }

            save_path = os.path.join(
                train_data_path, "image_" + str(accum_train_num).zfill(width_train)
            )

            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_npy_files(save_path, train_set)

            accum_train_num += 1 
            #index = end_index
        
        # valid
        #while index < image_num:
        for index in local_valid_idx:

            start_index = index
            end_index = index + 1

            valid_set = {
                "AtomType": atom_type[image_index[start_index] : image_index[end_index]],
                "ImageDR": image_dR[image_index[start_index] : image_index[end_index]],
                "ListNeighbor": list_neigh[
                    image_index[start_index] : image_index[end_index]
                ],
                "Ei": Ei[image_index[start_index] : image_index[end_index]],
                #"Egroup": Egroup[image_index[start_index] : image_index[end_index]],
                #"Divider": divider[image_index[start_index] : image_index[end_index]],
                #"Egroup_weight": Egroup_weight[image_index[start_index] : image_index[end_index]],
                "Ri": Ri[image_index[start_index] : image_index[end_index]],
                "Ri_d": Ri_d[image_index[start_index] : image_index[end_index]],
                "Force": Force[image_index[start_index] : image_index[end_index]],
                "Virial": Virial[start_index:end_index],
                "ImageAtomNum": atom_num_per_image[start_index:end_index],
            }

            save_path = os.path.join(
                valid_data_path, "image_" + str(accum_valid_num).zfill(width_valid)
            )

            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_npy_files(save_path, valid_set)

            accum_valid_num += 1
            #index = end_index   

    print("Saving npy file done")


def main():

    with open("config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Read Config successful")

    gen_train_data(config)
    sepper_data(config)


if __name__ == "__main__":
    main()
