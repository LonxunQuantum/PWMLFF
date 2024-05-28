import torch
from src.mods.infer import Inference
import os 
import glob
import numpy as np
def infer_main(ckpt_file, structures_file, format= "config", atom_typs=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    infer = Inference(ckpt_file, device)
    if infer.model_type == "DP":
        infer.inference(structures_file, format, atom_typs)
    elif infer.model_type == "NEP":
        Etot, Ei, Force, Egroup, Virial = infer.inference_nep(structures_file, format, atom_typs)
        
def model_devi(ckpt_file_list, structure_dir, format, save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if os.path.isdir(structure_dir):
        traj_list = glob.glob(os.path.join(structure_dir, "*"))
        trajs = sorted(traj_list, key = lambda x : int(os.path.basename(x).split(".")[0]))
    else:
        trajs = [structure_dir]
    force_list = []
    Etot_list = []
    Ei_list = []
    for mi, model in enumerate(ckpt_file_list):
        force_i = []
        ei_i = []
        Etot_i = []
        infer = Inference(model, device)
        for ti, traj in enumerate(trajs):
            Etot, Ei, Force, Egroup, Virial = infer.inference(traj, format)
            force_i.append(Force)
            ei_i.append(Ei)
            Etot_i.append(Etot)
        force_list.append(force_i)
        Etot_list.append(Etot_i)
        Ei_list.append(ei_i)
    # calculate model deviation
    print()
    ei = np.squeeze(np.array(Ei_list))
    force = np.squeeze(np.array(force_list))
    etot = np.squeeze(np.array(Etot_list))

    avg_force = np.mean(force, axis=0)
    # max_force = np.full([len(ckpt_file_list), avg_force.shape[0]], 0)
    max_force = []
    for i in range(0, len(ckpt_file_list)):
        tmp_error_1  = force[i]-avg_force
        tmp_error_2  = np.square(tmp_error_1)
        tmp_error_3  = np.sqrt(np.sum(tmp_error_2, axis=-1))
        max_force.append(np.max(tmp_error_3, axis=-1))
    d_max_force = np.array(max_force)
    res_devi_foce = np.max(d_max_force, axis=0)
    avg_ei = np.mean(ei, axis=0)
    # max_ei = np.full([len(ckpt_file_list), avg_ei.shape[0]], 0)
    max_ei = []
    for i in range(0, len(ckpt_file_list)):
        tmp_error_1 = ei[i]-avg_ei
        tmp_error_2 = np.abs(tmp_error_1)
        max_ei.append(np.max(tmp_error_2, axis=-1))
    d_max_ei = np.array(max_ei)
    res_devi_ei = np.max(d_max_ei, axis=0)
    print("model deviation of Ei:")
    print(res_devi_ei)
    print("model deviation of Force:")
    print(res_devi_foce)
