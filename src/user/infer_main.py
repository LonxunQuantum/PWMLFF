import torch
from src.mods.infer import Inference
import os 
import glob
import numpy as np
def infer_main(sys_cmd:list[str]):
    ckpt_file = sys_cmd[0]
    use_nep_txt = False
    sys_index = 0
    nep_in_txt = None
    if "nep.txt" in ckpt_file:
        use_nep_txt= True
    if use_nep_txt:
        nep_in_txt = sys_cmd[1]
        structures_file = sys_cmd[2]
        format = sys_cmd[3] if len(sys_cmd) > 3 else "pwmat/config"
        sys_index = 3
    else:
        structures_file = sys_cmd[1]
        format = sys_cmd[2] if len(sys_cmd) > 2 else "pwmat/config"
        sys_index = 2
    if format is not None and format.lower() == "lammps/dump":
        atom_typs = sys_cmd[sys_index+1:]
        if isinstance(atom_typs, list) is False:
            atom_typs = [atom_typs]
        print("Structure atom type is ", atom_typs)
    else:
        atom_typs = None
    
    if use_nep_txt is False:
        model_checkpoint = torch.load(ckpt_file, map_location = torch.device("cpu"))
        model_type = model_checkpoint['json_file']['model_type'].upper()
    else:
        model_type = "NEP"

    if model_type == "DP":
        device = torch.device("cpu")
        if torch.cuda.is_available():
            print("Warnning! Modify the GPU device to CPU for the DP infer interface!")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    infer = Inference(ckpt_file, device, nep_in_txt)
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
