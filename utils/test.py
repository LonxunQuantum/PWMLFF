import torch
import numpy as np
def adjust_order_same_as_user_input():
    ckpt = torch.load('/data/home/wuxingxing/codespace/PWMLFF/pwmat_mlff_workdir/alloy/alloy/alloy_type_5/model_record/dp_model.ckpt',map_location=torch.device("cpu"))
    davg = ckpt['davg']
    dstd = ckpt['dstd']
    energy_shift = ckpt['energy_shift']
    atom_type_order = [44, 45, 77, 46, 28]
    atom_type_list = [28, 44, 45, 46, 77]
    davg_res, dstd_res, energy_shift_res = [], [], []
    for i, atom in enumerate(atom_type_list):
        davg_res.append(davg[atom_type_order.index(atom)])
        dstd_res.append(dstd[atom_type_order.index(atom)])
        energy_shift_res.append(energy_shift[atom_type_order.index(atom)])
    ckpt['davg'] = np.array(davg_res)
    ckpt['dstd'] = np.array(dstd_res)
    ckpt['energy_shift'] = np.array(energy_shift_res)
    ckpt["atom_type_order"] = np.array(atom_type_list)
    torch.save(ckpt, "/data/home/wuxingxing/codespace/PWMLFF/pwmat_mlff_workdir/alloy/alloy/alloy_type_5/model_record/dp_model_res.ckpt")
    return davg_res, dstd_res, energy_shift_res, atom_type_list

if __name__ == "__main__":
    adjust_order_same_as_user_input()