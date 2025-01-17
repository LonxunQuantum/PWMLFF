import numpy as np
import matplotlib.pyplot as plt
import os

def inference_plot(data_dir:str):
    size = 20
    num_atom = np.loadtxt(os.path.join(data_dir, "image_atom_nums.txt"))
    dft_E   = np.loadtxt(os.path.join(data_dir, "dft_total_energy.txt")) / num_atom
    MLFF_E  = np.loadtxt(os.path.join(data_dir, "inference_total_energy.txt")) / num_atom
    rmse_E = np.sqrt(np.square(dft_E - MLFF_E).mean())
    E_min = min(dft_E.min(), MLFF_E.min())
    E_max = max(dft_E.max(), MLFF_E.max())

    dft_F   = np.loadtxt(os.path.join(data_dir, "dft_force.txt")).flatten()
    MLFF_F  = np.loadtxt(os.path.join(data_dir, "inference_force.txt")).flatten()
    rmse_F = np.sqrt(np.square(dft_F - MLFF_F).mean())
    F_min = min(dft_F.min(), MLFF_F.min())
    F_max = max(dft_F.max(), MLFF_F.max())

    if os.path.exists(os.path.join(data_dir, "dft_virial.txt")):
        _dft_V  = np.loadtxt(os.path.join(data_dir, "dft_virial.txt"))
        _MLFF_V = np.loadtxt(os.path.join(data_dir, "inference_virial.txt"))

        filtered_indices = _dft_V > -1e6
        
        dft_V = _dft_V[filtered_indices].reshape(-1, 6)
        MLFF_V= _MLFF_V[filtered_indices].reshape(-1, 6)

        if len(dft_V) > 0:
            atom_idx = np.repeat(num_atom[filtered_indices[:,0]][:, np.newaxis], 6, axis=1)
            dft_V = dft_V/atom_idx
            MLFF_V = MLFF_V/atom_idx
            rmse_V = np.sqrt(np.square(dft_V - MLFF_V).mean())
            V_min = min(dft_V.min(), MLFF_V.min())
            V_max = max(dft_V.max(), MLFF_V.max())
        else:
            rmse_V = None
            dft_V = []
    else:
        rmse_V = None
        dft_V = []
    plt.plot(dft_E.flatten(), MLFF_E.flatten(),"o",markersize=3,c="C0")
    plt.plot([E_min,E_max],[E_min,E_max],"--",lw=1.2,c="C1")
    plt.axis([E_min*1.02,E_max*0.98,E_min*1.02,E_max*0.98])
    s = "RMSE of Energy is %.1f meV/atom" % (rmse_E * 1e3)
    ax = plt.gca()
    plt.text(.15,.05,s,fontsize=size-2,transform=ax.transAxes)
    plt.xticks(size=size-4)
    plt.yticks(size=size-4)
    plt.xlabel("DFT Energy (eV/atom)",size=size)
    plt.ylabel("MLFF Energy (eV/atom)",size=size)
    #plt.title(title,size=size+4)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "Energy.png"), dpi=360)
    plt.close()

    plt.plot(dft_F, MLFF_F,"o",markersize=3,c="C0")
    plt.plot([F_min,F_max],[F_min,F_max],"--",lw=1.2,c="C1")
    plt.axis([F_min+1.02,F_max*1.02,F_min*1.02,F_max*1.02])
    s = r"RMSE of Force is %.3f eV/$\mathrm{\AA}$" % rmse_F
    ax = plt.gca()
    plt.text(.15,.05,s,fontsize=size,transform=ax.transAxes)
    plt.xticks(size=size-4)
    plt.yticks(size=size-4)
    plt.xlabel(r"DFT Force (eV/$\mathrm{\AA}$)",size=size)
    plt.ylabel(r"MLFF Force (eV/$\mathrm{\AA}$)",size=size)
    #plt.title(title,size=size+4)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "Force.png"), dpi=360)
    plt.close()

    if rmse_V is not None and len(dft_V) > 0 :
        plt.plot(dft_V, MLFF_V,"o",markersize=3,c="C0")
        plt.plot([V_min,V_max],[V_min,V_max],"--",lw=1.2,c="C1")
        plt.axis([V_min+1.02,V_max*1.02,V_min*1.02,V_max*1.02])
        s = r"RMSE of Virial is %.3f eV/atom" % rmse_V
        ax = plt.gca()
        plt.text(.15,.05,s,fontsize=size,transform=ax.transAxes)
        plt.xticks(size=size-4)
        plt.yticks(size=size-4)
        plt.xlabel(r"DFT Virial (eV)",size=size)
        plt.ylabel(r"MLFF Virial (eV)",size=size)
        #plt.title(title,size=size+4)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "Virial.png"), dpi=360)
        plt.close()

    return rmse_E, rmse_F, rmse_V
if __name__=="__main__":
    inference_plot("/data/home/wuxingxing/codespace/PWMLFF_grad_opt/src/test/test_workdir/2_dev_gpu_NN_LKF/work_test_dir/test_result")