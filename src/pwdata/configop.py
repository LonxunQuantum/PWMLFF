import os

from src.pwdata.poscar import POSCAR
from src.pwdata.atomconfig import CONFIG
from src.pwdata.build.supercells import make_supercell
from src.pwdata.pertub.perturbation import BatchPerturbStructure
from src.pwdata.pertub.scale import BatchScaleCell
from src.pwdata.pwdata import Configs
from src.pwdata.lammpsdata import LMP
from src.pwdata.dump import DUMP

def extract_config(config_path:str, format:str, pbc:list[int]=[1,1,1], atom_names: list[str] = None):
    if format.lower() == "config" or format.lower() == "pwmat":
        image = CONFIG(config_path, pbc).image_list[0]
    elif format.lower() == "poscar" or format.lower() == "vasp":
        image = POSCAR(config_path, pbc).image_list[0]
    elif format.lower() == "lmp" or format.lower() == "lammps":
        image = LMP(config_path).image_list[0]
    elif format.lower() == "dump":# for lammps single structure traj file
        image = DUMP(dump_file=config_path, atom_names=atom_names).image_list[0]
    else:
        raise ValueError("Invalid format")
    return image

def save_config(config, input_format:str = None, wrap = False, direct = True, sort = True, \
        save_format:str=None, save_path:str=None, save_name:str=None, atom_names: list[str] = None):
    if isinstance(config, str):
        config = extract_config(config_path=config, format=input_format, atom_names=atom_names)
        
    config.to(file_path=save_path, 
            file_name  =save_name, 
            file_format=save_format, 
            direct     =direct, 
            sort       =sort, 
            wrap       =wrap
            )
    return os.path.join(save_path, save_name)

def do_super_cell(config, input_format:str=None, supercell_matrix:list[int]=None, pbc:list[int]=[1, 1, 1], direct = True, sort = True, \
                    save_format:str=None, save_path:str=None, save_name:str=None):
    if isinstance(config, str):
        config = extract_config(config_path=config, format=input_format)
    # Make a supercell     
    supercell = make_supercell(config, supercell_matrix, pbc)
    # Write out the structure
    supercell.to(file_path = save_path,
                    file_name = save_name,
                    file_format = save_format,
                    direct = direct,
                    sort = sort)
    return os.path.join(save_path, save_name)

def do_scale(config, input_format:str=None, scale_factor:float=None, 
            direct:bool=True, sort:bool=True, save_format:str=None, save_path:str=None, save_name:str=None):
    if isinstance(config, str):
        config = extract_config(config_path=config, format=input_format)
    scaled_struct = BatchScaleCell.batch_scale(config, scale_factor)
    scaled_struct.to(file_path = save_path,
                    file_name = save_name,
                    file_format = save_format,
                    direct = direct,
                    sort = sort)
    return os.path.join(save_path, save_name)

def do_pertub(config, input_format:str=None, pert_num:int=None, cell_pert_fraction:float=None, atom_pert_distance:float=None, \
        direct:bool=True, sort:bool=True, save_format:str=None, save_path:str=None, save_name:str=None):
    if isinstance(config, str):
        config = extract_config(config_path=config, format=input_format)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    perturbed_structs = BatchPerturbStructure.batch_perturb(config, pert_num, cell_pert_fraction, atom_pert_distance)
    for tmp_perturbed_idx, tmp_pertubed_struct in enumerate(perturbed_structs):
        tmp_pertubed_struct.to(file_path = save_path,
                                file_name = "{}_{}".format(tmp_perturbed_idx, save_name),
                                file_format = save_format,
                                direct = direct,
                                sort = sort) 
        print("pertub {} done!".format(os.path.join(save_path, "{}_{}".format(tmp_perturbed_idx, save_name))))

'''
description: 
    if merge is ture, save pwdata to datasets_path/data_name ...
    else:
        save pwdata to datasets_path/data_name/train or valid
return {*}
author: wuxingxing
'''
def extract_pwdata(data_list:list[str], 
                data_format:str="movement", 
                datasets_path="PWdata", 
                train_valid_ratio:float=0.8, 
                data_shuffle:bool=True,
                merge_data:bool=False
                ):
    data_name = None
    if merge_data:
        data_name = os.path.basename(datasets_path)
        if not os.path.isabs(datasets_path):
            # data_name = datasets_path
            datasets_path = os.path.dirname(os.path.join(os.getcwd(), datasets_path))
        else:
            datasets_path = os.path.dirname(datasets_path)
        multi_data = []
        for data_path in data_list:
            image_data = Configs.read(data_format, data_path)
            multi_data += image_data
        get_all = Configs.get(multi_data)
        Configs.save(image_data_dict=get_all, 
                    datasets_path=datasets_path, 
                    train_data_path="train", 
                    valid_data_path="valid", 
                    train_ratio = train_valid_ratio, 
                    random=data_shuffle,
                    seed = 2024, 
                    retain_raw = False,
                    data_name = data_name
                    )
    else:
        for data_path in data_list:
            image_data = Configs.read(data_format, data_path)
            get_all = Configs.get(image_data)
            Configs.save(image_data_dict=get_all, 
                        datasets_path=datasets_path,
                        train_data_path="train", 
                        valid_data_path="valid", 
                        train_ratio = train_valid_ratio, 
                        random=data_shuffle,
                        seed = 2024, 
                        retain_raw = False
                    )
    
if __name__ == "__main__":
    # in_config = "/data/home/wuxingxing/datas/al_dir/si_5_pwmat/sys_config/structures/1.config"
    # save_path = "/data/home/wuxingxing/datas/al_dir/si_5_pwmat/sys_config/structures"
    # image = extract_config(config_path=in_config, format="pwmat")
    
    # save_config(image, wrap=False, direct=True, sort=True,\
    #     save_format="pwmat", save_path=save_path, save_name="temp_atom.config")
    # save_config(image, wrap=False, direct=True, sort=True,\
    #     save_format="vasp", save_path=save_path, save_name="temp_poscar")
    
    # do_super_cell(config=image,
    #         supercell_matrix=[[2,0,0], [0,2,0], [0,0,2]], 
    #         pbc=[1, 1, 1], 
    #         direct = True, 
    #         sort = True, \
    #         save_format="pwmat", save_path=save_path, save_name="temp_super_atom.config")
    
    # do_super_cell(config="/data/home/wuxingxing/datas/al_dir/si_5_pwmat/sys_config/structures/44_POSCAR",
    #     input_format="vasp",
    #     supercell_matrix=[[2,0,0], [0,2,0], [0,0,2]], 
    #     pbc=[1, 1, 1], 
    #     direct = True, 
    #     sort = True, \
    #     save_format="pwmat", save_path=save_path, save_name="temp_super_atom_from_poscar.config")
    
    
    # do_super_cell(config="/data/home/wuxingxing/datas/al_dir/si_5_pwmat/sys_config/structures/44_POSCAR",
    #     input_format="vasp",
    #     supercell_matrix=[[2,0,0], [0,2,0], [0,0,2]], 
    #     pbc=[1, 1, 1], 
    #     direct = True, 
    #     sort = True, \
    #     save_format="vasp", save_path=save_path, save_name="temp_super_poscar")

    # do_scale(config=image,
    #         scale_factor=0.99, 
    #         direct=True,
    #         sort=True, 
    #         save_format="pwmat", save_path=save_path, save_name="temp_scale_atom.config")

    # do_scale(config=image,
    #         scale_factor=0.99, 
    #         direct=True,
    #         sort=True, 
    #         save_format="vasp", save_path=save_path, save_name="temp_scale_poscar")
    
    # save_path = "/data/home/wuxingxing/datas/al_dir/si_5_pwmat/sys_config/structures/temp_pwmat_pertub"
    # do_pertub(config=image, 
    #     pert_num=50, 
    #     cell_pert_fraction=0.01, 
    #     atom_pert_distance=0.04, 
    #     direct=True,
    #     sort=True, 
    #     save_format="pwmat", save_path=save_path, save_name="pertub.config")
    
    # save_path = "/data/home/wuxingxing/datas/al_dir/si_5_pwmat/sys_config/structures/temp_vasp_pertub"
    # do_pertub(config=image, 
    #     pert_num=50, 
    #     cell_pert_fraction=0.01, 
    #     atom_pert_distance=0.04, 
    #     direct=True,
    #     sort=True, 
    #     save_format="vasp", save_path=save_path, save_name="pertub.poscar")

    # convert trajs to config for model_deviation calculate
    import glob
    traj_dir = "/data/home/wuxingxing/datas/al_dir/si_5_pwmat/iter.0000/temp_run_iter_work/explore/md/md.000.sys.000/md.000.sys.000.t.000/traj"
    save_dir = "/data/home/wuxingxing/datas/al_dir/si_5_pwmat/iter.0000/temp_run_iter_work/explore/md/md.000.sys.000/md.000.sys.000.t.000/traj2config"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    trajs = glob.glob(os.path.join(traj_dir, "*.lammpstrj"))
    trajs = sorted(trajs)
    for traj in trajs:
        save_name = "{}.config".format(os.path.basename(traj).split('.')[0])
        save_config(traj,  input_format="dump", wrap=False, direct=True, sort=True,\
        save_format="pwmat", save_path=save_dir, save_name=save_name)
        print("{} to {} done!".format(traj, save_name))