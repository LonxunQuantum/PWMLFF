import torch
import numpy as np
import os

from PWMLFF.dp_network import dp_network
from src.user.nep_work import tonepckpt
from PWMLFF.nep_network import nep_network
from user.input_param import InputParam
from pwdata import Save_Data
from pre_data.dp_data_loader import type_map, find_neighbore
from pwdata import Config
from utils.nep_to_gpumd import extract_model, get_atomic_number_from_name

class Inference(object):
    def __init__(self, 
                 ckpt_file: str, 
                 device: torch.device = None,
                 nep_txt:bool = False) -> None:
        self.ckpt_file = ckpt_file
        self.device = device
        self.model_atom_type = None
        if nep_txt is False: # the input model is dp_model.ckpt
            self.model, self.model_type, self.input_param = self.load_model(ckpt_file)
            self.model_atom_type = self.model.atom_type
        else: # load from nep.txt and nep.in
            try: # convert nep_model.ckpt to nep.txt
                _model = torch.load(ckpt_file, map_location=torch.device('cpu'))
                nep_content, self.model_atom_type, atom_names = extract_model(ckpt_file)
                self.ckpt_file = os.path.join(os.path.dirname(os.path.abspath(ckpt_file)), "tmp_pwmlff_nep.txt")
                with open(self.ckpt_file, 'w') as wf:
                    wf.write(nep_content)
            except Exception as e: # the input file is nep.txt, extract atom_type
                with open(ckpt_file, 'r') as rf:
                    line = rf.readline()
                if "nep" not in line:
                    raise Exception("Error parsing model file! {}".format(ckpt_file))
                atom_name = line.split()[2:]
                self.model_atom_type = get_atomic_number_from_name(atom_name)

            # self.model, self.model_type, self.input_param = tonepckpt([ckpt_file], save_ckpt=False)

    '''
    description: 
        load dp_model.ckpt
        the nep_model.ckpt does not use this function
    param {*} self
    param {str} ckpt_file
    return {*}
    author: wuxingxing
    '''
    def load_model(self, ckpt_file: str):
        model_checkpoint = torch.load(ckpt_file, map_location = torch.device("cpu"))
        model_checkpoint["json_file"]["model_load_file"] = ckpt_file
        model_checkpoint["json_file"]["datasets_path"] = []
        if "optimizer" not in model_checkpoint["json_file"].keys() or \
            model_checkpoint["json_file"]["optimizer"] is None:
            model_checkpoint["json_file"]["optimizer"] = {}
            model_checkpoint["json_file"]["optimizer"]["optimizer"] = "LKF"

        if model_checkpoint['json_file']['model_type'].upper() == "DP".upper():
            stat = [model_checkpoint["davg"], model_checkpoint["dstd"], model_checkpoint["energy_shift"]]
            dp_param = InputParam(model_checkpoint["json_file"], "train".upper())
            dp_param.inference = True 
            dp_trainer = dp_network(dp_param)
            model = dp_trainer.load_model_with_ckpt(davg=stat[0], dstd=stat[1], energy_shift=stat[2])
            model.load_state_dict(model_checkpoint["state_dict"])
            if "compress" in model_checkpoint.keys():
                model.set_comp_tab(model_checkpoint["compress"])

        # elif model_checkpoint['json_file']['model_type'].upper() == "NEP".upper():
        #     dp_param = InputParam(model_checkpoint["json_file"], "train".upper())
        #     nep_trainer = nep_network(dp_param)
        #     model, optimizer = nep_trainer.load_model_optimizer(model_checkpoint['energy_shift'])

        model.to(self.device)
        model.eval()
        return model, model_checkpoint['json_file']['model_type'].upper(), dp_param

    def inference(self, structrue_file, format="config", atom_names=None):
        model_config = self.model.config
        Ei = np.zeros(1)
        Egroup = 0
        nghost = 0
        # list_neigh, type_maps, atom_types, ImageDR = self.processed_data(structrue_file, model_config, Ei, Egroup, format, atom_names)
        # infer = Save_Data(data_path=structrue_file, format=format, atom_names=atom_names, train_ratio=1)

        image_read = Config(data_path=structrue_file, format=format, atom_names=atom_names).images
        if not isinstance(image_read, list): # for lammps/dumps or movement .images will be list
            image_read = [image_read]
        input_atom_types = np.array(self.model_atom_type)
        img_max_types = len(self.model_atom_type)

        for idx, image in enumerate(image_read):
            atom_types_struc = image.atom_types_image
            atom_types = image.atom_type
            ntypes = len(atom_types)
            # print("=========position==========\n")
            # print(image.position)
            # cart_postion = image.position
            if image.cartesian is True:
                image._set_fractional()
            # if image.cartesian is False:
            #     image._set_cartesian()
            # atom_nums = image.atom_nums
            if ntypes > img_max_types:
                raise Exception("Error! the atom types in structrue file is larger than the max atom types in model!")
            type_maps = np.array(type_map(atom_types_struc, input_atom_types)).reshape(1, -1)


            atom_types_struc = image.atom_types_image
            atom_types = image.atom_type
            ntypes = len(atom_types)
            position = np.array(image.position).reshape(1, -1, 3)
            natoms = position.shape[1]
            lattice = np.array(image.lattice).reshape(1, 3, 3)
            input_atom_types = np.array(self.model.atom_type)
            img_max_types = self.model.ntypes
            if ntypes > img_max_types:
                raise Exception("Error! the atom types in structrue file is larger than the max atom types in model!")
            m_neigh = self.model.maxNeighborNum
            Rc_M = self.model.Rmax
            Rc_type = np.array([(_['Rc']) for _ in model_config["atomType"]])
            Rm_type = np.array([(_['Rm']) for _ in model_config["atomType"]])
            type_maps = np.array(type_map(atom_types_struc, input_atom_types)).reshape(1, -1)
            list_neigh, dR_neigh, _, _, _, _ = find_neighbore(type_maps, position, lattice, natoms, Ei, 
                                                            img_max_types, Rc_type, Rm_type, m_neigh, Rc_M, Egroup)   
            
            list_neigh = self.to_tensor(list_neigh).unsqueeze(0)
            type_maps = self.to_tensor(type_maps).squeeze(0)
            atom_types = self.to_tensor(np.array(atom_types))
            ImageDR = self.to_tensor(dR_neigh).unsqueeze(0)

            Etot, Ei, Force, Egroup, Virial = self.model(list_neigh, type_maps, atom_types, ImageDR, nghost)
            Etot = Etot.cpu().detach().numpy()
            Ei = Ei.cpu().detach().numpy()
            Force = Force.cpu().detach().numpy()
            Virial = Virial.cpu().detach().numpy()
            try:
                Egroup = Egroup.cpu().detach().numpy()
            except:
                Egroup = None
            print("----------image   {}  -------".format(idx))
            print("----------Total Energy-------\n", Etot)
            print("----------Atomic Energy------\n", Ei)
            print("----------Force--------------\n", Force)
            print("----------Virial-------------\n", Virial)
            print("\n")
        # return Etot, Ei, Force, Egroup, Virial

    '''
    description: 
    not used, this function is replaced by Config when doing infer work
    param {*} self
    param {*} structrue_file
    param {*} model_config
    param {*} Ei
    param {*} Egroup
    param {*} format
    param {*} atom_names
    return {*}
    author: wuxingxing
    '''                                                
    def processed_data(self, structrue_file, model_config, Ei, Egroup, format, atom_names=None):
        infer = Save_Data(data_path=structrue_file, format=format, atom_names=atom_names)
        struc_num = 1
        if infer.image_nums != struc_num:
            raise Exception("Error! the image num in structrue file is not 1!")
        atom_types_struc = infer.atom_types_image
        atom_types = infer.atom_type[0]
        ntypes = len(atom_types)
        position = infer.position.reshape(struc_num, -1, 3)
        natoms = position.shape[1]
        lattice = infer.lattice.reshape(struc_num, 3, 3)
        input_atom_types = np.array(self.model.atom_type)
        img_max_types = self.model.ntypes
        if ntypes > img_max_types:
            raise Exception("Error! the atom types in structrue file is larger than the max atom types in model!")
        m_neigh = self.model.maxNeighborNum
        Rc_M = self.model.Rmax
        Rc_type = np.array([(_['Rc']) for _ in model_config["atomType"]])
        Rm_type = np.array([(_['Rm']) for _ in model_config["atomType"]])
        type_maps = np.array(type_map(atom_types_struc, input_atom_types)).reshape(1, -1)
        list_neigh, dR_neigh, _, _, _, _ = find_neighbore(type_maps, position, lattice, natoms, Ei, 
                                                          img_max_types, Rc_type, Rm_type, m_neigh, Rc_M, Egroup)   
        
        list_neigh = self.to_tensor(list_neigh).unsqueeze(0)
        type_maps = self.to_tensor(type_maps).squeeze(0)
        atom_types = self.to_tensor(atom_types)
        ImageDR = self.to_tensor(dR_neigh).unsqueeze(0)
        return list_neigh, type_maps, atom_types, ImageDR
    
    def to_tensor(self, data):
        data = torch.from_numpy(data).to(self.device)
        return data

    def inference_nep_txt(self, structrue_file, format="pwmat/config", atom_names=None):
        if torch.cuda.is_available():
            from src.feature.NEP_GPU import nep3_module
            calc = nep3_module.NEP3()
            calc.init_from_file(self.ckpt_file, is_rank_0=True, in_device_id=0)
        else:
            from src.feature.nep_find_neigh.findneigh import FindNeigh
            calc = FindNeigh()
            calc.init_model(self.ckpt_file)
        # infer = Save_Data(data_path=structrue_file, format=format)
        image_read = Config(data_path=structrue_file, format=format, atom_names=atom_names).images
        if not isinstance(image_read, list): # for lammps/dumps or movement .images will be list
            image_read = [image_read]
        input_atom_types = np.array(self.model_atom_type)
        img_max_types = len(self.model_atom_type)

        for idx, image in enumerate(image_read):
            atom_types_struc = image.atom_types_image
            atom_types = image.atom_type
            ntypes = len(atom_types)
            # print("=========position==========\n")
            # print(image.position)
            # cart_postion = image.position
            # if image.cartesian is True:
            #     image._set_fractional()
            if image.cartesian is False:
                image._set_cartesian()
            atom_nums = image.atom_nums

            if ntypes > img_max_types:
                raise Exception("Error! the atom types in structrue file is larger than the max atom types in model!")
            type_maps = np.array(type_map(atom_types_struc, input_atom_types)).reshape(1, -1)

            if torch.cuda.is_available():
                ei_predict = np.zeros(atom_nums, dtype=np.float64)
                force_predict = np.zeros(atom_nums*3, dtype=np.float64)
                virial_predict = np.zeros(9, dtype=np.float64)
                calc.compute_pwmlff(
                    atom_nums, 
                    ntypes*100, 
                    list(type_maps[0]), 
                    list(np.array(image.lattice).transpose(1, 0).reshape(-1)), 
                    list(np.array(image.position).transpose(1, 0).reshape(-1)), 
                    ei_predict, 
                    force_predict, 
                    virial_predict)
            else:
                ei_predict, force_predict, virial_predict = calc.inference(
                        list(type_maps[0]), 
                        list(np.array(image.lattice).transpose(1, 0).reshape(-1)), 
                        np.array(image.position).transpose(1, 0).reshape(-1)
                )

            ei_predict   = np.array(ei_predict).reshape(atom_nums)
            force_predict = np.array(force_predict).reshape(3, atom_nums).transpose(1, 0)
            virial_predict = np.array(virial_predict)
            etot_predict = np.sum(ei_predict)
            print("----------image   {}  -------".format(idx))
            print("----------Total Energy-------\n", etot_predict)
            print("----------Atomic Energy------\n", ei_predict)
            print("----------Force--------------\n", force_predict)
            print("----------Virial-------------\n", virial_predict)
            print("\n")

        if "tmp_pwmlff_nep" in self.ckpt_file:
            os.remove(self.ckpt_file)
        # return etot_predict, ei_predict, force_predict, None, virial_predict

    def inference_nep(self, structrue_file, format="pwmat/config", atom_names=None):
        Ei = np.zeros(1)
        Egroup = 0
        nghost = 0
        from src.feature.nep_find_neigh.findneigh import FindNeigh

        calc = FindNeigh()

        # infer = Save_Data(data_path=structrue_file, format=format)
        image_read = Config(data_path=structrue_file, format=format, atom_names=atom_names).images
        if isinstance(image_read, list): # for lammps/dumps or movement .images will be list
            image = image_read[0]
        else:
            image = image_read
        struc_num = 1
        atom_types_struc = image.atom_types_image
        atom_types = image.atom_type
        ntypes = len(atom_types)
        # print("=========position==========\n")
        # print(image.position)
        cart_postion = image.position
        if image.cartesian is True:
            image._set_fractional()
        atom_nums = image.atom_nums
        input_atom_types = np.array(self.model.atom_type)
        img_max_types = self.model.ntypes
        if ntypes > img_max_types:
            raise Exception("Error! the atom types in structrue file is larger than the max atom types in model!")
        type_maps = np.array(type_map(atom_types_struc, input_atom_types)).reshape(1, -1)
        # type_maps list 1 dim
        # Lattice [10.104840279, 0.0, -1.7274452448, 0.0, 10.28069973, 0.0, -5.1064545896e-16, 0.0, 10.275204659] 做转置后拉成一列
        # Position [96,3] 转置后拉成一列
        # 34622.19498329725 d12_radial
        d12_radial, d12_agular, NL_radial, NL_angular, NLT_radial, NLT_angular = calc.getNeigh(
                           self.input_param.descriptor.cutoff[0],self.input_param.descriptor.cutoff[1], 
                            len(self.input_param.atom_type)*self.input_param.max_neigh_num, list(type_maps[0]), list(np.array(image.lattice).transpose(1, 0).reshape(-1)), np.array(image.position).transpose(1, 0).reshape(-1))

        neigh_radial_rij   = np.array(d12_radial).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num, 4)
        neigh_angular_rij  = np.array(d12_agular).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num, 4)
        neigh_radial_list  = np.array(NL_radial).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num)
        neigh_angular_list = np.array(NL_angular).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num)
        neigh_radial_type_list  =  np.array(NLT_radial).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num)
        neigh_angular_type_list = np.array(NLT_angular).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num)

        neigh_radial_rij = self.to_tensor(neigh_radial_rij).unsqueeze(0)
        neigh_angular_rij = self.to_tensor(neigh_angular_rij).unsqueeze(0)

        neigh_radial_list = self.to_tensor(neigh_radial_list).unsqueeze(0)
        neigh_angular_list = self.to_tensor(neigh_angular_list).unsqueeze(0)

        neigh_radial_type_list = self.to_tensor(neigh_radial_type_list).unsqueeze(0)
        neigh_angular_type_list = self.to_tensor(neigh_angular_type_list).unsqueeze(0)
        
        type_maps = self.to_tensor(type_maps).squeeze(0)
        atom_types = self.to_tensor(np.array(atom_types))

        # Etot, Ei, Force, Egroup, Virial = self.model(neigh_radial_list, type_maps, atom_types, neigh_radial_rij, neigh_radial_type_list, nghost)

        Etot, Ei, Force, Egroup, Virial = self.model(
            neigh_radial_list, 
            neigh_radial_rij,
            neigh_radial_type_list,
            neigh_angular_list,
            neigh_angular_rij,
            neigh_angular_type_list,
            type_maps, 
            atom_types, 
            0
            )

        ### debug start
        # dR_neigh_txt = "/data/home/wuxingxing/datas/lammps_test/nep_hfo2_lmps/lmp_dug/dR_neigh.txt"
        # dR_neigh = np.loadtxt(dR_neigh_txt).reshape(atom_nums, len(self.input_param.atom_type) * self.input_param.max_neigh_num, 4)
        # imagetype_map_txt = "/data/home/wuxingxing/datas/lammps_test/nep_hfo2_lmps/lmp_dug/imagetype_map.txt"
        # imagetype_map = np.loadtxt(imagetype_map_txt, dtype=int).reshape(atom_nums)
        # neighbor_list_txt = "/data/home/wuxingxing/datas/lammps_test/nep_hfo2_lmps/lmp_dug/neighbor_list.txt"
        # neighbor_list = np.loadtxt(neighbor_list_txt, dtype=int).reshape(atom_nums, len(self.input_param.atom_type) * self.input_param.max_neigh_num)
        # neighbor_type_list_txt = "/data/home/wuxingxing/datas/lammps_test/nep_hfo2_lmps/lmp_dug/neighbor_type_list.txt"
        # neighbor_type_list = np.loadtxt(neighbor_type_list_txt, dtype=int).reshape(atom_nums, len(self.input_param.atom_type) * self.input_param.max_neigh_num)

        # neigh_radial_rij2 = self.to_tensor(dR_neigh).unsqueeze(0)
        # neigh_radial_list2 = self.to_tensor(neighbor_list).unsqueeze(0)
        # neigh_radial_type_list2 = self.to_tensor(neighbor_type_list).unsqueeze(0)
        # type_maps2 = self.to_tensor(imagetype_map).squeeze(0)
        # # atom_types = self.to_tensor(np.array(atom_types))

        # Etot2, Ei2, Force2, Egroup2, Virial2 = self.model(neigh_radial_list2, type_maps2, atom_types, neigh_radial_rij2, neigh_radial_type_list2, 1214)
        
        ### debug end
        Etot = Etot.cpu().detach().numpy()
        Ei = Ei.cpu().detach().numpy()
        Force = Force.cpu().detach().numpy()
        Virial = Virial.cpu().detach().numpy()
        try:
            Egroup = Egroup.cpu().detach().numpy()
        except:
            Egroup = None
        print("----------Total Energy-------\n", Etot)
        print("----------Atomic Energy------\n", Ei)
        print("----------Force--------------\n", Force)
        print("----------Virial-------------\n", Virial)
        return Etot, Ei, Force, Egroup, Virial    


