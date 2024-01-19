import torch
import numpy as np

from src.PWMLFF.dp_network import dp_network
from src.user.input_param import InputParam
from src.pre_data.pwdata import Save_Data
from src.pre_data.dp_data_loader import type_map, find_neighbore

class Inference(object):
    def __init__(self, 
                 ckpt_file: str, 
                 device: torch.device) -> None:
        self.ckpt_file = ckpt_file
        self.device = device
        self.model = self.load_model(ckpt_file)

    def load_model(self, ckpt_file: str):
        model_checkpoint = torch.load(ckpt_file, map_location = torch.device("cpu"))
        stat = [model_checkpoint["davg"], model_checkpoint["dstd"], model_checkpoint["energy_shift"]]
        model_checkpoint["json_file"]["model_load_file"] = ckpt_file
        model_checkpoint["json_file"]["datasets_path"] = []
        dp_param = InputParam(model_checkpoint["json_file"], "train".upper())
        dp_param.inference = True 
        dp_trainer = dp_network(dp_param)
        model = dp_trainer.load_model_with_ckpt(davg=stat[0], dstd=stat[1], energy_shift=stat[2])
        model.load_state_dict(model_checkpoint["state_dict"])
        if "compress" in model_checkpoint.keys():
            model.set_comp_tab(model_checkpoint["compress"])
        model.to(self.device)
        model.eval()
        return model
            
    def inference(self, structrue_file, format="config"):
        model_config = self.model.config
        Ei = np.zeros(1)
        Egroup = 0
        nghost = 0
        list_neigh, type_maps, atom_types, ImageDR = self.processed_data(structrue_file, model_config, Ei, Egroup, format)
        Etot, Ei, Force, Egroup, Virial = self.model(list_neigh, type_maps, atom_types, ImageDR, nghost)
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
                                                              
    def processed_data(self, structrue_file, model_config, Ei, Egroup, format):
        infer = Save_Data(data_path=structrue_file, format=format)
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
    
if __name__ == "__main__":
    ckpt_file = "/data/home/hfhuang/2_MLFF/2-DP/19-json-version/4-CH4-dbg/model_record/dp_model.ckpt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inference = Inference(ckpt_file, device)
    inference.load_model(ckpt_file)
    structrues_file = "/data/home/hfhuang/2_MLFF/2-DP/19-json-version/4-CH4-dbg/atom.config"
    inference.inference(structrues_file)
