import os
from utils.atom_type_emb_dict import element_table
from utils.json_operation import get_parameter
'''
description: 
 this NEP params default value is the same as 
    https://gpumd.org/nep/input_files/nep_in.html#nep-in

return {*}
author: wuxingxing
'''
class NepParam(object):
    '''
    description: 
        if nep.in file provided by user, prioritize using it
        elif nep.in file not exisited, use params set in user input json file
        else use default params
    param {*} self
    param {*} nep_dict
    param {str} nep_in_file
    param {list} type_list
    param {list} type_list_weight
    param {str} nep_save_file
    return {*}
    author: wuxingxing
    '''    
    def __init__(self) -> None:
        pass

    '''
    description: 
        extract nep params from nep.in file
    param {*} self
    param {*} nep_in_file
    param {list} type_list
    return {*}
    author: wuxingxing
    '''
    def set_nep_param_from_nep_in(self, nep_in_file, type_list:list[int]=[]) -> None:
        nep_file_dict = {}
        self.nep_in_file = nep_in_file
        if nep_in_file is not None:
            if not os.path.exists(nep_in_file):
                raise Exception("ERROR! the nep.in file is not exist, please check the file: {}".format(nep_in_file))
            nep_file_dict = self.read_nep_param_from_nep_file(nep_in_file)
            
        self.version = get_parameter("version", nep_file_dict, 4) # select between NEP2, NEP3, and NEP4
        type_str = self.set_atom_type(type_list)
        self.type = get_parameter("type", nep_file_dict, type_str) # number of atom types and list of chemical species
        self.type_num = len(self.type.split()[1:])
        type_list_weight_default = [1.0 for _ in range(0, self.type_num)]
        self.type_weight = get_parameter("type_weight", nep_file_dict, type_list_weight_default) # force weights for different atom types
        self.model_type = get_parameter("model_type", nep_file_dict, 0) # select to train potential 0, dipole 1, or polarizability 2
        self.prediction = get_parameter("prediction", nep_file_dict, 0) # select between training and prediction (inference)
        self.zbl = get_parameter("zbl", nep_file_dict, None) # outer cutoff for the universal ZBL potential [Ziegler1985]
        self.cutoff = get_parameter("cutoff", nep_file_dict, [8, 6]) # radial () and angular () cutoffs # use dp rcut, default to 6
        self.n_max = get_parameter("n_max", nep_file_dict, [4, 4]) # size of radial () and angular () basis
        if len(self.n_max) != 2:
            raise Exception("the input 'n_max' should has 2 values, such as [4, 4]")
        self.basis_size = get_parameter("basis_size", nep_file_dict, [12, 12]) # number of radial () and angular () basis functions
        if len(self.basis_size) != 2:
            raise Exception("the input 'basis_size' should has 2 values, such as [12, 12]")
        self.l_max = get_parameter("l_max", nep_file_dict, [4, 2, 1]) # expansion order for angular terms
        if len(self.l_max) != 3 or (self.l_max[0] != 4) or (self.l_max[1] != 2) or (self.l_max[2] > 1) :
            error_log = "the input 'l_max' should has 3 values. The values should be [4, 2, 0] or [4, 2, 1]. The last num '1', means use 5 body features.\n"
            raise Exception(error_log)
        if self.l_max[2] != 0 and self.l_max[2] != 1:
            error_log = "the input 'l_max' should has 3 values. The values should be [4, 2, 0] or [4, 2, 1]. The last num '1', means use 5 body features, '0' means not use 5 body features\n"
            raise Exception(error_log)

        self.neuron = get_parameter("neuron", nep_file_dict, 100) # number of neurons in the hidden layer
        self.neuron = [self.neuron, 1] # 1 is the output layer of fitting net
        lambda_1 = get_parameter("lambda_1", nep_file_dict, -1) # weight of regularization term
        if lambda_1 != -1 and lambda_1 < 0:
            raise Exception("ERROR! the lambda_1 should >= 0 or lambda_1 = -1 for automatically determined in training!")
        else:
            self.lambda_1 = lambda_1

        lambda_2 = get_parameter("lambda_2", nep_file_dict, -1) # weight of norm regularization term
        if lambda_2 != -1 and lambda_2 < 0:
            raise Exception("ERROR! the lambda_2 should >= 0 or lambda_2 = -1 for automatically determined in training!")
        else:
            self.lambda_2 = lambda_2

        self.lambda_ei = get_parameter("lambda_ei", nep_file_dict, 1.0) # weight of energy loss term
        self.lambda_eg = get_parameter("lambda_eg", nep_file_dict, 0.1) # weight of energy loss term
        self.lambda_e = get_parameter("lambda_e", nep_file_dict, 1.0) # weight of energy loss term
        self.lambda_f = get_parameter("lambda_f", nep_file_dict, 1.0) # weight of force loss term
        self.lambda_v = get_parameter("lambda_v", nep_file_dict, 0.1) # weight of virial loss term
        self.force_delta = get_parameter("force_delta", nep_file_dict, None) # bias term that can be used to make smaller forces more accurate
        self.batch = get_parameter("batch", nep_file_dict, 1000) # batch size for training
        self.population = get_parameter("population", nep_file_dict, 50) # population size used in the SNES algorithm [Schaul2011]
        self.generation = get_parameter("generation", nep_file_dict, 100000) # number of generations used by the SNES algorithm [Schaul2011]
        self.set_feature_params()

    '''
    description: 
    extract nep params from input json file
    param {*} self
    param {dict} json_dict
    param {list} type_list
    return {*}
    author: wuxingxing
    '''
    def set_nep_param_from_json(self, json_dict:dict, type_list:list[int]=[]):
        model_dict = get_parameter("model", json_dict, {})
        descriptor_dict = get_parameter("descriptor", model_dict, {})
        # optimizer_dict = get_parameter("optimizer", json_dict, {})

        self.version = 4 # select between NEP2, NEP3, and NEP4
        type_str = self.set_atom_type(type_list)
        self.type = type_str # number of atom types and list of chemical species
        self.type_num = len(type_list)
        type_list_weight_default = [1.0 for _ in range(0, self.type_num)]
        self.type_weight = get_parameter("type_weight", descriptor_dict, type_list_weight_default) # force weights for different atom types
        self.model_type = 0 # select to train potential 0, dipole 1, or polarizability 2
        self.prediction = 0 # select between training and prediction (inference)
        self.zbl = None # outer cutoff for the universal ZBL potential [Ziegler1985]
        self.cutoff = get_parameter("cutoff", descriptor_dict, [6, 6]) # radial () and angular () cutoffs # use dp rcut, default to 6
        self.n_max = get_parameter("n_max", descriptor_dict, [4, 4]) # size of radial () and angular () basis
        if len(self.n_max) != 2:
            raise Exception("the input 'n_max' should has 2 values, such as [4, 4]")
        self.basis_size = get_parameter("basis_size", descriptor_dict, [12, 12]) # number of radial () and angular () basis functions
        if len(self.basis_size) != 2:
            raise Exception("the input 'basis_size' should has 2 values, such as [12, 12]")
        self.l_max = get_parameter("l_max", descriptor_dict, [4, 2, 0]) # expansion order for angular terms
        if len(self.l_max) != 3 or (self.l_max[0] != 4) or (self.l_max[1] != 2) or (self.l_max[2] > 1) :
            error_log = "the input 'l_max' should has 3 values. The values should be [4, 2, 0] or [4, 2, 1]. The last num '1', means use 5 body features.\n"
            raise Exception(error_log)
        if self.l_max[2] != 0 and self.l_max[2] != 1:
            error_log = "the input 'l_max' should has 3 values. The values should be [4, 2, 0] or [4, 2, 1]. The last num '1', means use 5 body features, '0' means not use 5 body features\n"
            raise Exception(error_log)
        if "fitting_net" in model_dict.keys():
            self.neuron = get_parameter("network_size", model_dict, [100]) # number of neurons in the hidden layer
            if not isinstance(self.neuron, list):
                self.neuron = [self.neuron]
        else:
            self.neuron = [100]
        if self.neuron[-1] != 1:
            self.neuron.append(1) # output layer of fitting net
        self.set_feature_params()

    def set_feature_params(self):
        # features
        self.two_feat_num = self.n_max[0]+1
        self.three_feat_num = (self.n_max[1]+1)*self.l_max[0] 
        self.four_feat_num = (self.n_max[1]+1) if (len(self.l_max) >= 2) else 0
        self.five_feat_num = (self.n_max[1]+1) if len(self.l_max) == 3 else 0
        self.feature_nums = self.two_feat_num + self.three_feat_num + self.four_feat_num + self.five_feat_num
        # c params, the 4-body and 5-body use the same c param of 3-body, their N_base_a the same
        self.two_c_num = self.type_num*self.type_num*(self.n_max[0]+1)*(self.basis_size[0]+1)
        self.three_c_num = self.type_num*self.type_num*(self.n_max[1]+1)*(self.basis_size[1]+1)
        self.c_num = self.two_c_num + self.three_c_num

        self.c2_param = None
        self.c3_param = None
        self.q_scaler = None
        self.model_wb = None

    '''
    read params from nep.txt
    description: 
    param {*} self
    param {*} file_path
    return {*}
    author: wuxingxing
    '''    
    def set_params_from_neptxt(self, file_path="nep.txt"):
        # self.c2_param = None
        # self.c3_param = None
        # self.q_scaler = None
        # self.model_wb = None
        pass
    
    def read_nep_param_from_nep_file(self, nep_in_file:str):
        with open(nep_in_file, 'r') as rf:
            lines = rf.readlines()
        nep_dict = {}
        for line in lines:
            substring = line.split("#")[0].strip()
            if len(substring.split()) > 1:
                key = substring.split()[0]
                value_str = substring.replace(key, "")
                nep_dict[key.lower()] = value_str.strip()
        return nep_dict

    # def to_dict(self):
    #     dicts = {}
    #     dicts["version"] = self.version
    #     dicts["type"] = self.type_num
    #     if self.type_weight is not None:
    #         dicts["type_weight"] = self.type_weight
    #     dicts["model_type"] = self.model_type
    #     dicts["prediction"] = self.prediction
    #     if self.zbl is not None:
    #         dicts["zbl"] = self.zbl
    #     dicts["cutoff"] = self.cutoff
    #     dicts["n_max"] = self.n_max
    #     dicts["basis_size"] = self.basis_size
    #     dicts["l_max"] = self.l_max
    #     dicts["neuron"] = self.neuron
    #     dicts["lambda_1"] = self.lambda_1
    #     dicts["lambda_2"] = self.lambda_2
    #     dicts["lambda_e"] = self.lambda_e
    #     dicts["lambda_f"] = self.lambda_f
    #     dicts["lambda_v"] = self.lambda_v
    #     if self.force_delta is not None:
    #         dicts["force_delta"] = self.force_delta
    #     dicts["batch"] = self.batch
    #     dicts["population"] = self.population
    #     dicts["generation"] = self.generation
    #     return dicts

    def to_nep_in_txt(self):
        content = ""
        content += "version     {}\n".format(self.version)
        content += "type        {}\n".format(self.type)
        # if self.type_weight is not None:
        #     content += "type_weight {}\n".format(self.type_weight)
        content += "model_type  {}\n".format(self.model_type)
        content += "prediction  {}\n".format(self.prediction)
        if self.zbl is not None: #' '.join(map(str, int_list))
            content += "zbl         {}\n".format(self.zbl)
        content += "cutoff      {}\n".format(" ".join(map(str, self.cutoff)))
        content += "n_max       {}\n".format(" ".join(map(str, self.n_max)))
        content += "basis_size  {}\n".format(" ".join(map(str, self.basis_size)))
        content += "l_max       {}\n".format(" ".join(map(str, self.l_max)))
        content += "neuron      {}\n".format(self.neuron[0]) # filter the output layer
        # these are from optimizer SNES
        # content += "lambda_1 {}\n".format(self.lambda_1)
        # content += "lambda_2 {}\n".format(self.lambda_2)
        # content += "lambda_e {}\n".format(self.lambda_e)
        # content += "lambda_f {}\n".format(self.lambda_f)
        # content += "lambda_v {}\n".format(self.lambda_v)
        # if self.force_delta is not None:
        #     content += "force_delta {}\n".format(self.force_delta)
        # content += "batch {}\n".format(self.batch)
        # content += "population {}\n".format(self.population)
        # content += "generation {}\n".format(self.generation)
        return content
    
    def to_nep_txt(self):
        content = ""
        content += "nep4   {}\n".format(self.type)    #line1
        content += "cutoff {}\n".format(" ".join(map(str, self.cutoff)))    #line2
        content += "n_max  {}\n".format(" ".join(map(str, self.n_max)))    #line3
        content += "basis_size {}\n".format(" ".join(map(str, self.basis_size)))    #line4
        content += "l_max  {}\n".format(" ".join(map(str, self.l_max)))    #line5
        content += "ANN    {} {}\n".format(self.neuron[0], 0)    #line6
        return content

    '''
    description: 
        the format of type in nep.in file is as:
            "type 2 Te Pb # this is a mandatory keyword"
    param {*} self
    param {*} atom_type_list
    return {*}
    author: wuxingxing
    '''    
    def set_atom_type(self, atom_type_list):
        atom_names = []
        for atom in atom_type_list:
            atom_names.append(element_table[atom])
        return str(len(atom_type_list)) + " " + " ".join(atom_names)