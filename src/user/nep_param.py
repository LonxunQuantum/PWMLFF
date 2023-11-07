import os
import shutil

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
        if nep.in file provided by user, just use it 
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
    def __init__(self, nep_dict, nep_in_file:str = None, type_list:list[int]=[], type_list_weight:list[float]=[], nep_save_file:str="nep.in") -> None:
        nep_file_dict = {}
        self.nep_in_file = nep_in_file
        if os.path.exists(nep_in_file):
            nep_file_dict = self.read_nep_param_from_nep_file(nep_in_file)
        
        self.version = self.get_parameter("version", nep_dict, 3, nep_file_dict, 0) # select between NEP2, NEP3, and NEP4
        
        self.type = self.get_parameter("type", nep_dict, type_list, nep_file_dict, 2) # number of atom types and list of chemical species
        self.type_weight = self.get_parameter("type_weight", nep_dict, type_list_weight, nep_file_dict, 3) # force weights for different atom types
        self.model_type = self.get_parameter("model_type", nep_dict, 0, nep_file_dict, 0) # select to train potential 0, dipole 1, or polarizability 2
        self.prediction = self.get_parameter("prediction", nep_dict, 0, nep_file_dict, 0) # select between training and prediction (inference)
        self.zbl = self.get_parameter("zbl", nep_dict, None, nep_file_dict, 1) # outer cutoff for the universal ZBL potential [Ziegler1985]
        self.cutoff = self.get_parameter("cutoff", nep_dict, [8, 4], nep_file_dict, 2) # radial () and angular () cutoffs
        self.n_max = self.get_parameter("n_max", nep_dict, [4, 4], nep_file_dict, 2) # size of radial () and angular () basis
        self.basis_size = self.get_parameter("basis_size", nep_dict, [12, 12], nep_file_dict, 2) # number of radial () and angular () basis functions
        self.l_max = self.get_parameter("l_max", nep_dict, [4, 2, 0], nep_file_dict, 2) # expansion order for angular terms
        self.neuron = self.get_parameter("neuron", nep_dict, 30, nep_file_dict, 0) # number of neurons in the hidden layer
        self.lambda_1 = self.get_parameter("lambda_1", nep_dict, -1, nep_file_dict, 1) # weight of regularization term
        self.lambda_2 = self.get_parameter("lambda_2", nep_dict, -1, nep_file_dict, 1) # weight of norm regularization term
        self.lambda_e = self.get_parameter("lambda_e", nep_dict, 1.0, nep_file_dict, 1) # weight of energy loss term
        self.lambda_f = self.get_parameter("lambda_f", nep_dict, 1.0, nep_file_dict, 1) # weight of force loss term
        self.lambda_v = self.get_parameter("lambda_v", nep_dict, 0.1, nep_file_dict, 1) # weight of virial loss term
        self.force_delta = self.get_parameter("force_delta", nep_dict, None, nep_file_dict, 1) # bias term that can be used to make smaller forces more accurate
        self.batch = self.get_parameter("batch", nep_dict, 1000, nep_file_dict, 0) # batch size for training
        self.population = self.get_parameter("population", nep_dict, 50, nep_file_dict, 0) # population size used in the SNES algorithm [Schaul2011]
        self.generation = self.get_parameter("generation", nep_dict, 100000, nep_file_dict, 0) # number of generations used by the SNES algorithm [Schaul2011]
        
        # if not os.path.exists(nep_in_file):
        #     self.to_nep_in_file(nep_save_file)
        # else:
        #     #copy nep.in file to work_dir
        #     shutil.copy(nep_in_file, nep_save_file)

    def read_nep_param_from_nep_file(self, nep_in_file:str):
        with open(nep_in_file, 'r') as rf:
            lines = rf.lines()
        nep_dict = {}
        for line in lines:
            substring = line.split("#")[0].strip()
            if len(substring.split()) > 1:
                key = substring.split()[0].lower()
                nep_dict[key] = substring.split()[1:]
        return nep_dict

    def to_nep_in_file(self, file_path:str):
        with open(file_path, 'w') as wf:
            self._write_to_line(wf, self.version, 0)
            self._write_to_line(wf, self.type, 2)
            self._write_to_line(wf, self.type_weight, 3)
            self._write_to_line(wf, self.model_type, 0)
            self._write_to_line(wf, self.prediction, 0)
            self._write_to_line(wf, self.zbl, 1)
            self._write_to_line(wf, self.cutoff, 2)
            self._write_to_line(wf, self.n_max, 2)
            self._write_to_line(wf, self.basis_size, 2)
            self._write_to_line(wf, self.l_max, 2)
            self._write_to_line(wf, self.neuron, 0)
            self._write_to_line(wf, self.lambda_1, 1)
            self._write_to_line(wf, self.lambda_2, 1)
            self._write_to_line(wf, self.lambda_e, 1)
            self._write_to_line(wf, self.lambda_f, 1)
            self._write_to_line(wf, self.lambda_v, 1)
            self._write_to_line(wf, self.force_delta, 1)
            self._write_to_line(wf, self.batch, 0)
            self._write_to_line(wf, self.population, 0)
            self._write_to_line(wf, self.generation, 0)
 
    '''
    description: 
    return line: "key value"
        example: "l_max         4 2 0"
    param {*} self
    param {*} wf
    param {*} key
    param {*} value
    param {str} data_type: int_value 0, float_value 1, int_list 2, float_list 3
    return {*}
    author: wuxingxing
    '''
    def _write_to_line(self, wf, key, value, data_type:str):
        line = ""
        if data_type == 0 or data_type == 1:
            line = "{} {}".format(key, value)
        elif data_type == 2 or data_type == 3:
            value_line = " ".join(value)
            line = "{} {}".format(key, value_line)
        else:
            raise Exception("ERROR {} data  type not supported!".format(data_type))
        wf.write(line)
        wf.write("\n")

    def to_dict(self):
        dicts = {}
        dicts["version"] = self.version
        dicts["type"] = self.type
        dicts["type_weight"] = self.type_weight
        dicts["model_type"] = self.model_type
        dicts["prediction"] = self.prediction
        dicts["zbl"] = self.zbl
        dicts["cutoff"] = self.cutoff
        dicts["n_max"] = self.n_max
        dicts["basis_size"] = self.basis_size
        dicts["l_max"] = self.l_max
        dicts["neuron"] = self.neuron
        dicts["lambda_1"] = self.lambda_1
        dicts["lambda_2"] = self.lambda_2
        dicts["lambda_e"] = self.lambda_e
        dicts["lambda_f"] = self.lambda_f
        dicts["lambda_v"] = self.lambda_v
        dicts["force_delta"] = self.force_delta
        dicts["batch"] = self.batch
        dicts["population"] = self.population
        dicts["generation"] = self.generation


    '''
    description: 
        if the value is boath in nep.in file and user input json file, the nep.in value will be used
    param {str} param
    param {dict} json_input
    param {*} default_value
    param {dict} nep_file_dict
    param {str} data_type: int_value 0, float_value 1, int_list 2, float_list 3
    return {*}
    author: wuxingxing
    '''
    def get_parameter(self, param:str, json_input:dict, default_value, nep_file_dict:dict={}, data_type:str="int_value"):
        if param.lower() in nep_file_dict.keys():
            if data_type == 0:
                return int(nep_file_dict[param])
            elif data_type == 1:
                return float(nep_file_dict[param])
            elif data_type == 2:
                return [int(_) for _ in nep_file_dict[param]]
            elif data_type == 3:
                return [float(_) for _ in nep_file_dict[param]]
        
        if param not in json_input.keys():
            return default_value
        else:
            return json_input[param]