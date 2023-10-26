import os
import json
from utils.json_operation import get_parameter, get_required_parameter
from src.user.nn_feature_type import Descriptor
from src.user.model_param import ModelParam
from src.user.optimizer_param import OptimizerParam
from src.user.work_file_param import WorkFileStructure
'''
description: 
    covert the input json file to class
return {*}
author: wuxingxing
'''
class InputParam(object):
    '''
    description: 
        covert the input json file to class
    param {*} self
    param {dict} json_input: the input json file from user
    return {*}
    author: wuxingxing
    '''
    def __init__(self, json_input:dict, cmd) -> None:
        self.cmd = cmd
        self.inference = True if self.cmd == "test".upper() else False
        self.model_type = get_required_parameter("model_type", json_input).upper()
        # if feature_type specified at first layer of JSON, use this value as feature type in descriptor
        self.feature_type = get_parameter("feature_type", json_input, None) 
        self.model_num = get_parameter("model_num", json_input, 1)
        # set fitting net, embeding net
        self.recover_train = get_parameter("recover_train", json_input, False)

        self.type_embedding = get_parameter("type_embedding",json_input, False)
        model_dict = get_parameter("model", json_input, {})
        self.descriptor = Descriptor(get_parameter("descriptor", model_dict, {}), self.model_type, self.cmd, self.feature_type, self.type_embedding)
        self.type_embedding = self.descriptor.type_embedding
        if self.model_type == "DP":
            self.model_param = ModelParam()
            self.model_param.set_type_embedding_net(
                                                    network_size=self.descriptor.type_network_size, 
                                                    bias=self.descriptor.type_bias, 
                                                    resnet_dt=self.descriptor.type_resnet_dt, 
                                                    activation=self.descriptor.type_activation,
                                                    physical_property=self.descriptor.type_physical_property)
            self.model_param.set_embedding_net(network_size=self.descriptor.network_size, 
                                               bias=self.descriptor.bias, 
                                               resnet_dt=self.descriptor.resnet_dt, 
                                               activation=self.descriptor.activation)
            self.model_param.set_dp_fitting_net(get_parameter("fitting_net",model_dict, {}))
        elif self.model_type == "NN":
            self.model_param = ModelParam()
            self.model_param.set_type_embedding_net(
                                        network_size=self.descriptor.type_network_size, 
                                        bias=self.descriptor.type_bias, 
                                        resnet_dt=self.descriptor.type_resnet_dt, 
                                        activation=self.descriptor.type_activation,
                                        physical_property=self.descriptor.type_physical_property)
            self.model_param.set_nn_fitting_net(get_parameter("fitting_net",model_dict, {}))
            self.is_dfeat_sparse = get_parameter("is_dfeat_sparse", json_input, False)  #'true' not realized
        elif self.model_type == "Linear".upper():
            pass
        else: # linear
            raise Exception("model_type {} not realized yet".format(self.model_type))

        # set optimizer
        self.optimizer_param = OptimizerParam()
        self.optimizer_param.set_optimizer(get_parameter("optimizer", json_input, {}))

        # set file structures
        self.file_paths = WorkFileStructure(json_dir=os.getcwd(), 
                                       work_dir=os.path.abspath(get_parameter("work_dir", json_input, "work_dir")), 
                                       reserve_work_dir=get_parameter("reserve_work_dir", json_input, False), 
                                       reserve_feature = get_parameter("reserve_feature", json_input, False), 
                                       model_type=self.model_type, 
                                       cmd=self.cmd)
        self.file_paths.set_file_paths(json_input)

        # set feature related params
        self.valid_shuffle = get_parameter("valid_shuffle", json_input, False)
        self.data_shuffle = get_parameter("data_shuffle", json_input, True)
        self.train_valid_ratio = get_parameter("train_valid_ratio", json_input, 0.8)
        self.dwidth = get_parameter("dwidth", json_input, 3.0)
        self.seed = get_parameter("seed", json_input, None)
        self.precision = get_parameter("precision", json_input, "float64")

        # required params
        self.atom_type = get_required_parameter("atom_type", json_input)
        self.atom_type_dict = []
        for idx in self.atom_type:
            #the last 4 param used in function gen_config_inputfile()
            single_dict = {'type': idx, 'Rc': self.descriptor.Rmax, 'Rm': self.descriptor.Rmin, 'iflag_grid': 3, 'fact_base': 0.2, 'dR1': 0.5, "b_init":166.0}    
            self.atom_type_dict.append(single_dict)
        self.max_neigh_num = get_parameter("max_neigh_num", json_input, 100)


        self.profiling = get_parameter("profiling", json_input, False)#not realized

        # multi GPU train params  these params not used
        # "number of data loading workers (default: 4)
        self.workers = get_parameter("workers", json_input, 1)
        # dist training by horovod, when multi GPU train, need True,
        self.hvd = get_parameter("hvd", json_input, False)
        self.world_size = get_parameter("world_size", json_input, -1)
        self.rank = get_parameter("rank", json_input, -1)
        self.dist_url = get_parameter("dist_url", json_input, "tcp://localhost:23456")  
        self.dist_backend = get_parameter("dist_backend", json_input, "nccl")
        self.distributed = get_parameter("distributed", json_input, False)
        self.gpu = get_parameter("gpu", json_input, None)

    def set_test_relative_params(self, json_input:dict):
        self.inference = True
        self.optimizer_param.batch_size = 1     # set batch size to 1, so that each image inference info will be saved
        self.train_valid_ratio = 1
        self.data_shuffle = False
        self.file_paths.set_inference_paths(json_input)

    def get_dp_net_dict(self):
        net_dict = {}
        net_dict["net_cfg"] = {}
        net_dict["net_cfg"][self.model_param.embedding_net.net_type] = self.model_param.embedding_net.to_dict()
        net_dict["net_cfg"][self.model_param.fitting_net.net_type] = self.model_param.fitting_net.to_dict()
        net_dict["net_cfg"][self.model_param.type_embedding_net.net_type] = self.model_param.type_embedding_net.to_dict()
        net_dict["M2"] = self.descriptor.M2
        net_dict["maxNeighborNum"] = self.max_neigh_num
        net_dict["atomType"]=self.atom_type_dict
        net_dict["training_type"] = self.precision
        net_dict["Rc_M"] = self.descriptor.Rmax
        return net_dict
    
    def get_data_file_dict(self):
        data_file_dict = self.file_paths.get_data_file_structure()
        data_file_dict["M2"] = self.descriptor.M2
        data_file_dict["maxNeighborNum"] = self.max_neigh_num
        data_file_dict["atomType"]=self.atom_type_dict
        data_file_dict["Rc_M"] = self.descriptor.Rmax
        data_file_dict["E_tolerance"] = self.descriptor.E_tolerance
        # data_file_dict["dwidth"] = self.dwidth
        if self.optimizer_param.train_egroup:           # setting calculate flag for egroup (Fortran code)
            data_file_dict["gen_egroup_input"] = 1      
        else:
            data_file_dict["gen_egroup_input"] = 0
        data_file_dict["ratio"] = self.train_valid_ratio

        return data_file_dict
    
    def to_dict(self):
        params_dict = {}
        params_dict["model_type"] = self.model_type
        params_dict["atom_type"] = self.atom_type
        params_dict["max_neigh_num"] = self.max_neigh_num
        if self.seed is not None:
            params_dict["seed"] = self.seed
        params_dict["model_num"] = self.model_num
        # params_dict["E_tolerance"] = self.descriptor.E_tolerance
        # params_dict["Rmax"] = self.descriptor.Rmax
        # params_dict["Rmin"] = self.Rmin
        # params_dict["M2"] = self.descriptor.M2
        # params_dict["data_shuffle"] = self.data_shuffle
        if self.cmd == "train".upper():
            params_dict["train_valid_ratio"] = self.train_valid_ratio
            
        # params_dict["dwidth"] = self.dwidth
        # params_dict["precision"] = self.precision

        # params_dict["profiling"] = self.profiling
        # params_dict["workers"] = self.workers
        # params_dict["hvd"] = self.hvd
        # params_dict["world_size"] = self.world_size
        # params_dict["rank"] = self.rank
        # params_dict["dist_url"] = self.dist_url
        # params_dict["dist_backend"] = self.dist_backend
        # params_dict["distributed"] = self.distributed

        # params_dict["inference"] = self.inference
        if self.cmd == "train".upper() and self.model_type != "Linear".upper():
            params_dict["recover_train"] = self.recover_train

        params_dict["model"] = {}
        params_dict["model"]["descriptor"] = self.descriptor.to_dict()

        if self.model_type == "Linear".upper():
            params_dict["optimizer"] = self.optimizer_param.to_linear_dict()
        else:
            params_dict["model"]["fitting_net"] = self.model_param.fitting_net.to_dict_std()
            params_dict["optimizer"] = self.optimizer_param.to_dict()
        
        file_dir_dict = self.file_paths.to_dict()
        for key in file_dir_dict:
            params_dict[key] = file_dir_dict[key]
        return params_dict
    
    def print_input_params(self, json_file_save_name="json_all.json"):
        params_dict = self.to_dict()
        json.dump(params_dict, open(os.path.join(self.file_paths.json_dir, json_file_save_name), "w"), indent=4)
        print(params_dict)
        
def help_info():
    print("train: do model training")
    print("test: do dp model inference")
    