import os
import json
from numpy import Inf
class TrainFileStructure(object):
    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir
        self.movement_name = "MOVEMENT"

    def _set_data_path_list(self, feature_path:list):
        self.feature_path = feature_path

    def _set_data_file_paths(self, trainSetDir:str, dRFeatureInputDir:str, dRFeatureOutputDir:str,\
                        trainDataPath:str, validDataPath:str):
        self.trainSetDir = trainSetDir
        self.dRFeatureInputDir = dRFeatureInputDir
        self.dRFeatureOutputDir = dRFeatureOutputDir
        self.trainDataPath = trainDataPath
        self.validDataPath = validDataPath

    def _set_p_matrix_paths(self, p_path, save_p_matrix:bool):
        self.save_p_matrix = save_p_matrix
        self.p_matrix_path = p_path

    def _set_model_paths(self, model_dir:str, model_name:str, model_path:str, best_model_path:str):
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_path = model_path
        self.best_model_path = best_model_path
    
    def _set_inference_paths(self, test_movement_path:list, test_dir: str):
        self.test_dir = test_dir
        self.test_movement_path = test_movement_path

    '''
    description: set feature paths
        the feature path list could from user input.json, \
            the feature path list could auto set after generating feature
    param {*} self
    param {list} feature_path
    return {*}
    author: wuxingxing
    '''
    def set_feature_path(self, feature_path:list):
        self.feature_path = feature_path

    def get_file_path(self, index:int, file_name):
        return os.path.join(self.data_path_list[index], file_name)
    
    def get_data_file_structure(self):
        file_dict = {}
        file_dict["trainSetDir"] = self.trainSetDir
        file_dict["dRFeatureInputDir"] = self.dRFeatureInputDir
        file_dict["dRFeatureOutputDir"] = self.dRFeatureOutputDir
        file_dict["trainDataPath"] = self.trainDataPath
        file_dict["validDataPath"] = self.validDataPath
        return file_dict
    
class ModelParam(object):
    def __init__(self, net_type:str) -> None:
        # set default parameters
        self.net_type = net_type

    def set_params(self, network_size: list, bias:bool, resnet_dt:bool, activation:str):
        self.network_size = network_size
        self.bias = bias
        self.resnet_dt = resnet_dt
        self.activation=activation      
        
    def to_dict(self):
        return \
            {
                    "network_size": self.network_size, 
                    "bias": self.bias, 
                    "resnet_dt": self. resnet_dt, 
                    "activation": self.activation
            }
        

class OptimizerParam(object):
    def __init__(self, optimizer:str) -> None:
        self.opt_name = optimizer.upper()
        
    def set_kf_params(self, block_size:int, kalman_lambda:float, kalman_nue:float, \
                   nselect:int, groupsize:int):
        self.block_size = block_size
        self.kalman_lambda = kalman_lambda
        self.kalman_nue = kalman_nue
        self.nselect = nselect
        self.groupsize = groupsize

    def set_adam_sgd_params(self, learning_rate:float, weight_decay:float, momentum:float):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

    def to_dict(self):
        opt_dict = {}
        opt_dict["optimizer"]=self.opt_name
        if "KF" in self.opt_name:
            opt_dict["block_size"]= self.block_size 
            opt_dict["kalman_lambda"]= self.kalman_lambda
            opt_dict["kalman_nue"]= self.kalman_nue
        else:
            opt_dict["learning_rate"]= self.learning_rate
            opt_dict["weight_decay"]= self.weight_decay
            opt_dict["momentum"]= self.momentum
        return opt_dict

'''
description: 
    covert the input json file to class
return {*}
author: wuxingxing
'''
class DpParam(object):
    '''
    description: 
        covert the input json file to class
    param {*} self
    param {dict} json_input: the input json file from user
    return {*}
    author: wuxingxing
    '''
    def __init__(self, json_input:dict, cmd) -> None:
        # set fitting net, embeding net
        model_dict = self._get_parameters("model", json_input, {})
        self.embedding_net, self.fitting_net = self._set_dp_net(model_dict)
        #resume the latest checkpoint model
        self.resume = self._get_parameters("resume", json_input, False)
        opt_dict = self._get_parameters("optimizer", json_input, {})
        self.optimizer_param = self._set_optimizer(opt_dict)

        # set file structures
        self.file_paths = self._set_file_paths(json_input)
        
        self._set_default_params(json_input)

        # set feature related params
        self.E_tolerance = self._get_parameters("E_tolerance", json_input, 9999999.0)
        self.Rmax = self._get_parameters("Rmax", json_input, 6.0) 
        self.Rmin = self._get_parameters("Rmin", json_input, 0.5) 
        self.M2 = self._get_parameters("M2", json_input, 16)
        self.data_shuffle = self._get_parameters("data_shuffle", json_input, True)
        self.train_valid_ratio = self._get_parameters("train_valid_ratio", json_input, 0.8)
        self.batch_size = self._get_parameters("batch_size", json_input, 4)
        self.epochs = self._get_parameters("epochs", json_input, 40)
        # the start epoch could be reset at the resume model code block
        self.start_epoch = self._get_parameters("start_epoch", json_input, 1)
        self.print_freq = self._get_parameters("print_freq", json_input, 10)
        
        # required params
        self.atom_type = self._get_required_parameters("atom_type", json_input)
        self.atom_type_dict = []
        for idx in self.atom_type:
            #the last 4 param used in function gen_config_inputfile()
            single_dict = {'type': idx, 'Rc': self.Rmax, 'Rm': self.Rmin, 'iflag_grid': 3, 'fact_base': 0.2, 'dR1': 0.5, "b_init":166.0}    
            self.atom_type_dict.append(single_dict)
        self.max_neigh_num = self._get_required_parameters("max_neigh_num", json_input)

        # set train params
        self.train_energy = self._get_parameters("train_energy", json_input, True) 
        self.train_force = self._get_parameters("train_force", json_input, True) 
        self.train_ei = self._get_parameters("train_ei", json_input, False) 
        self.train_virial = self._get_parameters("train_virial", json_input, False) 
        self.train_egroup = self._get_parameters("train_egroup", json_input, False) 

        self.pre_fac_force = self._get_parameters("pre_fac_force", json_input, 2.0) 
        self.pre_fac_etot = self._get_parameters("pre_fac_etot", json_input, 1.0) 
        self.pre_fac_ei = self._get_parameters("pre_fac_ei", json_input, 1.0) 
        self.pre_fac_virial = self._get_parameters("pre_fac_virial", json_input, 1.0) 
        self.pre_fac_egroup = self._get_parameters("pre_fac_egroup", json_input, 0.1) 

        self.precision = self._get_parameters("precision", json_input, "float64")

        self.profiling = self._get_parameters("profiling", json_input, False)

        # multi GPU train params  these params not used
        # "number of data loading workers (default: 4)
        self.workers = self._get_parameters("workers", json_input, 1)
        # dist training by horovod, when multi GPU train, need True,
        self.hvd = self._get_parameters("hvd", json_input, False)
        self.world_size = self._get_parameters("world_size", json_input, -1)
        self.rank = self._get_parameters("rank", json_input, -1)
        self.dist_url = self._get_parameters("dist_url", json_input, "tcp://localhost:23456")  
        self.dist_backend = self._get_parameters("dist_backend", json_input, "nccl")
        self.distributed = self._get_parameters("distributed", json_input, False)
        self.gpu = self._get_parameters("gpu", json_input, None)

        # set cmd prams
        self.cmd = cmd
        self._set_cmd_relative_params(json_input)

    def _set_cmd_relative_params(self, json_input:dict):
        if self.cmd == "dp_train".upper():
            self.inference = False

        elif self.cmd == "dp_gen_feat".upper():
            self.inference = False

        elif self.cmd == "dp_test".upper():
            self.inference = True
            self.resume = True   #load model from specified model path
            self.batch_size = 1     # set batch size to 1, so that each image inference info will be saved
            self.train_valid_ratio = 1
            self.data_shuffle = False
            test_movement_path = self._get_required_parameters("test_movement_path", json_input)
            if isinstance(test_movement_path, list) is False:
                test_movement_path = [test_movement_path]
            for mvm in test_movement_path:
                if os.path.exists(mvm) is False:
                    raise Exception("{} file is not exists, please check!".format(mvm))
            test_dir_name = self._get_parameters("test_dir_name", json_input, "test_result")
            if os.path.isabs(test_dir_name) is False:
                test_dir = os.path.join(self.file_paths.root_dir, test_dir_name)
            else:
                test_dir = test_dir_name
            self.file_paths._set_inference_paths(test_movement_path, test_dir)
        else:
            error_info = "error! The command {} does not exist and currently only includes the following commands:\
                dp_train\tdp_gen_feat\tdp_inference\n".format(self.cmd)
            raise Exception(error_info) 
            
            

    def _set_file_paths(self, json_input:dict):
        root_path = self._get_parameters("work_dir", json_input, os.getcwd())
        file_paths = TrainFileStructure(root_path)
        # model paths
        model_path = self._get_parameters("model_path", json_input, "./model_record")
        if os.path.isfile(model_path):
            model_name = os.path.basename(model_path)
            model_dir = os.path.dirname(model_path)
        else:
            model_name = self._get_parameters("model_name", json_input, "checkpoint.pth.tar")
            model_dir = model_path
            model_path = os.path.join(model_dir, model_name)
        file_paths._set_model_paths(model_dir, model_name, model_path, os.path.join(model_dir, "best.pth.tar"))

        self.store_path = model_dir

        # data_paths how about data path list ? which means multi data sources
        data_path = self._get_parameters("data_path", json_input, ["./"])
        # data files under data path
        file_paths._set_data_path_list(data_path)

        trainSetDir = self._get_parameters("trainSetDir", json_input, 'PWdata')
        dRFeatureInputDir = self._get_parameters("dRFeatureInputDir", json_input, 'input')
        dRFeatureOutputDir = self._get_parameters("dRFeatureOutputDir", json_input, 'output')
        trainDataPath = self._get_parameters("trainDataPath", json_input, 'train')
        validDataPath = self._get_parameters("validDataPath", json_input, 'valid')
        file_paths._set_data_file_paths(trainSetDir, dRFeatureInputDir, dRFeatureOutputDir, trainDataPath, validDataPath)
        
        # p matix, resume p matrix when recover is not realized
        save_p_matrix = self._get_parameters("save_p_matrix", json_input, False)
        if save_p_matrix is not False:
            Pmatrix_path = os.path.join(self.store_path, "P.pt")
            file_paths._set_p_matrix_paths(Pmatrix_path, True)
        else:
            file_paths._set_p_matrix_paths(None, False)

        return file_paths
    
    def _set_dp_net(self, json_input:dict):
        # set dp embedding net params
        embedding_json = self._get_parameters("embedding_net",json_input, {})
        network_size = self._get_parameters("network_size", embedding_json, [25, 25, 25])
        bias = self._get_parameters("bias", embedding_json, True)
        resnet_dt = self._get_parameters("resnet_dt", embedding_json, False)
        activation = self._get_parameters("activation", embedding_json, "tanh")
        embedding_net = ModelParam("embedding_net")
        embedding_net.set_params(network_size, bias, resnet_dt, activation)

        # set dp fitting net params
        fitting_net_dict = self._get_parameters("fitting_net",json_input, {})
        network_size = self._get_parameters("network_size", fitting_net_dict, [50, 50, 50, 1])
        bias = self._get_parameters("bias", fitting_net_dict, True)
        resnet_dt = self._get_parameters("resnet_dt", fitting_net_dict, False)
        activation = self._get_parameters("activation", fitting_net_dict, "tanh")
        fitting_net = ModelParam("fitting_net")
        fitting_net.set_params(network_size, bias, resnet_dt, activation)
        return embedding_net, fitting_net

    def _set_optimizer(self, json_input:dict):
        optimizer_type = self._get_parameters("optimizer", json_input, "LKF")
        optimizer = OptimizerParam(optimizer_type)
        if "KF" in optimizer_type.upper():  #set Kalman Filter Optimizer params
            kalman_lambda = self._get_parameters("kalman_lambda", json_input, 0.98)
            kalman_nue = self._get_parameters("kalman_nue", json_input, 0.9987)
            block_size = self._get_parameters("block_size", json_input, 5120)
            nselect = self._get_parameters("nselect", json_input, 24)
            groupsize = self._get_parameters("groupsize", json_input, 6)
            optimizer.set_kf_params(block_size, kalman_lambda, kalman_nue, nselect, groupsize)
        else:   # set ADAM Optimizer params
            learning_rate = self._get_parameters("learning_rate", json_input, 0.001)
            weight_decay = self._get_parameters("weight_decay", json_input, 1e-4)
            momentum = self._get_parameters("momentum", json_input, 0.9)
            optimizer.set_adam_sgd_params(learning_rate, weight_decay, momentum)
        return optimizer

    '''
    description: these params use default value, but also could be set by user.
    param {*} self
    param {dict} json_input
    return {*}
    author: wuxingxing
    '''    
    def _set_default_params(self, json_input:dict):
        self.best_loss = self._get_parameters("best_loss", json_input, 1e10)    #may be not need
        self.min_loss = Inf
        self.dwidth = 3.0
        self.seed = self._get_parameters("seed", json_input, None)

    '''
    description: 
        get value of param in json_input which is required parameters which need input by user
            if the parameter is not specified in json_input, raise error and print error log to user.
    param {*} self
    param {str} param
    param {dict} json_input
    param {str} info 
    return {*}
    author: wuxingxing
    '''
    def _get_required_parameters(self, param:str, json_input:dict):
        if param not in json_input.keys():
            raise Exception("Input error! : The {} parameter is missing and must be specified in input json file!".format(param))
        return json_input[param]

    '''
    description: 
        get value of param in json_input,
            if the parameter is not specified in json_input, return the default parameter value 'default_value'
    param {*} self
    param {str} param
    param {dict} json_input
    param {*} default_value
    return {*}
    author: wuxingxing
    '''
    def _get_parameters(self, param:str, json_input:dict, default_value):
        if param not in json_input.keys():
            return default_value
        else:
            return json_input[param]
    
    def _to_dict(self):
        pass

    def get_dp_net_dict(self):
        net_dict = {}
        net_dict["net_cfg"] = {}
        net_dict["net_cfg"][self.fitting_net.net_type]=self.fitting_net.to_dict()
        net_dict["net_cfg"][self.embedding_net.net_type]=self.embedding_net.to_dict()
        net_dict["M2"] = self.M2
        net_dict["maxNeighborNum"] = self.max_neigh_num
        net_dict["atomType"]=self.atom_type_dict
        net_dict["training_type"] = self.precision
        net_dict["Rc_M"] = self.Rmax
        return net_dict
    
    def get_data_file_dict(self):
        data_file_dict = self.file_paths.get_data_file_structure()
        data_file_dict["M2"] = self.M2
        data_file_dict["maxNeighborNum"] = self.max_neigh_num
        data_file_dict["atomType"]=self.atom_type_dict
        data_file_dict["Rc_M"] = self.Rmax
        data_file_dict["E_tolerance"] = self.E_tolerance
        data_file_dict["dwidth"] = self.dwidth
        data_file_dict["ratio"] = self.train_valid_ratio

        return data_file_dict
    
    def to_dict(self):
        params_dict = {}
        params_dict["atom_type"] = self.atom_type
        # params_dict["atom_type_dict"] = self.atom_type_dict
        params_dict["max_neigh_num"] = self.max_neigh_num

        params_dict["model"] = {}
        params_dict["resume"] = self.resume
        params_dict["model"]["embedding_net"] = self.embedding_net.to_dict()
        params_dict["model"]["fitting_net"] = self.fitting_net.to_dict()
        params_dict["optimizer"] = self.optimizer_param.to_dict()
        
        params_dict["E_tolerance"] = self.E_tolerance
        params_dict["Rmax"] = self.Rmax
        params_dict["Rmin"] = self.Rmin
        params_dict["M2"] = self.M2
        params_dict["data_shuffle"] = self.data_shuffle
        params_dict["train_valid_ratio"] = self.train_valid_ratio

        params_dict["batch_size"] = self.batch_size
        params_dict["epochs"] = self.epochs
        params_dict["start_epoch"] = self.start_epoch
        params_dict["print_freq"] = self.print_freq

        params_dict["train_energy"] = self.train_energy
        params_dict["train_force"] = self.train_force
        params_dict["train_ei"] = self.train_ei
        params_dict["train_virial"] = self.train_virial
        params_dict["train_egroup"] = self.train_egroup

        params_dict["pre_fac_force"] = self.pre_fac_force
        params_dict["pre_fac_etot"] = self.pre_fac_etot
        params_dict["pre_fac_ei"] = self.pre_fac_ei
        params_dict["pre_fac_virial"] = self.pre_fac_virial
        params_dict["pre_fac_egroup"] = self.pre_fac_egroup

        params_dict["precision"] = self.precision

        # params_dict["profiling"] = self.profiling
        # params_dict["workers"] = self.workers
        # params_dict["hvd"] = self.hvd
        # params_dict["world_size"] = self.world_size
        # params_dict["rank"] = self.rank
        # params_dict["dist_url"] = self.dist_url
        # params_dict["dist_backend"] = self.dist_backend
        # params_dict["distributed"] = self.distributed
        
        params_dict["inference"] = self.inference
        params_dict["resume"] = self.resume
        params_dict["batch_size"] = self.batch_size
        params_dict["train_valid_ratio"] = self.train_valid_ratio
        params_dict["data_shuffle"] = self.data_shuffle

        params_dict["best_loss"] = self.best_loss
        params_dict["min_loss"] = self.min_loss
        params_dict["dwidth"] = self.dwidth

        return params_dict
    
    def print_input_params(self):
        params_dict = self.to_dict()
        json_dict = json.dumps(params_dict, indent=4)
        print(json_dict)
        
def help_info():
    print("dp_train: do dp model training")
    print("dp_test: do dp model inference")
    print("dp_gen_feat: generate feature for dp model")
 