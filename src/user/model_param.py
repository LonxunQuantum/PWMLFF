import os
import json
from numpy import Inf
from src.user.file_structure import TrainFileStructure, ModelParam, NetParam, OptimizerParam
from utils.json_operation import get_parameter, get_required_parameter

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
        self.recover_train = get_parameter("recover_train", json_input, False)
        model_dict = get_parameter("model", json_input, {})
        self.model_param = self._set_model_params(model_dict)

        # set optimizer
        opt_dict = get_parameter("optimizer", json_input, {})
        self.optimizer_param = self._set_optimizer(opt_dict)

        # set file structures
        self.file_paths = self._set_file_paths(json_input)
        
        self._set_default_params(json_input)

        self.model_type = get_parameter("model_type", json_input, "DP")
        # set feature related params
        self.E_tolerance = get_parameter("E_tolerance", json_input, 9999999.0)
        self.Rmax = get_parameter("Rmax", json_input, 6.0) 
        self.Rmin = get_parameter("Rmin", json_input, 0.5) 
        self.M2 = get_parameter("M2", json_input, 16)
        self.data_shuffle = get_parameter("data_shuffle", json_input, True)
        self.train_valid_ratio = get_parameter("train_valid_ratio", json_input, 0.8)

        
        # required params
        self.atom_type = get_required_parameter("atom_type", json_input)
        self.atom_type_dict = []
        for idx in self.atom_type:
            #the last 4 param used in function gen_config_inputfile()
            single_dict = {'type': idx, 'Rc': self.Rmax, 'Rm': self.Rmin, 'iflag_grid': 3, 'fact_base': 0.2, 'dR1': 0.5, "b_init":166.0}    
            self.atom_type_dict.append(single_dict)
        self.max_neigh_num = get_required_parameter("max_neigh_num", json_input)

        self.precision = get_parameter("precision", json_input, "float64")

        self.profiling = get_parameter("profiling", json_input, False)

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

        # set cmd prams
        self.cmd = cmd
        self._set_cmd_relative_params(json_input)

    def _set_cmd_relative_params(self, json_input:dict):
        if self.cmd == "train".upper():
            self.inference = False

        elif self.cmd == "gen_feat".upper():
            self.inference = False

        elif self.cmd == "test".upper():
            self.inference = True
            self.optimizer_param.batch_size = 1     # set batch size to 1, so that each image inference info will be saved
            self.train_valid_ratio = 1
            self.data_shuffle = False
            test_movement_path = get_required_parameter("test_movement_path", json_input)
            if isinstance(test_movement_path, list) is False:
                test_movement_path = [test_movement_path]
            for mvm in test_movement_path:
                if os.path.exists(mvm) is False:
                    raise Exception("{} file is not exists, please check!".format(mvm))
            test_dir_name = get_parameter("test_dir_name", json_input, "test_result")
            if os.path.isabs(test_dir_name) is False:
                test_dir = os.path.join(self.file_paths.work_dir, test_dir_name)
            else:
                test_dir = test_dir_name
            self.file_paths._set_inference_paths(test_movement_path, test_dir)
        else:
            error_info = "error! The command {} does not exist and currently only includes the following commands:\
                train\t gen_feat\t inference\n".format(self.cmd)
            raise Exception(error_info) 

    def _set_file_paths(self, json_input:dict):
        json_dir = os.getcwd()
        work_dir = get_parameter("work_dir", json_input, None)
        file_paths = TrainFileStructure(json_dir=json_dir, work_dir=work_dir)
        # model paths
        model_load_path = get_parameter("model_load_path", json_input, "")
        # if self.recover_train is True and os.path.isfile(model_load_path):
        #     raise Exception("Error! The recover_train and model_load_path are simultaneously specified, please set recover_train to False or remove param model_load_path")
        
        model_name = get_parameter("model_name", json_input, "checkpoint.pth.tar")
        model_store_dir = get_parameter("model_store_dir", json_input, "model_record")
        model_store_dir = os.path.join(file_paths.work_dir, model_store_dir)
        file_paths._set_model_paths(model_store_dir = model_store_dir, model_load_path=model_load_path, \
                                    model_name = model_name, best_model_path=os.path.join(work_dir, "best.pth.tar"))

        # set trian movement file path
        train_movement_path = get_parameter("train_movement_path", json_input, [])
        for mvm in train_movement_path:
            if os.path.exists(mvm) is False:
                raise Exception("Error! train movement: {} file not exist!".format(mvm))
        # set train feature path
        train_feature_path = get_parameter("train_feature_path", json_input, [])
        for feat_path in train_feature_path:
            if os.path.exists(feat_path) is False:
                raise Exception("Error! train movement: {} file not exist!".format(feat_path)) 
        file_paths._set_training_path(train_movement_path=train_movement_path, 
                                      train_feature_path=train_feature_path,
                                      train_dir=os.path.join(file_paths.work_dir, "feature"))
        # set Pwdata dir file structure, they are used in feature generation
        trainSetDir = get_parameter("trainSetDir", json_input, 'PWdata')
        dRFeatureInputDir = get_parameter("dRFeatureInputDir", json_input, 'input')
        dRFeatureOutputDir = get_parameter("dRFeatureOutputDir", json_input, 'output')
        trainDataPath = get_parameter("trainDataPath", json_input, 'train')
        validDataPath = get_parameter("validDataPath", json_input, 'valid')
        file_paths._set_data_file_paths(trainSetDir, dRFeatureInputDir, dRFeatureOutputDir, trainDataPath, validDataPath)
        
        # set 
        forcefield_name = get_parameter("forcefield_name", json_input, "forcefield_ff")
        forcefield_dir = get_parameter("forcefield_dir", json_input, "forcefield")
        file_paths.set_forcefield_path(forcefield_dir, forcefield_name)
        
        # p matix, resume p matrix when recover is not realized
        # p matrix should extract to checkpoint files or a single file.
        # current not realized
        save_p_matrix = get_parameter("save_p_matrix", json_input, False)
        if save_p_matrix is not False:
            Pmatrix_path = os.path.join(file_paths.work_dir, "P.pkl")
            file_paths._set_p_matrix_paths(Pmatrix_path, True)
        else:
            file_paths._set_p_matrix_paths(None, False)
        return file_paths
    
    def _set_model_params(self, json_input:dict):
        model_param = ModelParam()
    
        # set dp embedding net params
        embedding_json = get_parameter("embedding_net",json_input, {})
        network_size = get_parameter("network_size", embedding_json, [25, 25, 25])
        bias = get_parameter("bias", embedding_json, True)
        resnet_dt = get_parameter("resnet_dt", embedding_json, True) # resnet in embedding net is true.
        activation = get_parameter("activation", embedding_json, "tanh")
        embedding_net = NetParam("embedding_net")
        embedding_net.set_params(network_size, bias, resnet_dt, activation)

        # set dp fitting net params
        fitting_net_dict = get_parameter("fitting_net",json_input, {})
        network_size = get_parameter("network_size", fitting_net_dict, [50, 50, 50, 1])
        bias = get_parameter("bias", fitting_net_dict, True)
        resnet_dt = get_parameter("resnet_dt", fitting_net_dict, False)
        activation = get_parameter("activation", fitting_net_dict, "tanh")
        fitting_net = NetParam("fitting_net")
        fitting_net.set_params(network_size, bias, resnet_dt, activation)

        model_param.set_net(fitting_net=fitting_net, embedding_net=embedding_net)
        return model_param

    def _set_optimizer(self, json_input:dict):
        optimizer_type = get_parameter("optimizer", json_input, "LKF")
        batch_size = get_parameter("batch_size", json_input, 1)
        epochs = get_parameter("epochs", json_input, 30)
        print_freq = get_parameter("print_freq", json_input, 10)
        # the start epoch could be reset at the resume model code block
        start_epoch = get_parameter("start_epoch", json_input, 1)
        best_loss = get_parameter("best_loss", json_input, 1e10)    #may be not need
        min_loss = Inf
        optimizer_param = OptimizerParam(optimizer_type, start_epoch=start_epoch, epochs=epochs, batch_size=batch_size, \
                                         print_freq=print_freq, best_loss=best_loss, min_loss=min_loss)

        if "KF" in optimizer_param.opt_name.upper():  #set Kalman Filter Optimizer params
            kalman_lambda = get_parameter("kalman_lambda", json_input, 0.98)
            kalman_nue = get_parameter("kalman_nue", json_input, 0.9987)
            block_size = get_parameter("block_size", json_input, 5120)
            nselect = get_parameter("nselect", json_input, 24)
            groupsize = get_parameter("groupsize", json_input, 6)
            optimizer_param.set_kf_params(block_size, kalman_lambda, kalman_nue, nselect, groupsize)
        else:   # set ADAM Optimizer params
            learning_rate = get_parameter("learning_rate", json_input, 0.001)
            weight_decay = get_parameter("weight_decay", json_input, 1e-4)
            momentum = get_parameter("momentum", json_input, 0.9)
            optimizer_param.set_adam_sgd_params(learning_rate, weight_decay, momentum)
        
        train_energy = get_parameter("train_energy", json_input, True) 
        train_force = get_parameter("train_force", json_input, True) 
        train_ei = get_parameter("train_ei", json_input, False) 
        train_virial = get_parameter("train_virial", json_input, False) 
        train_egroup = get_parameter("train_egroup", json_input, False) 
        pre_fac_force = get_parameter("pre_fac_force", json_input, 2.0) 
        pre_fac_etot = get_parameter("pre_fac_etot", json_input, 1.0) 
        pre_fac_ei = get_parameter("pre_fac_ei", json_input, 1.0) 
        pre_fac_virial = get_parameter("pre_fac_virial", json_input, 1.0) 
        pre_fac_egroup = get_parameter("pre_fac_egroup", json_input, 0.1) 
        
        optimizer_param.set_train_pref(train_energy = train_energy, train_force = train_force, 
            train_ei = train_ei, train_virial = train_virial, train_egroup = train_egroup, 
                pre_fac_force = pre_fac_force, pre_fac_etot = pre_fac_etot, 
                    pre_fac_ei = pre_fac_ei, pre_fac_virial = pre_fac_virial, pre_fac_egroup = pre_fac_egroup
                )
        return optimizer_param

    '''
    description: these params use default value, but also could be set by user.
    param {*} self
    param {dict} json_input
    return {*}
    author: wuxingxing
    '''    
    def _set_default_params(self, json_input:dict):

        self.dwidth = 3.0
        self.seed = get_parameter("seed", json_input, None)


    def _to_dict(self):
        pass

    def get_dp_net_dict(self):
        net_dict = {}
        net_dict["net_cfg"] = {}
        net_dict["net_cfg"][self.model_param.embedding_net.net_type] = self.model_param.embedding_net.to_dict()
        net_dict["net_cfg"][self.model_param.fitting_net.net_type] = self.model_param.fitting_net.to_dict()
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
        params_dict["model_type"] = self.model_type
        params_dict["atom_type"] = self.atom_type
        params_dict["max_neigh_num"] = self.max_neigh_num
       
        params_dict["E_tolerance"] = self.E_tolerance
        params_dict["Rmax"] = self.Rmax
        params_dict["Rmin"] = self.Rmin
        params_dict["M2"] = self.M2
        params_dict["data_shuffle"] = self.data_shuffle
        params_dict["train_valid_ratio"] = self.train_valid_ratio

        params_dict["precision"] = self.precision

        # params_dict["profiling"] = self.profiling
        # params_dict["workers"] = self.workers
        # params_dict["hvd"] = self.hvd
        # params_dict["world_size"] = self.world_size
        # params_dict["rank"] = self.rank
        # params_dict["dist_url"] = self.dist_url
        # params_dict["dist_backend"] = self.dist_backend
        # params_dict["distributed"] = self.distributed
        
        params_dict["train_valid_ratio"] = self.train_valid_ratio
        params_dict["data_shuffle"] = self.data_shuffle
        params_dict["dwidth"] = self.dwidth

        if self.cmd == "train".upper():
            params_dict["recover_train"] = self.recover_train
        params_dict["inference"] = self.inference

        params_dict["model"] = self.model_param.to_dict()
        params_dict["optimizer"] = self.optimizer_param.to_dict()
        
        file_dir_dict = self.file_paths.to_dict()
        for key in file_dir_dict:
            params_dict[key] = file_dir_dict[key]
        return params_dict
    
    def print_input_params(self):
        params_dict = self.to_dict()
        json.dump(params_dict, open(os.path.join(self.file_paths.json_dir, "json_all.json"), "w"), indent=4)
        print(params_dict)
        
def help_info():
    print("dp_train: do dp model training")
    print("dp_test: do dp model inference")
    print("dp_gen_feat: generate feature for dp model")
 