import os
import json
from numpy import Inf
from utils.json_operation import get_parameter, get_required_parameter
from src.user.nn_feature_type import Descriptor
from src.user.param_structure import TrainFileStructure, ModelParam, NetParam, OptimizerParam

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
        self.cmd = cmd
        self.model_type = get_required_parameter("model_type", json_input).upper()
        self.model_num = get_parameter("model_num", json_input, 1)
        # set fitting net, embeding net
        self.recover_train = get_parameter("recover_train", json_input, False)
        model_dict = get_parameter("model", json_input, {})
        self.descriptor = Descriptor(get_parameter("descriptor", model_dict, {}), self.model_type, self.cmd)

        if self.model_type == "DP":
            self.model_param = self._set_dp_model_params(model_dict)
        elif self.model_type == "NN":
            self.model_param = self._set_nn_model_params(model_dict)
            self._set_nn_personal_params(json_input)
        elif self.model_type == "Linear".upper():
            pass
        else: # linear
            raise Exception("model_type {} not realized yet".format(self.model_type))

        # set optimizer
        opt_dict = get_parameter("optimizer", json_input, {})
        self.optimizer_param = self._set_optimizer(opt_dict)

        # set file structures
        self.file_paths = self._set_file_paths(json_input)

        # set feature related params

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
        self.max_neigh_num = get_required_parameter("max_neigh_num", json_input)


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

        # set cmd prams
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
            test_movement_path = get_required_parameter("test_movement_file", json_input)
            if isinstance(test_movement_path, list) is False:
                test_movement_path = [test_movement_path]
            for mvm in test_movement_path:
                if os.path.exists(mvm) is False:
                    raise Exception("{} file is not exists, please check!".format(mvm))
            test_dir_name = get_parameter("test_dir_name", json_input, "test_result")
            test_dir = os.path.join(self.file_paths.work_dir, test_dir_name)

            test_feature_path = get_parameter("test_feature_path", json_input, [])
            for feat_path in test_feature_path:
                if os.path.exists(feat_path) is False:
                    raise Exception("Error! test_feature_path {} does not exist!".format(feat_path))
            test_feature_path = [os.path.abspath(_) for _ in test_feature_path]
            test_movement_path = [os.path.abspath(_) for _ in test_movement_path]

            if os.path.exists(self.file_paths.model_load_path) is False:
                raise Exception("the model_load_path is not exist: {}".format(self.file_paths.model_load_path))
            
            self.file_paths._set_inference_paths(test_movement_path, test_dir)
        else:
            error_info = "error! The command {} does not exist and currently only includes the following commands:\
                train\t gen_feat\t inference\n".format(self.cmd)
            raise Exception(error_info) 

    def _set_file_paths(self, json_input:dict):
        json_dir = os.getcwd()
        work_dir = os.path.abspath(get_parameter("work_dir", json_input, "work_dir"))
        reserve_work_dir = get_parameter("reserve_work_dir", json_input, False)
        file_paths = TrainFileStructure(json_dir=json_dir, work_dir=work_dir, reserve_work_dir=reserve_work_dir, model_type=self.model_type, cmd=self.cmd)
        # model paths
        model_load_path = os.path.abspath(get_parameter("model_load_file", json_input, ""))
        # if self.recover_train is True and os.path.isfile(model_load_path):
        #     raise Exception("Error! The recover_train and model_load_path are simultaneously specified, please set recover_train to False or remove param model_load_path")
        if self.model_type == "NN":
            model_name = get_parameter("model_name", json_input, "nn_model.ckpt")
        else:
            model_name = get_parameter("model_name", json_input, "dp_model.ckpt")
        
        model_store_dir = get_parameter("model_store_dir", json_input, "model_record")
        model_store_dir = os.path.join(file_paths.work_dir, model_store_dir)
        file_paths._set_model_paths(model_store_dir = model_store_dir, model_load_path=model_load_path, \
                                    model_name = model_name, best_model_path=os.path.join(work_dir, "best.pth.tar"))

        # set trian movement file path
        train_movement_path = get_parameter("train_movement_file", json_input, [])
        for mvm in train_movement_path:
            if os.path.exists(mvm) is False:
                raise Exception("Error! train movement: {} file not exist!".format(mvm))
        # set train feature path
        train_movement_path = [os.path.abspath(_) for _ in train_movement_path]

        train_feature_path = get_parameter("train_feature_path", json_input, [])
        for feat_path in train_feature_path:
            if os.path.exists(feat_path) is False:
                raise Exception("Error! train movement: {} file not exist!".format(feat_path))
        train_feature_path = [os.path.abspath(_) for _ in train_feature_path]
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
        forcefield_name = get_parameter("forcefield_name", json_input, "forcefield.ff")
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
    
    def _set_dp_model_params(self, json_input:dict):
        model_param = ModelParam()
        # set dp embedding net params
        embedding_json = get_parameter("descriptor", json_input, {})
        network_size = get_parameter("network_size", embedding_json, [25, 25, 25])
        bias = get_parameter("bias", embedding_json, True)
        resnet_dt = get_parameter("resnet_dt", embedding_json, False) # resnet in embedding net is False.
        activation = get_parameter("activation", embedding_json, "tanh")
        embedding_net = NetParam("embedding_net")
        embedding_net.set_params(network_size, bias, resnet_dt, activation)
       
        # set dp fitting net params
        fitting_net_dict = get_parameter("fitting_net",json_input, {})
        network_size = get_parameter("network_size", fitting_net_dict, [50, 50, 50, 1])
        bias = get_parameter("bias", fitting_net_dict, True)
        resnet_dt = get_parameter("resnet_dt", fitting_net_dict, True)
        activation = get_parameter("activation", fitting_net_dict, "tanh")
        fitting_net = NetParam("fitting_net")
        fitting_net.set_params(network_size, bias, resnet_dt, activation)

        model_param.set_net(fitting_net=fitting_net, embedding_net=embedding_net)
        return model_param

    def _set_descriptor(self, json_input:dict):
        pass

    def _set_nn_model_params(self, json_input:dict):
        model_param = ModelParam()
        fitting_net_dict = get_parameter("fitting_net",json_input, {})
        network_size = get_parameter("network_size", fitting_net_dict,[15,15,1])
        bias = True # get_parameter("bias", fitting_net_dict, True)
        resnet_dt = False # get_parameter("resnet_dt", fitting_net_dict, False)
        activation = "tanh" #get_parameter("activation", fitting_net_dict, )
        fitting_net = NetParam("fitting_net")
        fitting_net.set_params(network_size, bias, resnet_dt, activation)

        model_param.set_net(fitting_net=fitting_net, embedding_net=None)
        return model_param

    def _set_optimizer(self, json_input:dict):
        optimizer_type = get_parameter("optimizer", json_input, "LKF")
        batch_size = get_parameter("batch_size", json_input, 1)
        epochs = get_parameter("epochs", json_input, 30)
        print_freq = get_parameter("print_freq", json_input, 10)
        # the start epoch could be reset at the resume model code block
        start_epoch = get_parameter("start_epoch", json_input, 1)
        optimizer_param = OptimizerParam(optimizer_type, start_epoch=start_epoch, epochs=epochs, batch_size=batch_size, \
                                         print_freq=print_freq)

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
            gamma = get_parameter("gamma", json_input, 0.99) # used in nn optimizer
            step = get_parameter("step", json_input, 100) # used in nn optimizer
            scheduler = get_parameter("scheduler", json_input, None) # used in nn optimizer

            stop_step = get_parameter("stop_step", json_input, 1000000)
            decay_step = get_parameter("decay_step", json_input, 5000)
            stop_lr = get_parameter("stop_lr",json_input, 3.51e-8)
            optimizer_param.set_adam_sgd_params(learning_rate, weight_decay, momentum,\
                                                 gamma, step, scheduler, stop_step, decay_step, stop_lr)
        
        train_energy = get_parameter("train_energy", json_input, True) 
        train_force = get_parameter("train_force", json_input, True) 
        train_ei = get_parameter("train_ei", json_input, False) 
        train_virial = get_parameter("train_virial", json_input, False) 
        train_egroup = get_parameter("train_egroup", json_input, False) 

        if "KF" in optimizer_param.opt_name:
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
        else:
            start_pre_fac_force = get_parameter("start_pre_fac_force", json_input, 1000) 
            start_pre_fac_etot = get_parameter("start_pre_fac_etot", json_input, 0.02) 
            start_pre_fac_ei = get_parameter("start_pre_fac_ei", json_input, 0.1) 
            start_pre_fac_virial = get_parameter("start_pre_fac_virial", json_input, 50.0) 
            start_pre_fac_egroup = get_parameter("start_pre_fac_egroup", json_input, 0.02) 

            end_pre_fac_force = get_parameter("end_pre_fac_force", json_input, 1.0) 
            end_pre_fac_etot = get_parameter("end_pre_fac_etot", json_input, 1.0) 
            end_pre_fac_ei = get_parameter("end_pre_fac_ei", json_input, 2.0) 
            end_pre_fac_virial = get_parameter("end_pre_fac_virial", json_input, 1.0) 
            end_pre_fac_egroup = get_parameter("end_pre_fac_egroup", json_input, 1.0) 

            optimizer_param.set_adam_sgd_train_pref(
                    train_energy = train_energy, train_force = train_force, 
                        train_ei = train_ei, train_virial = train_virial, train_egroup = train_egroup, 
                    start_pre_fac_force = start_pre_fac_force, start_pre_fac_etot = start_pre_fac_etot, 
                        start_pre_fac_ei = start_pre_fac_ei, start_pre_fac_virial = start_pre_fac_virial, 
                        start_pre_fac_egroup = start_pre_fac_egroup,
                    end_pre_fac_force = end_pre_fac_force, end_pre_fac_etot = end_pre_fac_etot, 
                        end_pre_fac_ei = end_pre_fac_ei, end_pre_fac_virial = end_pre_fac_virial, 
                        end_pre_fac_egroup = end_pre_fac_egroup,
                    )      
        return optimizer_param

    def _set_nn_personal_params(self, json_input:dict):
        self.is_dfeat_sparse = get_parameter("is_dfeat_sparse", json_input, False)  #if true not realized

    def _to_dict(self):
        pass

    def get_dp_net_dict(self):
        net_dict = {}
        net_dict["net_cfg"] = {}
        net_dict["net_cfg"][self.model_param.embedding_net.net_type] = self.model_param.embedding_net.to_dict()
        net_dict["net_cfg"][self.model_param.fitting_net.net_type] = self.model_param.fitting_net.to_dict()
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
        data_file_dict["dwidth"] = self.dwidth
        data_file_dict["ratio"] = self.train_valid_ratio

        return data_file_dict
    
    def to_dict(self):
        params_dict = {}
        params_dict["model_type"] = self.model_type
        params_dict["atom_type"] = self.atom_type
        params_dict["max_neigh_num"] = self.max_neigh_num
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
            params_dict["model"]["fitting_net"] = self.model_param.to_dict()
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
    