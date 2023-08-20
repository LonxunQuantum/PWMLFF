import json
import os
from varname import nameof

class TrainFileStructure(object):
    '''
    description: 
    param {*} self
    param {*} work_dir:  is the work path, model training, feature generation, and inference work \
                            are all carried out under the modified directory\
                                if the user does not set it, it defaults to json_dir
    param {*} json_dir: The trained models, features, and inference results are collected in this directory
    return {*}
    author: wuxingxing
    '''    
    def __init__(self, json_dir:str=None, work_dir:str=None) -> None:
        self.json_dir = json_dir
        if work_dir is None:
            self.work_dir = json_dir
        else:
            self.work_dir = work_dir
        self.movement_name = "MOVEMENT"
        self.test_movement_path = []
        self.train_movement_path = []
        self.train_feature_path = []
        self.test_feature_path = []

    def _set_training_path(self, train_movement_path:list, train_feature_path:list, train_dir: str):
        self.train_movement_path = train_movement_path
        self.train_feature_path = train_feature_path
        self.train_dir = os.path.join(self.work_dir, train_dir)

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

    def _set_model_paths(self,model_store_dir:str, model_load_path:str, model_name:str, best_model_path:str):
        self.model_store_dir = model_store_dir
        self.model_load_path = model_load_path
        self.model_name = model_name
        self.best_model_path = best_model_path
        self.model_save_path = os.path.join(model_store_dir, self.model_name)

    def _set_inference_paths(self, test_movement_path:list=[], test_dir: str=""):
        self.test_dir = os.path.join(self.work_dir,test_dir)
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
    def set_train_feature_path(self, feature_path:list):
        self.train_feature_path.extend(feature_path)

    def set_test_feature_path(self, feature_path:list):
        self.test_feature_path.extend(feature_path)

    def get_data_file_structure(self):
        file_dict = {}
        file_dict["trainSetDir"] = self.trainSetDir
        file_dict["dRFeatureInputDir"] = self.dRFeatureInputDir
        file_dict["dRFeatureOutputDir"] = self.dRFeatureOutputDir
        file_dict["trainDataPath"] = self.trainDataPath
        file_dict["validDataPath"] = self.validDataPath
        return file_dict

    def set_forcefield_path(self, forcefield_dir:str, forcefield_name:str):
        self.forcefield_dir = os.path.join(self.work_dir, forcefield_dir)
        self.forcefield_name = forcefield_name

    def to_dict(self):
        dicts = {}
        dicts["work_dir"] = self.work_dir
        if os.path.exists(self.model_load_path):
            dicts["model_load_path"] = self.model_load_path
        if len(self.train_movement_path) > 0:
            dicts["train_movement_path"] = self.train_movement_path
            # dicts["model_store_dir"] = self.model_store_dir
        
        if len(self.train_feature_path) > 0:
            dicts["train_feature_path"] = self.train_feature_path
            # dicts["train_dir_name"] = self.train_dir

        if len(self.test_movement_path) > 0:
            dicts["test_movement_path"] = self.test_movement_path
            dicts["test_dir_name"] = self.test_dir

        return dicts

class NetParam(object):
    def __init__(self, net_type:str) -> None:
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

class ModelParam(object):
    def __init__(self) -> None:
        pass
        
    def set_net(self, fitting_net:NetParam=None, embedding_net:NetParam=None):
        self.fitting_net = fitting_net
        self.embedding_net = embedding_net
    
    def to_dict(self):
        dicts = {}
        if self.embedding_net is not None:
            dicts[self.embedding_net.net_type] = self.embedding_net.to_dict()
        if self.fitting_net is not None:
            dicts[self.fitting_net.net_type] = self.fitting_net.to_dict()
        return dicts

class OptimizerParam(object):
    def __init__(self, optimizer:str, start_epoch:int, epochs:int, batch_size:int, \
                 print_freq:int, best_loss:int, min_loss:int) -> None:
        self.opt_name = optimizer.upper()
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.batch_size = batch_size

        self.print_freq = print_freq
        self.best_loss=best_loss
        self.min_loss=min_loss

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

    '''
    description: these params are used when optimizer is KF
    return {*}
    author: wuxingxing
    '''
    def set_train_pref(self,train_energy:bool=True, train_force:bool=True, 
                       train_ei:bool=False, train_virial:bool=False, train_egroup:bool=False, 
                            pre_fac_etot:float=1.0, pre_fac_force:float=2.0, 
                                pre_fac_ei:float=1.0, pre_fac_virial:float=1.0, pre_fac_egroup:float=0.1):
        self.train_energy = train_energy
        self.train_force = train_force
        self.train_ei = train_ei
        self.train_virial = train_virial
        self.train_egroup = train_egroup

        self.pre_fac_etot = pre_fac_etot
        self.pre_fac_force = pre_fac_force
        self.pre_fac_ei = pre_fac_ei
        self.pre_fac_virial = pre_fac_virial
        self.pre_fac_egroup = pre_fac_egroup

    def to_dict(self):
        opt_dict = {}
        opt_dict["optimizer"]=self.opt_name
        opt_dict["start_epoch"] = self.start_epoch
        opt_dict["epochs"] = self.epochs
        opt_dict["batch_size"] = self.batch_size
        opt_dict["print_freq"] = self.print_freq
        opt_dict["best_loss"] = self.best_loss
        opt_dict["min_loss"] = self.min_loss

        if "KF" in self.opt_name:
            opt_dict["block_size"] = self.block_size 
            opt_dict["kalman_lambda"] = self.kalman_lambda
            opt_dict["kalman_nue"] = self.kalman_nue
            #prefect:
            opt_dict["train_energy"] = self.train_energy
            opt_dict["train_force"] = self.train_force
            opt_dict["train_ei"] = self.train_ei
            opt_dict["train_virial"] = self.train_virial
            opt_dict["train_egroup"] = self.train_egroup
            opt_dict["pre_fac_force"] = self.pre_fac_force
            opt_dict["pre_fac_etot"] = self.pre_fac_etot
            opt_dict["pre_fac_ei"] = self.pre_fac_ei
            opt_dict["pre_fac_virial"] = self.pre_fac_virial
            opt_dict["pre_fac_egroup"] = self.pre_fac_egroup
        else:
            opt_dict["learning_rate"]= self.learning_rate
            opt_dict["weight_decay"]= self.weight_decay
            opt_dict["momentum"]= self.momentum
        return opt_dict
