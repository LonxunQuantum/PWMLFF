import json
import os
from utils.json_operation import get_parameter, get_required_parameter
from utils.file_operation import is_alive_atomic_energy

class WorkFileStructure(object):
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
    def __init__(self, json_dir:str, work_dir:str, reserve_work_dir:bool, reserve_feature:bool, model_type:str, cmd:str) -> None:
        self.model_type = model_type
        self.cmd = cmd
        self.json_dir = json_dir
        self.work_dir = work_dir
        self.reserve_work_dir = reserve_work_dir
        self.reserve_feature = reserve_feature
        self.movement_name = "MOVEMENT"
        self.test_movement_path = []
        self.train_movement_path = []
        self.train_feature_path = []
        self.test_feature_path = []

    def _set_training_path(self, train_movement_path:list, train_feature_path:list, train_dir: str):
        self.train_movement_path = train_movement_path
        self.train_feature_path = train_feature_path
        self.train_dir = os.path.join(self.work_dir, train_dir)

    def _set_alive_atomic_energy(self, alive_atomic_energy:bool):
        self.alive_atomic_energy = alive_atomic_energy

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

    def set_inference_paths(self, json_input:dict):
        # load test files and check if they are exist
        test_movement_path = get_required_parameter("test_movement_file", json_input)
        if isinstance(test_movement_path, list) is False:
            test_movement_path = [test_movement_path]
        for mvm in test_movement_path:
            if os.path.exists(mvm) is False:
                raise Exception("{} file is not exists, please check!".format(mvm))
        
        test_dir_name = get_parameter("test_dir_name", json_input, "test_result")
        self.test_dir = os.path.join(self.work_dir, test_dir_name)

        test_feature_path = get_parameter("test_feature_path", json_input, [])
        for feat_path in test_feature_path:
            if os.path.exists(feat_path) is False:
                raise Exception("Error! test_feature_path {} does not exist!".format(feat_path))
        test_feature_path = [os.path.abspath(_) for _ in test_feature_path]
        self.test_movement_path = [os.path.abspath(_) for _ in test_movement_path]

        if os.path.exists(self.model_load_path) is False:
            raise Exception("the model_load_path is not exist: {}, please speccified 'model_load_path' at json file".format(self.model_load_path))
        
        alive_atomic_energy = is_alive_atomic_energy(test_movement_path)
        self._set_alive_atomic_energy(alive_atomic_energy)
    

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

    def set_file_paths(self, json_input:dict):
        model_load_path = get_parameter("model_load_file", json_input, " ")
        if os.path.exists(model_load_path):
            model_load_path = os.path.abspath(model_load_path)
        # if self.recover_train is True and os.path.isfile(model_load_path):
        #     raise Exception("Error! The recover_train and model_load_path are simultaneously specified, please set recover_train to False or remove param model_load_path")
        if self.model_type == "NN":
            model_name = get_parameter("model_name", json_input, "nn_model.ckpt")
        else:
            model_name = get_parameter("model_name", json_input, "dp_model.ckpt")
        model_store_dir = get_parameter("model_store_dir", json_input, "model_record")
        model_store_dir = os.path.join(self.work_dir, model_store_dir)
        self._set_model_paths(model_store_dir = model_store_dir, model_load_path=model_load_path, \
                                    model_name = model_name, best_model_path=os.path.join(self.work_dir, "best.pth.tar"))

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
        self._set_training_path(train_movement_path=train_movement_path, 
                                      train_feature_path=train_feature_path,
                                      train_dir=os.path.join(self.work_dir, "feature"))
        
        alive_atomic_energy = get_parameter("alive_atomic_energy", json_input, False)
        alive_atomic_energy = is_alive_atomic_energy(train_movement_path)
        self._set_alive_atomic_energy(alive_atomic_energy)

        # set Pwdata dir file structure, they are used in feature generation
        trainSetDir = get_parameter("trainSetDir", json_input, 'PWdata')
        dRFeatureInputDir = get_parameter("dRFeatureInputDir", json_input, 'input')
        dRFeatureOutputDir = get_parameter("dRFeatureOutputDir", json_input, 'output')
        trainDataPath = get_parameter("trainDataPath", json_input, 'train')
        validDataPath = get_parameter("validDataPath", json_input, 'valid')
        self._set_data_file_paths(trainSetDir, dRFeatureInputDir, dRFeatureOutputDir, trainDataPath, validDataPath)
        
        # set 
        forcefield_name = get_parameter("forcefield_name", json_input, "forcefield.ff")
        forcefield_dir = get_parameter("forcefield_dir", json_input, "forcefield")
        self.set_forcefield_path(forcefield_dir, forcefield_name)
        
        # p matix, resume p matrix when recover is not realized
        # p matrix should extract to checkpoint files or a single file.
        # current not realized
        save_p_matrix = get_parameter("save_p_matrix", json_input, False)
        if save_p_matrix is not False:
            Pmatrix_path = os.path.join(self.work_dir, "P.pkl")
            self._set_p_matrix_paths(Pmatrix_path, True)
        else:
            self._set_p_matrix_paths(None, False)
            
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
        dicts["reserve_work_dir"] = self.reserve_work_dir

        if os.path.exists(self.model_load_path):
            dicts["model_load_file"] = self.model_load_path
        if len(self.train_movement_path) > 0 and self.cmd == "train".upper():
            dicts["train_movement_file"] = self.train_movement_path
            # dicts["model_store_dir"] = self.model_store_dir
        
        # if len(self.train_feature_path) > 0:
        #     dicts["train_feature_path"] = self.train_feature_path
            # dicts["train_dir_name"] = self.train_dir

        if len(self.test_movement_path) > 0 and self.cmd == "test".upper():
            dicts["test_movement_file"] = self.test_movement_path
            # dicts["test_dir_name"] = self.test_dir

        return dicts
