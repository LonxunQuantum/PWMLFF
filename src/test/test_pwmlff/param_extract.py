from test_util.json_operation import get_parameter, get_required_parameter
from test_pwmlff.params import Resource, TrainInput, LmpsInput
from typing import List, Tuple

class TrainParam(object):
    def __init__(self, json_dict:dict) -> None:
        self.work_dir = "./"
        # extract env lists
        env_list = get_required_parameter("envs", json_dict)
        if isinstance(env_list, dict):
            env_list = [env_list]
        self.resource_list = []
        for env in env_list:
            self.resource_list.append(Resource(env))
        # extract train jsons
        self.train_inputs = []
        path_prefix = get_parameter("path_prefix", json_dict, None)
        train_inputs = get_required_parameter("train_inputs", json_dict)
        if isinstance(train_inputs, dict):
            train_inputs = [train_inputs]
        for input_dict in train_inputs:
            self.train_inputs.append(TrainInput(input_dict, path_prefix))
        # get work lists
        work_list = get_required_parameter("work_list", json_dict)
        if isinstance(work_list, dict):
            work_list = [work_list]
        self.work_list = self.get_work_list(work_list, self.train_inputs, self.resource_list)

    def get_work_list(self, json_dict:dict, train_list:list[TrainInput], resource:list[Resource]):
        work_list: List[Tuple[TrainInput, Resource, int]] = []
        for work_dict in json_dict:
            env_list = get_required_parameter("envs", work_dict)
            train_inputs = get_required_parameter("train_inputs", work_dict)
            epoch_list = get_parameter("epochs", work_dict, [])
            if isinstance(epoch_list, int):
                epoch_list = [epoch_list for _ in range(len(env_list))]
            for id, env_idx in enumerate(env_list):
                epoch = epoch_list[id] if len(epoch_list) > 0 else None
                for train_idx in train_inputs:
                    work_list.append((train_list[train_idx], resource[env_idx], epoch))
        return work_list  


class LmpsParam(object):
    def __init__(self, json_dict:dict) -> None:
        self.work_dir = "./"
        # extract env lists
        env_list = get_required_parameter("envs", json_dict)
        if isinstance(env_list, dict):
            env_list = [env_list]
        self.resource_list = []
        for env in env_list:
            self.resource_list.append(Resource(env))
        # extract lmps input
        self.lmps_inputs = []
        path_prefix = get_parameter("path_prefix", json_dict, None)
        lmps_inputs = get_required_parameter("lmps_inputs", json_dict)
        if isinstance(lmps_inputs, dict):
            lmps_inputs = [lmps_inputs]
        for input_dict in lmps_inputs:
            self.lmps_inputs.append(LmpsInput(input_dict, path_prefix))
        # get work lists
        work_list = get_required_parameter("work_list", json_dict)
        if isinstance(work_list, dict):
            work_list = [work_list]
        self.work_list = self.get_lmps_work_list(work_list, self.lmps_inputs, self.resource_list)

    def get_train_work_list(self, json_dict:dict, train_list:list[TrainInput], resource:list[Resource]):
        work_list: List[Tuple[TrainInput, Resource, int]] = []
        for work_dict in json_dict:
            env_list = get_required_parameter("envs", work_dict)
            train_inputs = get_required_parameter("train_inputs", work_dict)
            epoch_list = get_parameter("epochs", work_dict, [])
            if isinstance(epoch_list, int):
                epoch_list = [epoch_list for _ in range(len(env_list))]
            for id, env_idx in enumerate(env_list):
                epoch = epoch_list[id] if len(epoch_list) > 0 else None
                for train_idx in train_inputs:
                    work_list.append((train_list[train_idx], resource[env_idx], epoch))
        return work_list  

    def get_lmps_work_list(self, json_dict:dict, lmps_list:list[LmpsInput], resource:list[Resource]):
        work_list: List[Tuple[LmpsInput, Resource]] = []
        for work_dict in json_dict:
            env_list = get_required_parameter("envs", work_dict)
            lmps_inputs = get_required_parameter("lmps_inputs", work_dict)
            for id, env_idx in enumerate(env_list):
                for lmps_idx in lmps_inputs:
                    work_list.append((lmps_list[lmps_idx], resource[env_idx]))
        return work_list  