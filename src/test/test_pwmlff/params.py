from test_util.json_operation import get_parameter, get_required_parameter
import os
import json

class Resource(object):
    def __init__(self, json_dict:dict) -> None:
        self.command = get_parameter("command", json_dict, None)
        self.env_type = get_required_parameter("env_type", json_dict)
        self.group_size = get_parameter("group_size", json_dict, 1)
        self.number_node = get_parameter("number_node", json_dict, 1)
        self.gpu_per_node = get_parameter("gpu_per_node", json_dict, 0)
        self.cpu_per_node = get_parameter("cpu_per_node", json_dict, 1)
        queue_name = get_required_parameter("queue_name", json_dict)
        self.queue_name = queue_name.replace(" ","")
        custom_flags = get_parameter("custom_flags", json_dict, [])
        source_list = get_parameter("source_list", json_dict, [])
        module_list = get_parameter("module_list", json_dict, [])
        env_list = get_parameter("env_list", json_dict, [])
        for i in range(len(custom_flags)):
            if "#SBATCH".lower() not in custom_flags[i].lower():
                custom_flags[i] = "#SBATCH {}".format(custom_flags[i])
        self.custom_flags = custom_flags

        env_script = ""
        if len(module_list) > 0:
            for source in module_list:
                if "module" != source.split()[0].lower():
                    tmp_source = "module load {}\n".format(source)
                else:
                    tmp_source = "{}\n".format(source)
                env_script += tmp_source

        if len(source_list) > 0:
            for source in source_list:
                if "source" != source.split()[0].lower() and \
                    "export" != source.split()[0].lower() and \
                        "module" != source.split()[0].lower() and \
                            "conda" != source.split()[0].lower():
                    tmp_source = "source {}\n".format(source)
                else:
                    tmp_source = "{}\n".format(source)
                env_script += tmp_source

        if len(env_list) > 0:
            for source in env_list:
                env_script += source + "\n"

        self.env_script = env_script

class TrainInput(object):
    def __init__(self, json_dict:dict, path_prefix:str=None) -> None:
        # data_dir = get_required_parameter("data_dir", json_dict)
        # get json_file path
        self.json_file = self.get_abs_path(get_required_parameter("json_file", json_dict), path_prefix)

        train_dict = json.load(open(self.json_file))
        # get model_type
        # self.model_type = get_required_parameter("model_type", json_dict).upper()
        self.model_type = train_dict["model_type"]
        # get optimizer type
        if "optimizer" in train_dict.keys():
            self.optimizer = get_parameter("optimizer", train_dict["optimizer"], "LKF")
        else:
            self.optimizer = "LKF"
        # get train_data
        train_data = get_parameter("train_data", json_dict, [])
        self.train_data = []
        if len(train_data) > 0:
            for raw_file in train_data:
                raw_file = self.get_abs_path(raw_file, path_prefix)
                self.train_data.append(raw_file)
            self.format = get_parameter("format", json_dict, "pwmat/movement")
            if self.model_type == "LINEAR" or self.model_type == "NN":
                if self.format != "pwmat/movement":
                    raise Exception("ERROR! the raw_file format {} should be pwmlff/movement for LINEAR or NN models!")

        self.do_test = get_required_parameter("do_test", json_dict, True)
        # print("extract {}, the optimizer is {}, the model type is {}\n\n".format(self.json_file, self.optimizer, self.model_type))

    def get_abs_path(self, file:str, path_prefix:str=None):
        if not os.path.isabs(file) and path_prefix is not None:
            res = os.path.join(path_prefix, file)
        elif not os.path.isabs(file):
            res = os.path.abspath(file)
        else:
            res = file
        if not os.path.exists(res):
            raise FileNotFoundError(res)
        return res


class LmpsInput(object):
    def __init__(self, json_dict:dict, path_prefix:str=None) -> None:
        self.model_type = get_required_parameter("model_type", json_dict).upper()
        self.files = []
        files = get_required_parameter("files", json_dict)
        for file in files:
            if not os.path.isabs(file) and path_prefix is not None:
                self.files.append(os.path.join(path_prefix, file))
            elif not os.path.isabs(file):
                self.files.append(os.path.abspath(file))
            else:
                self.files.append(file)
            if not os.path.exists(self.files[-1]):
                raise FileNotFoundError(self.files[-1])

        