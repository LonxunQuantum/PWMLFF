class ModelParam(object):
    def __init__(self) -> None:
        # set default parameters
        self.net_shape = []
        self.is_bias = False
        self.is_resnet = False
        self.act_funtion="tanh"

    def set_params(self, net_shape:list, is_bias:bool, is_resnet: bool, act_funtion:str):
        self.net_shape = net_shape
        self.is_bias = is_bias
        self.is_resnet = is_resnet
        self.act_funtion=act_funtion      

    def to_dict(self):
        return {"net_shape": self.net_shape, "is_bias": self.is_bias, "is_resnet": self. is_resnet, "act_funtion": self.act_funtion}

class OptimizerParam(object):
    def __init__(self) -> None:
        self.optimizer = ""
        self.block_size = 5120
        self.kalman_lambda = 0.98
        self.kalman_nue = 0.9987

    def set_params(self, optimizer:str, block_size:int, kalman_lambda:float, kalman_nue:float):
        self.optimizer = optimizer
        self.block_size = block_size
        self.kalman_lambda = kalman_lambda
        self.kalman_nue = kalman_nue

    def to_dict(self):
        return {"optimizer": self.optimizer, "block_size":self.block_size, "kalman_lambda": self.kalman_lambda, "kalman_nue": self.kalman_nue}
    
class DpParam(object):
    def __init__(self, model_dict:dict) -> None:
        # set fitting net, embeding net
        self.fitting_net, self.embedding_net = self._set_dp_net(model_dict)
        self.optimizer = self._set_optimizer(model_dict)

        # required params
        self.atom_type = self._get_required_parameters("atom_type", model_dict)
        self.max_neigh_num = self._get_required_parameters("max_neigh_num", model_dict)

        # set feature related params
        self.Rmax = self._get_parameters("Rmax", model_dict, 6.0) 
        self.Rmin = self._get_parameters("Rmin", model_dict, 0.5) 
        self.M2 = self._get_parameters("M2", model_dict, 16)

        # set train params
        self.train_energy = self._get_parameters("train_energy", model_dict, True) 
        self.train_force = self._get_parameters("train_force", model_dict, True) 
        self.train_ei = self._get_parameters("train_ei", model_dict, False) 
        self.train_virial = self._get_parameters("train_virial", model_dict, False) 
        self.train_egroup = self._get_parameters("train_egroup", model_dict, True) 

        self.pre_fac_force = self._get_parameters("pre_fac_force", model_dict, 2.0) 
        self.pre_fac_etot = self._get_parameters("pre_fac_etot", model_dict, 1.0) 
        self.pre_fac_ei = self._get_parameters("pre_fac_ei", model_dict, 1.0) 
        self.pre_fac_virial = self._get_parameters("pre_fac_virial", model_dict, 1.0) 
        self.pre_fac_egroup = self._get_parameters("pre_fac_egroup", model_dict, 0.1) 

        self.precision = self._get_parameters("precision", model_dict, "float64")

    def _set_dp_net(self, model_dict:dict):
        fitting_net = ModelParam()
        embedding_net = ModelParam()
        net_shape = [50, 50, 50, 1] # default dp fitting net params
        is_bias = True # default dp params
        is_resnet = False # default dp params
        act_funtion = "tanh" # default dp params

        if "fitting_net" in model_dict.keys():
            fitting_param = model_dict["fitting_net"]
            if "net_shape" in fitting_param.keys():
                net_shape = fitting_param["net_shape"]
            if "is_bias" in fitting_param.keys():
                is_bias = fitting_param["is_bias"]
            if "is_resnet" in fitting_param.keys():
                is_resnet = fitting_param["is_resnet"]
            if "act_funtion" in fitting_param.keys():
                act_funtion = fitting_param["act_funtion"]
        fitting_net.set_params(net_shape, is_bias, is_resnet, act_funtion)

        net_shape = [25, 25, 25] # default dp embedding net params
        if "embedding_net" in model_dict.keys():
            embedding_net = model_dict["embedding_net"]
            if "net_shape" in embedding_net.keys():
                net_shape = embedding_net["net_shape"]
            if "is_bias" in embedding_net.keys():
                is_bias = embedding_net["is_bias"]
            if "is_resnet" in embedding_net.keys():
                is_resnet = embedding_net["is_resnet"]
            if "act_funtion" in embedding_net.keys():
                act_funtion = embedding_net["act_funtion"]
        embedding_net.set_params(net_shape, is_bias, is_resnet, act_funtion)
        return fitting_net, embedding_net
    
    def _set_optimizer(self, model_dict:dict):
        optimizer = OptimizerParam()
        opt_type = "LKF"
        block_size = 5120
        kalman_lambda = 0.98
        kalman_nue = 0.9987
        if "optimizer" in model_dict.keys():
            optimizer_parm = model_dict["optimizer"]
            if "optimizer" in optimizer_parm.keys():
                opt_type = optimizer_parm["optimizer"] 
            if "block_size" in optimizer_parm.keys():
                block_size = optimizer_parm["block_size"]
            if "kalman_lambda" in optimizer_parm.keys():
                kalman_lambda = optimizer_parm["kalman_lambda"] 
            if "kalman_nue" in optimizer_parm.keys():
                kalman_nue = optimizer_parm["kalman_nue"]
        optimizer.set_params(opt_type, block_size, kalman_lambda, kalman_nue)
        return optimizer

    '''
    description: 
        check the param is required parameters which need input by user
    param {*} self
    param {str} param
    param {dict} model_dict
    param {str} info 
    return {*}
    author: wuxingxing
    '''
    def _get_required_parameters(self, param:str, model_dict:dict):
        if param not in model_dict.keys():
            raise Exception("Input error! : The {} parameter is missing and must be specified in input json file!".format(param))
        return model_dict[param]

    def _get_parameters(self, param:str, model_dict:dict, default_value):
        if param not in model_dict[param]:
            return default_value
        else:
            return model_dict[param]