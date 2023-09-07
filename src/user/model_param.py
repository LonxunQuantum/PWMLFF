from utils.json_operation import get_parameter, get_required_parameter

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
    
    def to_dict_std(self):
        return \
            {
            "network_size": self.network_size
            # "bias": self.bias, 
            # "resnet_dt": self. resnet_dt, 
            # "activation": self.activation
            }

class ModelParam(object):
    def __init__(self) -> None:
        pass

    def set_type_embedding_net(self, type_embedding_dict:dict):
        # set dp embedding net params
        network_size = get_parameter("network_size", type_embedding_dict, [25, 25, 25])
        bias = get_parameter("bias", type_embedding_dict, True)
        resnet_dt = get_parameter("resnet_dt", type_embedding_dict, False) # resnet in embedding net is False.
        activation = get_parameter("activation", type_embedding_dict, "tanh")
        self.type_embedding_net = NetParam("type_embedding_net")
        self.type_embedding_net.set_params(network_size, bias, resnet_dt, activation)

    def set_embedding_net(self, embedding_json:dict):
        # embedding_json = get_parameter("descriptor", json_input, {})
        network_size = get_parameter("network_size", embedding_json, [25, 25, 25])
        bias = get_parameter("bias", embedding_json, True)
        resnet_dt = get_parameter("resnet_dt", embedding_json, False) # resnet in embedding net is False.
        activation = get_parameter("activation", embedding_json, "tanh")
        self.embedding_net = NetParam("embedding_net")
        self.embedding_net.set_params(network_size, bias, resnet_dt, activation)

    def set_dp_fitting_net(self, fitting_net_dict:dict):
        # fitting_net_dict = get_parameter("fitting_net",json_input, {})
        network_size = get_parameter("network_size", fitting_net_dict, [50, 50, 50, 1])
        if network_size[-1] != 1:
            raise Exception("Error: The last layer of the fitting network should have a size of 1 for etot energy, but the input size is {}!".format(network_size[-1]))
        bias = get_parameter("bias", fitting_net_dict, True)
        resnet_dt = get_parameter("resnet_dt", fitting_net_dict, True)
        activation = get_parameter("activation", fitting_net_dict, "tanh")
        self.fitting_net = NetParam("fitting_net")
        self.fitting_net.set_params(network_size, bias, resnet_dt, activation)

    def set_nn_fitting_net(self, fitting_net_dict:dict):
        # fitting_net_dict = get_parameter("fitting_net",json_input, {})
        network_size = get_parameter("network_size", fitting_net_dict,[15,15,1])
        if network_size[-1] != 1:
            raise Exception("Error: The last layer of the fitting network should have a size of 1 for etot energy, but the input size is {}!".format(network_size[-1]))
        bias = True # get_parameter("bias", fitting_net_dict, True)
        resnet_dt = False # get_parameter("resnet_dt", fitting_net_dict, False)
        activation = "tanh" #get_parameter("activation", fitting_net_dict, )
        self.fitting_net = NetParam("fitting_net")
        self.fitting_net.set_params(network_size, bias, resnet_dt, activation)

    # def to_dict(self):
    #     # dicts = {}
    #     # if self.embedding_net is not None:
    #     #     dicts[self.embedding_net.net_type] = self.embedding_net.to_dict()
    #     # if self.fitting_net is not None:
    #     #     dicts[self.fitting_net.net_type] = self.fitting_net.to_dict()
    #     return self.fitting_net.to_dict_std()


