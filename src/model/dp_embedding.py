from numpy.core.fromnumeric import std
import torch
import torch.nn as nn
import collections
from torch.nn.init import normal_ as normal
import numpy as np
from typing import List

# logging and o ur extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# setup logging module
logger = logging.getLogger('train.DeepMD')

def dump(msg, *args, **kwargs):
    logger.log(logging_level_DUMP, msg, *args, **kwargs)
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)
def summary(msg, *args, **kwargs):
    logger.log(logging_level_SUMMARY, msg, *args, **kwargs)
def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)
def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)
def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs, exc_info=True)

class LayerModule(nn.Module):
    def __init__(self, 
                 weight, 
                 bias = None, 
                 resnet_dt = None):
        super(LayerModule, self).__init__()
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.bias = nn.Parameter(bias, requires_grad=True) if bias is not None else None
        self.resnet_dt = nn.Parameter(resnet_dt, requires_grad=True) if resnet_dt is not None else None

class EmbeddingNet(nn.Module):
    def __init__(self, 
                 network_size: List[int], 
                 bias: bool, 
                 resnet_dt: bool, 
                 activation: str, 
                 device,
                 magic = False):
        super(EmbeddingNet, self).__init__()
        # self.cfg = cfg
        self.network_size = [1] + network_size
        self.bias_flag = bias
        self.resnet_dt_flag = resnet_dt
        self.activation = torch.tanh if activation == "tanh" else None

        # self.weights = nn.ParameterDict()
        # self.weights = nn.ParameterList()
        # self.bias = nn.ParameterList() if bias else None
        # self.resnet_dt = nn.ParameterList() if resnet_dt else None

        self.layers = nn.ModuleList()

        # if cfg['bias']:
        #     self.bias = nn.ParameterDict()
        
        # if self.cfg['resnet_dt']:
        #         self.resnet_dt = nn.ParameterDict()

        # self.network_size = [1] + self.cfg['network_size']

        # if cfg["activation"] == "tanh":
        #     cfg["activation"] = torch.tanh
        # else:
        #     pass
        
        # initial weight normalization
        for i in range(1, len(self.network_size)):
            wij = torch.Tensor(self.network_size[i-1], self.network_size[i])
            normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))
            # self.weights["weight" + str(i-1)] = nn.Parameter(wij, requires_grad=True) 
            # self.weights.append(nn.Parameter(wij, requires_grad=True)).to('cuda:2')
            bias = None
            if self.bias_flag:
                bias = torch.Tensor(1, self.network_size[i])
                normal(bias, mean=0, std=1)
                # self.bias["bias" + str(i-1)] = nn.Parameter(bias, requires_grad=True) 
                # self.bias.append(nn.Parameter(bias, requires_grad=True)).to('cuda:2')
            resnet_dt = None
            if self.resnet_dt_flag:
                resnet_dt = torch.Tensor(1, self.network_size[i])
                normal(resnet_dt, mean=1, std=0.001)
                # self.resnet_dt["resnet_dt" + str(i-1)] = nn.Parameter(resnet_dt, requires_grad=True)
                # self.resnet_dt.append(nn.Parameter(resnet_dt, requires_grad=True)).to('cuda:2')
            self.layers.append(LayerModule(wij, bias, resnet_dt)).to(device)

    def forward(self, x):
        '''
        for i in range(1, len(self.network_size)):
            weight = self.weights[i-1]
            if self.bias_flag:
                bias = self.bias[i-1]
                hiden = torch.matmul(x, weight) + bias
                # hiden = torch.matmul(x, self.weights['weight' + str(i-1)]) + self.bias['bias' + str(i-1)]
            else:
                # hiden = torch.matmul(x, self.weights['weight' + str(i-1)])
                hiden = torch.matmul(x, weight)
            
            # hiden = self.cfg['activation'](hiden)
            hiden = self.activation(hiden)
            
            if self.network_size[i] == self.network_size[i-1]:
                if self.resnet_dt_flag:
                    resnet_dt = self.resnet_dt[i-1]
                    # x = hiden * self.resnet_dt['resnet_dt' + str(i-1)] + x
                    x = hiden * resnet_dt + x
                else:
                    x = hiden + x
            elif self.network_size[i] == 2 * self.network_size[i-1]:
                if self.resnet_dt_flag:
                    resnet_dt = self.resnet_dt[i-1]
                    # x = torch.cat((x, x), dim=-1)  + hiden * self.resnet_dt['resnet_dt' + str(i-1)]
                    x = torch.cat((x, x), dim=-1)  + hiden * resnet_dt
                else:
                    x = torch.cat((x, x), dim=-1)  + hiden
            else:
                x = hiden
        '''
        for i, layer in enumerate(self.layers):
            if self.bias_flag and layer.bias is not None:
                hiden = torch.matmul(x, layer.weight) + layer.bias
            else:
                hiden = torch.matmul(x, layer.weight)
            
            hiden = self.activation(hiden)
            
            if self.network_size[i+1] == self.network_size[i]:
                if self.resnet_dt_flag and layer.resnet_dt is not None:
                    x = hiden * layer.resnet_dt + x
                else:
                    x = hiden + x
            elif self.network_size[i+1] == 2 * self.network_size[i]:
                if self.resnet_dt_flag and layer.resnet_dt is not None:
                    x = torch.cat((x, x), dim=-1)  + hiden * layer.resnet_dt
                else:
                    x = torch.cat((x, x), dim=-1)  + hiden
            else:
                x = hiden

        return x


class FittingNet(nn.Module):

    def __init__(self,  
                 network_size: List[int], 
                 bias: bool, 
                 resnet_dt: bool, 
                 activation: str, 
                 device, 
                 input_dim, 
                 ener_shift, 
                 magic = False):
        super(FittingNet, self).__init__()
        # self.cfg = cfg
        self.network_size = [input_dim] + network_size
        self.bias_flag = bias
        self.resnet_dt_flag = resnet_dt
        self.activation = torch.tanh if activation == "tanh" else None

        # self.weights = nn.ParameterList()
        # self.bias = nn.ParameterList() if bias else None
        # self.resnet_dt = nn.ParameterList() if resnet_dt else None
        # self.weights = nn.ParameterDict()

        self.layers = nn.ModuleList()

        # if cfg['bias']:
        #     self.bias = nn.ParameterDict()
        # if self.cfg['resnet_dt']:
        #     self.resnet_dt = nn.ParameterDict()
        
        # self.network_size = [input_dim] + self.cfg['network_size']

        # if cfg["activation"] == "tanh":
        #     cfg["activation"] = torch.tanh
        # else:
        #     pass

        for i in range(1, len(self.network_size)-1):
            wij = torch.Tensor(self.network_size[i-1], self.network_size[i])
            normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))
            # self.weights["weight" + str(i-1)] = nn.Parameter(wij, requires_grad=True) 
            # self.weights.append(nn.Parameter(wij, requires_grad=True)).to('cuda:2')
            bias = None
            if self.bias_flag:
                bias = torch.Tensor(1, self.network_size[i])
                normal(bias, mean=0, std=1)
                # self.bias["bias" + str(i-1)] = nn.Parameter(bias, requires_grad=True)
                # self.bias.append(nn.Parameter(bias, requires_grad=True)).to('cuda:2')
            resnet_dt = None
            if i > 1 and self.resnet_dt_flag:
                resnet_dt = torch.Tensor(1, self.network_size[i])
                normal(resnet_dt, mean=0.1, std=0.001)
                # self.resnet_dt["resnet_dt" + str(i-1)] = nn.Parameter(resnet_dt, requires_grad=True)
                # self.resnet_dt.append(nn.Parameter(resnet_dt, requires_grad=True)).to('cuda:2')
            self.layers.append(LayerModule(wij, bias, resnet_dt)).to(device)
        
        i = len(self.network_size) - 1
        wij = torch.randn(self.network_size[i-1], self.network_size[i])
        normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))
        # self.weights["weight" + str(i-1)] = nn.Parameter(wij, requires_grad=True)
        # self.weights.append(nn.Parameter(wij, requires_grad=True)).to('cuda:2')
        if self.bias_flag:
            bias_init = torch.randn(1, self.network_size[i])
            normal(bias_init, mean=ener_shift, std=1.0)
            # import ipdb;ipdb.set_trace()
            # bias_init = torch.Tensor(ener_shift)
            # self.bias["bias" + str(i-1)] = nn.Parameter(bias_init, requires_grad=True)       # 初始化指定均值
            # self.bias.append(nn.Parameter(bias_init, requires_grad=True)).to('cuda:2')
        
        self.layers.append(LayerModule(wij, bias_init, None)).to(device)
        

    def forward(self, x):
        '''
        for i in range(1, len(self.network_size) - 1):
            weight = self.weights[i-1]
            if self.bias_flag:
                bias = self.bias[i-1]
                # hiden = torch.matmul(x, self.weights['weight' + str(i-1)]) + self.bias['bias' + str(i-1)]
                hiden = torch.matmul(x, weight) + bias
            else:
                # hiden = torch.matmul(x, self.weights['weight' + str(i-1)])
                hiden = torch.matmul(x, weight)
            
            # hiden = self.cfg['activation'](hiden)
            hiden = self.activation(hiden)
            
            if i > 1:
                # if self.network_size[i] == self.network_size[i-1] and self.cfg['resnet_dt']:
                if self.network_size[i] == self.network_size[i-1] and self.resnet_dt_flag:
                    resnet_dt = self.resnet_dt[i-2]                             # resnet_dt index from 0
                    # x = hiden * self.resnet_dt['resnet_dt' + str(i-1)] + x    # resnet_dt index from 1
                    x = hiden * resnet_dt + x
                else:
                    x = hiden + x
            else:
                x = hiden
        '''
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:        # 对于非最后一层
                if self.bias_flag and layer.bias is not None:
                    hiden = torch.matmul(x, layer.weight) + layer.bias
                else:
                    hiden = torch.matmul(x, layer.weight)

                hiden = self.activation(hiden)
                if i > 0:
                    if self.network_size[i+1] == self.network_size[i] and layer.resnet_dt is not None:
                        x = hiden * layer.resnet_dt + x
                    else:
                        x = hiden + x
                else:
                    x = hiden
                
        # i = len(self.network_size) - 1
        # weight = self.weights[i-1]

        for i, layer in enumerate(self.layers):
            if i == len(self.network_size) - 2:
                if self.bias_flag:
                    x = torch.matmul(x, layer.weight) + layer.bias
                else:
                    x = torch.matmul(x, layer.weight)

        return x
    
class EmbeddingNet0(nn.Module):
    def __init__(self, 
                 network_size: List[int], 
                 bias: bool, 
                 resnet_dt: bool, 
                 activation: str,
                 device, 
                 magic = False):
        super(EmbeddingNet, self).__init__()
        # self.cfg = cfg
        self.network_size = [1] + network_size
        self.bias_flag = bias
        self.resnet_dt_flag = resnet_dt
        self.activation = torch.tanh if activation == "tanh" else None

        self.weights = nn.ParameterDict()
        self.bias = nn.ParameterDict() if bias else None
        self.resnet_dt = nn.ParameterDict() if resnet_dt else None

        # if cfg['bias']:
        #     self.bias = nn.ParameterDict()
        
        # if self.cfg['resnet_dt']:
        #         self.resnet_dt = nn.ParameterDict()

        # self.network_size = [1] + self.cfg['network_size']

        # if cfg["activation"] == "tanh":
        #     cfg["activation"] = torch.tanh
        # else:
        #     pass
        
        # initial weight normalization
        for i in range(1, len(self.network_size)):
            wij = torch.Tensor(self.network_size[i-1], self.network_size[i])
            normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))
            self.weights["weight" + str(i-1)] = nn.Parameter(wij, requires_grad=True) 
            if self.bias_flag:
                bias = torch.Tensor(1, self.network_size[i])
                normal(bias, mean=0, std=1)
                self.bias["bias" + str(i-1)] = nn.Parameter(bias, requires_grad=True) 
            if self.resnet_dt_flag:
                resnet_dt = torch.Tensor(1, self.network_size[i])
                normal(resnet_dt, mean=1, std=0.001)
                self.resnet_dt["resnet_dt" + str(i-1)] = nn.Parameter(resnet_dt, requires_grad=True)


    def forward(self, x):
        # check for embedding input: first layer
        # for num in range(10):
        #     print(x[0,0,num,0])
        for i in range(1, len(self.network_size)):
            # 
            #if (True in torch.isnan(self.bias['bias' + str(i-1)])):
            #    print(torch.isnan(self.bias['bias' + str(i-1)]))

            #if (True in torch.isnan(self.weights['weight' + str(i-1)])):
            #    print(torch.isnan(self.weights['weight' + str(i-1)]))

            if self.bias_flag:
                hiden = torch.matmul(x, self.weights['weight' + str(i-1)]) + self.bias['bias' + str(i-1)]
            else:
                hiden = torch.matmul(x, self.weights['weight' + str(i-1)])
            
            hiden = self.activation(hiden)
            
            if self.network_size[i] == self.network_size[i-1]:
                if self.resnet_dt_flag:
                    x = hiden * self.resnet_dt['resnet_dt' + str(i-1)] + x
                else:
                    x = hiden + x
            elif self.network_size[i] == 2 * self.network_size[i-1]:
                if self.resnet_dt_flag:
                    x = torch.cat((x, x), dim=-1)  + hiden * self.resnet_dt['resnet_dt' + str(i-1)]
                else:
                    x = torch.cat((x, x), dim=-1)  + hiden
            else:
                x = hiden
            
            # if i == 1:
            #     for num in range(10):
            #         print(x[0,0,num,:])

        return x


class FittingNet0(nn.Module):

    def __init__(self,  
                 network_size: List[int], 
                 bias: bool, 
                 resnet_dt: bool, 
                 activation: str,
                 device, 
                 input_dim, 
                 ener_shift, 
                 magic = False):
        super(FittingNet, self).__init__()
        # self.cfg = cfg
        self.network_size = [input_dim] + network_size
        self.bias_flag = bias
        self.resnet_dt_flag = resnet_dt
        self.activation = torch.tanh if activation == "tanh" else None

        self.weights = nn.ParameterDict()
        self.bias = nn.ParameterDict() if bias else None
        self.resnet_dt = nn.ParameterDict() if resnet_dt else None

        # if cfg['bias']:
        #     self.bias = nn.ParameterDict()
        # if self.cfg['resnet_dt']:
        #     self.resnet_dt = nn.ParameterDict()
        
        # self.network_size = [input_dim] + self.cfg['network_size']

        # if cfg["activation"] == "tanh":
        #     cfg["activation"] = torch.tanh
        # else:
        #     pass

        for i in range(1, len(self.network_size)-1):
            wij = torch.Tensor(self.network_size[i-1], self.network_size[i])
            normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))
            self.weights["weight" + str(i-1)] = nn.Parameter(wij, requires_grad=True) 
            if self.bias_flag:
                bias = torch.Tensor(1, self.network_size[i])
                normal(bias, mean=0, std=1)
                self.bias["bias" + str(i-1)] = nn.Parameter(bias, requires_grad=True)
            if i > 1 and self.resnet_dt_flag:
                resnet_dt = torch.Tensor(1, self.network_size[i])
                normal(resnet_dt, mean=0.1, std=0.001)
                self.resnet_dt["resnet_dt" + str(i-1)] = nn.Parameter(resnet_dt, requires_grad=True)
        
        i = len(self.network_size) - 1
        wij = torch.randn(self.network_size[i-1], self.network_size[i])
        normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))
        self.weights["weight" + str(i-1)] = nn.Parameter(wij, requires_grad=True) 
        if self.bias_flag:
            bias_init = torch.randn(1, self.network_size[i])
            normal(bias_init, mean=ener_shift, std=1.0)
            # import ipdb;ipdb.set_trace()
            # bias_init = torch.Tensor(ener_shift)
            self.bias["bias" + str(i-1)] = nn.Parameter(bias_init, requires_grad=True)       # 初始化指定均值
        

    def forward(self, x):
        for i in range(1, len(self.network_size) - 1):
            if self.bias_flag:
                hiden = torch.matmul(x, self.weights['weight' + str(i-1)]) + self.bias['bias' + str(i-1)]
            else:
                hiden = torch.matmul(x, self.weights['weight' + str(i-1)])
            
            hiden = self.activation(hiden)
            
            if i > 1:
                if self.network_size[i] == self.network_size[i-1] and self.resnet_dt_flag:
                    x = hiden * self.resnet_dt['resnet_dt' + str(i-1)] + x
                else:
                    x = hiden + x
            else:
                x = hiden
                
        i = len(self.network_size) - 1

        if self.bias_flag:
            x = torch.matmul(x, self.weights['weight' + str(i-1)]) + self.bias['bias' + str(i-1)]
        else:
            x = torch.matmul(x, self.weights['weight' + str(i-1)])
        return x