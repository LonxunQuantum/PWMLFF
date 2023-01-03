from numpy.core.fromnumeric import std
import torch
import torch.nn as nn
import collections
from torch.nn.init import normal_ as normal
import numpy as np

# logging and our extension
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

class EmbeddingNet(nn.Module):
    def __init__(self, cfg, magic=False):
        super(EmbeddingNet, self).__init__()
        self.cfg = cfg
        self.weights = nn.ParameterDict()

        if cfg['bias']:
            self.bias = nn.ParameterDict()
        
        if self.cfg['resnet_dt']:
                self.resnet_dt = nn.ParameterDict()

        self.network_size = [1] + self.cfg['network_size']

        if cfg["activation"] == "tanh":
            cfg["activation"] = torch.tanh
        else:
            pass

        # 初始化权重 normalization
        for i in range(1, len(self.network_size)):
            wij = torch.Tensor(self.network_size[i-1], self.network_size[i])
            normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))
            self.weights["weight" + str(i-1)] = nn.Parameter(wij, requires_grad=True) 
            if self.cfg['bias']:
                bias = torch.Tensor(1, self.network_size[i])
                normal(bias, mean=0, std=1)
                self.bias["bias" + str(i-1)] = nn.Parameter(bias, requires_grad=True) 
            if self.cfg['resnet_dt']:
                resnet_dt = torch.Tensor(1, self.network_size[i])
                normal(resnet_dt, mean=1, std=0.001)
                self.resnet_dt["resnet_dt" + str(i-1)] = nn.Parameter(resnet_dt, requires_grad=True)


    def forward(self, x):
        for i in range(1, len(self.network_size)):
            if self.cfg['bias']:
                hiden = torch.matmul(x, self.weights['weight' + str(i-1)]) + self.bias['bias' + str(i-1)]
            else:
                hiden = torch.matmul(x, self.weights['weight' + str(i-1)])
            
            hiden = self.cfg['activation'](hiden)
            
            if self.network_size[i] == self.network_size[i-1]:
                if self.cfg['resnet_dt']:
                    x = hiden * self.resnet_dt['resnet_dt' + str(i-1)] + x
                else:
                    x = hiden + x
            elif self.network_size[i] == 2 * self.network_size[i-1]:
                if self.cfg['resnet_dt']:
                    x = torch.cat((x, x), dim=-1)  + hiden * self.resnet_dt['resnet_dt' + str(i-1)]
                else:
                    x = torch.cat((x, x), dim=-1)  + hiden
            else:
                x = hiden
        return x


class FittingNet(nn.Module):

    def __init__(self, cfg, input_dim, ener_shift, magic=False):
        super(FittingNet, self).__init__()
        self.cfg = cfg
        self.weights = nn.ParameterDict()
        if cfg['bias']:
            self.bias = nn.ParameterDict()
        if self.cfg['resnet_dt']:
            self.resnet_dt = nn.ParameterDict()
        
        self.network_size = [input_dim] + self.cfg['network_size']

        if cfg["activation"] == "tanh":
            cfg["activation"] = torch.tanh
        else:
            pass

        for i in range(1, len(self.network_size)-1):
            wij = torch.Tensor(self.network_size[i-1], self.network_size[i])
            normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))
            self.weights["weight" + str(i-1)] = nn.Parameter(wij, requires_grad=True) 
            if self.cfg['bias']:
                bias = torch.Tensor(1, self.network_size[i])
                normal(bias, mean=0, std=1)
                self.bias["bias" + str(i-1)] = nn.Parameter(bias, requires_grad=True)
            if i > 1 and self.cfg['resnet_dt']:
                resnet_dt = torch.Tensor(1, self.network_size[i])
                normal(resnet_dt, mean=0.1, std=0.001)
                self.resnet_dt["resnet_dt" + str(i-1)] = nn.Parameter(resnet_dt, requires_grad=True)
        
        i = len(self.network_size) - 1
        wij = torch.randn(self.network_size[i-1], self.network_size[i])
        normal(wij, mean=0, std=(1.0 / np.sqrt(self.network_size[i-1] + self.network_size[i])))
        self.weights["weight" + str(i-1)] = nn.Parameter(wij, requires_grad=True) 
        if self.cfg['bias']:
            bias_init = torch.randn(1, self.network_size[i])
            normal(bias_init, mean=ener_shift, std=1.0)
            # import ipdb;ipdb.set_trace()
            # bias_init = torch.Tensor(ener_shift)
            self.bias["bias" + str(i-1)] = nn.Parameter(bias_init, requires_grad=True)       # 初始化指定均值
        

    def forward(self, x):
        for i in range(1, len(self.network_size) - 1):
            if self.cfg['bias']:
                hiden = torch.matmul(x, self.weights['weight' + str(i-1)]) + self.bias['bias' + str(i-1)]
            else:
                hiden = torch.matmul(x, self.weights['weight' + str(i-1)])
            
            hiden = self.cfg['activation'](hiden)
            
            if i > 1:
                if self.network_size[i] == self.network_size[i-1] and self.cfg['resnet_dt']:
                    x = hiden * self.resnet_dt['resnet_dt' + str(i-1)] + x
                else:
                    x = hiden + x
            else:
                x = hiden
                
        i = len(self.network_size) - 1

        if self.cfg['bias']:
            x = torch.matmul(x, self.weights['weight' + str(i-1)]) + self.bias['bias' + str(i-1)]
        else:
            x = torch.matmul(x, self.weights['weight' + str(i-1)])
        return x