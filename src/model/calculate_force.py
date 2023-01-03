from torch import nn
import torch
from torch.autograd import Function
import op, op_grad

class CalculateForce(Function):
    @staticmethod
    def forward(ctx, list_neigh, dE, Ri_d, F):
        dims = list_neigh.shape
        batch_size = dims[0]
        natoms = dims[1]
        neigh_num = dims[2] * dims[3]
        ctx.save_for_backward(list_neigh, dE, Ri_d)
        op.calculate_force(list_neigh, dE, Ri_d, batch_size, natoms, neigh_num, F)
        return F

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        list_neigh = inputs[0]
        dE = inputs[1]
        Ri_d = inputs[2]
        dims = list_neigh.shape
        batch_size = dims[0]
        natoms = dims[1]
        neigh_num = dims[2] * dims[3]
        grad = torch.zeros_like(dE)
        op_grad.calculate_force_grad(list_neigh, Ri_d, grad_output, batch_size, natoms, neigh_num, grad)
        return (None, grad, None, None)