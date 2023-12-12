#include <torch/torch.h>
#include <iostream>
#include "../include/CalcOps.h"
#include "../include/op_declare.h"

torch::autograd::variable_list CalculateForceFuncs::forward(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F) 
    {
        auto dims = list_neigh.sizes();
        int batch_size = dims[0];
        int natoms = dims[1];
        int neigh_num = dims[2] * dims[3];
        torch_launch_calculate_force(list_neigh, dE, Ri_d, batch_size, natoms, neigh_num, F);
        return {F};
        
    }

torch::autograd::variable_list CalculateForceFuncs::backward(
    torch::autograd::variable_list grad_output,
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d) 
    {
        auto dims = list_neigh.sizes();
        int batch_size = dims[0];
        int natoms = dims[1];
        int neigh_num = dims[2] * dims[3];
        auto grad = torch::zeros_like(dE);
        torch_launch_calculate_force_grad(list_neigh, Ri_d, grad_output[0], batch_size, natoms, neigh_num, grad);
        return {torch::autograd::Variable(), grad, torch::autograd::Variable(), torch::autograd::Variable()};
    }

torch::autograd::variable_list CalculateForce::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F) 
    {
        ctx->save_for_backward({list_neigh, dE, Ri_d});
        return CalculateForceFuncs::forward(list_neigh, dE, Ri_d, F);
    }

torch::autograd::variable_list CalculateForce::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto list_neigh = saved[0];
        auto dE = saved[1];
        auto Ri_d = saved[2];
        return CalculateForceFuncs::backward(grad_output, list_neigh, dE, Ri_d);
    }
    
torch::autograd::variable_list calculateForce(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F) 
    {
        return CalculateForce::apply(list_neigh, dE, Ri_d, F);
    }

// the following is the code virial
torch::autograd::variable_list CalculateVirialFuncs::forward(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d) 
    {
        auto dims = list_neigh.sizes();
        int batch_size = dims[0];
        int natoms = dims[1];
        int neigh_num = dims[2] * dims[3];
        auto virial = torch::zeros({batch_size, 9}, dE.options());
        auto atom_virial = torch::zeros({batch_size, natoms, 9}, dE.options());
        torch_launch_calculate_virial_force(list_neigh, dE, Rij, Ri_d, batch_size, natoms, neigh_num, virial, atom_virial);
        return {virial};
    }

torch::autograd::variable_list CalculateVirialFuncs::backward(
    torch::autograd::variable_list grad_output,
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d) 
    {
        auto dims = list_neigh.sizes();
        int batch_size = dims[0];
        int natoms = dims[1];
        int neigh_num = dims[2] * dims[3];
        auto grad = torch::zeros_like(dE);
        torch_launch_calculate_virial_force_grad(list_neigh, Rij, Ri_d, grad_output[0], batch_size, natoms, neigh_num, grad);
        return {torch::autograd::Variable(), grad, torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable()};
    }

torch::autograd::variable_list CalculateVirial::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d) 
    {
        ctx->save_for_backward({list_neigh, dE, Rij, Ri_d});
        return CalculateVirialFuncs::forward(list_neigh, dE, Rij, Ri_d);
    }

torch::autograd::variable_list CalculateVirial::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto list_neigh = saved[0];
        auto dE = saved[1];
        auto Rij = saved[2];
        auto Ri_d = saved[3];
        return CalculateVirialFuncs::backward(grad_output, list_neigh, dE, Rij, Ri_d);
    }

torch::autograd::variable_list calculateVirial(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d) 
    {
        return CalculateVirial::apply(list_neigh, dE, Rij, Ri_d);
    }
    
// the following is the code compress
torch::autograd::variable_list CalculateCompressFuncs::forward(
    at::Tensor f2,
    at::Tensor coefficient) 
    {
        auto dims = coefficient.sizes();
        int64_t sij_num = dims[0];
        int64_t layer_node = dims[1];
        int64_t coe_num = dims[2];
        auto G = torch::empty({sij_num, layer_node}, f2.options());
        torch_launch_calculate_compress(f2, coefficient, sij_num, layer_node, coe_num, G);
        return {G};
    }

torch::autograd::variable_list CalculateCompressFuncs::backward(
    torch::autograd::variable_list grad_output,
    at::Tensor f2,
    at::Tensor coefficient) 
    {
        auto dims = coefficient.sizes();
        int64_t sij_num = dims[0];
        int64_t layer_node = dims[1];
        int64_t coe_num = dims[2];
        auto grad = torch::zeros({sij_num, layer_node}, f2.options());
        torch_launch_calculate_compress_grad(f2, coefficient, grad_output[0], sij_num, layer_node, coe_num, grad);
        auto grad_out = torch::sum(grad_output[0]*grad, -1).unsqueeze(-1); //对应坐标位置相乘，然后算加法
        return {grad_out, torch::autograd::Variable()};
    }

torch::autograd::variable_list CalculateCompress::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor f2,
    at::Tensor coefficient) 
    {
        ctx->save_for_backward({f2, coefficient});
        return CalculateCompressFuncs::forward(f2, coefficient);
    }

torch::autograd::variable_list CalculateCompress::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto f2 = saved[0];
        auto coefficient = saved[1];
        return CalculateCompressFuncs::backward(grad_output, f2, coefficient);
    }

torch::autograd::variable_list calculateCompress(
    at::Tensor f2,
    at::Tensor coefficient) 
    {
        return CalculateCompress::apply(f2, coefficient);
    }