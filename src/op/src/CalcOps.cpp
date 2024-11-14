#include <torch/torch.h>
#include <iostream>
#include "../include/CalcOps.h"
#include "../include/op_declare.h"

torch::autograd::variable_list CalculateForceFuncs::forward(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F,
    at::Tensor nghost_tensor) 
    {
        auto dims = list_neigh.sizes();
        int batch_size = dims[0];
        int natoms = dims[1];
        int neigh_num = dims[2] * dims[3];
        int nghost = nghost_tensor.item<int>();
        auto force = torch::zeros({batch_size, natoms + nghost, 3}, dE.options());
        force.slice(1, 0, natoms) = F;
        torch_launch_calculate_force(list_neigh, dE, Ri_d, batch_size, natoms, neigh_num, force, nghost);
        return {force};
        
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
        return {torch::autograd::Variable(), grad, torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable()};
    }

torch::autograd::variable_list CalculateForce::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F,
    at::Tensor nghost_tensor) 
    {
        ctx->save_for_backward({list_neigh, dE, Ri_d});
        return CalculateForceFuncs::forward(list_neigh, dE, Ri_d, F, nghost_tensor);
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
    at::Tensor F,
    at::Tensor nghost_tensor) 
    {
        return CalculateForce::apply(list_neigh, dE, Ri_d, F, nghost_tensor);
    }

// the following is the code virial
torch::autograd::variable_list CalculateVirialFuncs::forward(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor nghost_tensor) 
    {
        auto dims = list_neigh.sizes();
        int batch_size = dims[0];
        int natoms = dims[1];
        int neigh_num = dims[2] * dims[3];
        int nghost = nghost_tensor.item<int>();
        auto virial = torch::zeros({batch_size, 9}, dE.options());
        auto atom_virial = torch::zeros({batch_size, natoms + nghost, 9}, dE.options());
        torch_launch_calculate_virial_force(list_neigh, dE, Rij, Ri_d, batch_size, natoms, neigh_num, virial, atom_virial, nghost);
        return {virial, atom_virial};
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
    at::Tensor Ri_d,
    at::Tensor nghost_tensor) 
    {
        ctx->save_for_backward({list_neigh, dE, Rij, Ri_d});
        return CalculateVirialFuncs::forward(list_neigh, dE, Rij, Ri_d, nghost_tensor);
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
    at::Tensor Ri_d,
    at::Tensor nghost_tensor) 
    {
        return CalculateVirial::apply(list_neigh, dE, Rij, Ri_d, nghost_tensor);
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
        return {grad_out, torch::autograd::Variable()}; // f2的梯度 coeff的梯度 因为不需要，所以不计算，只占位
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


// the following is the code nep feature
torch::autograd::variable_list CalculateNepFeatFuncs::forward(
    at::Tensor coeff2,
    at::Tensor d12_radial,
    at::Tensor NL_radial,
    at::Tensor atom_map,
    double rcut_radial) 
    {
        auto dims_2b = coeff2.sizes();
        int64_t atom_types = dims_2b[0];
        int64_t n_max_2b = dims_2b[2];
        int64_t n_base_2b = dims_2b[3];

        auto dims_image = d12_radial.sizes();
        int64_t batch_size = dims_image[0];
        int64_t atom_nums = dims_image[1];
        int64_t maxneighs = dims_image[2];
        // options() 集成dtype和device
        auto feat_2b  = torch::zeros({batch_size, atom_nums, n_max_2b}, d12_radial.options());
        auto dfeat_c2 = torch::zeros({batch_size, atom_nums, atom_types, n_base_2b}, d12_radial.options());
        auto dfeat_2b = torch::zeros({batch_size, atom_nums, maxneighs, n_max_2b}, d12_radial.options());
        
        torch_launch_calculate_nepfeat(coeff2, d12_radial, NL_radial, atom_map, 
                                rcut_radial, feat_2b, dfeat_c2, dfeat_2b, 
                                    batch_size, atom_nums, maxneighs, n_max_2b, n_base_2b, atom_types);
                     // 交换coeff2维度，方便在核函数中检索c下标
        printf("dfeat_c2 [0,0,0,0] %f [0,0,1,0] %f\n", dfeat_c2[0,0,0,0], dfeat_c2[0,0,1,0]);
        return {feat_2b, dfeat_c2, dfeat_2b};
    }

torch::autograd::variable_list CalculateNepFeatFuncs::backward(
    torch::autograd::variable_list grad_output,
    at::Tensor coeff2,
    at::Tensor d12_radial,
    at::Tensor dfeat_c2,
    at::Tensor dfeat_2b,
    at::Tensor atom_map) 
    {
        auto dims_2b = coeff2.sizes();
        int64_t atom_types = dims_2b[0];
        int64_t n_max_2b = dims_2b[2];
        int64_t n_base_2b = dims_2b[3];

        auto dims_image = d12_radial.sizes();
        int64_t batch_size = dims_image[0];
        int64_t atom_nums = dims_image[1];
        int64_t maxneighs = dims_image[2];
        // options() 集成dtype和device
        auto grad_coeff2 = torch::zeros({atom_types, n_max_2b, atom_types, n_base_2b}, d12_radial.options());
        auto grad_d12_radial = torch::zeros({batch_size, atom_nums, maxneighs, 4}, d12_radial.options());
        // printf("back grad_output[0,0,:] %f %f %f %f %f\n", grad_output[0][0,0,0], grad_output[0][0,0,1], grad_output[0][0,0,2], grad_output[0][0,0,3], grad_output[0][0,0,4]);
        
        // torch_launch_calculate_nepfeat_grad(grad_output[0], dfeat_c2, dfeat_2b, atom_map, 
        //              batch_size, atom_nums, maxneighs, n_max_2b, n_base_2b, atom_types, grad_coeff2, grad_d12_radial);
        grad_coeff2 = grad_coeff2.permute({0, 2, 1, 3});
        return {grad_coeff2, grad_d12_radial, torch::autograd::Variable(), torch::autograd::Variable()}; // f2的梯度 coeff的梯度 因为不需要，所以不计算，只占位
    }

torch::autograd::variable_list CalculateNepFeat::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor coeff2,
    at::Tensor d12_radial,
    at::Tensor NL_radial,
    at::Tensor atom_map,
    double rcut_radial) 
    {
        auto results = CalculateNepFeatFuncs::forward(coeff2, d12_radial, NL_radial, atom_map, rcut_radial);
        auto feat_2b = results[0];
        auto dfeat_c2 = results[1];
        auto dfeat_2b = results[2];
        ctx->save_for_backward({coeff2, d12_radial, dfeat_c2, dfeat_2b, atom_map});
        return {feat_2b};
    }

torch::autograd::variable_list CalculateNepFeat::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto coeff2 = saved[0];
        auto d12_radial = saved[1];
        auto dfeat_c2 = saved[2];
        auto dfeat_2b = saved[3];
        auto atom_map = saved[4];
        return CalculateNepFeatFuncs::backward(grad_output, coeff2, d12_radial, dfeat_c2, dfeat_2b, atom_map);
    }

torch::autograd::variable_list calculateNepFeat(
    at::Tensor coeff2,
    at::Tensor d12_radial,
    at::Tensor NL_radial,
    at::Tensor atom_map,
    double rcut_radial) 
    {
        return CalculateNepFeat::apply(coeff2, d12_radial, NL_radial, atom_map, rcut_radial);
    }