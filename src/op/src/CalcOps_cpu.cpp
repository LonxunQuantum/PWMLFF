#include <torch/torch.h>
#include <iostream>
#include "../include/CalcOps.h"
// #include "../include/op_declare.h"


torch::autograd::variable_list calculateForce_cpu(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F,
    at::Tensor nghost_tensor) 
    {
        return {torch::autograd::Variable()};
    }

// the following is the code virial

torch::autograd::variable_list calculateVirial_cpu(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor nghost_tensor) 
    {
        return {torch::autograd::Variable()};
    }
    
// the following is the code compress

torch::autograd::variable_list calculateCompress_cpu(
    at::Tensor f2,
    at::Tensor coefficient) 
    {
        return {torch::autograd::Variable()};
    }