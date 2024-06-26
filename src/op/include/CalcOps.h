#include <torch/torch.h>
#include <iostream>

class CalculateForceFuncs {
    public:
        static torch::autograd::variable_list forward(
            at::Tensor list_neigh,
            at::Tensor dE,
            at::Tensor Ri_d,
            at::Tensor F,
            at::Tensor nghost_tensor);

        static torch::autograd::variable_list backward(
            torch::autograd::variable_list grad_output,
            at::Tensor list_neigh,
            at::Tensor dE,
            at::Tensor Ri_d);
};

class CalculateForce : public torch::autograd::Function<CalculateForce> {
    public:
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor list_neigh,
            at::Tensor dE,
            at::Tensor Ri_d,
            at::Tensor F,
            at::Tensor nghost_tensor);

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::variable_list grad_output);
};

torch::autograd::variable_list calculateForce(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F,
    at::Tensor nghost_tensor);

torch::autograd::variable_list calculateForce_cpu(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F,
    at::Tensor nghost_tensor);

// the following is the code virial
class CalculateVirialFuncs {
    public:
        static torch::autograd::variable_list forward(
            at::Tensor list_neigh,
            at::Tensor dE,
            at::Tensor Rij,
            at::Tensor Ri_d,
            at::Tensor nghost_tensor
        );

        static torch::autograd::variable_list backward(
            torch::autograd::variable_list grad_output,
            at::Tensor list_neigh,
            at::Tensor dE,
            at::Tensor Rij,
            at::Tensor Ri_d
        );
};

class CalculateVirial : public torch::autograd::Function<CalculateVirial> {
    public:
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor list_neigh,
            at::Tensor dE,
            at::Tensor Rij,
            at::Tensor Ri_d,
            at::Tensor nghost_tensor);

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::variable_list grad_output);
};

torch::autograd::variable_list calculateVirial(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor nghost_tensor);

torch::autograd::variable_list calculateVirial_cpu(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor nghost_tensor);

// the following is the code compress
class CalculateCompressFuncs {
    public:
        static torch::autograd::variable_list forward(
            at::Tensor f2,
            at::Tensor coefficient);

        static torch::autograd::variable_list backward(
            torch::autograd::variable_list grad_output,
            at::Tensor f2,
            at::Tensor coefficient);
};

class CalculateCompress : public torch::autograd::Function<CalculateCompress> {
    public:
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor f2,
            at::Tensor coefficient);

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx,
            torch::autograd::variable_list grad_output);
};

torch::autograd::variable_list calculateCompress(
    at::Tensor f2,
    at::Tensor coefficient);

torch::autograd::variable_list calculateCompress_cpu(
    at::Tensor f2,
    at::Tensor coefficient);