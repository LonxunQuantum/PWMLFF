#include <torch/extension.h>

void torch_launch_calculate_compress(
                       const torch::Tensor &f2,
                       const torch::Tensor &coefficient,
                       int64_t sij_num,
                       int64_t layer_node,
                       int64_t coe_num,
                       const torch::Tensor &G
);

void torch_launch_calculate_compress_grad(
                       const torch::Tensor &f2,
                       const torch::Tensor &coefficient,
                       const torch::Tensor &grad_output,
                       int64_t sij_num,
                       int64_t layer_node,
                       int64_t coe_num,
                       const torch::Tensor &Grad
);

void torch_launch_calculate_force(torch::Tensor &nblist,
                       const torch::Tensor &dE,
                       const torch::Tensor &Ri_d,
                       int64_t batch_size,
                       int64_t natoms,
                       int64_t neigh_num,
                       const torch::Tensor &force,
                       int64_t nghost
);

void torch_launch_calculate_force_grad(torch::Tensor &nblist,
                       const torch::Tensor &Ri_d,
                       const torch::Tensor &net_grad,
                       int64_t batch_size,
                       int64_t natoms,
                       int64_t neigh_num,
                       const torch::Tensor &grad
);

void torch_launch_calculate_virial_force(torch::Tensor &nblist,
                       const torch::Tensor &dE,
                       const torch::Tensor &Rij,
                       const torch::Tensor &Ri_d,
                       int64_t batch_size,
                       int64_t natoms,
                       int64_t neigh_num,
                       const torch::Tensor &virial_force,
                       const torch::Tensor &atom_virial_force,
                       int64_t nghost
);

void torch_launch_calculate_virial_force_grad(torch::Tensor &nblist,
                       const torch::Tensor &Rij,
                       const torch::Tensor &Ri_d,
                       const torch::Tensor &net_grad,
                       int64_t batch_size,
                       int64_t natoms,
                       int64_t neigh_num,
                       const torch::Tensor &grad
);