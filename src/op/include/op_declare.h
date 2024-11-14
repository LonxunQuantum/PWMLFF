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

void torch_launch_calculate_nepfeat(
                        const torch::Tensor &coeff2,
                        const torch::Tensor &d12_radial,
                        const torch::Tensor &NL_radial,
                        const torch::Tensor &atom_map,
                        const double rcut_radial,
                        torch::Tensor &feat_2b,
                        torch::Tensor &dfeat_c2,
                        torch::Tensor &dfeat_2b,
                        int64_t batch_size,
                        int64_t natoms,
                        int64_t neigh_num,
                        int64_t n_max,
                        int64_t n_base,
                        int64_t n_types
);

void torch_launch_calculate_nepfeat_grad(
                        const torch::Tensor &grad_output,
                        const torch::Tensor &dfeat_c2,
                        const torch::Tensor &dfeat_2b,
                        const torch::Tensor atom_map,
                        int64_t batch_size, 
                        int64_t atom_nums, 
                        int64_t maxneighs, 
                        int64_t n_max_2b, 
                        int64_t n_base_2b,
                        int64_t n_types,
                        torch::Tensor &grad_coeff2,
                        torch::Tensor &grad_d12_radial
);