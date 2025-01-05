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
                        torch::Tensor &dfeat_2b_noc,
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
                        int64_t multi_feat_num,
                        torch::Tensor &grad_coeff2,
                        torch::Tensor &grad_d12_radial
);

void torch_launch_calculate_nepfeat_secondgradout(
                        const torch::Tensor &grad_second,
                        const torch::Tensor &dfeat_b,
                        const int64_t batch_size, 
                        const int64_t atom_nums, 
                        const int64_t maxneighs, 
                        const int64_t n_max, 
                        torch::Tensor &gradsecond_gradout
);

void torch_launch_calculate_nepfeat_secondgradout_c2(
                        const torch::Tensor &grad_second,
                        const torch::Tensor &de_feat,
                        const torch::Tensor &dfeat_2b_noc,
                        const torch::Tensor &atom_map,
                        const torch::Tensor &NL_radial,
                        const int64_t batch_size, 
                        const int64_t atom_nums, 
                        const int64_t maxneighs, 
                        const int64_t n_max_2b, 
                        const int64_t n_base_2b,
                        const int64_t atom_types,
                        const int64_t multi_feat_num,
                        torch::Tensor &gradsecond_c2
);


void torch_launch_calculate_nepmbfeat(
                        const torch::Tensor &coeff3,
                        const torch::Tensor &d12,
                        const torch::Tensor &NL,
                        const torch::Tensor &atom_map,
                        torch::Tensor &feat_3b,
                        torch::Tensor &dfeat_c3,
                        torch::Tensor &dfeat_3b,
                        torch::Tensor &dfeat_3b_noc,
                        torch::Tensor &sum_fxyz,
                        const double rcut,
                        int64_t batch_size,
                        int64_t natoms,
                        int64_t neigh_num,
                        int64_t n_max_3b, 
                        int64_t n_base_3b,
                        int64_t lmax_3,
                        int64_t lmax_4,
                        int64_t lmax_5,
                        int64_t n_types
);

void torch_launch_calculate_nepmbfeat_grad(
                        const torch::Tensor &grad_output,
                        const torch::Tensor &coeff3,
                        const torch::Tensor &d12,
                        const torch::Tensor &NL,                        
                        const torch::Tensor &atom_map,
                        const double rcut_angular,
                        int64_t batch_size, 
                        int64_t atom_nums, 
                        int64_t maxneighs, 
                        int64_t feat_2b_num, 
                        int64_t n_max_3b, 
                        int64_t n_base_3b,
                        int64_t lmax_3,
                        int64_t lmax_4,
                        int64_t lmax_5,
                        int64_t n_types,
                        torch::Tensor &sum_fxyz,
                        torch::Tensor &grad_coeff3,
                        torch::Tensor &grad_d12_3b,
                        torch::Tensor &dsnlm_dc,
                        torch::Tensor &dfeat_drij
);

void torch_launch_calculate_nepmbfeat_secondgradout(
                        const torch::Tensor &grad_second,
                        const torch::Tensor &dfeat_b,
                        const int64_t batch_size, 
                        const int64_t atom_nums, 
                        const int64_t maxneighs, 
                        const int64_t feat_mb_nums, 
                        torch::Tensor &gradsecond_gradout
);

void torch_launch_calculate_nepmbfeat_secondgradout_c3(
                        const torch::Tensor &grad_second,
                        const torch::Tensor &d12,
                        const torch::Tensor &NL,
                        const torch::Tensor &de_feat,
                        const torch::Tensor &dsnlm_dc,
                        const torch::Tensor &sum_fxyz,
                        const torch::Tensor &atom_map,
                        const torch::Tensor &coeff3,
                        const double rcut_angular,
                        const int64_t batch_size, 
                        const int64_t atom_nums, 
                        const int64_t maxneighs, 
                        const int64_t n_max_3b, 
                        const int64_t n_base_3b,
                        const int64_t atom_types,
                        const int64_t lmax_3,
                        const int64_t lmax_4,
                        const int64_t lmax_5,
                        const int64_t feat_2b_num,
                        const int64_t multi_feat_num,
                        torch::Tensor &gradsecond_c3
);

//abandon
void torch_launch_calculate_nepfeatmb(
                        const torch::Tensor &coeff2,
                        const torch::Tensor &coeff3,
                        const torch::Tensor &d12,
                        const torch::Tensor &NL,
                        const torch::Tensor &atom_map,
                        const double rcut_radial,
                        const double rcut_angular,
                        torch::Tensor &feats,
                        torch::Tensor &sum_fxyz,
                        int64_t batch_size,
                        int64_t natoms,
                        int64_t neigh_num,
                        int64_t n_max_2b,
                        int64_t n_base_2b,
                        int64_t n_max_3b,
                        int64_t n_base_3b,
                        int64_t lmax_3,
                        int64_t lmax_4,
                        int64_t lmax_5,
                        int64_t n_types
);

void torch_launch_calculate_nepfeatmb_grad(
                        const torch::Tensor &grad_output,
                        const torch::Tensor &coeff2,
                        const torch::Tensor &coeff3,
                        const torch::Tensor &d12,
                        const torch::Tensor &NL,                        
                        const torch::Tensor &atom_map,
                        const double rcut_radial,
                        const double rcut_angular,
                        int64_t batch_size, 
                        int64_t atom_nums, 
                        int64_t maxneighs, 
                        int64_t n_max_2b, 
                        int64_t n_base_2b,
                        int64_t n_max_3b, 
                        int64_t n_base_3b,
                        int64_t lmax_3,
                        int64_t lmax_4,
                        int64_t lmax_5,
                        int64_t n_types,
                        torch::Tensor &sum_fxyz,
                        torch::Tensor &grad_coeff2,
                        torch::Tensor &grad_d12_radial,
                        torch::Tensor &grad_coeff3,
                        torch::Tensor &grad_d12_3b
);
