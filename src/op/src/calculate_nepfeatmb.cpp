#include <torch/extension.h>
// #include "op_declare.h"
#include "../include/calculate_nepfeatmb.h"
// #include "../include/calculate_nepfeat_grad.h"

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
){
    int device_id = d12.device().index();
    launch_calculate_nepfeatmb(
    (const double *) coeff2.data_ptr(),
    (const double *) coeff3.data_ptr(),
    (const double *) d12.data_ptr(),
    (const int *) NL.data_ptr(),
    (const int *) atom_map.data_ptr(),
    rcut_radial,
    rcut_angular,
    (double *) feats.data_ptr(),
    (double *) sum_fxyz.data_ptr(),
    batch_size,
    natoms,
    neigh_num,
    n_max_2b,
    n_base_2b,
    n_max_3b,
    n_base_3b,
    lmax_3,
    lmax_4,
    lmax_5,
    n_types,
    device_id
    );
}

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
                        torch::Tensor &grad_d12_3b)
{
    int device_id = d12.device().index();
    launch_calculate_nepfeatmb_grad(
    (const double *) grad_output.data_ptr(),
    (const double *) coeff2.data_ptr(),
    (const double *) coeff3.data_ptr(),
    (const double *) d12.data_ptr(),
    (const int *) NL.data_ptr(),
    (const int *) atom_map.data_ptr(),
    (double *) sum_fxyz.data_ptr(),
    (double *) grad_coeff2.data_ptr(),
    (double *) grad_d12_radial.data_ptr(),
    (double *) grad_coeff3.data_ptr(),
    (double *) grad_d12_3b.data_ptr(),
    rcut_radial,
    rcut_angular,
    batch_size, 
    atom_nums, 
    maxneighs, 
    n_max_2b, 
    n_base_2b,
    n_max_3b, 
    n_base_3b,
    lmax_3,
    lmax_4,
    lmax_5,
    n_types,
    device_id
    );
}
