#include <torch/extension.h>
// #include "op_declare.h"
#include "../include/calculate_nepfeat.h"
// #include "../include/calculate_nepfeat_grad.h"

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
){
    auto dtype = d12_radial.dtype();
    int device_id = d12_radial.device().index();
    launch_calculate_nepfeat(
        (const double *) coeff2.data_ptr(),
        (const double *) d12_radial.data_ptr(),
        (const int *) NL_radial.data_ptr(),
        (const int *) atom_map.data_ptr(),
        rcut_radial, 
        (double *) feat_2b.data_ptr(),
        (double *) dfeat_c2.data_ptr(),
        (double *) dfeat_2b.data_ptr(),
        (double *) dfeat_2b_noc.data_ptr(),
        batch_size, natoms, neigh_num, n_max, n_base, n_types, 
        device_id
    );
}

void torch_launch_calculate_nepfeat_grad(const torch::Tensor &grad_output, 
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
                                torch::Tensor &grad_d12_radial)
{
    auto dtype = dfeat_c2.dtype();
    int device_id = dfeat_c2.device().index();
    launch_calculate_nepfeat_grad(
        (const double *) grad_output.data_ptr(),
        (const double *) dfeat_c2.data_ptr(),
        (const double *) dfeat_2b.data_ptr(),
        (const int *) atom_map.data_ptr(),
        (double *) grad_coeff2.data_ptr(),
        (double *) grad_d12_radial.data_ptr(),
        batch_size, atom_nums, maxneighs, n_max_2b, n_base_2b, n_types, 
        device_id
    );
}


void torch_launch_calculate_nepfeat_secondgradout(
                        const torch::Tensor &grad_second,
                        const torch::Tensor &dfeat_b,
                        const int64_t batch_size, 
                        const int64_t atom_nums, 
                        const int64_t maxneighs, 
                        const int64_t n_max, 
                        torch::Tensor &gradsecond_gradout)
{
    auto dtype = dfeat_b.dtype();
    int device_id = dfeat_b.device().index();
    launch_calculate_nepfeat_secondgradout(
        (const double *) grad_second.data_ptr(),
        (const double *) dfeat_b.data_ptr(),
        (double *) gradsecond_gradout.data_ptr(),
        batch_size, atom_nums, maxneighs, n_max, device_id
    );
}


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
                        torch::Tensor &gradsecond_c2
)
{
    auto dtype = de_feat.dtype();
    int device_id = de_feat.device().index();
    launch_calculate_nepfeat_secondgradout_c2(
        (const double *) grad_second.data_ptr(),
        (const double *) de_feat.data_ptr(),
        (const double *) dfeat_2b_noc.data_ptr(),
        (const int *) atom_map.data_ptr(),
        (const int *) NL_radial.data_ptr(),
        (double *) gradsecond_c2.data_ptr(),
        batch_size, atom_nums, maxneighs, n_max_2b, n_base_2b, atom_types, device_id
    );
}
