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
                        int64_t batch_size,
                        int64_t natoms,
                        int64_t neigh_num,
                        int64_t n_max,
                        int64_t n_base,
                        int64_t n_types
){
    auto dtype = d12_radial.dtype();
    int device_id = d12_radial.device().index();
    if (dtype == torch::kFloat32)
    {
        launch_calculate_nepfeat<float>(
            (const float *) coeff2.data_ptr(),
            (const float *) d12_radial.data_ptr(),
            (const int *) NL_radial.data_ptr(),
            (const int *) atom_map.data_ptr(),
            rcut_radial, 
            (float *) feat_2b.data_ptr(),
            (float *) dfeat_c2.data_ptr(),
            (float *) dfeat_2b.data_ptr(),
            batch_size, natoms, neigh_num, n_max, n_base, n_types, 
            device_id
        );
    } else if (dtype == torch::kFloat64)
    {
        launch_calculate_nepfeat<double>(
            (const double *) coeff2.data_ptr(),
            (const double *) d12_radial.data_ptr(),
            (const int *) NL_radial.data_ptr(),
            (const int *) atom_map.data_ptr(),
            rcut_radial, 
            (double *) feat_2b.data_ptr(),
            (double *) dfeat_c2.data_ptr(),
            (double *) dfeat_2b.data_ptr(),
            batch_size, natoms, neigh_num, n_max, n_base, n_types, 
            device_id
        );
    }
    else
        printf("data type error!");
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
    if (dtype == torch::kFloat32)
    {
        launch_calculate_nepfeat_grad<float>(
            (const float *) grad_output.data_ptr(),
            (const float *) dfeat_c2.data_ptr(),
            (const float *) dfeat_2b.data_ptr(),
            (const int *) atom_map.data_ptr(),
            (float *) grad_coeff2.data_ptr(),
            (float *) grad_d12_radial.data_ptr(),
            batch_size, atom_nums, maxneighs, n_max_2b, n_base_2b, n_types, 
            device_id
        );
    } else if (dtype == torch::kFloat64)
    {
        launch_calculate_nepfeat_grad<double>(
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
    else
        printf("data type error!");
}
