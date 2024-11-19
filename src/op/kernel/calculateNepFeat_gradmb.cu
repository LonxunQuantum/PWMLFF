#include "./utilities/common.cuh"
#include "./utilities/nep_utilities.cuh"
#include <iostream>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                             __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

void launch_calculate_nepfeatmb_grad(
            const double * grad_output,
            const double * coeff2, 
            const double * coeff3, 
            const double * d12,
            const int   * NL, 
            const int   * atom_map, 
            double * sum_fxyz, 
            double * grad_coeff2, 
            double * grad_d12_radial, 
            double * grad_coeff3, 
            double * grad_d12_3b,
            const int rcut_radial, 
            const int rcut_angular,
            const int batch_size, 
            const int atom_nums, 
            const int maxneighs, 
            const int n_max_2b, 
            const int n_base_2b, 
            const int n_max_3b, 
            const int n_base_3b,
            const int lmax_3,
            const int lmax_4,
            const int lmax_5,
            const int n_types, 
            const int device_id
) {
    cudaSetDevice(device);
    const int BLOCK_SIZE = 64;
    const int N = atom_map.size();// N = natoms * batch_size
    const int grid_size = (N - 1) / BLOCK_SIZE + 1;
    const int num_types_sq = n_types * n_types;
    double rcinv_radial = 1.0 / rcut_radial;
    double rcinv_angular = 1.0 / rcut_angular;
    const int size_x12 = atom_map.size() * neigh_num;
    
    int feat_2b_num = 0;
    int feat_3b_num = 0;
    feat_2b_num = n_max_2b * n_base_2b;
    if (lmax_3 > 0) feat_3b_num += n_max_3b * n_base_3b;
    if (lmax_4 > 0) feat_3b_num += n_max_3b;
    if (lmax_5 > 0) feat_3b_num += n_max_3b;
    
    const size_t array_size = static_cast<size_t>(N) * n_types * n_base_2b;
    GPU_Vector<double> dfeat_c2(array_size);

    find_force_radial_small_box<<<grid_size, BLOCK_SIZE>>>(
        N,
        n_types,
        num_types_sq,
        neigh_num,
        lmax_3,
        lmax_4,
        lmax_5,
        feat_3b_num,
        rcut_radial,
        rcinv_radial,
        n_max_2b,
        n_base_2b,
        NL.data(),
        r12.data(),
        coeff2.data(),
        atom_map.data(),
        grad_output.data(),
        dfeat_c2.data(),
        grad_d12_radial.data()
    );
    CUDA_CHECK_KERNEL
    // calculate dfeat_c2
    dfeat_2c_calc<<<grid_size, BLOCK_SIZE>>>(
            grad_output.data(),
            dfeat_c2.data(),
            atom_map.data(),
            grad_coeff2.data(),
            N,
            n_max,
            n_base,
            feat_2b_num,
            n_types
    );
    CUDA_CHECK_KERNEL

    if lmax_3 > 0:
        find_force_angular_small_box<<<grid_size, BLOCK_SIZE>>>(
            paramb,
            annmb,
            N,
            N1,
            N2,
            NN_angular.data(),
            NL_angular.data(),
            type.data(),
            r12.data() + size_x12 * 3,
            r12.data() + size_x12 * 4,
            r12.data() + size_x12 * 5,
            nep_data.Fp.data(),
            nep_data.sum_fxyz.data(),
            is_dipole,

            force_per_atom.data(),
            force_per_atom.data() + N,
            force_per_atom.data() + N * 2,
            virial_per_atom.data());
        CUDA_CHECK_KERNEL

}
