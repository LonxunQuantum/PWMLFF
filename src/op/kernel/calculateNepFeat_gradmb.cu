#include "./utilities/common.cuh"
#include "./utilities/nep_utilities.cuh"
#include "./utilities/error.cuh"
#include "./utilities/gpu_vector.cuh"
#include "./utilities/nep3_small_box.cuh"
#include <iostream>

void print_grad_coeff3(double* d_grad_coeff3, int n_types, int n_max_3b, int n_base_3b) {
    int total_size = n_types * n_types * n_max_3b * n_base_3b;
    
    // 在主机上分配内存
    std::vector<double> h_grad_coeff3(total_size);
    
    // 从设备拷贝到主机
    cudaMemcpy(h_grad_coeff3.data(), d_grad_coeff3, total_size * sizeof(double), cudaMemcpyDeviceToHost);
    
    // 打印内容
    for (int t1 = 0; t1 < n_types; ++t1) {
        for (int t2 = 0; t2 < n_types; ++t2) {
            for (int n = 0; n < n_max_3b; ++n) {
                for (int b = 0; b < n_base_3b; ++b) {
                    int idx = t1 * n_types * n_max_3b * n_base_3b
                              + t2 * n_max_3b * n_base_3b
                              + n * n_base_3b
                              + b;
                    std::cout << "grad_coeff3[" << t1 << "][" << t2 << "][" << n << "][" << b << "] = "
                              << h_grad_coeff3[idx] << std::endl;
                }
            }
        }
    }
}

void print_dfeat_c3(double* d_dfeat_c3, int N, int n_types, int n_max_3b, int n_base_3b) {
    int total_size = N * n_types * n_max_3b * n_base_3b;
    
    // 在主机上分配内存
    std::vector<double> h_dfeat_c3(total_size);
    
    // 从设备拷贝到主机
    cudaMemcpy(h_dfeat_c3.data(), d_dfeat_c3, total_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); // 确保拷贝完成

    // 打印内容
    for (int i = 0; i < N; ++i) {
        for (int t = 0; t < n_types; ++t) {
            for (int n = 0; n < n_max_3b; ++n) {
                for (int b = 0; b < n_base_3b; ++b) {
                    int idx = i * n_types * n_max_3b * n_base_3b 
                              + t * n_max_3b * n_base_3b 
                              + n * n_base_3b 
                              + b;
                    std::cout << "dfeat_c3[" << i << "][" << t << "][" << n << "][" << b << "] = "
                              << h_dfeat_c3[idx] << std::endl;
                }
            }
        }
    }
}

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
            const int neigh_num, 
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
    cudaSetDevice(device_id);
    const int BLOCK_SIZE = 64;
    const int N = batch_size * atom_nums; // N = natoms * batch_size
    const int grid_size = (N - 1) / BLOCK_SIZE + 1;
    const int num_types_sq = n_types * n_types;
    double rcinv_radial = 1.0 / rcut_radial;
    double rcinv_angular = 1.0 / rcut_angular;
    
    int feat_2b_num = 0;
    int feat_3b_num = 0;
    feat_2b_num = n_max_2b;
    if (lmax_3 > 0) feat_3b_num += n_max_3b * lmax_3;
    if (lmax_4 > 0) feat_3b_num += n_max_3b;
    if (lmax_5 > 0) feat_3b_num += n_max_3b;
    
    // GPU_Vector<double> dfeat_c2(N * n_types * n_base_2b);

    GPU_Vector<double> dfeat_c2(N * n_types * n_max_2b * n_base_2b);

    find_force_radial_small_box<<<grid_size, BLOCK_SIZE>>>(
        N,
        n_types,
        num_types_sq,
        neigh_num,
        lmax_3,
        lmax_4,
        lmax_5,
        feat_2b_num + feat_3b_num,
        rcut_radial,
        rcinv_radial,
        n_max_2b,
        n_base_2b,
        NL,
        d12,
        coeff2,
        atom_map,
        grad_output,
        dfeat_c2.data(),
        grad_d12_radial
    );
    CUDA_CHECK_KERNEL
    
    int total_elements = N * n_max_2b * n_base_2b;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    aggregate_features<<<num_blocks, threads_per_block>>>(
        dfeat_c2.data(), 
        atom_map, 
        grad_coeff2, 
        N, 
        n_types, 
        n_max_2b, 
        n_base_2b);
    CUDA_CHECK_KERNEL

    if (lmax_3 > 0){
        GPU_Vector<double> dfeat_c3(N * n_types * n_max_3b * n_base_3b, 0.0);
        GPU_Vector<double> dsnlm_dc(N * n_types * n_max_3b * n_base_3b, 0.0);
        
        // find_force_angular_small_box<<<grid_size, BLOCK_SIZE>>>(
        //     N,
        //     n_types,
        //     num_types_sq,
        //     neigh_num,
        //     lmax_3,
        //     lmax_4,
        //     lmax_5,
        //     feat_2b_num + feat_3b_num,
        //     feat_3b_num,
        //     rcut_angular,
        //     rcinv_angular,
        //     n_max_2b, 
        //     n_base_2b, 
        //     n_max_3b, 
        //     n_base_3b,
        //     NL,
        //     d12,
        //     coeff3,
        //     atom_map,
        //     grad_output,
        //     sum_fxyz,
        //     dfeat_c3.data(),
        //     dsnlm_dc.data(),
        //     grad_d12_3b
        // );
        CUDA_CHECK_KERNEL
        
        // print_dfeat_c3(dfeat_c3.data(), N, n_types, n_max_3b, n_base_3b);

        total_elements = N * n_max_3b * n_base_3b;
        threads_per_block = 256;
        num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        aggregate_features<<<num_blocks, threads_per_block>>>(
        dfeat_c3.data(), 
        atom_map, 
        grad_coeff3, 
        N, 
        n_types, 
        n_max_3b, 
        n_base_3b);
        CUDA_CHECK_KERNEL
    }

    // print_grad_coeff3(grad_coeff3, n_types, n_max_3b, n_base_3b);

}
