#include "./utilities/common.cuh"
#include "./utilities/nep_utilities.cuh"
#include "./utilities/nep3_small_box.cuh"
#include <iostream>

void launch_calculate_nepfeatmb(
            const double * coeff2,
            const double * coeff3,
            const double * r12,
            const int * NL,
            const int * atom_map,
            const double rcut_radial,
            const double rcut_angular,
            double * feats,
            double * sum_fxyz,
            const int batch_size,
            const int natoms,
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

    find_descriptor_small_box<<<grid_size, BLOCK_SIZE>>>(
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
        rcut_angular,
        rcinv_angular,
        n_max_2b,
        n_base_2b,
        n_max_3b,
        n_base_3b,
        NL.data(),
        coeff2.data(),
        coeff3.data(),
        feats.data(),
        atom_map.data(),
        r12.data(),
        sum_fxyz.data());
    CUDA_CHECK_KERNEL
    
    // cudaDeviceSynchronize();
    // 打印 dfeat_c2 数据 (将 dfeat_c2 从设备内存复制到主机内存)
    // double * h_dfeat_c2 = new double[batch_size * natoms * num_types * n_base];  // 主机内存中的副本
    // cudaMemcpy(h_dfeat_c2, dfeat_c2, batch_size * natoms * num_types * n_base * sizeof(double), cudaMemcpyDeviceToHost);
    
    // // 打印 dfeat_c2 的一部分 (例如前 5 个元素)
    // printf("dfeat_c2 (first few values):\n");
    // for (int i = 0; i < batch_size; ++i) {
    //     for (int j = 0; j < natoms; ++j) {
    //         for (int k = 0; k < num_types; ++k) {
    //             for (int p = 0; p < n_base; ++p) {
    //                 printf("dfeat_c2[%d][%d][%d][%d] = %f\n", i, j, k, p, h_dfeat_c2[i * natoms * num_types * n_base + j* num_types * n_base + k * n_base + p]);
    //             }
    //         }
    //     }
    // }

    // 打印 dfeat_2b 数据 (将 dfeat_2b 从设备内存复制到主机内存)
    // double * h_dfeat_2b = new double[batch_size * natoms * neigh_num * n_max];  // 主机内存中的副本
    // cudaMemcpy(h_dfeat_2b, dfeat_2b, batch_size * natoms * neigh_num * n_max * sizeof(double), cudaMemcpyDeviceToHost);
    
    // 打印 dfeat_2b 的一部分 (例如前 5 个元素)
    // printf("dfeat_2b (first few values):\n");
    // for (int i = 0; i < batch_size; ++i) {
    //     for (int j = 0; j < 10; ++j) {
    //         for (int k = 0; k < neigh_num; ++k) {
    //             printf("dfeat_2b[%d][%d][%d] = ", i, j, k);
    //             for (int l = 0; l < n_max; ++l) {
    //                 printf(" %f ", h_dfeat_2b[i * natoms * neigh_num * n_max + j * neigh_num * n_max + k * n_max + l]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }
}
