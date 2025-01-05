#include "./utilities/common.cuh"
#include "./utilities/nep3_small_box.cuh"
#include "./utilities/nep3_small_box_mbgrad.cuh"
#include "./utilities/error.cuh"
#include "./utilities/gpu_vector.cuh"
#include <iostream>
#include <cuda_runtime.h>

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

__global__ void compute_gradsecond_mbgradout(
    const double *grad_second, // Shape: [batch_size, atom_nums, maxneighs, 4]
    const double *dfeat_2b,    // Shape: [batch_size, atom_nums, maxneighs, feat_mb_num, 4]
    double *gradsecond_gradout, // Shape: [batch_size, atom_nums, feat_mb_num]
    int batch_size,
    int atom_nums,
    int maxneighs,
    int feat_mb_num)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int atom_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int grad_idx = 0;
    int feat_idx = 0;
    double tmp_val = 0.0;
    double tmp_grad0 = 0.0;
    double tmp_grad1 = 0.0;
    double tmp_grad2 = 0.0;
    double tmp_grad3 = 0.0;
    if (batch_idx < batch_size && atom_idx < atom_nums) {
        for (int neigh = 0; neigh < maxneighs; ++neigh) {
            grad_idx = (batch_idx * atom_nums + atom_idx) * maxneighs * 4 + neigh * 4;
            tmp_grad0 = grad_second[grad_idx+0];
            tmp_grad1 = grad_second[grad_idx+1];
            tmp_grad2 = grad_second[grad_idx+2];
            tmp_grad3 = grad_second[grad_idx+3];
            for (int n = 0; n < feat_mb_num; ++n) {
                feat_idx = (batch_idx * atom_nums + atom_idx) * maxneighs * feat_mb_num * 4 + neigh * feat_mb_num * 4 + n * 4;
                gradsecond_gradout[(batch_idx * atom_nums + atom_idx) * feat_mb_num + n] += tmp_grad0 * dfeat_2b[feat_idx] + 
                                                                                            tmp_grad1 * dfeat_2b[feat_idx + 1] +
                                                                                            tmp_grad2 * dfeat_2b[feat_idx + 2] +
                                                                                            tmp_grad3 * dfeat_2b[feat_idx + 3];
            }
        }
    }
}


void launch_calculate_nepmbfeat_secondgradout(
    const double * grad_second,
    const double * dfeat_b,
    double * gradsecond_gradout,
    const int batch_size, 
    const int atom_nums, 
    const int maxneighs, 
    const int feat_mb_nums, 
    const int device
) {
    cudaSetDevice(device);
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, (atom_nums + threadsPerBlock.y - 1) / threadsPerBlock.y);

    compute_gradsecond_mbgradout<<<numBlocks, threadsPerBlock>>>(
        grad_second, 
        dfeat_b, 
        gradsecond_gradout,
        batch_size, 
        atom_nums, 
        maxneighs, 
        feat_mb_nums
        );

    CUDA_CHECK_KERNEL
}

void launch_calculate_nepmbfeat_secondgradout_c3(
    const double * grad_second,
    const double * d12,
    const int * NL,
    const double * de_dfeat,
    const double * dsnlm_dc,
    const double * sum_fxyz,
    const int * atom_map,
    const double * coeff3,
    double * gradsecond_c3,
    const double rcut_angular,
    const int batch_size, 
    const int atom_nums, 
    const int maxneighs, 
    const int n_max_3b, 
    const int n_base_3b, 
    const int atom_types, 
    const int lmax_3,
    const int lmax_4,
    const int lmax_5,
    const int feat_2b_num,
    const int multi_feat_num,
    const int device
){
    cudaSetDevice(device);
    // // 每个线程块的线程数 (这里选择 8 * 16 * 2 = 256，保证不会超过 1024)
    double rcinv_angular = 1.0 / rcut_angular;
    int atom_types_sq = atom_types * atom_types;
    // 按照邻居粒度存在一个问题，线程内分配的数组太多了，可能炸显存，但是速度会快，后面可以试试
    // int total_elements = batch_size * atom_nums * maxneighs;
    // int threads_per_block = 256;
    // int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    const int BLOCK_SIZE = 64;
    const int N = batch_size * atom_nums; // N = natoms * batch_size
    const int grid_size = (N - 1) / BLOCK_SIZE + 1;

    GPU_Vector<double> dfeat_c3(N * atom_types * n_max_3b * n_base_3b, 0.0);
    printf("==start launch_calculate_nepmbfeat_secondgradout_c3 === N=%d\n", N);
    find_angular_gardc_small_box<<<grid_size, BLOCK_SIZE>>>(
        N,
        grad_second,
        d12, 
        NL,
        de_dfeat-feat_2b_num, 
        dsnlm_dc,
        sum_fxyz,
        atom_map,
        coeff3,
        dfeat_c3.data(),
        rcut_angular,
        rcinv_angular,
        batch_size, 
        atom_nums, 
        maxneighs, 
        n_max_3b,
        n_base_3b,
        atom_types,
        atom_types_sq,
        lmax_3,
        lmax_4,
        lmax_5,
        feat_2b_num,
        multi_feat_num
        );
    cudaDeviceSynchronize();
    printf("==end launch_calculate_nepmbfeat_secondgradout_c3 ===\n");
    int total_elements = N * n_max_3b * n_base_3b;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    printf("==== tmp_c3 =====\n");
    std::vector<double> tmp_c3(N * atom_types * n_max_3b * n_base_3b);
    dfeat_c3.copy_to_host(tmp_c3.data());
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < atom_types; j++) {
            for (int n = 0; n < n_max_3b; n++) {
                printf("tmp_c3[i %d][J %d][n %d][k:] = ", i, j, n);
                for(int k = 0; k < n_base_3b; k++) {
                    printf("%f ", tmp_c3[i * atom_types * n_max_3b * n_base_3b + j * n_max_3b * n_base_3b + n * n_base_3b + k]);
                }
                printf("\n");
            }
        }
    }

    aggregate_features<<<num_blocks, threads_per_block>>>(
    dfeat_c3.data(), 
    atom_map, 
    gradsecond_c3, 
    N, 
    atom_types, 
    n_max_3b, 
    n_base_3b);
    CUDA_CHECK_KERNEL
    // 检查下对 atom_type 的处理，是否复制了多分份？
    // 先搞定三体，再累加四体和五体
    // 先计算gradsecond * de/dfeat 后面的
    // 然后再和gradsecond * de/dfeat合并
    // 再规约到对C的梯度
    // 注意规约 de/dfeat时候的起始位置偏移

    // reduce_kernel<<<num_blocks, threads_per_block>>>(
    //     tmp_grad.data(), atom_map, NL_radial,
    //     batch_size, atom_nums, maxneighs,
    //     n_max_2b, n_base_2b, atom_types, gradsecond_c2
    // );
    
    // CUDA_CHECK_KERNEL
    // cudaDeviceSynchronize();
}
