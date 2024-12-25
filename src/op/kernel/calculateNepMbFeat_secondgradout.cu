#include "./utilities/common.cuh"
#include "./utilities/nep_utilities.cuh"
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

// __global__ void compute_gradsecond_c2(
//     const double *grad_second, // Shape: [batch_size, atom_nums, maxneighs, 4]
//     const double *de_feat, // Shape: [batch_size, atom_nums, n_max_2b]
//     const double *dfeat_2b_noc, // Shape: [batch_size, atom_nums, maxneighs, n_base_2b, 4]
//     double *tmp_grad, // Shape: [batch_size, atom_nums, maxneighs, n_base_2b]
//     int batch_size,
//     int atom_nums,
//     int maxneighs,
//     int n_max_2b,
//     int n_base_2b,
//     int multi_feat_num)
// {
//     // 计算总元素数目
//     int total_elements = batch_size * atom_nums * maxneighs;
//     int elem_idx = threadIdx.x + blockIdx.x * blockDim.x; // 网格中的元素索引

//     if (elem_idx < total_elements) {
//         // 计算对应的 atom_idx 和 maxneigh_idx
//         int batch_idx = elem_idx / (atom_nums * maxneighs);
//         int remaining = elem_idx % (atom_nums * maxneighs);
//         int atom_idx = remaining / maxneighs;
//         int maxneigh_idx = remaining % maxneighs;
        
//         int dfeat_start = batch_idx * atom_nums * (n_max_2b + multi_feat_num) + atom_idx * (n_max_2b + multi_feat_num);
//         int dnoc_start = (batch_idx * atom_nums + atom_idx) * maxneighs * n_base_2b * 4 + maxneigh_idx * n_base_2b * 4;
//         int grad2_start = (batch_idx * atom_nums + atom_idx) * maxneighs * 4 + maxneigh_idx * 4;
//         int tmp_grad_start = (batch_idx * atom_nums + atom_idx) * maxneighs * n_max_2b * n_base_2b + maxneigh_idx * n_max_2b * n_base_2b;

//         double noc0 = 0.0, noc1 = 0.0, noc2 = 0.0, noc3 = 0.0;
//         double dfeat_val = 0.0;
        
//         double grad0 = grad_second[grad2_start];
//         double grad1 = grad_second[grad2_start + 1];
//         double grad2 = grad_second[grad2_start + 2];
//         double grad3 = grad_second[grad2_start + 3];

//         for (int n = 0; n < n_max_2b; ++n) {
//             dfeat_val = de_feat[dfeat_start + n];
//             for (int k = 0; k < n_base_2b; ++k) {
//                 noc0 = dfeat_2b_noc[dnoc_start + k * 4];       // 2b is 0
//                 noc1 = dfeat_2b_noc[dnoc_start + k * 4 + 1];   // 2b is 0
//                 noc2 = dfeat_2b_noc[dnoc_start + k * 4 + 2];   // 2b is 0
//                 noc3 = dfeat_2b_noc[dnoc_start + k * 4 + 3];   // 2b is 0

//                 // 更新tmp_grad数组
//                 tmp_grad[tmp_grad_start + n * n_base_2b + k] += dfeat_val * (noc0 * grad0 + noc1 * grad1 + noc2 * grad2 + noc3 * grad3);
                
//                 // if (batch_idx == 0 and atom_idx == 1 and maxneigh_idx < 20){
//                 //     printf("tmp_grad[b %d i %d j %d n %d k %d] = %f dfeat_val %f secondgrad %f %f %f %f noc0-4 %f %f %f %f\n", 
//                 //         batch_idx, atom_idx, maxneigh_idx, n, k, 
//                 //             tmp_grad[tmp_grad_start + n * n_base_2b + k], dfeat_val, grad0, grad1, grad2, grad3, noc0, noc1, noc2,noc3 );
//                 // }
//             }
//         }
//     }
// }

// __global__ void reduce_kernel(
//     double *tmp_grad,          // 输入的tmp_grad数组，维度为(batch_size, atom_nums, maxneighs, n_max_2b, n_base_2b)
//     const int *atom_map,            // atom_map数组，维度为(atom_nums)
//     const int *NL_radial,           // NL_radial数组，维度为(batch_size, atom_nums, maxneighs)
//     const int batch_size,           // batch_size的大小
//     const int atom_nums,            // atom_nums的大小
//     const int maxneighs,            // maxneighs的大小
//     const int n_max_2b,             // n_max_2b的大小
//     const int n_base_2b,            // n_base_2b的大小
//     const int atom_types,           // atom_types的大小
//     double *output             // 输出数组，维度为(atom_types, atom_types, n_max_2b, n_base_2b)
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 线程索引
//     // 计算对应的 batch_idx, atom_idx 和 maxneigh_idx
//     int b = idx / (atom_nums * maxneighs);  
//     int i = (idx % (atom_nums * maxneighs)) / maxneighs;  
//     int maxneighs_j = idx % maxneighs;
//     if (b < batch_size && i < atom_nums && maxneighs_j < maxneighs) {
//         int n2 = NL_radial[b * atom_nums * maxneighs + i * maxneighs + maxneighs_j] -1;
//         if (n2 < 0) return;
//         int tmp_grad_start = b * atom_nums * maxneighs * n_max_2b * n_base_2b
//                             + i * maxneighs * n_max_2b * n_base_2b
//                                 + maxneighs_j * n_max_2b * n_base_2b;
//         // 临时变量用于存储计算结果
//         // double result[MAX_NUM_BEADS] = {0.0};

//         // 第一步：根据NL_radial规约tmp_grad的maxneighs维度
//         // 获取NL_radial中对应邻居的元素类型
        
//         int atom_type_j = atom_map[NL_radial[b * atom_nums * maxneighs + i * maxneighs + maxneighs_j] - 1];  // 使用NL_radial索引获取邻居的元素类型
//         int atom_type_i = atom_map[i];
        
//         for (int k = 0; k < n_max_2b; ++k) {
//             for (int l = 0; l < n_base_2b; ++l) {
//                 // if (b == 0 and i == 0 and maxneighs_j < 10){
//                 // printf("t1 %d t2 %d tmp_grad[b %d i %d j %d n %d k %d] = %f\n", 
//                 //     atom_type_i, atom_type_j, b, i, maxneighs_j, k, l, 
//                 //         tmp_grad[tmp_grad_start + k * n_base_2b + l]);
//                 // }

//                 // 将之前的结果累加到输出
//                 atomicAdd(&output[atom_type_i * atom_types * n_max_2b * n_base_2b
//                                 + atom_type_j * n_max_2b * n_base_2b
//                                 + k * n_base_2b
//                                 + l], 
//                                 tmp_grad[tmp_grad_start + k * n_base_2b + l]
//                 );
//             }
//         }
//     }
// }

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
    const double * NL,
    const double * de_dfeat,
    const double * sum_fxyz,
    const int * atom_map,
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
    int total_elements = batch_size * atom_nums * maxneighs;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    // GPU_Vector<double> tmp_grad(batch_size * atom_nums * maxneighs * n_max_2b * n_base_2b, 0.0);
    find_angular_gardc_small_box<<<num_blocks, threads_per_block>>>(
        d12, 
        NL,
        atom_map,
        batch_size, 
        atom_nums, 
        maxneighs, 
        n_max_3b,
        n_base_3b,
        atom_types,
        lmax_3,
        lmax_4,
        lmax_5,
        feat_2b_num,
        multi_feat_num
        );
    cudaDeviceSynchronize();

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
