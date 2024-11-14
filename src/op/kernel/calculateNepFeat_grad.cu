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

template<typename DType>
__global__ void dfeat_2c_calc(
            const DType * grad_output,
            const DType * dfeat_c2,
            const int * atom_map,
            DType * grad_coeff2,
            int64_t batch_size,
            int64_t atoms,
            int64_t n_max,
            int64_t n_base,
            int64_t n_types,
            int64_t n_types_sq)
{  
    const uint batch_idx = blockIdx.x;
    const uint atom_idx = blockIdx.y;
    const uint n_max_idx = threadIdx.x;
    const uint n_type_idx = threadIdx.y;
    const uint n_base_idx = threadIdx.z;
    if (batch_idx >= batch_size || atom_idx >= atoms || n_max_idx >= n_max ||
    n_type_idx >= n_types || n_base_idx >= n_base) return;
    const uint atom_type = atom_map[atom_idx];  // 获取原子类型
    // if (n_type_idx != atom_type) return; // 只累加对应的类型
    const uint A_idx = batch_idx * atoms * n_max + atom_idx * n_max + n_max_idx;
    const uint B_idx = batch_idx * atoms * n_types * n_base + atom_idx * n_types * n_base + n_type_idx * n_base + n_base_idx;
    const uint C_idx = atom_type * n_max * n_types * n_base + n_max_idx * n_types * n_base + n_type_idx * n_base + n_base_idx;
    // 将 A 和 B 的元素相乘并累加到 C 中
    // Dtype res = 0.f;
    // res = grad_output[A_idx] * dfeat_c2[B_idx];
    atomicAdd(grad_coeff2+C_idx, grad_output[A_idx] * dfeat_c2[B_idx]);
    // if (atom_idx == 0){
    //     printf("batch %d atom %d n %d j_t %d k %d grad[%d] = %f dfeat_c2[%d] = %f\n", batch_idx, atom_idx, n_max_idx, n_type_idx, n_base_idx, A_idx, grad_output[A_idx], B_idx, dfeat_c2[B_idx]);
    // }
    // 针对double类型使用专门的atomicAdd
    // if (std::is_same<DType, double>::value) {
    //     atomicAdd(reinterpret_cast<double*>(&grad_coeff2[C_idx]), grad_output[A_idx] * dfeat_c2[B_idx]);
    // } 
    // else {
    //     atomicAdd(&grad_coeff2[C_idx], grad_output[A_idx] * dfeat_c2[B_idx]);
    // }
}

template<typename DType>
__global__ void dfeat_2b_calc_bk(
            const DType * grad_output,
            const DType * dfeat_2b,
            const int * atom_map,
            DType * grad_d12_radial,
            int64_t batch_size,
            int64_t natoms,
            int64_t neigh_num,
            int64_t n_max,
            int64_t n_base,
            int64_t n_types,
            int64_t n_types_sq)
{  
    int batch_idx = blockIdx.x;
    int atom_idx = blockIdx.y;
    int neigh_num_idx = threadIdx.x;
    // 检查边界条件
    if (batch_idx >= batch_size || atom_idx >= natoms || neigh_num_idx >= neigh_num) return;
    // 初始化累加器
    float sum = 0.0f;
    // 对 n_max 维度进行累加
    for (int n_max_idx = 0; n_max_idx < n_max; ++n_max_idx) {
        int grad_output_idx = batch_idx * natoms * n_max + atom_idx * n_max + n_max_idx;
        int dfeat_2b_idx = batch_idx * natoms * n_max * neigh_num + atom_idx * n_max * neigh_num + n_max_idx * neigh_num + neigh_num_idx;
        
        sum += grad_output[grad_output_idx] * dfeat_2b[dfeat_2b_idx];
    }
    // 将结果写入 result 张量
    int result_idx = batch_idx * natoms * neigh_num * 4 + atom_idx * neigh_num * 4 + neigh_num_idx * 4;
    grad_d12_radial[result_idx] = sum;
}

template<typename DType>
__global__ void dfeat_2b_calc(
            const DType * grad_output,
            const DType * dfeat_2b,
            DType * grad_d12_radial,
            int64_t batch_size,
            int64_t natoms,
            int64_t neigh_num,
            int64_t n_max
            )
{  
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batch_size * natoms * neigh_num) {
        int batch_idx = index / (natoms * neigh_num);
        int atom_idx = (index % (natoms * neigh_num)) / neigh_num;
        int neigh_idx = index % neigh_num;

        DType sum = 0;
        // 对 n_max 维度加和，逐元素乘法并加和
        for (int i = 0; i < n_max; ++i) {
            sum += grad_output[batch_idx * natoms * n_max + atom_idx * n_max + i] * 
                   dfeat_2b[batch_idx * natoms * neigh_num * n_max + atom_idx * neigh_num * n_max + neigh_idx * n_max + i];
        }

        grad_d12_radial[batch_idx * natoms * neigh_num * 4 + atom_idx * neigh_num * 4 + neigh_idx * 4] = sum;
    }
}

// grad_output shape is [batch_size, natoms, n_max_2b]
// dfeat_c2 shape is    [batch_size, n_types_J, n_base_2b]
// dfeat_2b shape is    [batch_size, natoms, n_max_2b, maxneighs]
// atom_map shape is    [natoms]
// grad_coeff2 shape is [n_types_I, n_max_2b, n_types_J, n_base_2b]
// grad_d12_radial shape is [batch_size, natoms, n_max_2b, maxneighs]
template<typename DType>
void launch_calculate_nepfeat_grad(
            const DType * grad_output,
            const DType * dfeat_c2,
            const DType * dfeat_2b,
            const int * atom_map,
            DType * grad_coeff2,
            DType * grad_d12_radial,
            const int batch_size, 
            const int natoms, 
            const int neigh_num, 
            const int n_max_2b, 
            const int n_base_2b,
            const int n_types, 
            const int device
) {
    cudaSetDevice(device);
    dim3 threads(n_max_2b, n_types, n_base_2b);
    dim3 blocks(batch_size, natoms);
    int n_types_sq = n_types * n_types;
    // printf("dfeat_2c_calc\n");
    dfeat_2c_calc<<<blocks, threads>>>(
                grad_output, dfeat_c2, atom_map, grad_coeff2, 
                            batch_size, natoms, n_max_2b, n_base_2b, n_types, n_types_sq);

    // threads = dim3(neigh_num);
    // blocks = dim3(batch_size, natoms);
    // // printf("dfeat_2b_calc\n");
    // dfeat_2b_calc_bk<<<blocks, threads>>>(
    //         grad_output, dfeat_2b, atom_map, grad_d12_radial, 
    //                     batch_size, natoms, neigh_num, n_max_2b, n_base_2b, n_types, n_types_sq);

    int BLOCK_SIZE = 64;
    int grid_size = (batch_size * natoms * neigh_num - 1) / BLOCK_SIZE + 1;
    dfeat_2b_calc<<<grid_size, BLOCK_SIZE>>>(
            grad_output, dfeat_2b, grad_d12_radial, 
                        batch_size, natoms, neigh_num, n_max_2b);
}

template void launch_calculate_nepfeat_grad(
            const float * grad_output,
            const float * dfeat_c2,
            const float * dfeat_2b,
            const int * atom_map,
            float * grad_coeff2,
            float * grad_d12_radial,
            const int batch_size, 
            const int atom_nums, 
            const int neigh_num, 
            const int n_max_2b, 
            const int n_base_2b,
            const int n_types, 
            const int device
                );

template void launch_calculate_nepfeat_grad(
            const double * grad_output,
            const double * dfeat_c2,
            const double * dfeat_2b,
            const int * atom_map,
            double * grad_coeff2,
            double * grad_d12_radial,
            const int batch_size, 
            const int atom_nums, 
            const int neigh_num, 
            const int n_max_2b, 
            const int n_base_2b,
            const int n_types, 
            const int device
                );