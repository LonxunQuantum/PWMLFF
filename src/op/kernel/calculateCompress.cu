#include <iostream>

template<typename DType>
__global__ void compress_calc(
    DType * G,
    const DType * f2,
    const DType * coefficient,
    const int sij_num,
    const int layer_node,
    const int coe_num
    )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < sij_num) {
        DType f2_val = f2[tid];
        for (int i = 0; i < layer_node; i++) {
            DType coeff_0 = coefficient[tid * layer_node * 4 + i * 4 + 0];
            DType coeff_1 = coefficient[tid * layer_node * 4 + i * 4 + 1];
            DType coeff_2 = coefficient[tid * layer_node * 4 + i * 4 + 2];
            DType coeff_3 = coefficient[tid * layer_node * 4 + i * 4 + 3];

            G[tid * layer_node + i] = f2_val * f2_val * f2_val * coeff_0 + f2_val * f2_val * coeff_1 +
                            f2_val * coeff_2 + coeff_3;
        }
    }
}

template<typename DType>
void launch_calculate_compress(
    const DType * f2,
    const DType * coefficient,
    const int sij_num,
    const int layer_node,
    const int coe_num,
    DType * G
) {
    const int blockSize = 256;
    const int gridSize = (sij_num + blockSize -1) / blockSize;
    // Launch the kernel
    compress_calc<<<gridSize, blockSize>>>(G, f2, coefficient, sij_num, layer_node, coe_num);
}

template void launch_calculate_compress(
    const float * f2,
    const float * coefficient,
    const int sij_num,
    const int layer_node,
    const int coe_num,
    float * G
    );

template void launch_calculate_compress(
    const double * f2,
    const double * coefficient,
    const int sij_num,
    const int layer_node,
    const int coe_num,
    double * G
    );