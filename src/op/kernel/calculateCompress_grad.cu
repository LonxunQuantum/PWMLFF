#include <iostream>

template<typename DType>
__global__ void compress_calc_grad(
    DType * Grad,
    const DType * f2,
    const DType * coefficient,
    const DType * grad_output,
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
            // DType coeff_3 = coefficient[tid * layer_node * 4 + i * 4 + 3];
            Grad[tid * layer_node + i] = 3 * f2_val * f2_val * coeff_0 + 2 * f2_val * coeff_1 + coeff_2;
            // Grad[tid * layer_node] = Grad[tid * layer_node]  + (3 * f2_val * f2_val * coeff_0 + 2 * f2_val * coeff_1 + coeff_2) * grad_output[tid * layer_node + i];
        }
    }
}

template<typename DType>
void launch_calculate_compress_grad(
    const DType * f2,
    const DType * coefficient,
    const DType * grad_output,
    const int sij_num,
    const int layer_node,
    const int coe_num,
    DType * Grad
) {
    const int blockSize = 256;
    const int gridSize = (sij_num + blockSize -1) / blockSize;
    // Launch the kernel
    compress_calc_grad<<<gridSize, blockSize>>>(Grad, f2, coefficient, grad_output, sij_num, layer_node, coe_num);
}

template void launch_calculate_compress_grad(
    const float * f2,
    const float * coefficient,
    const float * grad_output,
    const int sij_num,
    const int layer_node,
    const int coe_num,
    float * Grad
    );

template void launch_calculate_compress_grad(
    const double * f2,
    const double * coefficient,
    const double * grad_output,
    const int sij_num,
    const int layer_node,
    const int coe_num,
    double * Grad
    );