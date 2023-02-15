#include <iostream>
// #include "calculate_force.h"

template <typename DType, int THREADS_PER_BLOCK>
__global__ void atom_virial_reduction(
    DType * virial, 
    const DType * atom_virial,
    const int nall)
{
    unsigned int batch_index = blockIdx.y;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;

    __shared__ DType data[THREADS_PER_BLOCK];
    
    data[tid] = (DType)0.;
    for (int ii = tid; ii < nall; ii += THREADS_PER_BLOCK) {
        data[tid] += atom_virial[batch_index * nall * 9 + ii * 9 + bid];
    }
    __syncthreads(); 
    // do reduction in shared memory
    for (int ii = THREADS_PER_BLOCK >> 1; ii > 0; ii >>= 1) {
        if (tid < ii) {
            data[tid] += data[tid + ii];
        }
        __syncthreads();
    }
    // write result for this block to global memory
    if (tid == 0) virial[batch_index * 9 +  bid] = data[0];
}

template<typename DType>
__global__ void virial_deriv_wrt_neighbors_a(
    DType * atom_virial,
    const DType * dE,
    const DType * Ri_d,
    const DType * Rij,
    const int * nlist,
    const int natoms,
    const int neigh_num) 
{
    const unsigned int block_id = blockIdx.x;
    const unsigned int atom_id = blockIdx.y;
    const unsigned int batch_id = blockIdx.z;
    const unsigned int neigh_index = threadIdx.x + block_id * blockDim.x;
    const unsigned int virial_index = threadIdx.y;
    const int ndescrpt = neigh_num * 4;

    if (neigh_index >= neigh_num)
        return;

    const unsigned int nlist_offset = batch_id * natoms * neigh_num + atom_id * neigh_num + neigh_index;
    const int neigh_id = nlist[nlist_offset];

    if (neigh_id < 0) {
        return;
    }

    DType virial_tmp = (DType)0.;
    const unsigned int dE_offset = batch_id * natoms * ndescrpt + atom_id * ndescrpt + neigh_index * 4;
    const unsigned int Ri_d_offset = batch_id * natoms * ndescrpt * 3 + atom_id * ndescrpt * 3 + neigh_index * 12;
    const unsigned int Rij_offset = batch_id * natoms * neigh_num * 3 + atom_id * neigh_num * 3 + neigh_index * 3;

    for (int idw = 0; idw < 4; ++idw) {
        // virial_tmp += dE[idx * ndescrpt + idy * 4 + idw] * Ri[idx * neigh_num * 3 + idy * 3 + idz % 3] * Ri_d[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz / 3];
        virial_tmp += dE[dE_offset + idw] * Ri_d[Ri_d_offset + idw * 3 + virial_index / 3] * Rij[Rij_offset + virial_index % 3];
    }

    const uint index = batch_id * natoms * 9 + neigh_id * 9 + virial_index;

    atomicAdd(atom_virial + index, virial_tmp);
}

template<typename DType>
void launch_calculate_virial_force(
    const int * nblist,
    const DType * dE,
    const DType * Rij,
    const DType * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    DType * virial,
    DType * atom_virial
)
{
    // cudaMemset(virial, 0, sizeof(DTYPE) * 9 * batch_size);
    // cudaMemset(atom_virial, 0, sizeof(DTYPE) * 9 * natoms * batch_size);
        
    const int LEN = 16;
    int nblock = (neigh_num + LEN - 1) / LEN;

    dim3 block_grid(nblock, natoms, batch_size);
    dim3 thread_grid(LEN, 9);
    // compute virial of a frame
    virial_deriv_wrt_neighbors_a<<<block_grid, thread_grid>>>(atom_virial, dE, Ri_d, Rij, nblist, natoms, neigh_num);

    block_grid = dim3(9, batch_size);
    // reduction atom_virial to virial
    atom_virial_reduction<DType, 256> <<<block_grid, 256>>>(virial, atom_virial, natoms);

}


template void launch_calculate_virial_force(
    const int * nblist,
    const float * dE,
    const float * Rij,
    const float * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    float * virial,
    float * atom_virial
);

template void launch_calculate_virial_force(
    const int * nblist,
    const double * dE,
    const double * Rij,
    const double * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    double * virial,
    double * atom_virial
);