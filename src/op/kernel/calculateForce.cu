#include <iostream>
__global__ void force_deriv_wrt_neighbors_a(
    double * force, 
    const double * net_deriv,
    const double * in_deriv,
    const int * nlist,
    const int nloc,
    const int nnei)
{  
    // idy -> nnei
    const unsigned int idx = blockIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned int idz = threadIdx.y;
    const int ndescrpt = nnei * 4;
    if (idy >= nnei) {
        return;
    }
    // deriv wrt neighbors
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    double force_tmp = 0.f;
    for (int idw = 0; idw < 4; ++idw) {
        force_tmp += net_deriv[idx * ndescrpt + idy * 4 + idw] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz];
    }
    const uint force_index = j_idx * 3 + idz;
    if (force_index >= 324) {
        printf("error index %d\n", j_idx);
    }
    atomicAdd(force + force_index, force_tmp);
}

template<typename DType>
__global__ void force_calc(
    DType * force, 
    const DType * net_deriv,
    const DType * in_deriv,
    const int * nlist,
    const int nloc,
    const int nnei)
{  
    const unsigned int block_id = blockIdx.x;
    const unsigned int atom_id = blockIdx.y;
    const unsigned int batch_id = blockIdx.z;
    const unsigned int neigh_index = threadIdx.x + block_id * blockDim.x;
    const unsigned int xyz_index = threadIdx.y;
    const int ndescrpt = nnei * 4;

    if (neigh_index >= nnei)
        return;
    const unsigned int nlist_offset = batch_id * nloc * nnei + atom_id * nnei + neigh_index;
    const int neigh_id = nlist[nlist_offset];

    if (neigh_id < 0) {
        return;
    }

    DType temp_a[4], temp_b[4];

    const unsigned int net_offset = batch_id * nloc * ndescrpt + atom_id * ndescrpt + neigh_index * 4;
    const unsigned int in_offset = batch_id * nloc * ndescrpt * 3 + atom_id * ndescrpt * 3 + neigh_index * 12;

    for (int i=0; i<4; i++) {
        temp_a[i] = net_deriv[net_offset + i];
        temp_b[i] = in_deriv[in_offset + i*3 + xyz_index];
    }

    DType res = 0.f;

    for (int i=0; i<4; i++) {
        res += temp_a[i] * temp_b[i];
    }

    const uint force_index = batch_id * nloc * 3 + neigh_id * 3 + xyz_index;

    atomicAdd(force + force_index, res);
}

template<typename DType>
void launch_calculate_force(
    const int * nblist,
    const DType * dE,
    const DType * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    DType * force
) {

#if 0
    const int LEN = 64;
    const int nblock = (neigh_num + LEN - 1) / LEN;
    dim3 block_grid(natoms, nblock);
    dim3 thread_grid(LEN, 3);
    force_deriv_wrt_neighbors_a<<<block_grid, thread_grid>>>(force, dE, Ri_d, nblist, natoms, neigh_num);
#endif

#if 1
    const int LEN = 256;
    const int nblock = (neigh_num + LEN - 1) / LEN;
    dim3 block_grid(nblock, natoms, batch_size);
    dim3 thread_grid(LEN, 3);
    force_calc<<<block_grid, thread_grid>>>(force, dE, Ri_d, nblist, natoms, neigh_num);
#endif
}

template void launch_calculate_force(const int * nblist,
    const float * dE,
    const float * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    float * force);

template void launch_calculate_force(const int * nblist,
    const double * dE,
    const double * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    double * force);