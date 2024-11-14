#include "./utilities/common.cuh"
#include "./utilities/nep_utilities.cuh"
#include <iostream>

template<typename DType>
__global__ void feat_2b_calc(
        const DType * coeff2,
        const DType * d12_radial,
        const int * NL_radial,
        const int * atom_map,
        const DType rcut_radial,
        const DType rcinv_radial,
        DType * feat_2b,
        DType * dfeat_c2,
        DType * dfeat_2b,
        const int batch_size,
        const int natoms,
        const int neigh_num,
        const int n_max,
        const int n_base,
        const int num_types,
        const int num_types_sq)
{  
    // 计算全局线程索引，每个线程处理一个中心原子
    int global_atom_index = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算批次和原子索引
    int batch_id = global_atom_index / natoms;
    int atom_id = global_atom_index % natoms;
    int c_index = 0;
    if (batch_id < batch_size && atom_id < natoms) {
        int t1 = atom_map[atom_id];
        DType q[MAX_DIM] = {static_cast<DType>(0.0)};
        int neigh_start_idx = batch_id * natoms * neigh_num + atom_id * neigh_num;
        int r12_start_idx =  batch_id * natoms * neigh_num * 4 + atom_id * neigh_num * 4;
        int feat_start_idx = batch_id * natoms * n_max + atom_id * n_max; 
        int dfeat_c_start_idx = batch_id * natoms * num_types * n_base + atom_id * num_types * n_base;
        int dfeat_2b_start_idx = batch_id * natoms * neigh_num * n_max + atom_id * neigh_num * n_max;
        int c_start_idx = t1 * num_types * n_max * n_base;

        for (int i1=0; i1 < neigh_num; ++i1) {
            int n2 = NL_radial[neigh_start_idx + i1]-1;
            if (n2 < 0) return;
            int t2 = atom_map[n2];
            int c_I_J_idx = c_start_idx + t2 * n_max * n_base;
            int rij_idx = r12_start_idx + i1*4;
            int d2b_idx = dfeat_2b_start_idx + i1 * n_max;
            DType d12 = d12_radial[rij_idx]; // [rij, x, y, z]
            DType fc12, fcp12;
            find_fc_and_fcp(rcut_radial, rcinv_radial, d12, fc12, fcp12);
            DType fn12[MAX_NUM_N];
            DType fnp12[MAX_NUM_N];
            find_fn_and_fnp(
                n_base, rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
            for (int n = 0; n < n_max; ++n) {
                DType gn12 = static_cast<DType>(0.0);
                for (int k = 0; k < n_base; ++k) {
                    // c2的维度为[Nmax, Nbas, I, J]对c的索引会更方便
                    c_index =  c_I_J_idx + n * n_base + k;
                    gn12 += fn12[k] * coeff2[c_index];
                    dfeat_2b[d2b_idx + k] += fnp12[k]*coeff2[c_index];
                    if (n == 0 and k == 0) {
                        printf("batch %d I %d J %d n %d k %d c %f cid %d rij %f rid %d\n", batch_id, atom_id, n2, n, k, coeff2[c_index], c_index, d12, rij_idx);
                    }
                    if (n == 0) {
                        dfeat_c2[dfeat_c_start_idx + t2 * n_base + k] += fc12; //[batch, n_atom, J_Ntypes, N_base]
                    }
                }
                feat_2b[feat_start_idx + n] += gn12;
            }
        }//neighs
        // printf("batch %d atom %d feat [%f %f %f %f %f] dfc [%f %f]\n", batch_id, atom_id, 
        //     feat_2b[feat_start_idx], feat_2b[feat_start_idx+1], feat_2b[feat_start_idx+2], feat_2b[feat_start_idx+3], feat_2b[feat_start_idx+4],
        //         dfeat_c2[dfeat_c_start_idx], dfeat_c2[dfeat_c_start_idx+1*n_base]);
    }
}

template<typename DType>
void launch_calculate_nepfeat(
        const DType * coeff2,
        const DType * d12_radial,
        const int * NL_radial,
        const int * atom_map,
        const double rcut_radial,
        DType * feat_2b,
        DType * dfeat_c2,
        DType * dfeat_2b,
        const int batch_size,
        const int natoms,
        const int neigh_num,
        const int n_max,
        const int n_base,
        const int num_types,
        const int device_id
) {
    cudaSetDevice(device_id);
    int num_types_sq = num_types * num_types;
    int BLOCK_SIZE = 64;
    int grid_size = (natoms * batch_size - 1) / BLOCK_SIZE + 1;
    // float rcinv_radial = 1/rcut_radial;
    DType rcinv_radial = static_cast<DType>(1.0 / rcut_radial);
    feat_2b_calc<<<grid_size, BLOCK_SIZE>>>(
                coeff2, d12_radial, NL_radial, atom_map, 
                    static_cast<DType>(rcut_radial), rcinv_radial,
                        feat_2b, dfeat_c2, dfeat_2b, 
                            batch_size, natoms, neigh_num, 
                                n_max, n_base, num_types, num_types_sq
                            );
    // 还是按照中心原子作为线程来处理，后期batchsize多了之后，中心原子数量会多，利用率就会上来
}

template void launch_calculate_nepfeat(
            const float * coeff2,
            const float * d12_radial,
            const int * NL_radial,
            const int * atom_map,
            const double rcut_radial,
            float * feat_2b,
            float * dfeat_c2,
            float * dfeat_2b,
            const int batch_size,
            const int natoms,
            const int neigh_num,
            const int n_max,
            const int n_base,
            const int num_types,
            const int device_id
                );

template void launch_calculate_nepfeat(
            const double * coeff2,
            const double * d12_radial,
            const int * NL_radial,
            const int * atom_map,
            const double rcut_radial,
            double * feat_2b,
            double * dfeat_c2,
            double * dfeat_2b,
            const int batch_size,
            const int natoms,
            const int neigh_num,
            const int n_max,
            const int n_base,
            const int num_types,
            const int device_id
    );