/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "model/box.cuh"
#include "nep3.cuh"
#include "utilities/common.cuh"
#include "utilities/nep_utilities.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
static __device__ __inline__ double atomicAdd(double* address, double val)
{
  unsigned long long* address_as_ull = (unsigned long long*)address;
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

static __global__ void dfeat_2c_calc(
    const double* grad_output,       // [N, feat_2b_num + feat_3b_num]
    const double* dfeat_c2,          // [N, n_types, n_base]
    const int* atom_map,             // [N]
    double* grad_coeff2,             // [n_types, n_types, n_max, n_base]
    int64_t N,
    int64_t n_max,
    int64_t n_base,
    int64_t feat_2b_num,
    int64_t n_types
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int atom_type = atom_map[idx]; // 获取当前原子类型
        if (atom_type < 0 || atom_type >= n_types) return; // 防止无效类型
        
        // 共享内存分配，用于存储 grad_output 和 dfeat_c2 切片
        extern __shared__ double shared_mem[];
        double* shared_grad_output = shared_mem; // [feat_2b_num]
        double* shared_dfeat_c2 = &shared_mem[feat_2b_num]; // [n_types * n_base]

        // 加载数据到共享内存
        const double* grad_output_slice = grad_output + idx * (feat_2b_num + feat_3b_num);
        const double* dfeat_c2_slice = dfeat_c2 + idx * n_types * n_base;
        
        // 将 grad_output 和 dfeat_c2 加载到共享内存
        for (int f = threadIdx.x; f < feat_2b_num; f += blockDim.x) {
            shared_grad_output[f] = grad_output_slice[f];
        }
        for (int t = threadIdx.x; t < n_types * n_base; t += blockDim.x) {
            shared_dfeat_c2[t] = dfeat_c2_slice[t];
        }

        __syncthreads(); // 等待所有线程加载数据

        // 计算梯度并更新 grad_coeff2
        for (int t2 = 0; t2 < n_types; ++t2) { // 遍历原子类型 t2
            for (int n = 0; n < n_max; ++n) {  // 遍历 n_max
                for (int b = 0; b < n_base; ++b) { // 遍历 n_base
                    double contrib = 0.0;
                    // 计算 grad_output 和 dfeat_c2 的点积
                    for (int f = 0; f < feat_2b_num; ++f) { // 遍历 feat_2b_num
                        contrib += shared_grad_output[f] * shared_dfeat_c2[t2 * n_base + b];
                    }

                    // 使用 atomicAdd 累加到 grad_coeff2
                    atomicAdd(
                        &grad_coeff2[atom_type * n_types * n_max * n_base +
                                     t2 * n_max * n_base +
                                     n * n_base + b],
                        contrib
                    );
                }
            }
        }
    }
}


static __global__ void find_descriptor_small_box(
  const int N,
  const int num_types,
  const int num_types_sq,
  const int neigh_num,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_nums,
  const double rc_radial,
  const double rcinv_radial,
  const double rc_angular,
  const double rcinv_angular,
  const int n_max_radial,
  const int basis_size_radial,
  const int n_max_angular,
  const int basis_size_angular,
  const int* g_NL_radial,
  const double * coeff2,
  const double * coeff3,
  const double * feats,
  const int* __restrict__ g_type,
  const double* __restrict__ g_d12_radial,
  double* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    // double q[MAX_DIM] = {static_cast<double>(0.0)};
    // get radial descriptors
    double q[MAX_DIM] = {0.0};
    int neigh_start_idx = n1 * neigh_num;
    int r12_start_idx =  n1 * neigh_num * 4;
    int feat_start_idx = n1 * feat_nums; 
    int c2_start_idx = t1 * num_types * n_max_radial * basis_size_radial;
    for (int i1 = 0; i1 < neigh_num; ++i1) {
      int n2 = g_NL_radial[neigh_start_idx + i1] - 1;
      if (n2 < 0) break;
      int t2 = g_type[n2];
      int c_I_J_idx = c2_start_idx + t2 * n_max_radial * basis_size_radial;
      int rij_idx = r12_start_idx + i1*4;
      double d12 = g_d12_radial[rij_idx];
      double fc12;
      find_fc(rc_radial, rcinv_radial, d12, fc12);
      
      double fn12[MAX_NUM_N];

      find_fn(basis_size_radial, rcinv_radial, d12, fc12, fn12);
      for (int n = 0; n <= n_max_radial; ++n) {
        double gn12 = 0.0;
        for (int k = 0; k <= basis_size_radial; ++k) {
          int c_index = c_I_J_idx + n * basis_size_radial + k;
          gn12 += fn12[k] * coeff2[c_index];
        }
        // 2b feats
        q[n] += gn12;
      }
    }

    // get angular descriptors
    int c3_start_idx = t1 * num_types * n_max_angular * basis_size_angular;
    int sum_s_start_idx = n1 * n_max_3b * NUM_OF_ABC;
    for (int n = 0; n <= n_max_angular; ++n) {
      double s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < neigh_num; ++i1) {
        int n2 = NL_radial[neigh_start_idx + i1]-1;
        if (n2 < 0) break;
        int t2 = g_type[n2];
        int rij_idx = r12_start_idx + i1*4;
        double d12 = g_d12_radial[rij_idx];
        if (d12 > rc_angular) break;
        double r12[3] = {g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2], g_d12_radial[rij_idx+3]};
        double fc12;
        find_fc(rc_angular, rcinv_angular, d12, fc12);
        double fn12[MAX_NUM_N];
        find_fn(basis_size_angular, rcinv_angular, d12, fc12, fn12);
        double gn12 = 0.0;
        int c_I_J_idx = c3_start_idx + t2 * n_max_angular * basis_size_angular;
        for (int k = 0; k <= basis_size_angular; ++k) {
          int c_index = c_I_J_idx + n * basis_size_angular + k;
          gn12 += fn12[k] * coeff3[c_index];
        }
        accumulate_s(d12, r12[0], r12[1], r12[2], gn12, s);
      }
      if (L_max5 == 1) {
          find_q_with_5body(n_max_angular + 1, n, s, q + (n_max_radial + 1));
      } else if (L_max4 ==2) {
        find_q_with_4body(n_max_angular + 1, n, s, q + (n_max_radial + 1));
      } else {
        find_q(n_max_angular + 1, n, s, q + (n_max_radial + 1));
      }
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[sum_s_start_idx + n * NUM_OF_ABC + abc] = s[abc];
      }
    }
    for (int n1 = 0; n1 < feat_nums; ++n1) {
      feats[feat_start_idx+n1] = q[n1];
    }
  }
}

static __global__ void find_force_radial_small_box(
  const int N,
  const int num_types,
  const int num_types_sq,
  const int neigh_num,
  const int feat_nums,
  const double rc_radial,
  const double rcinv_radial,
  const int n_max_radial,
  const int basis_size_radial,
  const int* g_NL,
  const double* g_d12_radial,
  const double * coeff2,
  const int* g_type,
  const double * grad_output,
  double* dfeat_c2,
  double* grad_d12_radial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    
    int neigh_start_idx = n1 * neigh_num;
    int r12_start_idx =  n1 * neigh_num * 4;
    int feat_start_idx = n1 * feat_nums; 
    int c2_start_idx = t1 * num_types * n_max_radial * basis_size_radial;
    int dfeat_c_start_idx = n1 * num_types * basis_size_radial;
    int de_start_idx = n1 * feat_nums;
    for (int i1 = 0; i1 < neigh_num; ++i1) {
      int n2 = g_NL_radial[neigh_start_idx + i1] - 1;
      if (n2 < 0) break;
      int t2 = g_type[n2];
      int c_I_J_idx = c2_start_idx + t2 * n_max_radial * basis_size_radial;
      int rij_idx = r12_start_idx + i1*4;
      double d12 = g_d12_radial[rij_idx];
      double d12inv = 1.0 / d12;
      double f12[4] = {0.0};

      double fc12, fcp12;
      find_fc_and_fcp(rc_radial, rcinv_radial, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];

      find_fn_and_fnp(
        basis_size_radial, rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
      double dfeat_rij = 0.0;
      for (int n = 0; n <= n_max_radial; ++n) {
        double gnp12 = 0.0;
        for (int k = 0; k <= basis_size_radial; ++k) {
          int c_index = c_I_J_idx + n * basis_size_radial + k;
          gnp12 += fnp12[k] * coeff2[c_index];
          if (n == 0) {
            dfeat_c2[dfeat_c_start_idx + t2 * n_base + k] += fn12[k]; //[batch, n_atom, J_Ntypes, N_base]
          }
          dfeat_rij = grad_output[de_start_idx + n] * gnp12;
        }
        grad_d12_radial[rij_idx] = dfeat_rij;
      }
    }
  }
}

static __global__ void find_force_angular_small_box(
  const int N,
  const int num_types,
  const int num_types_sq,
  const int neigh_num,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_nums,
  const double rc_radial,
  const double rcinv_radial,
  const double rc_angular,
  const double rcinv_angular,
  const int n_max_radial,
  const int basis_size_radial,
  const int n_max_angular,
  const int basis_size_angular,
  const int* g_NL,
  const double * coeff2,
  const double * coeff3,
  const int* __restrict__ g_type,
  const double* __restrict__ g_d12_radial,
  const double* __restrict__ g_Fp,
  const double* __restrict__ g_sum_fxyz,
  double* __restrict__ grad_coeff2,
  double* __restrict__ grad_d12_radial,
  double* __restrict__ grad_coeff3,
  double* __restrict__ grad_d12_3b
  )
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {

    double Fp[MAX_DIM_ANGULAR] = {0.0};
    double sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }

    int t1 = g_type[n1];

    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double f12[3] = {0.0};

      double fc12, fcp12;
      find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
      int t2 = g_type[n2];
      if (paramb.version == 2) {
        for (int n = 0; n <= paramb.n_max_angular; ++n) {
          double fn;
          double fnp;
          find_fn_and_fnp(n, paramb.rcinv_angular, d12, fc12, fcp12, fn, fnp);
          const double c =
            (paramb.num_types == 1)
              ? 1.0
              : annmb.c
                  [((paramb.n_max_radial + 1 + n) * paramb.num_types + t1) * paramb.num_types + t2];
          fn *= c;
          fnp *= c;
          accumulate_f12(n, paramb.n_max_angular + 1, d12, r12, fn, fnp, Fp, sum_fxyz, f12);
        }
      } else {
        double fn12[MAX_NUM_N];
        double fnp12[MAX_NUM_N];
        find_fn_and_fnp(
          paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= paramb.n_max_angular; ++n) {
          double gn12 = 0.0;
          double gnp12 = 0.0;
          for (int k = 0; k <= paramb.basis_size_angular; ++k) {
            int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
            gn12 += fn12[k] * annmb.c[c_index];
            gnp12 += fnp12[k] * annmb.c[c_index];
          }
          if (paramb.num_L == paramb.L_max) {
            accumulate_f12(n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
          } else if (paramb.num_L == paramb.L_max + 1) {
            accumulate_f12_with_4body(
              n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
          } else {
            accumulate_f12_with_5body(
              n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
          }
        }
      }

      // double s_sxx = 0.0;
      // double s_sxy = 0.0;
      // double s_sxz = 0.0;
      // double s_syx = 0.0;
      // double s_syy = 0.0;
      // double s_syz = 0.0;
      // double s_szx = 0.0;
      // double s_szy = 0.0;
      // double s_szz = 0.0;
      // if (is_dipole) {
      //   double r12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      //   s_sxx -= r12_square * f12[0];
      //   s_syy -= r12_square * f12[1];
      //   s_szz -= r12_square * f12[2];
      // } else {
      //   s_sxx -= r12[0] * f12[0];
      //   s_syy -= r12[1] * f12[1];
      //   s_szz -= r12[2] * f12[2];
      // }
      // s_sxy -= r12[0] * f12[1];
      // s_sxz -= r12[0] * f12[2];
      // s_syz -= r12[1] * f12[2];
      // s_syx -= r12[1] * f12[0];
      // s_szx -= r12[2] * f12[0];
      // s_szy -= r12[2] * f12[1];

      // atomicAdd(&g_fx[n1], double(f12[0]));
      // atomicAdd(&g_fy[n1], double(f12[1]));
      // atomicAdd(&g_fz[n1], double(f12[2]));
      // atomicAdd(&g_fx[n2], double(-f12[0]));
      // atomicAdd(&g_fy[n2], double(-f12[1]));
      // atomicAdd(&g_fz[n2], double(-f12[2]));
      // // save virial
      // // xx xy xz    0 3 4
      // // yx yy yz    6 1 5
      // // zx zy zz    7 8 2
      // atomicAdd(&g_virial[n2 + 0 * N], s_sxx);
      // atomicAdd(&g_virial[n2 + 1 * N], s_syy);
      // atomicAdd(&g_virial[n2 + 2 * N], s_szz);
      // atomicAdd(&g_virial[n2 + 3 * N], s_sxy);
      // atomicAdd(&g_virial[n2 + 4 * N], s_sxz);
      // atomicAdd(&g_virial[n2 + 5 * N], s_syz);
      // atomicAdd(&g_virial[n2 + 6 * N], s_syx);
      // atomicAdd(&g_virial[n2 + 7 * N], s_szx);
      // atomicAdd(&g_virial[n2 + 8 * N], s_szy);
    }
  }
}
