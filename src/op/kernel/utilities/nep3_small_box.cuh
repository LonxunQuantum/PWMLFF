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

#include "common.cuh"
#include "nep_utilities.cuh"

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

static __global__ void aggregate_features(
    double* dfeat_c2,     // 输入张量，维度 [N, n_types, n_max_2b, n_base_2b]
    const int* atom_map,       // 原子类型映射，维度 [N]
    double* output,             // 输出张量，维度 [n_types, n_types, n_max_2b, n_base_2b]
    int N,                     // 中心原子数量
    int n_types,
    int n_max_2b,
    int n_base_2b) {

    // 获取线程索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 根据总元素数限制线程数
    int total_elements = n_max_2b * n_base_2b;
    if (tid >= N * total_elements) return;

    // 计算当前线程负责的元素在 dfeat_c2 中的位置
    int atom_idx = tid / total_elements; // 当前中心原子索引
    int base_idx = tid % total_elements; // n_max_2b 和 n_base_2b 的线性索引

    // 获取原子类型
    int center_type = atom_map[atom_idx];

    // 遍历 n_types（对应所有类型）
    for (int neighbor_type = 0; neighbor_type < n_types; ++neighbor_type) {
        // 累加到输出张量的指定位置
        atomicAdd(&output[center_type * n_types * total_elements +
                          neighbor_type * total_elements + base_idx],
                  dfeat_c2[atom_idx * n_types * total_elements +
                           neighbor_type * total_elements + base_idx]);
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
  double * feats,
  const int* g_type,
  const double* g_d12_radial,
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
      for (int n = 0; n < n_max_radial; ++n) {
        double gn12 = 0.0;
        for (int k = 0; k < basis_size_radial; ++k) {
          int c_index = c_I_J_idx + n * basis_size_radial + k;
          gn12 += fn12[k] * coeff2[c_index];
        }
        // 2b feats
        q[n] += gn12;
      }
    }

    // get angular descriptors
    int c3_start_idx = t1 * num_types * n_max_angular * basis_size_angular;
    int sum_s_start_idx = n1 * n_max_angular * NUM_OF_ABC;
    for (int n = 0; n < n_max_angular; ++n) {
      double s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < neigh_num; ++i1) {
        int n2 = g_NL_radial[neigh_start_idx + i1]-1;
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
        for (int k = 0; k < basis_size_angular; ++k) {
          int c_index = c_I_J_idx + n * basis_size_angular + k;
          gn12 += fn12[k] * coeff3[c_index];
        }
        accumulate_s(d12, r12[0], r12[1], r12[2], gn12, s);
      }
      if (L_max5 == 1) {
          find_q_with_5body(n_max_angular, n, s, q + n_max_radial);
      } else if (L_max4 ==2) {
        find_q_with_4body(n_max_angular, n, s, q + n_max_radial);
      } else {
        find_q(n_max_angular, n, s, q + n_max_radial);
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
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_nums,
  const double rc_radial,
  const double rcinv_radial,
  const int n_max_radial,
  const int basis_size_radial,
  const int* g_NL_radial,
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
    int dc_start_idx = n1 * n_max_radial * num_types * basis_size_radial;
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

      int dc_index =  dc_start_idx + t2 * n_max_radial * basis_size_radial;
      for (int n = 0; n < n_max_radial; ++n) {
        double gnp12 = 0.0;
        for (int k = 0; k < basis_size_radial; ++k) {
          int c_index = c_I_J_idx + n * basis_size_radial + k;
          gnp12 += fnp12[k] * coeff2[c_index];
          dfeat_c2[dc_index + n * basis_size_radial + k] += grad_output[de_start_idx + n] * fn12[k]; 
        }
        grad_d12_radial[rij_idx] += gnp12*grad_output[de_start_idx + n];
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
  const int feat_3b_nums, // 3b + 4b + 5b
  const double rc_angular,
  const double rcinv_angular,
  const int n_max_radial,
  const int basis_size_radial,
  const int n_max_angular,
  const int basis_size_angular,
  const int* g_NL_radial,
  const double* g_d12_radial,
  const double * coeff3,
  const int* g_type,
  const double * grad_output,
  const double* g_sum_fxyz,
  double* dfeat_c3,
  double* grad_d12_angular
  )
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int g_sum_start = n1 * n_max_angular * NUM_OF_ABC;
    int r12_start_idx =  n1 * neigh_num * 4;
    int dc_start_idx = n1 * num_types * n_max_angular * basis_size_angular;
    int de_start = n1 * feat_nums;// dE/dq
    int neigh_start_idx = n1 * neigh_num;
    double Fp[MAX_DIM_ANGULAR] = {0.0};
    double sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    int b3_nums = n_max_angular * L_max3;
    int dd = 0;
    if (n1 == 0) {
      for (int nn=0; nn < feat_3b_nums; nn++) {//all
        printf("grad[%d + %d]=%f\n", de_start + n_max_radial, nn, grad_output[de_start + n_max_radial + nn]);
      }
    }
    for (int nn=0; nn < n_max_angular; ++nn) {
      for (int ll = 0; ll < L_max3; ++ll) {
        Fp[dd] = grad_output[de_start + n_max_radial + ll * n_max_angular + nn];// i -> nmax_3b*l_max+2?
        if (n1==0){
          printf("3b Fp[%d] = %f from grad_output[%d + %d] = %f\n", dd, Fp[dd], de_start, n_max_radial + ll * n_max_angular + nn, grad_output[de_start + n_max_radial + n_max_radial + ll * n_max_angular + nn]);
        }
        dd++;
      }
    }
    if (L_max4 > 0) {
      for (int ll = 0; ll < n_max_angular; ++ll) {
        Fp[b3_nums + ll] = grad_output[de_start + n_max_radial + b3_nums + ll];
        if (n1==0){
          printf("4b Fp[%d + %d] = %f from grad_output[%d + %d] = %f\n", b3_nums, ll, Fp[b3_nums + ll], de_start + n_max_radial, b3_nums + ll, grad_output[de_start + n_max_radial + b3_nums + ll]);
        }
      }
    }
    if (L_max5 > 0) {
      for (int ll = 0; ll < n_max_angular; ++ll) {
        Fp[b3_nums + n_max_angular + ll] = grad_output[de_start + n_max_radial + b3_nums + n_max_angular + ll];
        if (n1==0){
          printf("5b Fp[%d + %d] = %f from grad_output[%d + %d] = %f\n", b3_nums, n_max_angular + ll, Fp[b3_nums + n_max_angular + ll], de_start+n_max_radial, b3_nums + n_max_angular + ll, grad_output[de_start + n_max_radial + b3_nums + n_max_angular + ll]);
        }
      }
    }

    for (int d = 0; d < n_max_angular * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[g_sum_start + d]; // g_sum is [N, n_max, 24]
    }

    int t1 = g_type[n1];
    int c3_start_idx = t1 * num_types * n_max_angular * basis_size_angular;
    for (int i1 = 0; i1 < neigh_num; ++i1) {
      int n2 = g_NL_radial[neigh_start_idx + i1]-1;
      if (n2 < 0) break;
      int t2 = g_type[n2];
      int rij_idx = r12_start_idx + i1*4;
      double d12 = g_d12_radial[rij_idx];
      if (d12 > rc_angular) break;
      double r12[3] = {g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2], g_d12_radial[rij_idx+3]};
      double f12[4] = {0.0};

      double fc12, fcp12;
      find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);

      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(
        basis_size_angular, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
      
      int c_I_J_idx = c3_start_idx + t2 * n_max_angular * basis_size_angular;
      double s[NUM_OF_ABC] = {0.0};
      accumulate_blm_rij(d12, r12[0], r12[1], r12[2], s);
      for (int n = 0; n < n_max_angular; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        for (int k = 0; k < basis_size_angular; ++k) {
          int c_index = c_I_J_idx + n * basis_size_angular + k;
          gn12 += fn12[k] * coeff3[c_index];
          gnp12 += fnp12[k] * coeff3[c_index];
        }
        if (L_max5 > 0) {
          accumulate_f12_with_5body(
            n, d12, r12, gn12, gnp12, Fp, sum_fxyz,
              s, f12, dfeat_c3, fn12, fnp12, 
              t2, num_types, L_max3, 
              n_max_angular, basis_size_angular, dc_start_idx, n1, i1);
        } else if (L_max4 > 0) {
          accumulate_f12_with_4body(
            n, d12, r12, gn12, gnp12, Fp, sum_fxyz,
              s, f12, dfeat_c3, fn12, fnp12, 
              t2, num_types, L_max3, 
              n_max_angular, basis_size_angular, dc_start_idx, n1, i1);
        } else {
          accumulate_f12(
            n, d12, r12, gn12, gnp12, Fp, sum_fxyz,
              s, f12, dfeat_c3, fn12, fnp12, 
              t2, num_types, L_max3, 
              n_max_angular, basis_size_angular, dc_start_idx, n1, i1);
        }
      }
      // copy f12 to dfeat_3rij
      grad_d12_angular[rij_idx]  += f12[3];
      grad_d12_angular[rij_idx+1]+= f12[0];
      grad_d12_angular[rij_idx+2]+= f12[1];
      grad_d12_angular[rij_idx+3]+= f12[2];
    }
  }
}
