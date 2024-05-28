/*
    Copyright 2022 Zheyong Fan, Junjie Wang, Eric Lindgren
    This file is part of NEP_CPU.
    NEP_CPU is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    NEP_CPU is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with NEP_CPU.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
A CPU implementation of the neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "nep.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// #if defined(_OPENMP)
// #include <omp.h>
// #endif

namespace
{

double get_area_one_direction(const double* a, const double* b)
{
  double s1 = a[1] * b[2] - a[2] * b[1];
  double s2 = a[2] * b[0] - a[0] * b[2];
  double s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

double get_area(const int d, const double* cpu_h)
{
  double area;
  double a[3] = {cpu_h[0], cpu_h[3], cpu_h[6]};
  double b[3] = {cpu_h[1], cpu_h[4], cpu_h[7]};
  double c[3] = {cpu_h[2], cpu_h[5], cpu_h[8]};
  if (d == 0) {
    area = get_area_one_direction(b, c);
  } else if (d == 1) {
    area = get_area_one_direction(c, a);
  } else {
    area = get_area_one_direction(a, b);
  }
  return area;
}

double get_det(const double* cpu_h)
{
  return cpu_h[0] * (cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7]) +
         cpu_h[1] * (cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8]) +
         cpu_h[2] * (cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6]);
}

double get_volume(const double* cpu_h) { return abs(get_det(cpu_h)); }

void get_inverse(double* cpu_h)
{
  cpu_h[9] = cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7];
  cpu_h[10] = cpu_h[2] * cpu_h[7] - cpu_h[1] * cpu_h[8];
  cpu_h[11] = cpu_h[1] * cpu_h[5] - cpu_h[2] * cpu_h[4];
  cpu_h[12] = cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8];
  cpu_h[13] = cpu_h[0] * cpu_h[8] - cpu_h[2] * cpu_h[6];
  cpu_h[14] = cpu_h[2] * cpu_h[3] - cpu_h[0] * cpu_h[5];
  cpu_h[15] = cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6];
  cpu_h[16] = cpu_h[1] * cpu_h[6] - cpu_h[0] * cpu_h[7];
  cpu_h[17] = cpu_h[0] * cpu_h[4] - cpu_h[1] * cpu_h[3];
  double det = get_det(cpu_h);
  for (int n = 9; n < 18; n++) {
    cpu_h[n] /= det;
  }
}

void get_expanded_box(const double rc, const double* box, int* num_cells, double* ebox)
{
  double volume = get_volume(box);
  double thickness_x = volume / get_area(0, box);
  double thickness_y = volume / get_area(1, box);
  double thickness_z = volume / get_area(2, box);
  num_cells[0] = int(ceil(2.0 * rc / thickness_x));
  num_cells[1] = int(ceil(2.0 * rc / thickness_y));
  num_cells[2] = int(ceil(2.0 * rc / thickness_z));

  ebox[0] = box[0] * num_cells[0];
  ebox[3] = box[3] * num_cells[0];
  ebox[6] = box[6] * num_cells[0];
  ebox[1] = box[1] * num_cells[1];
  ebox[4] = box[4] * num_cells[1];
  ebox[7] = box[7] * num_cells[1];
  ebox[2] = box[2] * num_cells[2];
  ebox[5] = box[5] * num_cells[2];
  ebox[8] = box[8] * num_cells[2];

  get_inverse(ebox);
}

void applyMicOne(double& x12)
{
  while (x12 < -0.5)
    x12 += 1.0;
  while (x12 > +0.5)
    x12 -= 1.0;
}

void apply_mic_small_box(const double* ebox, double& x12, double& y12, double& z12)
{
  double sx12 = ebox[9] * x12 + ebox[10] * y12 + ebox[11] * z12;
  double sy12 = ebox[12] * x12 + ebox[13] * y12 + ebox[14] * z12;
  double sz12 = ebox[15] * x12 + ebox[16] * y12 + ebox[17] * z12;
  applyMicOne(sx12);
  applyMicOne(sy12);
  applyMicOne(sz12);
  x12 = ebox[0] * sx12 + ebox[1] * sy12 + ebox[2] * sz12;
  y12 = ebox[3] * sx12 + ebox[4] * sy12 + ebox[5] * sz12;
  z12 = ebox[6] * sx12 + ebox[7] * sy12 + ebox[8] * sz12;
}

void find_neighbor_list_small_box(
  const double rc_radial,
  const double rc_angular,
  const int N,
  const int MN,
  const std::vector<int>& atom_type_map,
  const std::vector<double>& box,
  const std::vector<double>& position,
  int* num_cells,
  double* ebox,
  std::vector<int>& g_NN_radial,
  std::vector<int>& g_NL_radial,
  std::vector<int>& g_NLT_radial,
  std::vector<int>& g_NN_angular,
  std::vector<int>& g_NL_angular,
  std::vector<int>& g_NLT_angular,
  // std::vector<double>& r12,
  std::vector<double>& r12_radial,
  std::vector<double>& r12_angular)
{
  get_expanded_box(rc_radial, box.data(), num_cells, ebox);

  const double* g_x = position.data();
  const double* g_y = position.data() + N;
  const double* g_z = position.data() + N * 2;

  double* g_r12_radial = r12_radial.data();
  double* g_x12_radial = r12_radial.data();
  double* g_y12_radial = r12_radial.data();
  double* g_z12_radial = r12_radial.data();

  double* g_r12_angular= r12_angular.data();
  double* g_x12_angular= r12_angular.data();
  double* g_y12_angular= r12_angular.data();
  double* g_z12_angular= r12_angular.data();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int n1 = 0; n1 < N; ++n1) {
    //ct position
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    if (atom_type_map[n1] == -1) continue;

    // fraction postion
    // double x1 = g_x[n1] * box[0] + g_y[n1] * box[1] + g_z[n1] * box[2];
    // double y1 = g_x[n1] * box[3] + g_y[n1] * box[4] + g_z[n1] * box[5];
    // double z1 = g_x[n1] * box[6] + g_y[n1] * box[7] + g_z[n1] * box[8];

    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = 0; n2 < N; ++n2) {
      if (atom_type_map[n2] == -1) continue;
      for (int ia = 0; ia < num_cells[0]; ++ia) {
        for (int ib = 0; ib < num_cells[1]; ++ib) {
          for (int ic = 0; ic < num_cells[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }
            double delta[3];
            delta[0] = box[0] * ia + box[1] * ib + box[2] * ic;
            delta[1] = box[3] * ia + box[4] * ib + box[5] * ic;
            delta[2] = box[6] * ia + box[7] * ib + box[8] * ic;

            //ct position
            double x12 = g_x[n2] + delta[0] - x1;
            double y12 = g_y[n2] + delta[1] - y1;
            double z12 = g_z[n2] + delta[2] - z1;
            
            // fraction postion
            // double x12 = g_x[n2] * box[0] + g_y[n2] * box[1] + g_z[n2] * box[2] + delta[0] - x1;
            // double y12 = g_x[n2] * box[3] + g_y[n2] * box[4] + g_z[n2] * box[5] + delta[1] - y1;
            // double z12 = g_x[n2] * box[6] + g_y[n2] * box[7] + g_z[n2] * box[8] + delta[2] - z1;

            apply_mic_small_box(ebox, x12, y12, z12);
            double distance_square = x12 * x12 + y12 * y12 + z12 * z12;

            // std::cout<< "n1 " << n1 << "|n2 " << n2 << "|ng_id " << (count_radial * N + n1) \
            // << "|ia " << ia << "|ib " << ib << "|ic " << ic \
            // << "|rij " << distance_square << "|x12 "<<x12 << "|y12 " << y12 << "|z12 " << z12 \
            // << "|x1 " << g_x[n1] << "|y1 " << g_y[n1] << "|z0 " << g_z[n1] \
            // << "|x2 " << g_x[n2] << "|y2 " << g_y[n2] << "|z2 " << g_z[n2] \
            // << std::endl;

            if (distance_square < rc_radial * rc_radial) {
              // std::cout<< "n1 " << n1 << "|n2 " << n2 << "|ng_id " << (count_radial * N + n1) \
              // << "|ia " << ia << "|ib " << ib << "|ic " << ic \
              // << "|rij " << distance_square << "|x12 "<<x12 << "|y12 " << y12 << "|z12 " << z12 \
              // << "|x1 " << g_x[n1] << "|y1 " << g_y[n1] << "|z0 " << g_z[n1] \
              // << "|x2 " << g_x[n2] << "|y2 " << g_y[n2] << "|z2 " << g_z[n2] \
              // << std::endl;

              // g_NL_radial[count_radial * N + n1] = n2;
              // g_x12_radial[count_radial * N + n1] = x12;
              // g_y12_radial[count_radial * N + n1] = y12;
              // g_z12_radial[count_radial * N + n1] = z12;
              g_NLT_radial[n1*MN   +   count_radial  ] = atom_type_map[n2];
              g_NL_radial[ n1*MN   +   count_radial  ] = n2+1;
              g_r12_radial[n1*MN*4 + 4*count_radial  ] = sqrt(distance_square);
              g_x12_radial[n1*MN*4 + 4*count_radial+1] = x12;
              g_y12_radial[n1*MN*4 + 4*count_radial+2] = y12;
              g_z12_radial[n1*MN*4 + 4*count_radial+3] = z12;

              // std::cout << " n1*MN + count_radial [" << n1*MN + 4*count_radial << " " << n1*MN + 4*count_radial+1 << " " << n1*MN + 4*count_radial + 2 << " " << n1*MN + 4*count_radial +3 <<  \
              // " ] n1 " << n1 << " n2 " << n2 << " MN " << MN << " count_r " << count_radial << std::endl;

              count_radial++;
            } 
            // else {
            //   std::cout<< "n1 " << n1 << "|n2 " << n2 << "|BIGLN " << "|ia " << ia << "|ib " << ib << "|ic " << ic << std::endl;
            // }
            if (distance_square < rc_angular * rc_angular) {
              // g_NL_angular[count_angular * N + n1] = n2;
              // g_x12_angular[count_angular * N + n1] = x12;
              // g_y12_angular[count_angular * N + n1] = y12;
              // g_z12_angular[count_angular * N + n1] = z12;
              g_NLT_angular[n1*MN   +   count_angular  ] = atom_type_map[n2];
              g_NL_angular[ n1*MN   +   count_angular  ] = n2+1;
              g_r12_angular[n1*MN*4 + 4*count_angular  ] = sqrt(distance_square);
              g_x12_angular[n1*MN*4 + 4*count_angular+1] = x12;
              g_y12_angular[n1*MN*4 + 4*count_angular+2] = y12;
              g_z12_angular[n1*MN*4 + 4*count_angular+3] = z12;
              count_angular++;
            }
          }
        }
      }
    }

    g_NN_radial[n1] = count_radial;
    g_NN_angular[n1] = count_angular;
  }
}

} // namespace

NEP3::NEP3() {

}

void NEP3::find_neigh(
  const double rc_radial,
  const double rc_angular,
  const int MN,
  const std::vector<int>& atom_type_map,
  const std::vector<double>& box,
  const std::vector<double>& position)
{
  int N = atom_type_map.size();
  // NN_radial.resize(N, 0);
  // NL_radial.resize(N * MN, 0);
  // NLT_radial.resize(N * MN, -1);
  // NN_angular.resize(N);
  // NL_angular.resize(N * MN, 0);
  // NLT_angular.resize(N * MN, -1);
  // // r12.resize(N * MN * 6);
  // r12_radial.resize(N * MN * 4, 0);
  // r12_angular.resize(N * MN * 4, 0);

  NN_radial.assign(N, 0);
  NL_radial.assign(N * MN, 0);
  NLT_radial.assign(N * MN, -1);
  NN_angular.assign(N, 0);
  NL_angular.assign(N * MN, 0);
  NLT_angular.assign(N * MN, -1);
  // r12.assign(N * MN * 6);
  r12_radial.assign(N * MN * 4, 0);
  r12_angular.assign(N * MN * 4, 0);

  
  find_neighbor_list_small_box(
    rc_radial, rc_angular, N, MN, atom_type_map, box, position, num_cells, ebox, NN_radial, NL_radial, NLT_radial, NN_angular, NL_angular, NLT_angular, r12_radial, r12_angular);
}
