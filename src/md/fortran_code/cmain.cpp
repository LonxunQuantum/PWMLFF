#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "diy.hpp"

using namespace std;

int main(int argc, char** argv)
{
    int imodel = 1;
    int n_atom = 108;
    int m_dg = 1;
    int n_dg = 3;
    int k_dg = 3;
    int lda = 1;
    int ldb = 3;
    int ldc = 1;
    char transa = 'N';
    char transb = 'N';
    double alpha_dg = 1.0;
    double beta_dg = 0.0;
    
    double tmp_v3[3];
    int* type_atom;
    double e_tot;
    double *e_atom;
    double **lat;
    double **reclat;
    double **x_frac;
    double **x_cart;
    double **f_atom;

    diy::create(type_atom,n_atom);
    diy::create(lat, 3, 3);
    diy::create(reclat, 3, 3);
    diy::create(x_frac, n_atom, 3);
    diy::create(x_cart, n_atom, 3);
    diy::create(f_atom, n_atom, 3);
    diy::create(e_atom, n_atom);

    // box.raw  coord.raw  energy.raw  force.raw  type_map.raw  type.raw
    fstream f_in;
    stringstream ss;
    string tmp_str;
    // read lattice
    printf("read lattice\n");
    f_in.open("box.raw", ios::in);
    getline(f_in, tmp_str);
    ss << tmp_str;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            ss >> lat[i][j];
            reclat[i][j] = lat[i][j];
        }
    }
    f_in.close();
    ss.clear();
    diy::dinv(&reclat[0][0],3);
    // read x_atom
    printf("read x_atom\n");
    f_in.open("coord.raw", ios::in);
    getline(f_in, tmp_str);
    ss << tmp_str;
    for(int i = 0; i < n_atom; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            ss >> x_cart[i][j];
        }
    }
    f_in.close();
    ss.clear();
    // convert cart to frac
    printf("convert cart to frac\n");
    for(int i = 0; i < n_atom; i++)
    {
        //dgemm_('N', 'N', 1, 3, 3, 1.0, &x_cart[i][0], 1, &reclat[0][0], 3, 0.0, &tmp_v3[0], 1);
        dgemm_(&transa, &transb, &m_dg, &n_dg, &k_dg, &alpha_dg, &x_cart[i][0], &lda, &reclat[0][0], &ldb, &beta_dg, &tmp_v3[0], &ldc);
        x_frac[i][0] = tmp_v3[0];
        x_frac[i][1] = tmp_v3[1];
        x_frac[i][2] = tmp_v3[2];
    }
    printf("convert cart to frac end\n");
    // read type
    printf("read type\n");
    f_in.open("type_atomic_num.raw", ios::in);
    for(int i = 0; i < n_atom; i++)
    {
        getline(f_in, tmp_str);
        ss << tmp_str;
        ss >> type_atom[i];
        ss.clear();
    }
    f_in.close();

    printf("x_frac:\n");
    for(int i = 0; i < 10; i++)
    {
        printf("%f    %f     %f\n", x_frac[i][0], x_frac[i][1], x_frac[i][2]);
    }
    printf("x_frac_end:\n");
    // use the address of the first element, and pass a continuous memory (1d array)
    MPI_Init(&argc, &argv);
    f2c_calc_energy_force(&imodel, &n_atom, &type_atom[0], &lat[0][0], &x_frac[0][0], &e_atom[0], &f_atom[0][0], &e_tot);
    MPI_Finalize();
    printf("total energy: %f\n", e_tot);
    printf("forces:\n");
    for(int i = 0; i < n_atom; i++)
    {
        printf("%f    %f    %f\n", f_atom[i][0], f_atom[i][1], f_atom[i][2]);
    }
    diy::destroy(lat);
    diy::destroy(reclat);
    diy::destroy(x_frac);
    diy::destroy(x_cart);
    diy::destroy(f_atom);
    diy::destroy(e_atom);
}
