#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>
#include <iomanip>
#include "descriptor.h"

Descriptor::Descriptor(){}; // default constructor

/**
 * @brief Constructor for the Radial class.
 *
 * @param beta The order of the Chebyshev polynomials.
 * @param m1, m2 The number of the radial basis functions.
 * @param rcut_max The maximum cutoff radius.
 * @param rcut_smooth The smoothing cutoff radius.
 * @param natoms The number of atoms in the system.
 * @param ntypes The number of atom types in the system.
 * @param max_neighbors The maximum number of neighbors.
 * @param type_map The atom type for all atoms.
 * @param num_neigh_all The number of neighbors for all atoms.
 * @param neighbors_list_all The neighbor list for all atoms.
 * @param dR_neigh_all The partial derivative of the neighbor list with respect to rij for all atoms.
 * @param c The Chebyshev coefficients with the shape (itypes, jtypes, m1, beta).
 */
Descriptor::Descriptor(int beta, int m1, int m2, float rcut_max, float rcut_smooth,
                       int natoms, int ntypes, int max_neighbors, int *type_map,
                       int *num_neigh_all, int *neighbors_list_all, double *dR_neigh_all, double **c)
    : beta(beta), m1(m1), m2(m2), natoms(natoms), ntypes(ntypes), max_neighbors(max_neighbors),
      rcut_max(rcut_max), rcut_smooth(rcut_smooth), type_map(type_map),
      radial(m1, beta, ntypes, rcut_max, rcut_smooth), smooth(rcut_max, rcut_smooth),
      num_neigh_all(num_neigh_all), neighbors_list_all(neighbors_list_all), dR_neigh_all(dR_neigh_all)
{
    this->nfeat = ntypes * m1 * m2;
    this->num_neigh_alltypes = new int[natoms];
    // this->neighbor_list_alltypes = new int *[natoms];
    this->neighbor_list_alltypes = new int[natoms * max_neighbors];
    std::fill_n(this->neighbor_list_alltypes, natoms * max_neighbors, -1);
    this->dR_neigh_alltypes = new Neighbor *[natoms];
    this->ind_neigh_alltypes = new int **[natoms];
    this->feat = new double[natoms * this->nfeat]();
    // this->dfeat_tmp = new double ***[3];
    this->dfeat_tmp = new double[3 * natoms * this->nfeat * max_neighbors]();
    // this->dfeat = new double ***[3];
    this->dfeat = new double[3 * natoms * this->nfeat * max_neighbors]();
    // this->dfeat2c = new double **[natoms];
    this->dfeat2c = new double[natoms * this->nfeat * max_neighbors]();

    for (int i = 0; i < natoms; i++)
    {
        this->num_neigh_alltypes[i] = 0;
        // this->neighbor_list_alltypes[i] = new int[max_neighbors];
        // std::fill_n(this->neighbor_list_alltypes[i], max_neighbors, -1);
        this->dR_neigh_alltypes[i] = new Neighbor[max_neighbors];
        std::fill_n(this->dR_neigh_alltypes[i], max_neighbors, Neighbor());
        this->ind_neigh_alltypes[i] = new int *[ntypes];

        for (int j = 0; j < ntypes; j++)
        {
            this->ind_neigh_alltypes[i][j] = new int[max_neighbors]();
            std::fill_n(this->ind_neigh_alltypes[i][j], max_neighbors, -1);
        }

        /*
        this->dfeat2c[i] = new double *[this->nfeat];
        for (int j = 0; j < this->nfeat; j++)
        {
            this->dfeat2c[i][j] = new double[max_neighbors];
            std::fill_n(this->dfeat2c[i][j], max_neighbors, 0.0);
        }
        */
    }

    /*
    for (int i = 0; i < 3; i++)
    {
        this->dfeat_tmp[i] = new double **[natoms];
        this->dfeat[i] = new double **[natoms];
        for (int j = 0; j < natoms; j++)
        {
            this->dfeat_tmp[i][j] = new double *[this->nfeat];
            this->dfeat[i][j] = new double *[this->nfeat];
            for (int k = 0; k < this->nfeat; k++)
            {
                this->dfeat_tmp[i][j][k] = new double[max_neighbors];
                this->dfeat[i][j][k] = new double[max_neighbors];
                std::fill_n(this->dfeat_tmp[i][j][k], max_neighbors, 0.0);
                std::fill_n(this->dfeat[i][j][k], max_neighbors, 0.0);
            }
        }
    }
    */

    if (c == nullptr)
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-0.5, 0.5);
        this->c = new double *[m1];
        for (int i = 0; i < m1; i++)
        {
            this->c[i] = new double[beta];
            for (int j = 0; j < beta; j++)
            {
                this->c[i][j] = distribution(generator);
                // this->c[i][j] = 1.0;
            }
        }
        // this->c = new double ***[ntypes];
        // for (int i = 0; i < ntypes; i++)
        // {
        //     this->c[i] = new double **[ntypes];
        //     for (int j = 0; j < ntypes; j++)
        //     {
        //         this->c[i][j] = new double *[m1];
        //         for (int k = 0; k < m1; k++)
        //         {
        //             this->c[i][j][k] = new double[beta];
        //             for (int l = 0; l < beta; l++)
        //             {
        //                 this->c[i][j][k][l] = distribution(generator);
        //                 // this->c[i][j][k][l] = 1.0;
        //             }
        //         }
        //     }
        // }
    }
    else
    {
        this->c = c;
    }

    this->radial.set_c(this->c);

    build(this->m1, this->m2, this->max_neighbors, this->ntypes, this->natoms);
} // constructor

Descriptor::Descriptor(const Descriptor &other)
    : beta(other.beta), m1(other.m1), m2(other.m2), natoms(other.natoms), ntypes(other.ntypes),
      max_neighbors(other.max_neighbors), rcut_max(other.rcut_max), rcut_smooth(other.rcut_smooth),
      radial(other.radial), smooth(other.smooth), type_map(other.type_map),
      num_neigh_all(other.num_neigh_all), neighbors_list_all(other.neighbors_list_all), dR_neigh_all(other.dR_neigh_all)
{
    this->num_neigh_alltypes = new int[natoms];
    // this->neighbor_list_alltypes = new int *[natoms];
    this->neighbor_list_alltypes = new int[natoms * max_neighbors];
    std::copy_n(other.neighbor_list_alltypes, natoms * max_neighbors, this->neighbor_list_alltypes);
    this->dR_neigh_alltypes = new Neighbor *[natoms];
    this->ind_neigh_alltypes = new int **[natoms];
    this->feat = new double[natoms * this->nfeat]();
    this->dfeat_tmp = new double[3 * natoms * this->nfeat * max_neighbors]();
    this->dfeat = new double[3 * natoms * this->nfeat * max_neighbors]();
    // this->dfeat2c = new double **[natoms];
    this->dfeat2c = new double[natoms * this->nfeat * max_neighbors]();
    for (int i = 0; i < natoms; i++)
    {
        this->num_neigh_alltypes[i] = other.num_neigh_alltypes[i];
        // this->neighbor_list_alltypes[i] = new int[max_neighbors];
        // std::copy_n(other.neighbor_list_alltypes[i], max_neighbors, this->neighbor_list_alltypes[i]);
        this->dR_neigh_alltypes[i] = new Neighbor[max_neighbors];
        std::copy_n(other.dR_neigh_alltypes[i], max_neighbors, this->dR_neigh_alltypes[i]);
        this->ind_neigh_alltypes[i] = new int *[ntypes];
        for (int j = 0; j < ntypes; j++)
        {
            this->ind_neigh_alltypes[i][j] = new int[max_neighbors];
            std::copy_n(other.ind_neigh_alltypes[i][j], max_neighbors, this->ind_neigh_alltypes[i][j]);
        }

        /*
        this->dfeat2c[i] = new double *[this->nfeat];
        for (int j = 0; j < this->nfeat; j++)
        {
            this->dfeat2c[i][j] = new double[max_neighbors];
            std::copy_n(other.dfeat2c[i][j], max_neighbors, this->dfeat2c[i][j]);
        }
        */
    }

    /*
    this->dfeat_tmp = new double ***[3];
    this->dfeat = new double ***[3];
    for (int i = 0; i < 3; i++)
    {
        this->dfeat_tmp[i] = new double **[natoms];
        this->dfeat[i] = new double **[natoms];
        for (int j = 0; j < natoms; j++)
        {
            this->dfeat_tmp[i][j] = new double *[this->nfeat];
            this->dfeat[i][j] = new double *[this->nfeat];
            for (int k = 0; k < this->nfeat; k++)
            {
                this->dfeat_tmp[i][j][k] = new double[max_neighbors];
                std::copy_n(other.dfeat_tmp[i][j][k], max_neighbors, this->dfeat_tmp[i][j][k]);
                this->dfeat[i][j][k] = new double[max_neighbors];
                std::copy_n(other.dfeat[i][j][k], max_neighbors, this->dfeat[i][j][k]);
            }
        }
    }
    */

    this->c = new double *[m1];
    for (int i = 0; i < m1; i++)
    {
        this->c[i] = new double[beta];
        std::copy_n(other.c[i], beta, this->c[i]);
    }
    // this->c = new double ***[ntypes];
    // for (int i = 0; i < ntypes; i++)
    // {
    //     this->c[i] = new double **[ntypes];
    //     for (int j = 0; j < ntypes; j++)
    //     {
    //         this->c[i][j] = new double *[m1];
    //         for (int k = 0; k < m1; k++)
    //         {
    //             this->c[i][j][k] = new double[beta];
    //             std::copy_n(other.c[i][j][k], beta, this->c[i][j][k]);
    //         }
    //     }
    // }

    this->radial.set_c(this->c);

} // copy constructor

Descriptor::~Descriptor()
{
    delete[] this->num_neigh_alltypes;
    for (int i = 0; i < this->natoms; i++)
    {
        // delete[] this->neighbor_list_alltypes[i];
        delete[] this->dR_neigh_alltypes[i];

        for (int j = 0; j < this->ntypes; j++)
        {
            delete[] this->ind_neigh_alltypes[i][j];
        }
        delete[] this->ind_neigh_alltypes[i];

        /*
        for (int j = 0; j < this->nfeat; j++)
        {
            delete[] this->dfeat2c[i][j];
        }
        delete[] this->dfeat2c[i];
        */
    }
    delete[] this->neighbor_list_alltypes;
    delete[] this->dR_neigh_alltypes;
    delete[] this->ind_neigh_alltypes;
    delete[] this->feat;
    delete[] this->dfeat2c;

    /*
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < this->natoms; j++)
        {
            for (int k = 0; k < this->nfeat; k++)
            {
                delete[] this->dfeat_tmp[i][j][k];
                delete[] this->dfeat[i][j][k];
            }
            delete[] this->dfeat_tmp[i][j];
            delete[] this->dfeat[i][j];
        }
        delete[] this->dfeat_tmp[i];
        delete[] this->dfeat[i];
    }
    */
    delete[] this->dfeat_tmp;
    delete[] this->dfeat;

    for (int i = 0; i < this->m1; i++)
    {
        delete[] this->c[i];
    }
    // for (int i = 0; i < this->ntypes; i++)
    // {
    //     for (int j = 0; j < this->ntypes; j++)
    //     {
    //         for (int k = 0; k < this->m1; k++)
    //         {
    //             delete[] this->c[i][j][k];
    //         }
    //         delete[] this->c[i][j];
    //     }
    //     delete[] this->c[i];
    // }
    delete[] this->c;
} // destructor

void Descriptor::build(int m1, int m2, int max_neighbors, int ntypes, int natoms)
{
    int num, nneigh, jj, ii, jat, jj2;
    int index, itype, index_m1, index_m2;
    double rij, delx, dely, delz, dd;
    // print neighbors
    for (int i = 0; i < natoms; i++)
    {
        num = 0;
        // this->neighbor_list_alltypes[i][0] = i; // self
        // this->neighbor_list_alltypes[i * max_neighbors] = i; // self
        for (int jtype = 0; jtype < ntypes; jtype++)
        {
            for (int j = 0; j < this->num_neigh_all[i * ntypes + jtype]; j++)
            {
                // The index of the neighbor atom
                index = i * ntypes * max_neighbors + jtype * max_neighbors + j;
                // this->neighbor_list_alltypes[i][num] = this->neighbors_list_all[index];
                this->neighbor_list_alltypes[i * max_neighbors + num] = this->neighbors_list_all[index];
                num++;
                if (num >= max_neighbors)
                {
                    std::cerr << "Error: the maximum number of neighbors is too small." << std::endl;
                    exit(1);
                }
                this->dR_neigh_alltypes[i][num] = {this->dR_neigh_all[index * 4 + 0], this->dR_neigh_all[index * 4 + 1], this->dR_neigh_all[index * 4 + 2], this->dR_neigh_all[index * 4 + 3]};
                this->ind_neigh_alltypes[i][jtype][j] = num;
                // std::cout << "ind_neigh_alltypes[" << i << "][" << jtype << "][" << j << "] = " << num << std::endl;
            }
        }
        this->num_neigh_alltypes[i] = num + 1;
    }

    for (int i = 0; i < natoms; i++)
    {
        nneigh = this->num_neigh_alltypes[i];
        itype = this->type_map[i];
        std::vector<std::vector<double>> T(m1 * ntypes, std::vector<double>(4, 0.0));
        std::vector<std::vector<std::vector<std::vector<double>>>> dT(3, std::vector<std::vector<std::vector<double>>>(nneigh, std::vector<std::vector<double>>(m1 * ntypes, std::vector<double>(4, 0.0)))); // partial derivative of T with respect to rij
        std::vector<std::vector<std::vector<double>>> dT2c(nneigh, std::vector<std::vector<double>>(m1 * ntypes, std::vector<double>(4, 0.0)));                                                              // partial derivative of T with respect to c parameters

        for (int jtype = 0; jtype < ntypes; jtype++)
        {
            for (int j = 0; j < this->num_neigh_all[i * ntypes + jtype]; j++)
            {
                // jj = ind_neigh_alltypes[i][jtype][j]; // index of the neighbor atom, starting from 0. atom0, type0, neighbor index..., atom0, type1, neighbor index (accummulated)..., atom2, type0, neighbor index...
                jj2 = this->neighbors_list_all[i * ntypes * max_neighbors + jtype * max_neighbors + j];
                index = i * ntypes * max_neighbors + jtype * max_neighbors + j;
                rij = this->dR_neigh_all[index * 4 + 0];
                delx = this->dR_neigh_all[index * 4 + 1];
                dely = this->dR_neigh_all[index * 4 + 2];
                delz = this->dR_neigh_all[index * 4 + 3];
                // build radial basis functions
                this->radial.build(rij);
                // this->radial.show();
                const double *rads = this->radial.get_rads();
                const double *drads = this->radial.get_drads();
                const double *drads2c = this->radial.get_drads2c();
                const double fc = this->smooth.get_smooth(rij);
                const double dfc = this->smooth.get_dsmooth(rij);
                const double s = fc / rij;
                for (int m = 0; m < m1; m++)
                {
                    ii = m + itype * m1; // index of the radial basis function
                    build_component(m, ii, jj2, delx, dely, delz, rads, drads, drads2c, fc, dfc, s, rij, T, dT, dT2c);
                }
            } // end of loop over neighbors
        }     // end of loop over atom types

        for (int ii1 = 0; ii1 < m1; ii1++)
        {
            index_m1 = itype * m1 + ii1;
            for (int ii2 = 0; ii2 < m2; ii2++)
            {
                index_m2 = itype * m1 + ii2; 
                
                double sum = 0.0;
                std::vector<std::vector<double>> dsum(3, std::vector<double>(nneigh, 0.0));
                std::vector<double> dsum2c(nneigh, 0.0);

                for (int k = 0; k < 4; k++)
                {
                    sum += T[index_m1][k] * T[index_m2][k];

                    for (jj = 0; jj < nneigh; jj++)
                    {
                        dsum[0][jj] += dT[0][jj][index_m1][k] * T[index_m2][k] + T[index_m1][k] * dT[0][jj][index_m2][k];
                        dsum[1][jj] += dT[1][jj][index_m1][k] * T[index_m2][k] + T[index_m1][k] * dT[1][jj][index_m2][k];
                        dsum[2][jj] += dT[2][jj][index_m1][k] * T[index_m2][k] + T[index_m1][k] * dT[2][jj][index_m2][k];
                    }
                }
                ii = itype * m1 * m2 + ii1 * m2 + ii2;
                feat[i * this->nfeat + ii] = sum;

                for (jj = 0; jj < nneigh; jj++)
                {
                    if (std::abs(dsum[0][jj]) + std::abs(dsum[1][jj]) + std::abs(dsum[2][jj]) > 1.0e-7)
                    {
                        // this->dfeat_tmp[0][i][ii][jj] = dsum[0][jj];
                        // this->dfeat_tmp[1][i][ii][jj] = dsum[1][jj];
                        // this->dfeat_tmp[2][i][ii][jj] = dsum[2][jj];
                        // 表示的是第 i 个原子的第 ii 个特征如何随着第 i 个原子与其第 jj 个邻居的x/y/z方向的距离变化而变化。
                        // this->dfeat2c[i][ii][jj] = dsum2c[jj];
                        index = i * this->nfeat * max_neighbors + ii * max_neighbors + jj;
                        this->dfeat_tmp[index * 3 + 0] = dsum[0][jj];
                        this->dfeat_tmp[index * 3 + 1] = dsum[1][jj];
                        this->dfeat_tmp[index * 3 + 2] = dsum[2][jj];
                        this->dfeat2c[index] = dsum2c[jj];
                    }
                }
            }
        }
    } // end of loop over atoms

    /*
    for (int i = 0; i < natoms; i++)
    {
        nneigh = this->num_neigh_alltypes[i];
        for (int jj = 0; jj < nneigh; jj++)
        {
            // jat = this->neighbor_list_alltypes[i][jj];
            jat = this->neighbor_list_alltypes[i * max_neighbors + jj];
            int nneigh2 = this->num_neigh_alltypes[jat];
            for (int jj2 = 0; jj2 < nneigh2; jj2++)
            {
                // if (this->neighbor_list_alltypes[jat][jj2] == i)
                if (this->neighbor_list_alltypes[jat * max_neighbors + jj2] == i)
                {
                    dd = std::pow(dR_neigh_alltypes[i][jj].delx + dR_neigh_alltypes[jat][jj2].delx, 2) + std::pow(dR_neigh_alltypes[i][jj].dely + dR_neigh_alltypes[jat][jj2].dely, 2) + std::pow(dR_neigh_alltypes[i][jj].delz + dR_neigh_alltypes[jat][jj2].delz, 2);
                }
                if (dd < 1.0e-8)
                {
                    for (int ii = 0; ii < this->nfeat; ii++)
                    {
                        // this->dfeat[0][jat][ii][jj2] = this->dfeat_tmp[0][i][ii][jj];
                        // this->dfeat[1][jat][ii][jj2] = this->dfeat_tmp[1][i][ii][jj];
                        // this->dfeat[2][jat][ii][jj2] = this->dfeat_tmp[2][i][ii][jj];
                        index = 3 * (jat * this->nfeat * max_neighbors + ii * max_neighbors + jj2);
                        this->dfeat[index + 0] = this->dfeat_tmp[3 * (i * this->nfeat * max_neighbors + ii * max_neighbors + jj) + 0];
                        this->dfeat[index + 1] = this->dfeat_tmp[3 * (i * this->nfeat * max_neighbors + ii * max_neighbors + jj) + 1];
                        this->dfeat[index + 2] = this->dfeat_tmp[3 * (i * this->nfeat * max_neighbors + ii * max_neighbors + jj) + 2];
                    }
                }
            }
        }
    }
    */
}

void Descriptor::build_component(int m, int ii, int jj, double delx, double dely, double delz,
                                 const double *rads, const double *drads, const double *drads2c, double fc, double dfc,
                                 double s, double rij, std::vector<std::vector<double>> &T,
                                 std::vector<std::vector<std::vector<std::vector<double>>>> &dT,
                                 std::vector<std::vector<std::vector<double>>> &dT2c)
{
    // build 4 components of T
    T[ii][0] += rads[m] * s;
    T[ii][1] += rads[m] * s * delx / rij;
    T[ii][2] += rads[m] * s * dely / rij;
    T[ii][3] += rads[m] * s * delz / rij;

    // partial derivative of T with respect to rij
    const double ff = (dfc / rij - fc / std::pow(rij, 2)) * rads[m] + s * drads[m];

    // partial derivative of T w.r.t. c parameters
    const double ff2 = drads2c[m] * s;

    // The derivative of the component (s * rads[m]),
    // = partial derivative of T with respect to rij * partial derivative of rij with respect to delx, dely, delz
    dT[0][jj][ii][0] += ff * delx / rij;
    dT[1][jj][ii][0] += ff * dely / rij;
    dT[2][jj][ii][0] += ff * delz / rij;

    // The derivative of the component (s * rads[m] / rij) * delx
    dT[0][jj][ii][1] += (ff - s * rads[m] / rij) * delx * delx / std::pow(rij, 2);
    dT[1][jj][ii][1] += (ff - s * rads[m] / rij) * delx * dely / std::pow(rij, 2);
    dT[2][jj][ii][1] += (ff - s * rads[m] / rij) * delx * delz / std::pow(rij, 2);
    dT[0][jj][ii][1] += rads[m] * s / rij;

    // The derivative of the component (s * rads[m] / rij) * dely
    dT[0][jj][ii][2] += (ff - s * rads[m] / rij) * dely * delx / std::pow(rij, 2);
    dT[1][jj][ii][2] += (ff - s * rads[m] / rij) * dely * dely / std::pow(rij, 2);
    dT[2][jj][ii][2] += (ff - s * rads[m] / rij) * dely * delz / std::pow(rij, 2);
    dT[1][jj][ii][2] += rads[m] * s / rij;

    // The derivative of the component (s * rads[m] / rij) * delz
    dT[0][jj][ii][3] += (ff - s * rads[m] / rij) * delz * delx / std::pow(rij, 2);
    dT[1][jj][ii][3] += (ff - s * rads[m] / rij) * delz * dely / std::pow(rij, 2);
    dT[2][jj][ii][3] += (ff - s * rads[m] / rij) * delz * delz / std::pow(rij, 2);
    dT[2][jj][ii][3] += rads[m] * s / rij;

    // minus itself
    // dT[0][0][ii][0] -= ff * delx / rij;
    // dT[1][0][ii][0] -= ff * dely / rij;
    // dT[2][0][ii][0] -= ff * delz / rij;

    // dT[0][0][ii][1] -= (ff - s * rads[m] / rij) * delx * delx / std::pow(rij, 2);
    // dT[1][0][ii][1] -= (ff - s * rads[m] / rij) * delx * dely / std::pow(rij, 2);
    // dT[2][0][ii][1] -= (ff - s * rads[m] / rij) * delx * delz / std::pow(rij, 2);
    // dT[0][0][ii][1] -= rads[m] * s / rij;

    // dT[0][0][ii][2] -= (ff - s * rads[m] / rij) * dely * delx / std::pow(rij, 2);
    // dT[1][0][ii][2] -= (ff - s * rads[m] / rij) * dely * dely / std::pow(rij, 2);
    // dT[2][0][ii][2] -= (ff - s * rads[m] / rij) * dely * delz / std::pow(rij, 2);
    // dT[1][0][ii][2] -= rads[m] * s / rij;

    // dT[0][0][ii][3] -= (ff - s * rads[m] / rij) * delz * delx / std::pow(rij, 2);
    // dT[1][0][ii][3] -= (ff - s * rads[m] / rij) * delz * dely / std::pow(rij, 2);
    // dT[2][0][ii][3] -= (ff - s * rads[m] / rij) * delz * delz / std::pow(rij, 2);
    // dT[2][0][ii][3] -= rads[m] * s / rij;

    // partial derivative of T with respect to c parameters
    dT2c[jj][ii][0] += ff2;
    dT2c[jj][ii][1] += ff2 * delx / rij;
    dT2c[jj][ii][2] += ff2 * dely / rij;
    dT2c[jj][ii][3] += ff2 * delz / rij;

    // dT2c[0][ii][0] -= ff2;
    // dT2c[0][ii][1] -= ff2 * delx / rij;
    // dT2c[0][ii][2] -= ff2 * dely / rij;
    // dT2c[0][ii][3] -= ff2 * delz / rij;
}

double *Descriptor::get_feat() const
{
    return (double *)this->feat;
}

int Descriptor::get_nfeat() const
{
    return this->nfeat;
}

int *Descriptor::get_neighbor_list() const
{
    return (int *)this->neighbor_list_alltypes;
}

/*
std::pair<std::vector<double>, std::vector<double>> Descriptor::get_dfeat() const
{
    std::vector<double> dfeat_out;
    std::vector<double> dfeat2c_out;

    for (int i = 0; i < this->natoms; i++)
    {
        for (int j = 0; j < this->nfeat; j++)
        {
            for (int k = 0; k < this->max_neighbors; k++)
            {
                dfeat_out.push_back(this->dfeat_tmp[0][i][j][k]);
                dfeat_out.push_back(this->dfeat_tmp[1][i][j][k]);
                dfeat_out.push_back(this->dfeat_tmp[2][i][j][k]);
                dfeat2c_out.push_back(this->dfeat2c[i][j][k]);
            }
        }
    }
    return std::make_pair(std::move(dfeat_out), std::move(dfeat2c_out));
}
*/
double *Descriptor::get_dfeat() const
{
    return (double *)this->dfeat_tmp;
}

double *Descriptor::get_dfeat2c() const
{
    return (double *)this->dfeat2c;
}

void Descriptor::show() const
{
    std::cout << "natoms: " << this->natoms << std::endl;
    std::cout << "ntypes: " << this->ntypes << std::endl;
    std::cout << "max_neighbors: " << this->max_neighbors << std::endl;
    std::cout << "m1: " << this->m1 << std::endl;
    std::cout << "m2: " << this->m2 << std::endl;
    std::cout << "rcut_max: " << this->rcut_max << std::endl;
    std::cout << "rcut_smooth: " << this->rcut_smooth << std::endl;

    std::cout << "num_neigh_alltypes: \n";
    for (int i = 0; i < this->natoms; i++)
    {
        std::cout << this->num_neigh_alltypes[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "neighbor_list_alltypes: \n";
    for (int i = 0; i < this->natoms; i++)
    {
        for (int j = 0; j < this->max_neighbors; j++)
        {
            std::cout << this->neighbor_list_alltypes[i * this->max_neighbors + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "dR_neigh_alltypes: \n";
    for (int i = 0; i < this->natoms; i++)
    {
        for (int j = 0; j < this->max_neighbors; j++)
        {
            std::cout << this->dR_neigh_alltypes[i][j].rij << " ";
            std::cout << this->dR_neigh_alltypes[i][j].delx << " ";
            std::cout << this->dR_neigh_alltypes[i][j].dely << " ";
            std::cout << this->dR_neigh_alltypes[i][j].delz << " ";
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    std::cout << "ind_neigh_alltypes: \n";
    for (int i = 0; i < this->natoms; i++)
    {
        for (int j = 0; j < this->ntypes; j++)
        {
            for (int k = 0; k < this->max_neighbors; k++)
            {
                std::cout << this->ind_neigh_alltypes[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    // std::cout << "feat: \n";
    // for (int i = 0; i < this->natoms; i++)
    // {
    //     for (int j = 0; j < this->nfeat; j++)
    //     {
    //         std::cout << std::setw(10) << std::setprecision(6) << "feat[" << i << "][" << j << "] = " << this->feat[i * this->nfeat + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // std::cout << "dfeat_out: \n";
    // for (int i = 0; i < this->natoms; i++)
    // {
    //     for (int j = 0; j < this->nfeat; j++)
    //     {
    //         for (int k = 0; k < this->max_neighbors; k++)
    //         {
    //             std::cout << std::setw(10) << std::setprecision(6) << "dfeat_tmp[" << i << "][" << j << "][" << k << "] = ";
    //             std::cout << this->dfeat_tmp[3 * (i * this->nfeat * this->max_neighbors + j * this->max_neighbors + k) + 0] << " ";
    //             std::cout << this->dfeat_tmp[3 * (i * this->nfeat * this->max_neighbors + j * this->max_neighbors + k) + 1] << " ";
    //             std::cout << this->dfeat_tmp[3 * (i * this->nfeat * this->max_neighbors + j * this->max_neighbors + k) + 2] << " ";
    //         }
    //     }
    // }
    // std::cout << std::endl;
    // std::cout << "dfeat2c_out: \n";
    // for (int i = 0; i < this->natoms; i++)
    // {
    //     for (int j = 0; j < this->nfeat; j++)
    //     {
    //         for (int k = 0; k < this->max_neighbors; k++)
    //         {
    //             std::cout << std::setw(10) << std::setprecision(6) << "dfeat2c[" << i << "][" << j << "][" << k << "] = ";
    //             std::cout << this->dfeat2c[i * this->nfeat * this->max_neighbors + j * this->max_neighbors + k] << " ";
    //         }
    //     }
    // }
    // std::cout << std::endl;
}

/*int main()
{
    // double coords[15] = {4.5292997, 4.8908755, 5.1718035,
    //                      5.1422234, 5.6168573, 5.7252511,
    //                      3.8329047, 4.2310279, 5.782918,
    //                      3.947507, 5.491898, 4.5658593,
    //                      5.0937183, 4.2990798, 4.4207844};
    double coords[15] = {4.53522153693573, 4.89748210138009, 5.15704871018876,
                         5.16314548021497, 5.56595917362901, 5.76462128362908,
                         3.87292748607096, 4.24760318502786, 5.78867733427430,
                         3.92898320330153, 5.51123203720754, 4.54097267559737,
                         5.14697840273998, 4.27511238945079, 4.47923037687864};
    // double coords[15] = {0.45008462, 0.48733209, 0.51992897,
    //                      0.51314639, 0.55968573, 0.57062167,
    //                      0.38582763, 0.4247631, 0.5800803,
    //                      0.37890853, 0.55349125, 0.45774762,
    //                      0.50137233, 0.43355653, 0.44128795};

    double box[9] = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
    int type_map[5] = {0, 1, 1, 1, 1};
    int natoms = 5;
    int ntypes = 2;
    int max_neighbors = 10;
    float rcut_max = 6.0;
    float rcut_smooth = 0.5;
    int m1 = 4, m2 = 2, beta = 4;
    const int image = 1;
    // NeighborList nblist(6, coords, box, 10, 2, 5, type_map);
    // nblist.show();
    MultiNeighborList mnblist(image, 6, 10, 2, 5, type_map, coords, box);
    mnblist.show();
    Descriptor desc(image, beta, m1, m2, rcut_max, rcut_smooth, natoms, ntypes, max_neighbors);
    // desc.show();
    return 0;
}*/