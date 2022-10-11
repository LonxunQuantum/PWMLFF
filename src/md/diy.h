#ifndef DIY_HPP
#define DIY_HPP

using namespace std;
extern "C"
{
    // use fortran-lapack please use ifort not icpc as linker, and add -mkl to link these subroutines.
    // LU decomoposition of a general matrix
    void dgetrf_(int* M, int *N, double* A, int* LDA, int* IPIV, int* INFO);
    // generate inverse of a matrix given its LU decomposition
    void dgetri_(int* N, double* A, int* LDA, int* IPIV, double* WORK, int* LWORK, int* INFO);
    // matmul, C = alpha * A*B + beta * C
    void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A, int* LDA, double* B, int* LDB, double* BETA, double* C, int* LDC);
}

// call fortran subroutine from cpp, diy fortran subroutine binded C name.
extern "C" void f2c_calc_energy_force(int* /*imodel*/,int* /*n_atom*/, int* /*type_atom*/, double* /*lat*/, double* /*x_frac*/, double* /*e_atom*/, double* /*f_atom*/, double* /*e_tot*/, int* /*iflag_reneighbor*/);
    
namespace diy
{
    void dinv(double*, int);
    template <typename TYPE> TYPE *create(TYPE *&array, int n);
    template <typename TYPE> void destroy(TYPE *&array);
    template <typename TYPE> TYPE **create(TYPE **&array, int n1, int n2);
    template <typename TYPE> void destroy(TYPE **&array);
}
#endif
