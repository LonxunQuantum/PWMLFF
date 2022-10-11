#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "diy.h"

namespace diy
{
    // inverse a matrix    
    void dinv(double* A, int n)
    {
        int *ipiv = new int[n];
        int lwork = n*n;
        double *work = new double[lwork];
        int info;
    
        dgetrf_(&n,&n,A,&n,ipiv,&info);
        dgetri_(&n,A,&n,ipiv,work,&lwork,&info);
    
        delete[] ipiv;
        delete[] work;
    }

    // memory operators
    // similar to lammps memory.h
    // 1d malloc and free
    template <typename TYPE> TYPE *create(TYPE *&array, int n)
    {                          
        int nbytes = ((int) sizeof(TYPE)) * n;
        array = (TYPE *) malloc(nbytes);
        return array;
    }
    template <typename TYPE> void destroy(TYPE *&array)
    {
        free(array);
        array = nullptr;                                 
    }                       
    
    // 2d malloc and free
    template <typename TYPE> TYPE **create(TYPE **&array, int n1, int n2)
    {
        // lmp similar malloc function for 2d array
        int nbytes = ((int)sizeof(TYPE)) * n1 * n2;
        TYPE *data = (TYPE *)malloc(nbytes);
        nbytes = ((int)sizeof(TYPE *)) * n1;
        array = (TYPE **)malloc(nbytes);
    
        int n = 0;
        for (int i = 0; i < n1; i++)
        {
          array[i] = &data[n];
          n += n2;
        }
        return array;
    }
    
    template <typename TYPE> void destroy(TYPE **&array)
    {
        // lmp similar free function for 2d array
        if (array == nullptr) return;
        free(array[0]);
        free(array);
        array = nullptr;
    }

}
