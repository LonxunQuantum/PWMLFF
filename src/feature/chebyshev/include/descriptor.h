#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include <iostream>
#include "radial.h"
#include "neighborList.h"

class Descriptor
{ // descriptor
public:
    Descriptor();

    Descriptor(int beta, int m1, int m2, float rcut_max, float rcut_smooth,
               int natoms, int ntypes, int max_neighbors, int *type_map, 
               int *num_neigh_all, int *neighbors_list_all, double *dR_neigh_all, double *c = nullptr);

    Descriptor(const Descriptor &other);

    ~Descriptor();

    void set_cparams(int ntypes, int natoms, int beta, int m1);

    void build(int max_neighbors, int ntypes, int natoms); // build descriptor

    void build_component(int m, int ii, int jj, double delx, double dely, double delz, int nneigh, int itype, int jtype,
                         double ***rads, double ***drads, double ****drads2c, double ****ddrads2c, double fc, double dfc,
                         double s, double rij, std::vector<std::vector<double>> &T,
                         std::vector<std::vector<std::vector<std::vector<double>>>> &dT,
                         std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> &dT2c,
                         std::vector<double> &ddT2c); // build 4-D descriptor component

    double *get_feat() const; // get descriptor

    int get_nfeat() const; // get number of descriptor

    int *get_neighbor_list() const; // get neighbor list for all atom types, including self

    // std::pair<std::vector<double>, std::vector<double>> get_dfeat() const; // get partial derivative of descriptor with respect to rij
    double *get_dfeat() const; // get partial derivative of descriptor with respect to rij

    double *get_dfeat2c() const; // get partial derivative of descriptor with respect to c

    double *get_ddfeat2c() const; // get partial derivative of dfeat with respect to c

    void show() const; // show descriptor

    const double PI = 3.141592653589793;

private:
    int natoms, ntypes, max_neighbors, beta, m1, m2, nfeat;
    float rcut_max, rcut_smooth;
    int *type_map;
    int *num_neigh_all;
    int *neighbors_list_all;
    double *dR_neigh_all;
    double *c;
    bool c_is_internal;
    SmoothFunc<double> smooth;
    Radial<double> radial;
    int *num_neigh_alltypes;      // number of neighbors for all atom types, including self
    // int **neighbor_list_alltypes; // neighbor list for all atom types, including self
    int *neighbor_list_alltypes; // neighbor list for all atom types, including self
    // Neighbor **dR_neigh_alltypes; // partial derivative of neighbor list with respect to rij
    double *dR_neigh_alltypes; // partial derivative of neighbor list with respect to rij
    // int ***ind_neigh_alltypes;    // index of neighbors for all atom types, including self
    int *ind_neigh_alltypes;    // index of neighbors for all atom types, including self
    double *feat;                 // descriptor
    // double ****dfeat_tmp;
    double *dfeat_tmp;
    // double ****dfeat;  // partial derivative of descriptor with respect to rij
    double *dfeat;   // partial derivative of descriptor with respect to rij
    double *dfeat2c; // partial derivative of descriptor with respect to c
    double *ddfeat2c; // partial derivative of dfeat with respect to c
};                   // end of class Descriptor

class MultiDescriptor
{ // descriptor
public:
    MultiDescriptor() {}

    MultiDescriptor(int images, int beta, int m1, int m2, float rcut_max, float rcut_smooth,
                    int natoms, int ntypes, int max_neighbors, int *type_map,
                    int *num_neigh_all, int *neighbors_list_all, double *dR_neigh_all, double *c = nullptr)
        : images(images), natoms(natoms), max_neighbors(max_neighbors), ntypes(ntypes), m1(m1), m2(m2), beta(beta)
    {
        int nfeat = m1 * m2;
        this->feat_all = new double[images * natoms * nfeat];
        this->dfeat_all = new double[images * 3 * natoms * nfeat * max_neighbors];
        this->dfeat2c_all = new double[images * natoms * nfeat * ntypes * m1 * beta];
        this->ddfeat2c_all = new double[images * 3 * natoms * nfeat * ntypes * m1 * beta * max_neighbors];
        this->neighbor_list_alltypes = new int[images * natoms * max_neighbors];
        this->descriptor = new Descriptor *[images];
        int offset1, offset2, offset3;
        for (int i = 0; i < images; i++)
        {
            offset1 = i * natoms * ntypes;
            offset2 = offset1 * max_neighbors;
            offset3 = offset2 * 4;       
            if (c != nullptr)
            {
                this->descriptor[i] = new Descriptor(beta, m1, m2, rcut_max, rcut_smooth, natoms, ntypes, max_neighbors, type_map, num_neigh_all + offset1, neighbors_list_all + offset2, dR_neigh_all + offset3, c);
            }
            else
            {     
                this->descriptor[i] = new Descriptor(beta, m1, m2, rcut_max, rcut_smooth, natoms, ntypes, max_neighbors, type_map, num_neigh_all + offset1, neighbors_list_all + offset2, dR_neigh_all + offset3);
            }
        }
    }
    ~MultiDescriptor()
    {
        for (int i = 0; i < this->images; i++)
        {
            delete descriptor[i];
        }
        delete[] this->descriptor;
        delete[] this->feat_all;
        delete[] this->dfeat_all;
        delete[] this->dfeat2c_all;
        delete[] this->ddfeat2c_all;
        delete[] this->neighbor_list_alltypes;
    }

    double *get_feat() const
    {
        double *feat;
        int nfeat, index;
        for (int i = 0; i < this->images; i++)
        {
            feat = this->descriptor[i]->get_feat();
            nfeat = this->descriptor[i]->get_nfeat();
            index = nfeat * this->natoms;
            std::copy(feat, feat + index, this->feat_all + i * index);
        }
        // int size = images * nfeat * natoms;
        // double *result_all = new double[size];
        // std::copy(feat_all, feat_all + size, result_all);
        return this->feat_all;
    }

    void show() const
    {
        for (int i = 0; i < this->images; i++)
        {
            this->descriptor[i]->show();
        }
    }

    int get_nfeat() const { return this->m1 * this->m2; } 

    // double *get_dfeat(int *size) const
    // {
    //     std::vector<double> results;
    //     for (int i = 0; i < images; i++)
    //     {
    //         std::pair<std::vector<double>, std::vector<double>> dfeat = descriptor[i]->get_dfeat();
    //         results.insert(results.end(), dfeat.first.begin(), dfeat.first.end());
    //         results.insert(results.end(), dfeat.second.begin(), dfeat.second.end());
    //     }
    //     *size = results.size();
    //     double *result_all = new double[*size];
    //     std::copy(results.begin(), results.end(), result_all);
    //     return result_all;
    // }
    double *get_dfeat() const
    {
        double *dfeat;
        int nfeat, index;
        for (int i = 0; i < this->images; i++)
        {
            dfeat = this->descriptor[i]->get_dfeat();
            nfeat = this->descriptor[i]->get_nfeat();
            index = 3 * this->natoms * nfeat * this->max_neighbors;
            std::copy(dfeat, dfeat + index, this->dfeat_all + i * index);
        }
        return this->dfeat_all;
    }

    double *get_dfeat2c() const
    {
        double *dfeat2c;
        int nfeat, index;
        for (int i = 0; i < this->images; i++)
        {
            dfeat2c = this->descriptor[i]->get_dfeat2c();
            nfeat = this->descriptor[i]->get_nfeat();
            index = this->natoms * nfeat * this->ntypes * this->m1 * this->beta;
            std::copy(dfeat2c, dfeat2c + index, this->dfeat2c_all + i * index);
        }
        return this->dfeat2c_all;
    }

    double *get_ddfeat2c() const
    {
        double *ddfeat2c;
        int nfeat, index;
        for (int i = 0; i < this->images; i++)
        {
            ddfeat2c = this->descriptor[i]->get_ddfeat2c();
            nfeat = this->descriptor[i]->get_nfeat();
            index = 3 * this->natoms * nfeat * this->ntypes * this->m1 * this->beta * this->max_neighbors;
            std::copy(ddfeat2c, ddfeat2c + index, this->ddfeat2c_all + i * index);
        }
        return this->ddfeat2c_all;
    }

    int *get_neighbor_list() const
    {
        int *neighbor_list;
        int index;
        for (int i = 0; i < this->images; i++)
        {
            neighbor_list = this->descriptor[i]->get_neighbor_list();
            index = this->natoms * this->max_neighbors;
            std::copy(neighbor_list, neighbor_list + index, this->neighbor_list_alltypes + i * index);
        }
        return this->neighbor_list_alltypes;
    }

    int images, natoms, max_neighbors, ntypes, m1, m2, beta;

private:
    Descriptor **descriptor;
    double *feat_all;
    double *dfeat_all;
    double *dfeat2c_all;
    double *ddfeat2c_all;
    int *neighbor_list_alltypes;
}; // end of class MultiDescriptor

extern "C"
{
    MultiDescriptor *CreateDescriptor(int images, int beta, int m1, int m2, float rcut_max, float rcut_smooth,
                                      int natoms, int ntypes, int max_neighbors, int *type_map,
                                      int *num_neigh_all, int *neighbors_list_all, double *dR_neigh_all, double *c = nullptr)
    {
        return new MultiDescriptor(images, beta, m1, m2, rcut_max, rcut_smooth, natoms, ntypes, max_neighbors, type_map, num_neigh_all, neighbors_list_all, dR_neigh_all, c);
    }

    void DestroyDescriptor(MultiDescriptor *descriptor)
    {
        delete descriptor; 
    }

    double *get_feat(MultiDescriptor *descriptor)
    {
        return descriptor->get_feat();
    }

    void show(MultiDescriptor *descriptor)
    {
        descriptor->show();
    }

    // double *get_dfeat(MultiDescriptor *descriptor, int *size)
    // {
    //     return descriptor->get_dfeat(size);
    // }
    double *get_dfeat(MultiDescriptor *descriptor)
    {
        return descriptor->get_dfeat();
    }

    double *get_dfeat2c(MultiDescriptor *descriptor)
    {
        return descriptor->get_dfeat2c();
    }

    double *get_ddfeat2c(MultiDescriptor *descriptor)
    {
        return descriptor->get_ddfeat2c();
    }

    int *get_neighbor_list(MultiDescriptor *descriptor)
    {
        return descriptor->get_neighbor_list();
    }

}
/*
In python, the following code will be executed:

from numpy.ctypeslib import ndpointer
import ctypes
lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib = ctypes.CDLL(os.path.join(lib_path, 'feature/chebyshev/build/lib/libdescriptor.so')) # multi-descriptor
lib.CreateDescriptor.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, 
                                  ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                  ndpointer(ctypes.c_int, ndim=1, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_int, ndim=3, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_int, ndim=4, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_double, ndim=5, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_double, ndim=4, flags="C_CONTIGUOUS")]
                                  
lib.CreateDescriptor.restype = ctypes.c_void_p

lib.show.argtypes = [ctypes.c_void_p]
lib.DestroyDescriptor.argtypes = [ctypes.c_void_p]

lib.get_feat.argtypes = [ctypes.c_void_p]
lib.get_feat.restype = ctypes.POINTER(ctypes.c_double)
lib.get_dfeat.argtypes = [ctypes.c_void_p]
lib.get_dfeat.restype = ctypes.POINTER(ctypes.c_double)
lib.get_dfeat2c.argtypes = [ctypes.c_void_p]
lib.get_dfeat2c.restype = ctypes.POINTER(ctypes.c_double)
lib.get_ddfeat2c.argtypes = [ctypes.c_void_p]
lib.get_ddfeat2c.restype = ctypes.POINTER(ctypes.c_double)
lib.get_neighbor_list.argtypes = [ctypes.c_void_p]
lib.get_neighbor_list.restype = ctypes.POINTER(ctypes.c_int)

descriptor = lib.CreateDescriptor(batch_size, self.beta, self.m1, self.m2, self.Rc_M, self.rcut_smooth, natoms_sum, ntypes, self.m_neigh, Imagetype_map.cpu().numpy(), num_neigh.cpu().numpy(), list_neigh.cpu().numpy(), ImagedR.cpu().numpy(), c)
# lib.show(descriptor)
feat = lib.get_feat(descriptor)
dfeat = lib.get_dfeat(descriptor)
dfeat2c = lib.get_dfeat2c(descriptor)
ddfeat2c = lib.get_ddfeat2c(descriptor)
list_neigh_alltype = lib.get_neighbor_list(descriptor)

feat = np.ctypeslib.as_array(feat, (batch_size * natoms_sum, nfeat))
dfeat = np.ctypeslib.as_array(dfeat, (batch_size, natoms_sum, nfeat, self.m_neigh, 3))
dfeat2c = np.ctypeslib.as_array(dfeat2c, (batch_size, natoms_sum, nfeat, ntypes, self.m1, self.beta))
ddfeat2c = np.ctypeslib.as_array(ddfeat2c, (batch_size, natoms_sum, nfeat, ntypes, self.m1, self.beta, self.m_neigh, 3))
list_neigh_alltype = np.ctypeslib.as_array(list_neigh_alltype, (batch_size, natoms_sum, self.m_neigh))
*/

#endif // DESCRIPTOR_H