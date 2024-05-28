#ifndef NEIGHBOR_LIST_H
#define NEIGHBOR_LIST_H

#include <cmath>
#include <vector>
#include <iostream>

struct Neighbor
{
    double rij;
    double delx;
    double dely;
    double delz;

    bool operator!=(const Neighbor &other) const
    {
        return rij != other.rij || delx != other.delx || dely != other.dely || delz != other.delz;
    }
};

class NeighborList
{
public:
    NeighborList();
    NeighborList(float cutoff, double *coords, double *box, int max_neighbors, int ntypes, int natoms, int *type_map);
    ~NeighborList();

    double get_volume(const double *box) const;
    double get_det(const double *H_matrix) const;
    double get_area(int dim, const double *box) const;
    double get_area_direction(const double *vec1, const double *vec2) const;
    void get_expanded_box(float cutoff, const double *box);
    void get_inversed_box(double *ebox);
    void applyMic(const double *ebox, double &x12, double &y12, double &z12);
    void applyMicOne(double &x12);

    void build(double *coords, double *box, int ***neighbors_list, int **num_neigh, Neighbor ***dR_neigh);
    void show() const;

    int **get_num_neigh() const;
    int ***get_neighbors_list() const;
    Neighbor ***get_dR_neigh() const;

private:
    float cutoff;
    int max_neighbors;
    int ntypes;
    int natoms;
    int *type_map;
    double ebox[18];
    int num_cells[3];
    int *num_types;
    int **num_neigh;
    int ***neighbors_list;
    Neighbor ***dR_neigh;

    void check_input_parameters(float cutoff, double *coords, double *box, int max_neighbors, int ntypes, int natoms, int *type_map) const;
};

int **NeighborList::get_num_neigh() const
{
    return this->num_neigh;
}

int ***NeighborList::get_neighbors_list() const
{
    return this->neighbors_list;
}

Neighbor ***NeighborList::get_dR_neigh() const
{
    return this->dR_neigh;
}

class MultiNeighborList
{
public:
    MultiNeighborList() : num_neigh_all(), neighbors_list_all(), dR_neigh_all(){};
    MultiNeighborList(int images, float cutoff, int max_neighbors, int ntypes, int natoms, int *type_map, double *coords_all, double *box_all)
        : cutoff(cutoff), max_neighbors(max_neighbors), ntypes(ntypes), natoms(natoms), type_map(type_map), images(images), coords_all(coords_all), box_all(box_all)
    {
        num_neigh_all = new int[images * natoms * ntypes]();
        neighbors_list_all = new int[images * natoms * ntypes * max_neighbors];
        std::fill_n(neighbors_list_all, images * natoms * ntypes * max_neighbors, -1);
        // dR_neigh_all = new Neighbor[images * natoms * ntypes * max_neighbors]();
        dR_neigh_all = new double[images * natoms * ntypes * max_neighbors * 4]();

        build();
    }
    ~MultiNeighborList()
    {
        delete[] num_neigh_all;
        delete[] neighbors_list_all;
        delete[] dR_neigh_all;
    }

    void build()
    {
        for (int i = 0; i < images; i++)
        {
            double coords[natoms * 3];
            double box[9];
            for (int j = 0; j < 9; j++)
            {
                box[j] = box_all[i * 9 + j];
            }
            for (int j = 0; j < natoms; j++)
            {
                double frac_coords[3] = {coords_all[i * natoms * 3 + j * 3], coords_all[i * natoms * 3 + j * 3 + 1], coords_all[i * natoms * 3 + j * 3 + 2]};
                double cart_coords[3];
                frac_to_cart(frac_coords, box, cart_coords); // Temporary use
                coords[j * 3] = cart_coords[0];
                coords[j * 3 + 1] = cart_coords[1];
                coords[j * 3 + 2] = cart_coords[2];
            }
            NeighborList *neighborList = new NeighborList(cutoff, coords, box, max_neighbors, ntypes, natoms, type_map);
            int **num_neigh = neighborList->get_num_neigh();
            int ***neighbors_list = neighborList->get_neighbors_list();
            Neighbor ***dR_neigh = neighborList->get_dR_neigh();
            for (int j = 0; j < natoms; j++)
            {
                for (int k = 0; k < ntypes; k++)
                {
                    num_neigh_all[i * natoms * ntypes + j * ntypes + k] = num_neigh[j][k];
                    for (int l = 0; l < num_neigh[j][k]; l++)
                    {
                        int indices = i * natoms * ntypes * max_neighbors + j * ntypes * max_neighbors + k * max_neighbors + l;
                        neighbors_list_all[indices] = neighbors_list[j][k][l];
                        // dR_neigh_all[indices] = dR_neigh[j][k][l];
                        dR_neigh_all[indices * 4] = dR_neigh[j][k][l].rij;
                        dR_neigh_all[indices * 4 + 1] = dR_neigh[j][k][l].delx;
                        dR_neigh_all[indices * 4 + 2] = dR_neigh[j][k][l].dely;
                        dR_neigh_all[indices * 4 + 3] = dR_neigh[j][k][l].delz;
                    }
                }
            }
        }
    }

    void show() const
    {
        for (int i = 0; i < images; i++)
        {
            std::cout << "Image " << i << std::endl;
            for (int j = 0; j < natoms; j++)
            {
                for (int k = 0; k < ntypes; k++)
                {
                    std::cout << "Atom " << j << " type " << k << " has " << num_neigh_all[i * natoms * ntypes + j * ntypes + k] << " neighbors" << std::endl;
                    for (int l = 0; l < num_neigh_all[i * natoms * ntypes + j * ntypes + k]; l++)
                    {
                        int indices = i * natoms * ntypes * max_neighbors + j * ntypes * max_neighbors + k * max_neighbors + l;
                        std::cout << "index " << neighbors_list_all[indices] << " ";
                        // std::cout << "rij " << dR_neigh_all[indices].rij << " ";
                        // std::cout << "delx " << dR_neigh_all[indices].delx << " ";
                        // std::cout << "dely " << dR_neigh_all[indices].dely << " ";
                        // std::cout << "delz " << dR_neigh_all[indices].delz << std::endl;
                        std::cout << "rij " << dR_neigh_all[indices * 4] << " ";
                        std::cout << "delx " << dR_neigh_all[indices * 4 + 1] << " ";
                        std::cout << "dely " << dR_neigh_all[indices * 4 + 2] << " ";
                        std::cout << "delz " << dR_neigh_all[indices * 4 + 3] << std::endl;
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                }
            }
        }
    }

    int *get_num_neigh_all() const
    {
        return num_neigh_all;
    }

    int *get_neighbors_list_all() const
    {
        return neighbors_list_all;
    }

    double *get_dR_neigh_all() const
    {
        return dR_neigh_all;
    }

    void frac_to_cart(double frac_coords[3], double box[9], double cart_coords[3])
    {
        cart_coords[0] = frac_coords[0] * box[0] + frac_coords[1] * box[1] + frac_coords[2] * box[2];
        cart_coords[1] = frac_coords[0] * box[3] + frac_coords[1] * box[4] + frac_coords[2] * box[5];
        cart_coords[2] = frac_coords[0] * box[6] + frac_coords[1] * box[7] + frac_coords[2] * box[8];
    }

    int max_neighbors;
    int ntypes;
    int natoms;
    int images;
    
private:
    int *num_neigh_all;
    int *neighbors_list_all;
    // Neighbor *dR_neigh_all;
    double *dR_neigh_all;
    float cutoff;
    int *type_map;
    double *coords_all;
    double *box_all;
};

extern "C"
{
    MultiNeighborList *CreateNeighbor(int images, float cutoff, int max_neighbors, int ntypes, int natoms, int *type_map, double *coords_all, double *box_all)
    {
        return new MultiNeighborList(images, cutoff, max_neighbors, ntypes, natoms, type_map, coords_all, box_all);
    }

    void DestroyNeighbor(MultiNeighborList *neighbor)
    {
        delete neighbor;
    }

    void ShowNeighbor(MultiNeighborList *neighbor)
    {
        neighbor->show();
    }

    int *GetNumNeighAll(MultiNeighborList *neighbor)
    {
        return neighbor->get_num_neigh_all();
    }

    int *GetNeighborsListAll(MultiNeighborList *neighbor)
    {
        return neighbor->get_neighbors_list_all();
    }

    double *GetDRNeighAll(MultiNeighborList *neighbor)
    {
        return neighbor->get_dR_neigh_all();
    }
}
#endif