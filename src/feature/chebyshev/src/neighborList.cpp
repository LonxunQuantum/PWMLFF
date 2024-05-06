#include "neighborList.h"
#include <algorithm>
#include <stdexcept>

void NeighborList::check_input_parameters(float cutoff, double* coords, double* box, int max_neighbors, int ntypes, int natoms, int* type_map) const {
    if (cutoff <= 0.0f) {
        throw std::invalid_argument("Cutoff radius must be positive.");
    }

    if (max_neighbors <= 0) {
        throw std::invalid_argument("Maximum number of neighbors must be positive.");
    }

    if (ntypes <= 0) {
        throw std::invalid_argument("Number of atom types must be positive.");
    }

    if (natoms <= 0) {
        throw std::invalid_argument("Number of atoms must be positive.");
    }

    if (coords == nullptr) {
        throw std::invalid_argument("Coordinates array cannot be null.");
    }

    if (box == nullptr) {
        throw std::invalid_argument("Box array cannot be null.");
    }

    if (type_map == nullptr) {
        throw std::invalid_argument("Type map array cannot be null.");
    }
}

NeighborList::NeighborList() {}; // default constructor
/**
 * @brief Constructor for the NeighborList class.
 * 
 * @param cutoff The cutoff radius.
 * @param coords The coordinates of the atoms.
 * @param box The lattice of the system.
 * @param max_neighbors The maximum number of neighbors.
 * @param ntypes The number of atom types.
 * @param natoms The number of atoms.
 * @param type_map The atom type of each atom. index from 0 to ntypes-1.
 */
NeighborList::NeighborList(float cutoff, double* coords, double* box, int max_neighbors, int ntypes, int natoms, int* type_map) {
    check_input_parameters(cutoff, coords, box, max_neighbors, ntypes, natoms, type_map);
    this->cutoff = cutoff;
    this->max_neighbors = max_neighbors;
    this->ntypes = ntypes;
    this->natoms = natoms;
    this->type_map = type_map;

    this->num_types = new int[ntypes];
    this->num_neigh = new int*[natoms];
    this->neighbors_list = new int**[natoms];
    this->dR_neigh = new Neighbor**[natoms];
    for (int i = 0; i < natoms; i++) {
        this->neighbors_list[i] = new int*[ntypes];
        this->num_neigh[i] = new int[ntypes];
        this->dR_neigh[i] = new Neighbor*[ntypes];
        for (int j = 0; j < ntypes; j++) {
            this->neighbors_list[i][j] = new int[max_neighbors];
            this->dR_neigh[i][j] = new Neighbor[max_neighbors];
            std::fill_n(this->neighbors_list[i][j], max_neighbors, -1);
            std::fill_n(this->dR_neigh[i][j], max_neighbors, Neighbor{0, 0, 0, 0});
        }
    }
    build(coords, box, this->neighbors_list, this->num_neigh, this->dR_neigh);
} // constructor

/**
 * @brief Destructor for the NeighborList class.
 */
NeighborList::~NeighborList() {
    for (int i = 0; i < this->natoms; i++) {
        for (int j = 0; j < this->ntypes; j++) {
            delete[] this->neighbors_list[i][j];
            delete[] this->dR_neigh[i][j];
        }
        delete[] this->dR_neigh[i];
        delete[] this->neighbors_list[i];
        delete[] this->num_neigh[i];
    }
    delete[] this->dR_neigh;
    delete[] this->neighbors_list;
    delete[] this->num_neigh;
    delete[] this->num_types;
} // destructor

/**
 * @brief Get the volume of the system.
 * 
 * @param box The lattice of the system.
 * @return The volume of the system. 
 * $V = |det(H)|$
 */
double NeighborList::get_volume(const double* box) const {
    return abs(get_det(box));
} // get_volume

/**
 * @brief Get the determinant of the lattice.
 * 
 * @param H_matrix The lattice of the system.
 * $H_matrix$ is a 3x3 matrix that represents the lattice of the system.
 * $H_matrix = [a_x, b_x, c_x, 
 *              a_y, b_y, c_y,
 *              a_z, b_z, c_z]$
 * @return The determinant of the lattice.
 */
double NeighborList::get_det(const double* H_matrix) const {
    double det = 0.0;
    det =  H_matrix[0] * (H_matrix[4] * H_matrix[8] - H_matrix[5] * H_matrix[7]) +
            H_matrix[1] * (H_matrix[5] * H_matrix[6] - H_matrix[3] * H_matrix[8]) + 
            H_matrix[2] * (H_matrix[3] * H_matrix[7] - H_matrix[4] * H_matrix[6]);
    return det;
} // get_det

/**
 * @brief Get the area of the system.
 * 
 * @param dim The dimension of the system.
 * @param box The lattice of the system.
 * @return The area of the system.
 */
double NeighborList::get_area(int dim, const double* box) const {
    double area;
    double a[3] = {box[0], box[3], box[6]};  
    double b[3] = {box[1], box[4], box[7]};
    double c[3] = {box[2], box[5], box[8]};
    if (dim == 0) {
        area = get_area_direction(b, c);
    } else if (dim == 1) {
        area = get_area_direction(c, a);
    } else if (dim == 2) {
        area = get_area_direction(a, b);
    } else {
        std::cerr << "Error: The dimension is out of range." << std::endl;
        exit(1);
    }
    return area;
} // get_area

/**
 * @brief Get the area of the system in a specific direction.
 * 
 * @param vec1 The first vector.
 * @param vec2 The second vector.
 * @return The area of the system in a specific direction.
 */
double NeighborList::get_area_direction(const double* vec1, const double* vec2) const {
    double s1 = vec1[1] * vec2[2] - vec1[2] * vec2[1];
    double s2 = vec1[2] * vec2[0] - vec1[0] * vec2[2];
    double s3 = vec1[0] * vec2[1] - vec1[1] * vec2[0];
    return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
} // get_area_direction

/**
 * @brief Get the expanded box.
 * 
 * @param cutoff The cutoff radius.
 * @param box The lattice of the system.
 */
void NeighborList::get_expanded_box(float cutoff, const double* box) {
    double volume = get_volume(box);
    double thickness_x = volume / get_area(0, box);
    double thickness_y = volume / get_area(1, box);
    double thickness_z = volume / get_area(2, box);
    this->num_cells[0] = int(ceil(2.0 * cutoff / thickness_x)); // number of cells in x direction
    this->num_cells[1] = int(ceil(2.0 * cutoff / thickness_y));
    this->num_cells[2] = int(ceil(2.0 * cutoff / thickness_z));

    ebox[0] = box[0] * this->num_cells[0];
    ebox[3] = box[3] * this->num_cells[0];
    ebox[6] = box[6] * this->num_cells[0];
    ebox[1] = box[1] * this->num_cells[1];
    ebox[4] = box[4] * this->num_cells[1];
    ebox[7] = box[7] * this->num_cells[1];
    ebox[2] = box[2] * this->num_cells[2];
    ebox[5] = box[5] * this->num_cells[2];
    ebox[8] = box[8] * this->num_cells[2];

    get_inversed_box(ebox);
}

/**
 * @brief Get the inversed box.
 * 
 * @param ebox The expanded box.
 */
void NeighborList::get_inversed_box(double* ebox) {
    ebox[9] = ebox[4] * ebox[8] - ebox[5] * ebox[7];
    ebox[10] = ebox[2] * ebox[7] - ebox[1] * ebox[8];
    ebox[11] = ebox[1] * ebox[5] - ebox[2] * ebox[4];
    ebox[12] = ebox[5] * ebox[6] - ebox[3] * ebox[8];
    ebox[13] = ebox[0] * ebox[8] - ebox[2] * ebox[6];
    ebox[14] = ebox[2] * ebox[3] - ebox[0] * ebox[5];
    ebox[15] = ebox[3] * ebox[7] - ebox[4] * ebox[6];
    ebox[16] = ebox[1] * ebox[6] - ebox[0] * ebox[7];
    ebox[17] = ebox[0] * ebox[4] - ebox[1] * ebox[3];
    double det = get_det(ebox);
    for (int i = 9; i < 18; i++) {
        ebox[i] /= det;
    }
} // get_inversed_box

/**
 * @brief The minimum mirror convention of the box is realized
 * 
 * @param ebox The expanded box.
 * @param x12 The displacement vector in the x direction of the atom i and j.
 * @param y12 The displacement vector in the y direction of the atom i and j.
 * @param z12 The displacement vector in the z direction of the atom i and j.
 */
void NeighborList::applyMic(const double* ebox, double& x12, double& y12, double& z12) {
    double sx12, sy12, sz12;
    sx12 = ebox[9] * x12 + ebox[10] * y12 + ebox[11] * z12; // convert to fractional coordinates
    sy12 = ebox[12] * x12 + ebox[13] * y12 + ebox[14] * z12;
    sz12 = ebox[15] * x12 + ebox[16] * y12 + ebox[17] * z12;

    applyMicOne(sx12);
    applyMicOne(sy12);
    applyMicOne(sz12);

    x12 = ebox[0] * sx12 + ebox[1] * sy12 + ebox[2] * sz12;  // convert to Cartesian coordinates
    y12 = ebox[3] * sx12 + ebox[4] * sy12 + ebox[5] * sz12;
    z12 = ebox[6] * sx12 + ebox[7] * sy12 + ebox[8] * sz12;
} // applyMic

/**
 * @brief The minimum image convention is applied to the displacement vector.
 * 
 * @param x12 The displacement vector.
 */
void NeighborList::applyMicOne(double& x12) {
    if (x12 > 0.5) {
        x12 -= 1.0;
    } else if (x12 < -0.5) {
        x12 += 1.0;
    }
} // applyMicOne


/**
 * @brief Build the neighbors list.
 * 
 * @param coords The coordinates of the atoms.
 * @param box The lattice of the system.
 * @param neighbors_list The neighbors list.
 * @param num_neigh The number of neighbors for each atom.
 * @param dR_neigh The displacement vector of the neighbors.
 */
void NeighborList::build(double* coords, double* box, int*** neighbors_list, int** num_neigh, Neighbor*** dR_neigh) {
    get_expanded_box(this->cutoff, box);
    double* ebox = this->ebox;
    int* ncells = this->num_cells;
    int ntypes = this->ntypes;
    int natoms = this->natoms;
    int max_neighbors = this->max_neighbors;
    int* type_map = this->type_map;
    double cutoff = this->cutoff;
    int* num_types = this->num_types;
    int itype;
    double rsq, delx, dely, delz, rij;

    /*
    在这段代码中，ia, ib, 和 ic 是循环变量，它们分别遍历了在x、y、z方向上的扩展单元格。这是为了考虑周期性边界条件，即当一个原子移动到盒子的一边时，它会从另一边出现。

    ncells[0], ncells[1], 和 ncells[2] 分别表示在x、y、z方向上的扩展单元格的数量。这个数量是根据截断距离和盒子的尺寸计算得出的，以确保在截断距离内的所有原子都被考虑到。

    在每个循环中，ia, ib, 和 ic 分别从0遍历到 ncells[0], ncells[1], 和 ncells[2]。对于每个 ia, ib, 和 ic 的组合，都计算了一个位移向量 delta，这个向量表示从原始盒子到扩展单元格的位移。

    然后，这个位移向量 delta 被用于计算原子 i 和 j 之间的距离，以确定它们是否是邻居。如果 i 和 j 是同一个原子，并且 ia, ib, 和 ic 都是0（也就是说，为原始盒子的时候），那么就跳过这次循环，因为一个原子不能是它自己的邻居。
    */
    for (int i = 0; i < natoms; i++) {
        std::fill_n(num_types, ntypes, 0);
        int count_radial = 0;   // number of neighbors
        for (int j = 0; j < natoms; j++) {
            for (int ia = 0; ia < ncells[0]; ia++) {
                for (int ib = 0; ib < ncells[1]; ib++) {
                    for (int ic = 0; ic < ncells[2]; ic++) {
                        if (i == j && ia == 0 && ib == 0 && ic == 0) {
                            continue;
                        }
                        double delta[3];
                        delta[0] = box[0] * ia + box[1] * ib + box[2] * ic;
                        delta[1] = box[3] * ia + box[4] * ib + box[5] * ic;
                        delta[2] = box[6] * ia + box[7] * ib + box[8] * ic;

                        delx = coords[3 * i] - coords[3 * j] + delta[0];
                        dely = coords[3 * i + 1] - coords[3 * j + 1] + delta[1];
                        delz = coords[3 * i + 2] - coords[3 * j + 2] + delta[2];

                        applyMic(ebox, delx, dely, delz);
                        rsq = delx * delx + dely * dely + delz * delz;
                        if (rsq < cutoff * cutoff) {
                            if (count_radial < max_neighbors) {
                                rij = sqrt(rsq);
                                itype = type_map[j];
                                num_types[itype]++;
                                neighbors_list[i][itype][count_radial] = j;
                                dR_neigh[i][itype][count_radial] = Neighbor{rij, delx, dely, delz};
                                count_radial++;
                            }
                        }
                    }
                }
            }
        }
        for (int j = 0; j < ntypes; j++) {
            num_neigh[i][j] = num_types[j];
            std::stable_partition(neighbors_list[i][j], neighbors_list[i][j] + max_neighbors, 
                [](int v) { return v != -1; });
            std::stable_partition(dR_neigh[i][j], dR_neigh[i][j] + max_neighbors, 
                [](const Neighbor& n) { return n != Neighbor{0, 0, 0, 0}; });
        }
    }
} // build

/**
 * @brief Get the neighbors list.
 * 
 * @return The neighbors list.
 */
void NeighborList::show() const {
    for (int i = 0; i < this->natoms; i++) {
        for (int j = 0; j < this->ntypes; j++) {
            std::cout << "Atom " << i << " type " << j << " has " << this->num_neigh[i][j] << " neighbors." << std::endl;
            for (int k = 0; k < this->max_neighbors; k++) {
                std::cout << this->neighbors_list[i][j][k] << " ";
                std::cout << this->dR_neigh[i][j][k].rij << " ";
                std::cout << this->dR_neigh[i][j][k].delx << " ";
                std::cout << this->dR_neigh[i][j][k].dely << " ";
                std::cout << this->dR_neigh[i][j][k].delz << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
} // show


// int main() {
//     double coords[15] = {4.5292997, 4.8908755, 5.1718035, 
//                         5.1422234, 5.6168573, 5.7252511,
//                         3.8329047, 4.2310279, 5.782918,
//                         3.947507, 5.491898, 4.5658593,
//                         5.0937183, 4.2990798, 4.4207844};
//     double box[9] = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
//     int type_map[5] = {0, 1, 1, 1, 1};
//     NeighborList nl(6, coords, box, 10, 2, 5, type_map);
//     nl.show();
//     int** numn = nl.get_num_neigh();
//     for (int i = 0; i < 5; i++) {
//         for (int j = 0; j < 2; j++) {
//             std::cout << numn[i][j] << " ";
//         }
//         std::cout << std::endl;
//     }
//     return 0;
// }
