template<typename DType>
void launch_calculate_nepfeat(
    const DType * coeff2,
    const DType * d12_radial,
    const int * NL_radial,
    const int * atom_map,
    const double rcut_radial,
    DType * feat_2b,
    DType * dfeat_c2,
    DType * dfeat_2b,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    const int n_max,
    const int n_base,
    const int num_types,
    const int device
);
