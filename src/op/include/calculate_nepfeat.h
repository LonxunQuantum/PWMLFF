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

template<typename DType>
void launch_calculate_nepfeat_grad(
    const DType * grad_output,
    const DType * dfeat_c2,
    const DType * dfeat_2b,
    const int * atom_map,
    DType * grad_coeff2,
    DType * grad_d12_radial,
    const int batch_size, 
    const int natoms, 
    const int neigh_num, 
    const int n_max_2b, 
    const int n_base_2b,
    const int n_types, 
    const int device
);
