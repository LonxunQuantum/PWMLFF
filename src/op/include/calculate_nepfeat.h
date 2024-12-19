void launch_calculate_nepfeat(
    const double * coeff2,
    const double * d12_radial,
    const int * NL_radial,
    const int * atom_map,
    const double rcut_radial,
    double * feat_2b,
    double * dfeat_c2,
    double * dfeat_2b,
    double * dfeat_2b_noc,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    const int n_max,
    const int n_base,
    const int num_types,
    const int device
);

void launch_calculate_nepfeat_grad(
    const double * grad_output,
    const double * dfeat_c2,
    const double * dfeat_2b,
    const int * atom_map,
    double * grad_coeff2,
    double * grad_d12_radial,
    const int batch_size, 
    const int natoms, 
    const int neigh_num, 
    const int n_max_2b, 
    const int n_base_2b,
    const int n_types, 
    const int device
);

void launch_calculate_nepfeat_secondgradout(
    const double * grad_second,
    const double * dfeat_b,
    double * gradsecond_out,
    const int batch_size, 
    const int atom_nums, 
    const int maxneighs, 
    const int n_max, 
    const int device
);

void launch_calculate_nepfeat_secondgradout_c2(
    const double * grad_second,
    const double * de_feat,
    const double * dfeat_2b_noc,
    const int * atom_map,
    const int * NL_radial,
    double * gradsecond_c2,
    const int batch_size, 
    const int atom_nums, 
    const int maxneighs, 
    const int n_max_2b, 
    const int n_base_2b, 
    const int atom_types, 
    const int device
);
