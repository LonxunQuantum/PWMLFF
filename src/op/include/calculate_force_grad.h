template <typename DType>
void launch_calculate_force_grad(
    const int * nblist,
    const DType * Ri_d,
    const DType * net_grad,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    DType * grad
);