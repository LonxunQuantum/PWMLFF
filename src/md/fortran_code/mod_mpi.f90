module mod_mpi
    include 'mpif.h'
    integer inode
    integer nnodes
    integer status(MPI_STATUS_SIZE)
    integer natom_n   ! divided number of atom
end module mod_mpi

