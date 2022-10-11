module mod_neighbor
    integer, allocatable, dimension(:,:), save :: num_neigh_saved ! (ntype,natom)
    integer, allocatable, dimension(:,:,:), save :: list_neigh_saved ! (m_neigh, ntype,natom)
    real(8), allocatable, dimension(:,:,:,:), save :: dR_neigh_saved ! (3, m_neigh, ntype, natom)
    integer, allocatable, dimension(:,:,:), save :: iat_neigh_saved  !(m_neigh, ntype, natom)
    integer ntype_saved
    integer iat_type_saved(100)
    
    integer, allocatable, dimension(:,:,:), save :: map2neigh_M_saved !(m_neigh, ntype, natom)
    integer, allocatable, dimension(:,:,:), save :: list_neigh_M_saved !(m_neigh, ntype, natom)
    integer, allocatable, dimension(:,:), save :: num_neigh_M_saved ! (ntype,natom)
    integer, allocatable, dimension(:,:,:), save :: iat_neigh_M_saved  !(m_neigh, ntype, natom)
    ! llp, new
    integer, allocatable, dimension(:,:,:,:), save :: period_saved  ! (3,m_neigh,ntype,natom)
    real(8), allocatable, dimension(:,:,:,:), save :: last_dR_neigh_saved ! (3, m_neigh, ntype, natom)
    
end module mod_neighbor
