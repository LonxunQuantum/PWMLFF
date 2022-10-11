module data_ewald
    real*8 :: Ecut = 60.0d0
    real*8 :: Ecut2 = 120.0d0
    integer :: n1 = 70
    integer :: n2 = 70
    integer :: n3 = 70
    real*8 :: vol
    real*8 AL_bohr(3,3)
    integer, parameter :: matom = 50000
    integer, parameter :: mtype = 120
    real*8, parameter :: Hartree2eV = 27.2114d0
    real*8, parameter :: Angstrom2Bohr = 1.8897259886d0
    integer n_born_type
    integer tmp_iatom
    real*8 tmp_zatom
    real*8 zatom_ewald(120)
    real*8 ewald
    real*8, allocatable, dimension(:) :: ewald_atom
    real*8, allocatable, dimension(:,:) :: fatom_ewald
    integer iflag_born_charge_ewald
    real*8  ewald_epsilon

end module data_ewald
