module mod_data
    use mod_parameter
    real*8 Vx_1(matom_1),Vy_1(matom_1),Vz_1(matom_1)
    real*8 xatom(3,matom_1)
    real*8 e_atom(matom_1) ! liuliping
    integer imov_at(3,matom_1)
    integer iatom(matom_1)
    real*8 iMDatom(matom_1)
    real*8 weight_atom_1(matom_1)
    real*8 langevin_factT(matom_1),langevin_factG(matom_1)
    integer natom
    integer ntype
    real*8 AL(3,3),ALI(3,3)
    real*8 stress_mask(3,3)
    real*8 stress_ext(3,3)
    character*20 f_xatom
    integer iflag_model
    integer :: iflag_reneighbor = 1
end module mod_data 

