subroutine f2c_calc_energy_force(i_model_lvn, n_atom, type_atom, lat,&
    x_frac, e_atom_out, f_atom_out, e_tot_out, iflag_reneigh_inout) bind(c,name="f2c_calc_energy_force")
    
    use iso_c_binding
    use mod_md
    use mod_mpi
    use data_ewald
    
    !use mod_control, only : MCTRL_iMD, MCTRL_AL,MCTRL_output_nstep 
    use mod_control
    use mod_data  ! xatom, iatom, natom, AL, f_atom, e_atom, iflag_reneigh_inout, iflag_born_charge_ewald and etc
    use calc_ftype1, only: load_model_type1, set_image_info_type1
    use calc_ftype2, only: load_model_type2, set_image_info_type2
    use calc_2bgauss_feature, only: load_model_type3, set_image_info_type3
    use calc_3bcos_feature, only: load_model_type4, set_image_info_type4
    use calc_MTP_feature, only: load_model_type5, set_image_info_type5
    use calc_SNAP_feature, only: load_model_type6, set_image_info_type6
    use calc_deepMD1_feature, only: load_model_type7, set_image_info_type7
    use calc_deepMD2_feature, only: load_model_type8, set_image_info_type8
    
    use calc_lin, only: set_paths_lin, load_model_lin, set_image_info_lin, nfeat_type_l, ifeat_type_l
    use calc_VV, only: set_paths_VV, load_model_VV, set_image_info_VV, nfeat_type_v, ifeat_type_v
    
    use calc_NN, only: set_paths_NN, load_model_NN, set_image_info_NN, nfeat_type_n, ifeat_type_n
    
    use calc_deepMD, only :set_paths_deepMD,load_model_deepMD,set_image_info_deepMD
    use calc_deepMD_f,only: load_model_deepMD_f,set_image_info_deepMD_f
    
    implicit none

    integer, intent(in) :: i_model_lvn
    integer, intent(in) :: n_atom
    integer, dimension(n_atom), intent(in) :: type_atom

    real*8, dimension(3,3), intent(in) :: lat
    real*8, dimension(3,n_atom), intent(in) :: x_frac
    real*8, dimension(n_atom), intent(inout) :: e_atom_out
    real*8, dimension(3,n_atom), intent(inout) :: f_atom_out
    real*8, allocatable, dimension(:, :) :: f_atom_predict
    real*8 e_tot_predict
    real*8, intent(inout) :: e_tot_out

    character(len=32) argv

    integer, intent(inout) :: iflag_reneigh_inout
    integer argc
    integer i, j, k, kk, ierr, ifile
    integer iat1
    integer i_image
    integer nfeat_type
    integer ifeat_type(100)

    logical :: scanit, is_reset

    integer iMD,MDstep
    real(8) dtMD, Temperature1, Temperature2
    logical right_logical
    integer ntype_mass
    integer itype_mass(100)
    real*8  mass_type(100) 
    integer nstep_temp_VVMD 
    integer iscale_temp_VVMD
    !call mpi_init(ierr)
    ! liuliping, is_ewald
    call get_zatom(n_atom)
    ! liuliping, is_ewald end

    call mpi_comm_rank(MPI_COMM_WORLD,inode,ierr)
    call mpi_comm_size(MPI_COMM_WORLD,nnodes,ierr)
    inode = inode + 1

    !open(unit=inode+4396, file='test'//char(nnodes+inode+48))
    
    !do i = 1, n_atom
    !    write(inode+4396, *) x_frac(1:3,i)
    !enddo
    
    !close(inode+4396)
    

    allocate(f_atom_predict(3,n_atom))
    iflag_model = i_model_lvn
    iflag_reneighbor = iflag_reneigh_inout
    if (iflag_reneigh_inout .eq. 1) iflag_reneigh_inout = 0


    ! 1  linear fitting
    ! 2  VV
    ! 3  NN

    !i_model_lvn, n_atom, type_atom, lat,&
    !  x_frac, e_atom_out, f_atom_out, e_tot_out

    ! we do not read md.input
    !xatom.config
    !100, 1000, 1.0, 600, 600
    !F
    !1
    !1
    !1
    !29 58

    f_xatom = 'xatom.config'
    iMD = 100
    MDstep = 1000
    dtMD = 1.0
    Temperature1 = 300
    Temperature2 = 300
    right_logical = .False.
    MCTRL_output_nstep = 1
    ntype_mass = 1 
    itype_mass = 8  ! not use
    mass_type = 16  ! not use
    
    !       open(9,file="md.input")
    !       rewind(9)
    !       read(9,*) f_xatom
    !       read(9,*)iMD,MDstep,dtMD,Temperature1,Temperature2
    !       read(9,*) right_logical
    !       read(9,*) iflag_model  ! 1: lineear; 2: VV; 3: NN 
    !       read(9,*) MCTRL_output_nstep
    !       read(9,*) ntype_mass
    !       do i=1,ntype_mass
    !       read(9,*) itype_mass(i),mass_type(i)
    !       enddo
    !       close(9)
    !       nstep_temp_VVMD=100
    !       iscale_temp_VVMD=0
            
    !        call readf_xatom_new(iMD,ntype_mass,itype_mass,mass_type)
    ! use intent(in) paras to initial the xatom

    natom = n_atom ! n_atom in 
    AL = lat
    
    call get_ALI(AL,ALI)

    iatom(1:n_atom) = type_atom(1:n_atom)
    xatom(1:3,1:n_atom) = x_frac(1:3,1:n_atom)

    iat1=0

    !
    do i=1,natom
        if(mod(i-1,nnodes).eq.inode-1) then
            iat1=iat1+1
        endif
    enddo

    natom_n=iat1      
    
    ! different prorcessor might have different natom_n
    ! main_MD: initialization

    if (iflag_model .eq. 1) then
        
        call set_paths_lin('.')
        call load_model_lin()
        call set_image_info_lin(iatom, is_reset, natom)
        nfeat_type = nfeat_type_l
        ifeat_type = ifeat_type_l
    end if

    if (iflag_model .eq. 2) then
        call set_paths_VV('.')
        call load_model_VV()
        call set_image_info_VV(iatom, is_reset, natom)
        nfeat_type = nfeat_type_v
        ifeat_type = ifeat_type_v
    end if

    if (iflag_model .eq. 3) then
        call set_paths_NN('.')
        call load_model_NN()
        call set_image_info_NN(iatom, is_reset, natom)
        nfeat_type = nfeat_type_n
        ifeat_type = ifeat_type_n
    end if


    if(iflag_model.eq.5) then
        call set_paths_deepMD('.')
        call load_model_deepMD()
        call set_image_info_deepMD(iatom,is_reset,natom)
        call load_model_deepMD_f()
        is_reset=.true.      ! why
        call set_image_info_deepMD_f(iatom,is_reset,natom)   ! why need this?
    endif

    is_reset = .true.
    
    if ((iflag_model.eq.1) .or. (iflag_model.eq.2) .or. (iflag_model.eq.3)) then 

        do kk = 1, nfeat_type
            if (ifeat_type(kk) .eq. 1) then
                call load_model_type1()      ! load up the parameter etc
                call set_image_info_type1(iatom, is_reset, natom)
            end if
            if (ifeat_type(kk) .eq. 2) then
                call load_model_type2()      ! load up the parameter etc
                call set_image_info_type2(iatom, is_reset, natom)
            end if
            if (ifeat_type(kk) .eq. 3) then
                call load_model_type3()      ! load up the parameter etc
                call set_image_info_type3(iatom, is_reset, natom)
            end if
            if (ifeat_type(kk) .eq. 4) then
                call load_model_type4()      ! load up the parameter etc
                call set_image_info_type4(iatom, is_reset, natom)
            end if
            if (ifeat_type(kk) .eq. 5) then
                call load_model_type5()      ! load up the parameter etc
                call set_image_info_type5(iatom, is_reset, natom)
            end if
            if (ifeat_type(kk) .eq. 6) then
                call load_model_type6()      ! load up the parameter etc
                call set_image_info_type6(iatom, is_reset, natom)
            end if
            if (ifeat_type(kk) .eq. 7) then
                call load_model_type7()      ! load up the parameter etc
                call set_image_info_type7(iatom, is_reset, natom)
            end if
            if (ifeat_type(kk) .eq. 8) then
                call load_model_type8()      ! load up the parameter etc
                call set_image_info_type8(iatom, is_reset, natom)
            end if

        end do
    endif 

    f_atom_predict = 0.0
    e_tot_predict = 0.0 
    
    call ML_FF_EF(e_tot_predict, f_atom_predict, x_frac, AL, n_atom)
    
    e_tot_out = e_tot_predict
 
    f_atom_out(1:3,1:n_atom) = -f_atom_predict(1:3,1:n_atom)
    e_atom_out(1:n_atom) = e_atom(1:n_atom) ! mod_data::e_atom, calculated by ML_FF_EF

    ! the following subroutine writes images to MD/md/MOVEMENT, and same format as old mlff movement,but different from PWmat
    !call write_to_mlff_movement(i_image, E_tot_predict, f_atom_predict, x_atom_predict, AL_predict, n_atom)
    
    deallocate(f_atom_predict)
    
    !call mpi_finalize(ierr) 
end
