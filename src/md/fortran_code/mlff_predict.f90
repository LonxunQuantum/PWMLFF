program mlff_predict
    use mod_md
    use mod_mpi
    !use mod_control, only : MCTRL_iMD, MCTRL_AL,MCTRL_output_nstep
    use mod_control
    use mod_data
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
    !!!
    implicit none
    integer argc
    character(len=32) argv
    integer i, j, k, kk, ierr, ifile
    integer i_image
    integer num_atoms
    integer nfeat_type
    integer ifeat_type(100)
    character(len=200) file_predict
    integer, allocatable, dimension(:) :: itype_atom_predict
    real(8) E_tot_predict
    real(8), allocatable, dimension(:, :) :: x_atom_predict
    real(8), allocatable, dimension(:, :) :: f_atom_predict
    real(8) AL_predict(3, 3)
    logical :: scanit, is_reset
    integer iMD,MDstep
    real(8) dtMD, Temperature1, Temperature2
    logical right_logical
    integer ntype_mass
    integer itype_mass(100)
    real*8  mass_type(100) 
    integer nstep_temp_VVMD 
    integer iscale_temp_VVMD
    ! not implement, but use it here
    call mpi_init(ierr)
    call mpi_comm_rank(MPI_COMM_WORLD,inode,ierr)
    call mpi_comm_size(MPI_COMM_WORLD,nnodes,ierr)

    iflag_model = 1 ! linear fitting
    call get_command_argument(1, argv)
    if (len_trim(argv) > 0) then
        read (argv, '(i3)') iflag_model
    end if

        open(9,file="md.input")
        rewind(9)
        read(9,*) f_xatom
        read(9,*)iMD,MDstep,dtMD,Temperature1,Temperature2
        read(9,*) right_logical
        read(9,*) iflag_model  ! 1: lineear; 2: VV; 3: NN 
        read(9,*) MCTRL_output_nstep
        read(9,*) ntype_mass
        do i=1,ntype_mass
        read(9,*) itype_mass(i),mass_type(i)
        enddo
        close(9)
        nstep_temp_VVMD=100
        iscale_temp_VVMD=0
        
        call readf_xatom_new(iMD,ntype_mass,itype_mass,mass_type)

    ! main_MD: initialization
    if (iflag_model .eq. 1) then
        call set_paths_lin('.')
        call load_model_lin()
        call set_image_info_lin(iatom, is_reset, natom)
        nfeat_type = nfeat_type_l
        ifeat_type = ifeat_type_l
        write(*,*) "iflag_model: ", iflag_model
    end if

    if (iflag_model .eq. 2) then
        call set_paths_VV('.')
        call load_model_VV()
        call set_image_info_VV(iatom, is_reset, natom)
        nfeat_type = nfeat_type_v
        ifeat_type = ifeat_type_v
        write(*,*) "iflag_model: ", iflag_model
    end if

    if (iflag_model .eq. 3) then
        call set_paths_NN('.')
        call load_model_NN()
        call set_image_info_NN(iatom, is_reset, natom)
        nfeat_type = nfeat_type_n
        ifeat_type = ifeat_type_n
        write(*,*) "iflag_model: ", iflag_model
    end if

    is_reset = .true.
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

    ifile = 1314
    file_predict = "MD/MOVEMENT"
    i_image = 0
    open (ifile, file=file_predict, iostat=ierr)
    if (ierr .ne. 0) then
        write(*,*) ierr
        goto 8889
    end if
    ! read DFT MOVEMENT
    call scan_key_words(ifile, "Iteration", len("Iteration"), scanit)
    if (scanit) then
        backspace (ifile)
        read (ifile, *) num_atoms
    else
        goto 8889
    end if
    allocate(itype_atom_predict(num_atoms))
    allocate(x_atom_predict(3,num_atoms))
    allocate(f_atom_predict(3,num_atoms))
    open(4396, file="flag")
    write(4396,*) 4
    close(4396)
        call scan_key_words(ifile, "LATTICE", len("LATTICE"), scanit)
        if (scanit) then
            read (ifile, *) (AL_predict(i, 1), i=1, 3)
            read (ifile, *) (AL_predict(i, 2), i=1, 3)
            read (ifile, *) (AL_predict(i, 3), i=1, 3)
        else
            write(*,*) "bad lattice"
            write (*, *) "incomplete image, end"
            goto 8888
        end if

        call scan_key_words(ifile, "POSITION", len("POSITION"), scanit)
        if (scanit) then
            do i = 1, natom
                read (ifile, *) itype_atom_predict(i), x_atom_predict(1, i), x_atom_predict(2, i), x_atom_predict(3, i)
            end do
        else
            write(*,*) "bad position"
            write (*, *) "incomplete image, end"
            goto 8888
        end if

        call ML_FF_EF(E_tot_predict, f_atom_predict, x_atom_predict, AL_predict, num_atoms)
        write(*, *) "wtf:"
        write(*, *) i_image, E_tot_predict
        goto 8889 ! end
        ! the following subroutine writes images to MD/md/MOVEMENT, and same format as old mlff movement,but different from PWmat
        !call write_to_mlff_movement(i_image, E_tot_predict, f_atom_predict, x_atom_predict, AL_predict, num_atoms)

8888 continue ! bad ending
8889 continue ! good ending
    write (*, *) " images predicted"
    call mpi_finalize(ierr)
end
