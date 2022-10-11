subroutine readf_xatom_new(iMD, ntype_mass, itype_mass, mass_type)

    use mod_data
    use mod_mpi
    use mod_control

    implicit none
    real*8, parameter :: A_AU_1 = 0.52917721067d0
    real*8, parameter :: Hartree_eV = 27.21138602d0 !(eV)

    real*8, allocatable, dimension(:, :) :: xatom_tmp
    integer, allocatable, dimension(:, :) :: imov_latt_vectmp
    integer, allocatable, dimension(:)   ::  iatom_tmp, iatom_tmp2
    real*8, allocatable, dimension(:) :: VX_1_tmp, VY_1_tmp, VZ_1_tmp
    real*8, allocatable, dimension(:) :: weight_mag_tmp
    real*8, allocatable, dimension(:, :) :: weight_mag_tmpxyz
    real*8, allocatable, dimension(:) ::langevin_factT_tmp, langevin_factG_tmp
    real*8, allocatable, dimension(:) :: constraint_mag_atom_tmp
    real*8, allocatable, dimension(:) :: constraint_mag_alpha_tmp
    real*8, allocatable, dimension(:, :) :: LDAU_lambda_tmp
    integer, allocatable, dimension(:)   :: ind_order_old2new_tmp, ind_order_new2old_tmp
    real*8, allocatable, dimension(:) :: weight_atom_tmp
    character*200 message

    integer ntype_mass, itype_mass(100)
    real*8 mass_type(100)

    integer ierr, i, ia, jjj
    integer iMD
    integer itype_tmp, imov_sum, ii, ncount

    logical :: scanit
    intrinsic :: sum

    open (10, file=f_xatom, status='old', action='read', iostat=ierr)

    if (ierr .ne. 0) then
        if (inode .eq. 1) write (message, *) "IN.ATOM", f_xatom, "not exist, stop"
        write (*, *) message
        stop
    end if

    rewind (10)
    read (10, *) natom
    if (natom .gt. matom_1) then
        if (inode .eq. 1) then
            write (message, *) "natom.gt.matom_1, increase matom_1 in data.f,stop ", f_xatom, natom, matom_1
            write (*, *) message
            stop
        end if
    end if

    allocate (xatom_tmp(3, natom))
    allocate (imov_latt_vectmp(3, natom))
    allocate (iatom_tmp(natom))
    allocate (iatom_tmp2(natom))
    allocate (VX_1_tmp(natom))
    allocate (VY_1_tmp(natom))
    allocate (VZ_1_tmp(natom))
    allocate (weight_mag_tmp(natom))
    allocate (weight_mag_tmpxyz(natom, 3))
    allocate (langevin_factT_tmp(natom))
    allocate (langevin_factG_tmp(natom))
    allocate (constraint_mag_atom_tmp(natom))
    allocate (constraint_mag_alpha_tmp(natom))
    allocate (ind_order_old2new_tmp(natom))
    allocate (ind_order_new2old_tmp(natom))
    allocate (LDAU_lambda_tmp(natom, 2))
    allocate (weight_atom_tmp(natom))


    call scan_key_words(10, "LATTICE", len("LATTICE"), scanit)
    if (scanit) then
        read (10, *) (AL(i, 1), i=1, 3)
        read (10, *) (AL(i, 2), i=1, 3)
        read (10, *) (AL(i, 3), i=1, 3)
    else
        write (message, *) "Must provide LATTICE in IN.ATOM file ", ADJUSTL(trim(f_xatom))
        write (*, *) message
        stop
    end if

    AL = AL/A_AU_1 !zhilin

    call scan_key_words(10, "STRESS_MASK", len("STRESS_MASK"), scanit)
    if (.not. scanit) then
        stress_mask = 1
    else
        read (10, *) (stress_mask(i, 1), i=1, 3)
        read (10, *) (stress_mask(i, 2), i=1, 3)
        read (10, *) (stress_mask(i, 3), i=1, 3)
    end if
    call scan_key_words(10, "STRESS_EXTERNAL", len("STRESS_EXTERNAL"), scanit)
    if (.not. scanit) then
        exist_stress_ext = .false.
    else
        exist_stress_ext = .true.
        read (10, *) (stress_ext(i, 1), i=1, 3)
        read (10, *) (stress_ext(i, 2), i=1, 3)
        read (10, *) (stress_ext(i, 3), i=1, 3)
        stress_ext = stress_ext/Hartree_ev*natom
    end if

    call scan_key_words(10, "POSITION", len("POSITION"), scanit)
    if (.not. scanit) then
        write (*, *) "keyword 'position' is needed at", f_xatom
        if (inode .eq. 1) then
            call mpi_abort(mpi_comm_world, ierr)
        end if
    else
        do i = 1, natom
            read (10, *) iatom_tmp(i), xatom_tmp(1, i), xatom_tmp(2, i), xatom_tmp(3, i), imov_latt_vectmp(1, i), imov_latt_vectmp(2, i), imov_latt_vectmp(3, i)
        end do
                    !!!!!  check imov avoiding imov==0 in POSITION-RELAX MD NEB TDDFT NAMD
        imov_sum = sum(imov_latt_vectmp)
                    !!!!!
    end if

    ! liuliping:
    ! MOVEMENT files contain "MD_VV_INFO: Basic Velocity Verlet Dynamics(NVE)"
    ! and xatom.config is the head of MOVEMENT
    ! This 'Velocity' conflicts with VELOCITY, so use 'Velocity (bohr/fs)' as keyword
    call scan_key_words(10, "VELOCITY (", len("VELOCITY ("), scanit)
    if (.not. scanit) then
        !write(*,*) "keyword 'velocity' is needed at",f_xatom
        !stop
        if ((iMD .eq. 11) .or. (iMD .eq. 22) .or. (iMD .eq. 33)) then
            write (*, *) "keyword 'velocity' is needed at", f_xatom
            if (inode .eq. 1) call mpi_abort(mpi_comm_world, ierr)
        end if
        exist_velocity = .false.
    end if
    if (scanit) then
        if (inode .eq. 1) then
            write (*, *) "velocity read in from ", f_xatom
        end if
        do i = 1, natom
            read (10, *) iatom_tmp2(i), VX_1_tmp(i), VY_1_tmp(i), VZ_1_tmp(i)
            if (iatom_tmp2(i) .ne. iatom_tmp(i)) then
                write (6, *) "order of iatom in position/velocity not the same", i
                if (inode .eq. 1) then
                    call mpi_abort(mpi_comm_world, ierr)
                end if
            end if
        end do
        exist_velocity = .true.
    end if

    call scan_key_words(10, "LANGEVIN_ATOMFACT_TG", len("LANGEVIN_ATOMFACT_TG"), scanit)
    if (.not. scanit) then

        do i = 1, natom
            langevin_factT_tmp(i) = 1.d0
            langevin_factG_tmp(i) = 1.d0
        end do
    else
        do i = 1, natom
            read (10, *) iatom_tmp2(i), langevin_factT_tmp(i), langevin_factG_tmp(i)
        end do
    end if

!********************************************************
!********************************************************
    call scan_key_words(10, "Weight_atom", len("Weight_atom"), scanit)
    if (.not. scanit) then
        do i = 1, natom
            weight_atom_tmp(i) = 1.d0
        end do
    else
        do i = 1, natom
            read (10, *) iatom_tmp2(i), weight_atom_tmp(i)
        end do
    end if
!********************************************************

    close (10)
    !ccccccccccccccccccccccccccccccccccccccccccccccccccc
    !ccc Now, re-arrange xatom, so the same atoms are consequentive together.
    !ccc This is useful to speed up the getwmask.f
    !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

    ii = 0
    if (inode .eq. 1) then
        open (9900, file='ORIGIN.INDEX')
        rewind (9900)
    end if
100 continue
    ncount = 0
    itype_tmp = -2
    do i = 1, natom
        if (itype_tmp .eq. -2 .and. iatom_tmp(i) .ne. -1) itype_tmp = iatom_tmp(i)

        if (iatom_tmp(i) .eq. itype_tmp) then
            ii = ii + 1
            ncount = ncount + 1
            iatom(ii) = iatom_tmp(i)
            iatom_tmp(i) = -1
            xatom(:, ii) = xatom_tmp(:, i)
            imov_at(:, ii) = imov_latt_vectmp(:, i)
            VX_1(ii) = VX_1_tmp(i)
            VY_1(ii) = VY_1_tmp(i)
            VZ_1(ii) = VZ_1_tmp(i)
            !     weight_mag(ii)=0.5d0+weight_mag_tmp(i)/2          ! weigh_mag: old meaning; input mag_tmp from - to +
            !     weight_magxyz(ii,1)=weight_mag_tmpxyz(i,1)
            !     weight_magxyz(ii,2)=weight_mag_tmpxyz(i,2)
            !     weight_magxyz(ii,3)=weight_mag_tmpxyz(i,3)
            langevin_factT(ii) = langevin_factT_tmp(i)
            langevin_factG(ii) = langevin_factG_tmp(i)
            !     constraint_mag_atom(ii)=constraint_mag_atom_tmp(i)
            !     constraint_mag_alpha(ii)=constraint_mag_alpha_tmp(i)
            !     LDAU_lambda(ii,:) = LDAU_lambda_tmp(i,:)
            ind_order_new2old_tmp(ii) = i
            ind_order_old2new_tmp(i) = ii
            weight_atom_1(ii) = weight_atom_tmp(i)

            iMDatom(ii) = 0.d0
            do jjj = 1, ntype_mass
                if (itype_mass(jjj) .eq. iatom(ii)) then
                    iMDatom(ii) = mass_type(jjj)
                end if
            end do

            if (iMDatom(ii) .eq. 0.d0) then
                iMDatom(ii) = iatom(ii)*2   ! to be fixed later
            end if

            if (inode .eq. 1) write (9900, '(i6,x,3i2,x,i6,x,i3)') i, imov_at(:, ii), ii, iatom(ii)
        end if
    end do
    if (ncount .gt. 0) goto 100
    if (inode .eq. 1) close (9900)
    if (ii .ne. natom) then
        write (message, *) "something wrong to rearrange xatom, stop"
        write (*, *) message
        stop
    end if

    if (inode .eq. 1) then
        open (unit=2200, file='final.config')
        rewind (2200)
        write (2200, *) natom
        write (2200, *) "Lattice vector (Angstrom), stress(eV/natom)"
        write (2200, "(3(E19.10,1x))") A_AU_1*AL(1, 1), A_AU_1*AL(2, 1), A_AU_1*AL(3, 1)
        write (2200, "(3(E19.10,1x))") A_AU_1*AL(1, 2), A_AU_1*AL(2, 2), A_AU_1*AL(3, 2)
        write (2200, "(3(E19.10,1x))") A_AU_1*AL(1, 3), A_AU_1*AL(2, 3), A_AU_1*AL(3, 3)
        write (2200, *) "Position, move_x, move_y, move_z"
        do ia = 1, natom
            write (2200, 1113) iatom(ia), xatom(1, ia), xatom(2, ia), xatom(3, ia), imov_at(1, ia), imov_at(2, ia), imov_at(3, ia)
        end do
        close (2200)
1113    format(i4, 1x, 3(f14.9, 1x), 4x, 3(i1, 2x))
    end if

    deallocate (xatom_tmp)
    deallocate (imov_latt_vectmp)
    deallocate (iatom_tmp)
    deallocate (iatom_tmp2)
    deallocate (VX_1_tmp)
    deallocate (VY_1_tmp)
    deallocate (VZ_1_tmp)
    deallocate (weight_mag_tmp)
    deallocate (weight_mag_tmpxyz)
    deallocate (langevin_factT_tmp)
    deallocate (langevin_factG_tmp)
    deallocate (constraint_mag_atom_tmp)
    deallocate (constraint_mag_alpha_tmp)
    deallocate (ind_order_old2new_tmp)
    deallocate (ind_order_new2old_tmp)
    deallocate (LDAU_lambda_tmp)
    deallocate (weight_atom_tmp)

    return
end subroutine readf_xatom_new

