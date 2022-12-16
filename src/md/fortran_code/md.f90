subroutine molecular_dynamics_kernel(Etot,fatom,e_stress)
        use mod_md
        use mod_mpi
        use mod_control, only : MCTRL_iMD, MCTRL_AL,MCTRL_output_nstep
        use mod_data
        use IFPORT
        !     use common_module_99,only : iatom_987,totNel_987
        !     use param_escan,only : inode_tot,n1L,n2L,n3L
        !     use data,only:is_SOM
        !     use mod_ldau,only:ido_ns,ldau_typ,is_LDAU
        implicit none
        integer precision_flag
        real(8) fatom_old(3,matom_1) 
        real(8) fatom(3,matom_1) 
        real(8) e_stress(3,3)
        real*8 Etot
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !mymd
        type(type_md) :: md
        integer istep
        character(len=200) message
        real*8 tt
        integer ierr
        integer iislda
       
        real*8 cpu_rss1, cpu_rss2, cpu_vm1, cpu_vm2
        
        !data for interpolation
        real*8,allocatable,dimension(:,:,:) :: r_his
        integer lp,llp,lllp 
        real*8 Aalpha,Abeta
        !data for plumed
        real*8 :: energyunits, lengthunits, timeunits
        real*8 :: h_plumed(3,3), f_plumed(3,matom_1),r_plumed(3,matom_1)
        real*8 :: stress_plumed(3,3)
        real*8 AL_tmp(3,3)

        real*8 tt1,tt2,tt_tmp
        real(8), allocatable, dimension(:) ::  const_vx,const_vy,const_vz
        integer(4),allocatable, dimension(:) :: const_vel_atom
        integer(4) :: const_vel_num, i,j
        ! INTEGER*4  access, status
        logical*2::alive

        character(100) :: cmd_name, flag, etot_name, ei_name, fi_name
        integer(2) :: s


        inquire(file='vel_constraint',exist=alive)
        !     status = access ("add_force",' ')    ! blank mode
        !   if (status .eq. 0 ) then
        if (alive) then
            open(10,file="vel_constraint")
            rewind(10)
            read(10,*) const_vel_num
            allocate(const_vel_atom(const_vel_num))
            ! allocate(direction(add_force_num))
            allocate(const_vx(const_vel_num))
            allocate(const_vy(const_vel_num))
            allocate(const_vz(const_vel_num))
            do i=1,const_vel_num
                read(10,*) const_vel_atom(i), const_vx(i), const_vy(i), const_vz(i)
            enddo
            close(10)
        else
            const_vel_num=0
        endif

        !
        allocate(r_his(3,matom_1,0:2))
        ! 
        
        call init_type_md(md)
        
        ! Here MD starts. Do a write_final for dp
        
        !if (iflag_model.eq.4) then
        !   call write_finalconfig(md)  
        !endif
         
        MCTRL_ido_stop=0
        MCTRL_ido_ns=1

        !  call Etotcalc_mdwrap(md,MCTRL_scf0,fatom_old,Etot,E_st,err_st,convergE,fatom,e_stress,precision_flag,0)
        !     fatom(:,1:natom/2)=0.01
        !     fatom(:,natom/2:natom)=-0.01
        ! e_stress=0.d0
        ! Etot=2.d0
        AL_tmp=MCTRL_AL*A_AU_1   ! convert to A

        e_stress = 0.d0 
        
        if ((iflag_model.eq.1) .or. (iflag_model.eq.2) .or. (iflag_model.eq.3).or. (iflag_model.eq.5)) then 
            call ML_FF_EF(Etot,fatom,MCTRL_xatom,AL_tmp,MCTRL_natom,e_stress)
        end if 

        !preprocessing for dp model 
        if (iflag_model.eq.4) then

            ! read atom.config as the initial image
            write(*,*) "calling pytorch..."
            s = runqq("predict.py","--dp=True -n DP_cfg_dp --outputname=atom.config > out 2> err")
            write(*,*) "pytorch routine ends"
            
            ! load from file 
            etot_name = "etot.tmp" 
            ei_name = "ei.tmp"
            fi_name = "force.tmp"


            open(9, file = etot_name, status = "old")
            read(9,*) Etot
            close(9)
            
            open(10, file = fi_name, status = "old")
            do i=1,MCTRL_natom
                read(10,*) fatom(1:3,i)
            end do
            close(10)

            open(11,file = ei_name, status = "old")
            do i=1,MCTRL_natom
                read(11,*) e_atom(i)
            end do 
            close(11)

        endif   

        write(*,*) "MLFF tot energy in eV: ", Etot

        Etot=Etot/Hartree_eV  ! convert back to Hartree
        
        !e_stress=0.d0
        fatom(:,1:natom)=fatom(:,1:natom)*A_AU_1/Hartree_eV  ! convert to Hatree/Bohr
                
        call exchange_data_scf2md(md,fatom,Etot,e_stress) 
        
        !write(*,*) "MLFF tot energy in Hartree: ", Etot 
        
        !return 

        if(inode .eq. 1) then
            print*, '***********************************************'
            print*, '*                                             *' 
            print*, '*         MD/MD100 starts Now!                *' 
            print*, '*                                             *' 
            print*, '***********************************************'
        endif

        call md_init(md)
        !
        !TODO: initialize plumed
        !
        !--------------------------------
        ! MD LOOP
        !--------------------------------
        
        do istep=1,MCTRL_MDstep
            if(inode.eq.1) then
                write(*,*) "MD step=",istep
            endif

            tt1=mpi_wtime()
            !
            call update_time(md)

            tt=mpi_wtime()

            call update_T(md,istep)

            call update_r(md,istep)
            
            ! write final config for dp
            if (iflag_model.eq.4) then
               call write_finalconfig(md)  
            endif 
            
            ! new force and Etot
            !--------------------------------
            call exchange_data_md2scf(md)
            
            call MPI_Bcast(MCTRL_xatom,MCTRL_natom*3,MPI_REAL8,0,MPI_COMM_WORLD,ierr)

            AL_tmp=MCTRL_AL*A_AU_1   ! convert to A
            
            ! calculate a new tensor in ML_FF_EF, and pass it into
            ! exchange_data_scf2md
            call ML_FF_EF(Etot,fatom,MCTRL_xatom,AL_tmp,MCTRL_natom,e_stress)
            
            !if md then a new routine
            !
            Etot=Etot/Hartree_eV  ! convert back to Hartree
            fatom(:,1:natom)=fatom(:,1:natom)*A_AU_1/Hartree_eV  ! convert to Hatree/Bohr
            
            
            call exchange_data_scf2md(md,fatom,Etot,e_stress)

            call update_v(md)

            call get_energy_kinetic(md)

            call get_temperature(md)

            ! the scaling must follow get_energy_kinetic & get_temperature
            call energy_scaling(md,istep)
            
            do j=1,const_vel_num

                md.v(1,const_vel_atom(j))= const_vx(j)   !give a force on x axis
                md.v(2,const_vel_atom(j))= const_vy(j)
                md.v(3,const_vel_atom(j))= const_vz(j)
                    
            enddo
            !
            !call get_diffusion_coeff(md)
            !call get_cell_info(md)
            call get_average_temperature(md)
            call get_average_pressure(md)
            call post_check(md)

            !--------------------------------
            ! output 
            !--------------------------------
            call write_MDSTEPS(md)
            
            if(mod(istep-1,MCTRL_output_nstep).eq.0) then
                call write_MOVEMENT(md,istep,MCTRL_MDstep)
                call write_finalconfig(md)    
            endif
            !            call write_finalconfig(md)    ! cost time !

            !if(.true.) then
            !    call write_diffusion_coeff(md)
            !    call write_average_temperature(md)
            !endif
            !
            tt=mpi_wtime()-tt
            !write(message,*) "md end istep, time = ",istep,md.curtime,"fs"
            ! call stdout_message(message,unit1=6,unit2=22)
            !write(*,*) message
            !write(message,*) "step MD used time: ",tt, "s"
            !   call stdout_message(message)
            !write(*,*) message
            !
!            call get_cpu_memory(cpu_vm2, cpu_rss2)
!           write(*,991) cpu_rss1, cpu_rss2, cpu_rss2-cpu_rss1
!           991   format("**MD CPU memleak: ", 3(f15.3, 1x))
            tt2=mpi_wtime()
            
            if(inode.eq.1) then
                write(6,*) "time MDstep", tt2-tt1
            endif
        enddo
        !    
        deallocate(r_his)
        !    
    contains
       ! subroutine interpolation_keeping()
       !         call get_push_pos(istep,lp,llp,lllp)
       !         call push_rho(rho_in_nL_5,istep,rho_his)
       !         call push_rho(rho_atom,istep,rho_atom_his)
       !         call push_r(md.r,istep,r_his)
       !         call push_ug(istep) ! use global ugIOBP
       !         if(is_SOM==1) then
       !             call push_rho_matrix(rho_nL_matrix_5,istep,rho_matrix_his)
       !         endif
       !         if(is_LDAU) then
       !             call push_ns(ldau_typ.ns,istep,ns_his)
       !         endif
       ! end subroutine interpolation_keeping
       ! subroutine interpolation()
        !        call getrhoatom(MCTRL_AL,md.r,MCTRL_ntype,iatom_987,totNel_987,1,MCTRL_islda,rho_atom)
       !         call get_interpolation_coeff(istep,Aalpha,Abeta,md.r,r_his,lp,llp,lllp)
                !open(1009,file='alphabeta',access='append')
                !write(1009,*) istep,Aalpha,Abeta
                !close(1009)
        !        call interpolation_ug(istep,Aalpha,Abeta)
        !        call interpolation_rho(istep,Aalpha,Abeta,rho_his,rho_atom,rho_atom_his,lp,llp,lllp)
        !        if(is_SOM==1) then
        !            call interpolation_rho_matrix(istep,Aalpha,Abeta,rho_matrix_his,lp,llp,lllp)
        !        endif
        !        if(is_LDAU) then
        !            call interpolation_ns(istep,Aalpha,Abeta,ns_his,lp,llp,lllp)
        !        endif
       ! end subroutine interpolation
end subroutine molecular_dynamics_kernel

