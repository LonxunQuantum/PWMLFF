program main_MD

        use mod_mpi
        use mod_data
        use mod_control
        use mod_md
        use data_ewald, only : zatom_ewald, tmp_iatom, tmp_zatom, n_born_type 
        use calc_ftype1, only : load_model_type1,set_image_info_type1
        use calc_ftype2, only : load_model_type2,set_image_info_type2
        use calc_2bgauss_feature, only : load_model_type3,set_image_info_type3
        use calc_3bcos_feature, only : load_model_type4,set_image_info_type4
        use calc_MTP_feature, only : load_model_type5,set_image_info_type5
        use calc_SNAP_feature, only : load_model_type6,set_image_info_type6
        use calc_deepMD1_feature, only : load_model_type7,set_image_info_type7
        use calc_deepMD2_feature, only : load_model_type8,set_image_info_type8
        use calc_lin, only : set_paths_lin,load_model_lin,set_image_info_lin,nfeat_type_l,ifeat_type_l
        use calc_VV, only : set_paths_VV,load_model_VV,set_image_info_VV,nfeat_type_v,ifeat_type_v
        use calc_NN, only : set_paths_NN,load_model_NN,set_image_info_NN,nfeat_type_n,ifeat_type_n

        implicit none

        integer ierr
        CHARACTER(LEN=200) :: right
        integer iMD,MDstep
        real*8 dtMD,Temperature1,Temperature2
        logical right_logical
        logical is_reset
        integer nstep_temp_VVMD
        integer iscale_temp_VVMD
        integer ntype_mass
        integer itype_mass(100)
        real*8  mass_type(100)
        integer i,iat1,kk
        integer nfeat_type
        integer ifeat_type(100)

        call mpi_init(ierr)
        call mpi_comm_rank(MPI_COMM_WORLD,inode,ierr)
        call mpi_comm_size(MPI_COMM_WORLD,nnodes,ierr)
        inode=inode+1

        write(6,*) "TEST1 inode=",inode
        write(6,*) "TEST1.1 nodes=",nnodes

        ! liuliping, is_ewald
        open(1314, file='input/ewald.input', action='read', iostat=ierr)
        if (ierr .ne. 0) then
            iflag_born_charge_ewald = 0
        else 
            read(1314, *) iflag_born_charge_ewald
            read(1314, *) n_born_type
            do i = 1, n_born_type
                read(1314, *) tmp_iatom, tmp_zatom
                zatom_ewald(tmp_iatom) = tmp_zatom
            end do
            close(1314)
        end if
        ! liuliping, is_ewald end

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

        iat1=0
        do i=1,natom
        if(mod(i-1,nnodes).eq.inode-1) then
        iat1=iat1+1
        endif
        enddo
        natom_n=iat1     ! different prorcessor might have different natom_n



        call get_ALI(AL,ALI)

          if(right_logical) then
          call read_mdopt(dtMD,Temperature1,temperature2,imov_at,natom,AL)
          else
          call default_mdopt(dtMD,Temperature1,temperature2,imov_at,natom,AL)
          endif

!ccccccccccccccccccccccccccccccccccccccc
         if(iMD==4 .or. iMD==44) MCTRL_stress=1
         if(iMD==5 .or. iMD==55) MCTRL_stress=1
         if(iMD==7 .or. iMD==77) MCTRL_stress=1
         if(iMD==8 .or. iMD==88) MCTRL_stress=1
         if(iMD==8 .or. iMD==88) then
         MCTRL_is_MSST=.true.
         else
         MCTRL_is_MSST=.false.
         endif
!ccccccccccccccccccccccccccccccccccccccc
         if(iflag_model.eq.1) then
         call set_paths_lin('.')
         call load_model_lin()
         call set_image_info_lin(iatom,is_reset,natom)
         nfeat_type=nfeat_type_l
         ifeat_type=ifeat_type_l
         endif
   
         if(iflag_model.eq.2) then
         call set_paths_VV('.')
         call load_model_VV()
         call set_image_info_VV(iatom,is_reset,natom)
         nfeat_type=nfeat_type_v
         ifeat_type=ifeat_type_v
         endif

         if(iflag_model.eq.3) then
         call set_paths_NN('.')
         call load_model_NN()
         call set_image_info_NN(iatom,is_reset,natom)
         nfeat_type=nfeat_type_n
         ifeat_type=ifeat_type_n
         endif
        
        
         is_reset=.true.
         do kk = 1, nfeat_type
          if (ifeat_type(kk)  .eq. 1) then
            write(*,*) "load1"
            call load_model_type1()      ! load up the parameter etc
            call set_image_info_type1(iatom,is_reset,natom)
          endif
          if (ifeat_type(kk)  .eq. 2) then
            call load_model_type2()      ! load up the parameter etc
            call set_image_info_type2(iatom,is_reset,natom)
          endif
          if (ifeat_type(kk)  .eq. 3) then
            call load_model_type3()      ! load up the parameter etc
            call set_image_info_type3(iatom,is_reset,natom)
          endif
          if (ifeat_type(kk)  .eq. 4) then
            call load_model_type4()      ! load up the parameter etc
            call set_image_info_type4(iatom,is_reset,natom)
          endif
          if (ifeat_type(kk)  .eq. 5) then
            call load_model_type5()      ! load up the parameter etc
            call set_image_info_type5(iatom,is_reset,natom)
          endif
          if (ifeat_type(kk)  .eq. 6) then
            call load_model_type6()      ! load up the parameter etc
            call set_image_info_type6(iatom,is_reset,natom)
          endif
          if (ifeat_type(kk)  .eq. 7) then
            call load_model_type7()      ! load up the parameter etc
            call set_image_info_type7(iatom,is_reset,natom)
          endif
          if (ifeat_type(kk)  .eq. 8) then
            call load_model_type8()      ! load up the parameter etc
            call set_image_info_type8(iatom,is_reset,natom)
          endif
  
          enddo
         !TODO: whether we need to set which one we use
        !  call load_model_type1()      ! load up the parameter etc
        !  call set_image_info_type1(iatom,is_reset,natom)
        !  call load_model_type2()      ! load up the parameter etc
        !  call set_image_info_type2(iatom,is_reset,natom)
        !  call load_model_type3()      ! load up the parameter etc
        !  call set_image_info_type3(iatom,is_reset,natom)
        !  call load_model_type4()      ! load up the parameter etc
        !  call set_image_info_type4(iatom,is_reset,natom)
        !  call load_model_type5()      ! load up the parameter etc
        !  call set_image_info_type5(iatom,is_reset,natom)
        !  call load_model_type6()      ! load up the parameter etc
        !  call set_image_info_type6(iatom,is_reset,natom)
        !  call load_model_type7()      ! load up the parameter etc
        !  call set_image_info_type7(iatom,is_reset,natom)
        !  call load_model_type8()      ! load up the parameter etc
        !  call set_image_info_type8(iatom,is_reset,natom)

        !write(6,*) "TEST2 inode=", inode
         call molecular_dynamics_new(dtMD,iMD,MDstep,xatom,iMDatom,temperature1,temperature2,iscale_temp_VVMD,nstep_temp_VVMD,imov_at,natom,AL,ALI,iatom,f_xatom)

        !write(6,*) "TEST3 inode=", inode

         call mpi_finalize(ierr)

         end
        

       

