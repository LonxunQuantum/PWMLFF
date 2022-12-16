module mod_md
    use mod_parameter
    use mod_mpi
!    use data_variable_1, only:   &
!    matom_1,mscf_1,vx_1,vy_1,vz_1,Hartree_eV,langevin_factT,langevin_factG, &
!    weight_atom_1
!    use data,only: mg_nx, mr_nL, mr_n, mst
!    use param_escan,only: nblock_band_mx
!    use mod_data
    use mod_control
!    use mod_ldau,only:ido_ns,ldau_typ,is_LDAU
    implicit none
    !real*8,parameter  :: HFs2vB2M=0.930895d0
    !real*8,parameter  :: Hdt=0.023538d0/273.15d0/27.211396d0
    real*8, parameter :: A_AU_1=0.52917721067d0
    real*8,parameter :: atomicmass_kg=1.660539040d-27 ! (kg)
    real*8,parameter :: bohr_m=0.52917721067d-10 !(m)
    real*8,parameter :: Hartree_J=4.359744650d-18 !(J)
    real*8,parameter :: Hartree_eV=27.21138602d0 !(eV)
    real*8,parameter :: HFs2vB2M=Hartree_J/(atomicmass_kg*(bohr_m*1.d15)**2)
    real*8,parameter :: Boltzmann_eVK=8.6173303d-5 !(eV K^-1)
    real*8,parameter :: Hdt=Boltzmann_eVK/Hartree_ev !(Hartree K^-1)

    integer :: iflag_MD_sp
    integer :: iflag_MD_sp_solvent
    real*8 x10_MD_sp,x20_MD_sp,x30_MD_sp,Rcut_MD_sp,dR_MD_sp,V_MD_sp
    real*8 frac_MD_sp,P_sph,P_MD_sp,R_rate_sp
    real*8 dRcut_MD_sp
    !velocity unit: bohr/fs
    !Hdt: boltzmann constant in unit Eh/K
    !HFs2vB2M: mv^2--u*(bohr/fs)^2--->kg*(m/s)^2/Eh [Hartree]
    ! the above constants come from:
    ! https://physics.nist.gov/cuu/Constants/index.html
    !
    integer :: Iseed=12345
    type :: type_md
        !const
        real*8 T1 ! temperature1
        real*8 T2 ! temperature2
        real*8 dtMD ! time step
        integer method
        logical restart
        integer num_degree
        integer num_degree_cell
        real*8 tau  ! time interval to calc average temperature
        !out
        !Berendsen method,whem method=6
        real*8 Berendsen_tau  ! time parameter for kinetic energy scaling 
        real*8 Berendsen_tauP  ! time parameter for cell scaking
        integer Berendsen_cell_steps ! the time steps per which to update cell
        real*8 diffusion_coeff
        real*8 totT
        real*8 totP
        real*8 averageT ! average temperature
        real*8 averageP ! average pressure
        real*8 dr(3,matom_1) ! xatom-xatom_old
        real*8 drtot(3,matom_1) ! used for diffusion_coeff
        real*8 stress(3,3) ! stress
        integer kinstress_flag  ! flag for whether to include kinetic stress, 0: no, 1: yes
        real*8 curT  ! current temperature 
        real*8 dL
        real*8 Fcheck
        real*8 Etot
        real*8 Ek
        real*8 f_scf(3,matom_1) ! origin force get from scf run
        real*8 f_scf_old(3,matom_1) ! origin force get from scf run
        !in
        real*8 curtime ! current time
        real*8 totalE_ini ! initialize Ep+Ek
        real*8 f_old(3,matom_1)
        real*8 Etot_old
        real*8 desiredT
        real*8 desiredT_atom(matom_1)
        real*8 cell_tau ! used to calc Wg
        real*8 ion_tau ! used to calc Q
        !inout
        real*8 v(3,matom_1) ! velocity
        real*8 r(3,matom_1) ! xatom
        real*8 f(3,matom_1) ! force
        real*8 m(matom_1) ! mass
        real*8 atomic_energy(matom_1)
        !for lvmd
            !const
        real*8 gamma(matom_1)
        real*8 gammaL
            !in
        real*8 f_s(3,matom_1) ! output of langevin_forces_new
        real*8 fL_s(3,3)
            !for npt
        real*8 pg(3,3)
        real*8 Wg
        real*8 h(3,3)
        real*8 h_old(3,3)
        real*8 hinv(3,3)
        real*8 Pext
        real*8 Pext_xyz(3)
        real*8 curpress
        ! npt(MSST with Q_is_infinity)
        logical :: is_MSST
        integer :: MSST_dir
        real*8 :: MSST_e0
        real*8 :: MSST_v0
        real*8 :: MSST_p0
        real*8 :: MSST_vs
        real*8 :: MSST_e
        real*8 :: MSST_p
        real*8 :: MSST_u
        real*8 :: MSST_M !total mass of atoms
        !
        !for nhmd
            !const
        real*8 Q ! Hartree * fs^2
        real*8 Q_cell ! Hartree * fs^2
            !out
        real*8 PS ! velocity_s/Q 
        real*8 PS_cell ! velocity_s/Q 
        !
        !constraints
        real*8 imov_at(3,matom_1)
        real*8 h0(3,3)
        integer isoscale_V
        real*8 stress_mask(3,3)
    end type type_md


contains


    subroutine init_type_md(mymd)

        use mod_data, only: langevin_factT,langevin_factG,stress_mask

                implicit none
                type(type_md) :: mymd
                integer :: ierr,clock
                character*200 message
                integer :: iat
                real*8 dx1,dx2,dx3,dx,dy,dz,d


        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if(iflag_MD_sp.eq.1) then   ! special case, for ML_FF development
            do iat=1,MCTRL_natom
            dx1=MCTRL_xatom(1,iat)-x10_MD_sp
            if(abs(dx1+1).lt.abs(dx1)) dx1=dx1+1
            if(abs(dx1-1).lt.abs(dx1)) dx1=dx1-1
            dx2=MCTRL_xatom(2,iat)-x20_MD_sp
            if(abs(dx2+1).lt.abs(dx2)) dx2=dx2+1
            if(abs(dx2-1).lt.abs(dx2)) dx2=dx2-1
            dx3=MCTRL_xatom(3,iat)-x30_MD_sp
            if(abs(dx3+1).lt.abs(dx3)) dx3=dx3+1
            if(abs(dx3-1).lt.abs(dx3)) dx3=dx3-1
            dx=MCTRL_AL(1,1)*dx1+MCTRL_AL(1,2)*dx2+MCTRL_AL(1,3)*dx3
            dy=MCTRL_AL(2,1)*dx1+MCTRL_AL(2,2)*dx2+MCTRL_AL(2,3)*dx3
            dz=MCTRL_AL(3,1)*dx1+MCTRL_AL(3,2)*dx2+MCTRL_AL(3,3)*dx3
            d=dsqrt(dx**2+dy**2+dz**2)
            if(d.lt.Rcut_MD_sp) then ! special
            MCTRL_imov_at(:,iat)=0
            endif
            enddo
        endif
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            
            mymd.diffusion_coeff=0.d0
            mymd.dr=0.d0
            mymd.drtot=0.d0
            mymd.v=0.d0
            mymd.f=0.d0
            mymd.r=MCTRL_xatom
            mymd.m=MCTRL_iMDatom/HFs2vB2M
            mymd.stress=0.d0
            mymd.Etot=0.d0
            mymd.Ek=0.d0
            mymd.T1=MCTRL_temperature1
            mymd.T2=MCTRL_temperature2
            mymd.curT=MCTRL_temperature1
            mymd.dL=0.d0
            mymd.Fcheck=0.d0
            mymd.f_old=0.d0
            mymd.f_scf_old=0.d0
            mymd.f_scf=0.d0
            mymd.Etot_old=0.d0
            mymd.totT=0.d0
            mymd.imov_at=MCTRL_imov_at
            mymd.stress_mask=stress_mask
            mymd.kinstress_flag=MCTRL_MD_kinstress_flag
            mymd.isoscale_V=MCTRL_MD_NPT_ISOSCALEV
            mymd.num_degree=sum(MCTRL_imov_at(1:3,1:MCTRL_natom))
            !
            if(MCTRL_is_MSST) then
                mymd.stress_mask=0.d0
                mymd.stress_mask(mymd.MSST_dir+1,mymd.MSST_dir+1)=1.d0
                mymd.is_MSST=MCTRL_is_MSST
                mymd.MSST_M=sum(mymd.m(1:MCTRL_natom))
            else
                mymd.is_MSST=.false.
            endif
            mymd.num_degree_cell=sum(mymd.stress_mask)
            if(MCTRL_iMD.eq.1.or.MCTRL_iMD.eq.11) then
                mymd.method=1
            else if(MCTRL_iMD.eq.2.or.MCTRL_iMD.eq.22) then 
                mymd.method=2
            else if(MCTRL_iMD.eq.3.or.MCTRL_iMD.eq.33) then 
                mymd.method=3
            else if(MCTRL_iMD.eq.4.or.MCTRL_iMD.eq.44) then 
                mymd.method=4
            else if(MCTRL_iMD.eq.5.or.MCTRL_iMD.eq.55) then 
                mymd.method=5
            else if(MCTRL_iMD.eq.6.or.MCTRL_iMD.eq.66) then 
                mymd.method=6
            else if(MCTRL_iMD.eq.7.or.MCTRL_iMD.eq.77) then 
                mymd.method=7
            else if(MCTRL_iMD.eq.8.or.MCTRL_iMD.eq.88) then 
                mymd.method=5
                mymd.is_MSST=.true.
            else if(MCTRL_iMD.eq.100.or.MCTRL_iMD.eq.101) then
                mymd.method=MCTRL_iMD
            else
                if(inode.eq.1) then
                    write(*,*) "Wrong MD method"
                end if
            endif
            mymd.curtime=0.d0
            mymd.dtMD=MCTRL_dtMD
            mymd.desiredT=mymd.curT
            mymd.desiredT_atom(:)=mymd.desiredT*langevin_factT(:)
            mymd.averageT=0.d0
            mymd.totalE_ini=0.d0
            !lvmd
            mymd.gamma=MCTRL_GAMMA_LVMD
            mymd.gamma(:)=mymd.gamma(:)*langevin_factG(:)
            mymd.f_s=0.d0
            !lv+npt
            mymd.gammaL=MCTRL_MD_NPT_GAMMA
            mymd.fL_s=0.d0
            mymd.Wg=MCTRL_MD_NPT_LMASS
            !mymd.Wg=100.d0*(mymd.num_degree+9)*Hdt*mymd.desiredT/(9.d0*1.d0/(100*mymd.dtMD)**2)
            mymd.pg=0.d0
            mymd.Pext=MCTRL_MD_NPT_PEXT
            mymd.Pext_xyz(:)=MCTRL_MD_NPT_PEXT_XYZ(:)
            !nh+npt
            mymd.cell_tau=MCTRL_MD_CELL_TAU
            mymd.ion_tau=MCTRL_MD_ION_TAU
            mymd.Berendsen_tau=MCTRL_MD_BERENDSEN_TAU
            mymd.Berendsen_tauP=MCTRL_MD_BERENDSEN_TAUP
            mymd.Berendsen_cell_steps=MCTRL_MD_BERENDSEN_CELL_STEPS
            mymd.MSST_vs=MCTRL_MSST_VS
            !
            mymd.h=MCTRL_AL
            mymd.hinv=MCTRL_ALI
            mymd.h0=MCTRL_AL
            !nhmd
            mymd.Q=MCTRL_Q_NHMD
            mymd.Q_cell=MCTRL_MD_NPT_Q_CELL
            mymd.PS=0.d0
            mymd.PS_cell=0.d0
            !reandom seed
            Iseed=MCTRL_SEED_MD
            if(Iseed .lt. 0.d0) then
                call system_clock(count=clock)
                Iseed=mod(clock,100000)
            endif
            call mpi_bcast(Iseed,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
            !
            mymd.tau=MCTRL_MD_AVET_TIMEINTERVAL
            !
            if(MCTRL_iMD.eq.11.or.MCTRL_iMD.eq.22.or.MCTRL_iMD.eq.33 .or. &
               MCTRL_iMD.eq.44 .or. MCTRL_iMD.eq.55 .or. MCTRL_iMD.eq.66 .or. &
               MCTRL_iMD.eq.77 .or. MCTRL_iMD.eq.88) then
                mymd.restart=.true.
            else
                mymd.restart=.false.
            endif
            if(mymd.restart) then
                call read_information(mymd) 
            endif
    end subroutine init_type_md
    subroutine init_Q(myT1,myT2,mydt,imov_at,natom)
            implicit none
            real*8 myT1,myT2,mydt
            integer imov_at(3,matom_1)
            integer natom
            integer mydegree
            mydegree=sum(imov_at(1:3,1:natom))
                if(MCTRL_Q_NHMD<= 1.d-15) then
                    MCTRL_Q_NHMD=2.d0*dble(mydegree)*max(myT1,myT2)*Hdt* &
                            (MCTRL_MD_ION_TAU/2.d0/3.1415926d0)**2
                endif
                if(MCTRL_is_MSST) then
                    MCTRL_Q_NHMD=1.d+30
                endif

    endsubroutine init_Q
    subroutine init_Q_cell(myT1,myT2,mydt,mydegree)
            implicit none
            real*8 myT1,myT2,mydt
            integer mydegree
                if(MCTRL_MD_NPT_Q_CELL<= 1.d-15) then
                    MCTRL_MD_NPT_Q_CELL=dble(mydegree)*max(myT1,myT2)*Hdt* &
                            (MCTRL_MD_CELL_TAU)**2
                endif
                if(MCTRL_is_MSST) then
                    MCTRL_MD_NPT_Q_CELL=1.d+30
                endif
    endsubroutine init_Q_cell
    subroutine init_lmass(myT1,myT2,mydt,imov_at,natom)
            implicit none
            real*8 myT1,myT2,mydt
            integer imov_at(3,matom_1)
            integer mydegree
            integer natom
            mydegree=sum(imov_at(1:3,1:natom))
                if(MCTRL_MD_NPT_LMASS<= 1.d-15) then
                    MCTRL_MD_NPT_LMASS=dble(mydegree+3)*max(myT1,myT2)*Hdt* &
                            (MCTRL_MD_CELL_TAU)**2/3.d0
                endif
    endsubroutine init_lmass


    subroutine molecular_dynamics_new(dtMD,iMD,MDstep,xatom,iMDatom,temperature1,temperature2,iscale_temp_VVMD,nstep_temp_VVMD,imov_at,natom,AL,ALI,iatom,f_xatom)

            implicit none

            !arguments
            !type_md

            real(8) dtMD ! time step
            integer iMD  ! MD method
            integer MDstep ! total num of MD loop   
            real(8) DeltR(3,matom_1) ! delta position of atoms
            real(8) errNH
            real(8) temperature1,temperature2 ! temperature
            integer iscale_temp_VVMD ! not used
            integer nstep_temp_VVMD ! delta step to scale the total energy when vvmd 
            integer imov_at(3,matom_1) ! whether move the atom
            integer natom

            character*20 f_xatom

            !type_atom
            real(8) xatom(3,matom_1) ! fractional coordinates of atom's position
            real(8) iMDatom(matom_1) ! atom's mass
            integer iatom(matom_1)
            !type_scf
            real(8) Etot  ! total energy
            real(8) Etot_old ! total energy of last step
            real(8) fatom(3,matom_1) ! atom's forces
            real(8) AL(3,3) ! lattice vector, in bohr unit
            real(8) ALI(3,3) ! lattice vector, in bohr unit

            integer ido_stop,istress_cal
            real(8) e_stress(3,3)
            !
            call init_global(AL,xatom,iMDatom,dtmd,iMD,MDstep,errNH,temperature1,temperature2,iscale_temp_VVMD,nstep_temp_VVMD,imov_at,MCTRL_stress,natom,ALI,iatom)
            ! call init_mctrl_scf()
            !
            MCTRL_XATOM_FILE=f_xatom
            !real md calling
            
            
            !!!!!
            call molecular_dynamics_kernel(Etot,fatom,e_stress)
            !
    end subroutine molecular_dynamics_new

    subroutine md_init(mymd)
            use mod_data, only: vx_1,vy_1,vz_1,langevin_factT,langevin_factG
            implicit none
            type(type_md) :: mymd
            integer i
            real*8 scaling
            real*8 :: Ek
            real*8 :: v(3,matom_1)

            real*8 RANFX

            real*8 woszi,tmp
            real*8 :: tmp3(3,3)
            !
            !
            call RANTEST(Iseed)
            do i = 1, MCTRL_natom
                if(exist_velocity) then
                    v(1,i) =vx_1(i) 
                    v(2,i) =vy_1(i) 
                    v(3,i) =vz_1(i) 
                else
                    v(:,i) = (RANFX(Iseed) - 0.5d0)/dsqrt(mymd.m(i))
                endif
                v(:,i) = v(:,i) * mymd.imov_at(:,i)
            enddo
            Ek=0.d0
            do i = 1, MCTRL_natom
                Ek = Ek + mymd.m(i)*dot_product(v(:,i),v(:,i))
            enddo
            Ek=Ek
            !
            if(exist_velocity .or. Ek.le.1.d-10) then
                scaling = 1.d0 
            else
            !    scaling = sqrt(dabs(mymd.T1*Hdt*dble(mymd.num_degree)/Ek))
                scaling = sqrt(2*dabs(mymd.T1*Hdt*dble(mymd.num_degree)/Ek))
            ! the factor of 2 is for the expectation that, hald of the initial
            ! kinetic energy will be given to potential energy when start from
            ! ground state geometry
            endif
            !------------------------
            Ek=0.d0
            do i=1,MCTRL_natom
                v(:,i)=v(:,i)*scaling
                Ek = Ek + mymd.m(i)*dot_product(v(:,i),v(:,i))
            enddo
            !
            mymd.r=MCTRL_xatom
            mymd.v=v
            mymd.Ek=0.5d0*Ek
            mymd.curT=2.d0*mymd.Ek/DBLE(mymd.num_degree*Hdt)
            mymd.totalE_ini=mymd.Etot+mymd.Ek
            !
            if(mymd.method==1) then
                if(inode.eq.1) then
                    write(6,*)
                    write(*,*) "Basic velocity verlet dynamics (NVE)"
                    write(6,*)
                endif
            endif
            if(mymd.method==2) then
                !if(mymd.Q <= 1.d-15) then
                !    mymd.Q=2.d0*dble(mymd.num_degree)*max(mymd.T1,mymd.T2)*Hdt* &
                !            ( 40.d0*mymd.dtMD/2.d0/3.1415926d0 )**2
                !endif
                call write_Q()
            endif

            if(mymd.method==3.or.mymd.method==4) then
              do i=1,MCTRL_natom
              mymd.v(:,i)=mymd.v(:,i)*dsqrt(langevin_factT(i))
              enddo
            endif

            if(mymd.method==3) then


                call langevin_forces_new(mymd)
                call write_GAMMA()
            endif
            if(mymd.method==4) then
                !if(mymd.wg <= 1.d-15) then
                !   mymd.wg=dble(mymd.num_degree+9)*Hdt*max(mymd.T1,mymd.T2)*(mymd.cell_tau)**2/9.d0
                !endif
                call langevin_forces_new(mymd)
                call langevin_forces_lattice_new(mymd)
                call write_GAMMAL()
                call write_NPT()
                !
            endif
            if(mymd.method==5) then
                !if(mymd.Q <= 1.d-15) then
                !   mymd.Q=dble(mymd.num_degree)*Hdt*max(mymd.T1,mymd.T2)*(mymd.ion_tau)**2
                !endif
                !if(mymd.Q_cell <= 1.d-15) then
                !   mymd.Q_cell=9.d0*Hdt*max(mymd.T1,mymd.T2)*(mymd.cell_tau)**2
                !endif
                !if(mymd.wg <= 1.d-15) then
                !   mymd.wg=dble(mymd.num_degree+9)*Hdt*max(mymd.T1,mymd.T2)*(mymd.cell_tau)**2/9.d0
                !endif
                call write_Q()
                call write_NPT()
                !
                if(mymd.is_MSST) then
                    if(.not.mymd.restart) then
                        mymd.MSST_e0=mymd.Etot+mymd.Ek
                        mymd.MSST_v0=det(mymd.h)
                        tmp3=Pinter(mymd)
                        if(mymd.MSST_DIR==0) then
                            mymd.MSST_p0=tmp3(1,1)
                        else if (mymd.MSST_DIR==1) then
                            mymd.MSST_p0=tmp3(2,2)
                        else if (mymd.MSST_DIR==2) then
                            mymd.MSST_p0=tmp3(3,3)
                        endif
                        mymd.MSST_vs=MCTRL_MSST_VS
                        mymd.MSST_DIR=MCTRL_MSST_DIR
                    endif
                    !mymd.stress_mask=0.d0
                    !mymd.stress_mask(mymd.MSST_dir+1,mymd.MSST_dir+1)=1.d0
                endif
            endif
            if(mymd.restart) then
                if(inode .eq. 1) then
                    write(*,*) "restart from the last run "
                endif
            endif

            if(mymd.method==100.or.mymd.method==101) then
                !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                ! liuliping, use a DFT MOVEMENT as it is
                ! do not add this line to IN.MOVEMENT
                ! add this line to md100.input instead
                !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                !read(88,*) MCTRL_MDstep,MCTRL_nskip_begin_AL,MCTRL_nskip_AL_x, &  
                !  MCTRL_nskip_x_end,MCTRL_jump100
                open(188,file='md100.input') 
                read(188,*) MCTRL_MDstep,MCTRL_nskip_begin_AL,MCTRL_nskip_AL_x, MCTRL_nskip_x_end,MCTRL_jump100
                read(188,'(A200)') MCTRL_md100_movement
                close(188)
                    if(inode .eq. 1) then                                        
                        print*, '***********************************************'
                        print*, '*                                             *'
                        print*, '*   RUNNING MD100 (MLFF Inference)            *'
                        print*, '*                                             *'
                        print*, '***********************************************'
                    endif                                                        

                ! MD/MOVEMENT  input file.
                open(88,file=MCTRL_md100_movement)
                rewind(88)
                ! keep this file open
            endif
        
        
        contains
            subroutine write_Q()
                implicit none
                if(inode .eq. 1) then
                    !mymd.Q=8.90
                    !mymd.Q=mymd.Q*atomicmass_kg*(dot_product(MCTRL_AL(:,1),MCTRL_AL(:,1))*bohr_m*bohr_m)/(Hartree_J*1.d-30)
                    woszi=sqrt(2.d0*Hdt*mymd.desiredT*mymd.num_degree/mymd.Q)
                    tmp=1.d0/(woszi*mymd.dtMD/2.d0/3.1415926d0)
                    write(6,*)
                    if(mymd.method==2) then
                        write(6,*) "Nose-Hoover dynamics (NVT)"
                        write(6,*) "MD_NH_Q(Hartree*fs^2)=",mymd.Q
                    else if(mymd.method==5) then
                        write(6,*) "Nose-Hoover dynamics (NPT)"
                        write(6,*) "MD_NH_Q(Hartree*fs^2)=",mymd.Q
                        write(6,*) "MD_NPT_Q_CELL(Hartree*fs^2)=",mymd.Q_cell
                    endif
                    !write(6,*) "MD_NH_Q(am)=",mymd.Q*Hartree_J*1.d-30 &
                    !        / (dot_product(MCTRL_AL(:,1),MCTRL_AL(:,1))*bohr_m*bohr_m)/atomicmass_kg
                    write(6,*)
                    !write(*,*) "woszi=",woszi
                    !write(*,*) "tmp=",tmp
                endif
            end subroutine write_Q
            subroutine write_GAMMA()
                implicit none
                if(inode .eq. 1) then
                    write(6,*)
                    write(6,*) "Langevin dynamics (NVT)"
                    write(6,*) "MD_LV_GAMMA(1/fs)=",mymd.gamma(1)
                    write(6,*)
                endif
            end subroutine write_GAMMA
            subroutine write_GAMMAL()
                implicit none
                if(inode .eq. 1) then
                    write(6,*)
                    write(6,*) "Langevin+PR dynamics (NPT) hydrostatic pressure"
                    write(6,*) "MD_LV_GAMMA(1/fs)=",mymd.gamma(1)
                    write(6,*) "MD_NPT_GAMMA(1/fs)=",mymd.gammaL
                    write(6,*) "MD_NPT_Q_CELL(Hartree*fs^2)=",mymd.Q_cell
                endif
            end subroutine write_GAMMAL
            subroutine write_NPT()
                implicit none
                if(inode .eq. 1) then
                    write(6,*) "MD_NPT_MASS(Hartree*fs^2)=",mymd.Wg
                    !write(6,'(1x,A,1x,E13.5)') "suggested MD_NPT_MASS=",100.d0*(mymd.num_degree+9)*Hdt*mymd.desiredT/(9.d0*1.d0/(100*mymd.dtMD)**2)
                    if(mymd.is_MSST) then
                        write(6,*) "MD_MSST_VS(bohr/fs)=",mymd.MSST_vs
                    else
                        write(6,*) "MD_NPT_PEXT(GPa)=",mymd.Pext
                        write(6,*) "MD_NPT_PEXT_XYZ(GPa)=",mymd.Pext_XYZ
                    endif
                    write(6,*)
                    !
                endif
            endsubroutine write_NPT
    end subroutine md_init
    
    subroutine check_constraints(mymd)
            implicit none
            type(type_md) :: mymd
            mymd.v(:,1:MCTRL_natom)=mymd.v(:,1:MCTRL_natom) * mymd.imov_at(:,1:MCTRL_natom)
    endsubroutine check_constraints

    subroutine update_r(mymd,istep)
            implicit none
            type(type_md) :: mymd
            integer istep
            integer i,i1,i2,iat_tmp,ii
            real*8 :: drf(3)=0.d0,dr(3)=0.d0
            real*8 :: hdot(3,3),tmp(3,3)
            real*8 :: rdot(3,MCTRL_natom)
            if(mymd.method==1.or.mymd.method==6.or.mymd.method==7) then ! basic verlet-velocity
                do i=1,MCTRL_natom
                    mymd.v(:,i)=mymd.v(:,i) - 0.5d0 * mymd.dtMD * mymd.f(:,i) / mymd.m(i)
                enddo
            else if(mymd.method==2) then ! Nose-Hoover
                do i=1,MCTRL_natom
                    mymd.v(:,i)=mymd.v(:,i) + 0.5d0 * mymd.dtMD * & 
                            (-mymd.f(:,i)/mymd.m(i) - mymd.PS * mymd.v(:,i))
                enddo

            else if(mymd.method==3) then ! Langevin
                do i=1,MCTRL_natom
                    mymd.v(:,i)=(1.d0-mymd.gamma(i)*0.5d0*mymd.dtMD)*mymd.v(:,i)  &
                            -0.5d0*mymd.dtMD*(mymd.f(:,i)-mymd.f_s(:,i))/mymd.m(i)
                enddo
            else if(mymd.method==4) then ! langevin+flexible cell dynamics with hydrostatic pressure
                mymd.pg(:,:)=mymd.pg(:,:) + 0.5d0 * mymd.dtMD * pgdot(mymd)
                do i=1,MCTRL_natom
                    mymd.v(:,i)=mymd.v(:,i) + 0.5d0 * mymd.dtMD * vdot(mymd,i)
                enddo
            else if(mymd.method==5) then ! nose-hoover+flexible cell dynamics with hydrostatic pressure
                mymd.ps_cell=mymd.ps_cell + 0.5d0 * mymd.dtMD * pscelldot(mymd)
                mymd.ps=mymd.ps + 0.5d0 * mymd.dtMD * psdot(mymd)
                mymd.pg(:,:)=mymd.pg(:,:) + 0.5d0 * mymd.dtMD * pgdot(mymd)
                mymd.pg=mymd.pg*mymd.stress_mask
                do i=1,MCTRL_natom
                    mymd.v(:,i)=mymd.v(:,i) + 0.5d0 * mymd.dtMD * vdot(mymd,i)
                enddo
            endif
            mymd.v=mymd.v*mymd.imov_at
            !
            if(mymd.method==4 .or. mymd.method==5) then
                hdot=1.d0/mymd.Wg*matmul(mymd.pg,mymd.h)
                rdot=mymd.v(:,1:MCTRL_natom) + 1.d0/mymd.Wg*matmul(mymd.pg,mymd.r(:,1:MCTRL_natom))
                !
                rdot=rdot*mymd.imov_at(:,1:MCTRL_natom)
            else
                rdot=mymd.v(:,1:MCTRL_natom)
                !
                rdot=rdot*mymd.imov_at(:,1:MCTRL_natom)
            endif
            !
            if(mymd.method==4 .or. mymd.method==5) then
                mymd.h_old=mymd.h
                mymd.h=mymd.h+mymd.dtMD*hdot*mymd.stress_mask
                call get_ALI(mymd.h,mymd.hinv)
                !scale V
                if(mymd.isoscale_V==1) then
                    mymd.h=mymd.h0*(det(mymd.h)/det(mymd.h0))**(1.d0/3)
                    call get_ALI(mymd.h,mymd.hinv)
                endif
                !
            endif

            if(mymd.method==7) then
              if(mod(istep-1,mymd.Berendsen_cell_steps).eq.0) then

                tmp=0.d0
                tmp(1,1)=mymd.Pext_xyz(1)
                tmp(2,2)=mymd.Pext_xyz(2)
                tmp(3,3)=mymd.Pext_xyz(3)
                hdot=(Pinter(mymd)-tmp)*27.211396d0/(0.529177**3)  ! convert to  eV/A^3


                do i1=1,3
                do i2=1,3
                hdot(i1,i2)=hdot(i1,i2)*mymd.dtMD/mymd.BERENDSEN_tauP
                hdot(i1,i2)=hdot(i1,i2)*mymd.Berendsen_cell_steps
                if(hdot(i1,i2).gt.0.01) hdot(i1,i2)=0.01
                if(hdot(i1,i2).lt.-0.01) hdot(i1,i2)=-0.01
                enddo
                enddo


                hdot=matmul(hdot,mymd.h)
                mymd.h_old=mymd.h
                do i1=1,3
                do i2=1,3
!                mymd.h(i1,i2)=mymd.h(i1,i2)*(1+hdot(i1,i2)*mymd.stress_mask(i1,i2))
                mymd.h(i1,i2)=mymd.h(i1,i2)+hdot(i1,i2)*mymd.stress_mask(i1,i2)
                ! already multiplied the hdot above, hdot=matmul(hdot,mymd.h)
                enddo
                enddo
                call get_ALI(mymd.h,mymd.hinv)
              endif
            endif

            ! The gkk update after cell change is done in update_box_relates in
            ! md.f90
            !
            do i=1,MCTRL_natom
                !
                if(mymd.method==4 .or. mymd.method==5) then
                    !
                    dr(:)=mymd.dtMD * rdot(:,i)-1.d0/mymd.Wg*matmul(hdot,mymd.r(:,i))*mymd.dtMD
                    dr=dr*mymd.imov_at(:,i)
                    drf(:)=matmul(transpose(mymd.hinv),dr)
                else
                    dr(:)=mymd.dtMD * rdot(:,i)
                    drf(:)=matmul(transpose(mymd.hinv),dr)
                endif
                mymd.r(:,i)=mymd.r(:,i)+drf(:)
                !
                mymd.drtot(:,i)=mymd.drtot(:,i)+dr(:)
                mymd.dr(:,i)=dr(:)
            enddo

            ! special ! 
            
            ! write(*,*) "MCTRL_nskip_begin_AL:",MCTRL_nskip_begin_AL

            if(mymd.method==100.or.mymd.method==101) then   ! this is special, read in the position from IN.MOVEMENT 
                mymd.h_old=mymd.h

                do ii=1,MCTRL_jump100

                    do i=1,MCTRL_nskip_begin_AL
                        read(88,*)       ! 88 is IN.MOVEMENT
                    enddo
                    
                    ! line below is somehow buggy 
                    read(88,*) mymd.h(1,1),mymd.h(2,1),mymd.h(3,1)
                    read(88,*) mymd.h(1,2),mymd.h(2,2),mymd.h(3,2)
                    read(88,*) mymd.h(1,3),mymd.h(2,3),mymd.h(3,3)

                    mymd.h=mymd.h/0.52917721067d0
                    
                    call get_ALI(mymd.h,mymd.hinv)

                    do i=1,MCTRL_nskip_AL_x
                        read(88,*)
                    enddo
                        
                    do i=1,MCTRL_natom
                        read(88,*) iat_tmp,mymd.r(1,i),mymd.r(2,i),mymd.r(3,i)      !no check on iat_tmp!!
                    enddo
                    
                    do i=1,MCTRL_nskip_x_end
                        read(88,*)
                    enddo
                enddo  ! ii

            endif 
            !  special ! 
              
            !
            !periodic boundary condition
            call periodic_boundary(mymd.r)
            !

    end subroutine update_r



    subroutine update_v(mymd)

            implicit none
            type(type_md) :: mymd
            integer i
            !
            if(mymd.method==1.or.mymd.method==6.or.mymd.method==7) then ! basic velocity verlet
                !
                do i=1,MCTRL_natom
                    mymd.v(:,i)=mymd.v(:,i) - 0.5d0 * mymd.dtMD * mymd.f(:,i) / mymd.m(i)
                enddo
                !
            else if(mymd.method==2) then
                !
                do i=1,MCTRL_natom
                    mymd.v(:,i) = ( mymd.v(:,i) - 0.5d0 * mymd.dtMD * mymd.f(:,i)/ mymd.m(i))&
                                  / ( 1.d0 + 0.5d0 * mymd.PS)
                enddo
                call get_energy_kinetic(mymd)
                mymd.PS = mymd.PS + 1.d0 * mymd.dtMD / mymd.Q * &
                        ( 2.d0*mymd.Ek - mymd.num_degree * Hdt * mymd.desiredT )
                !
            else if(mymd.method==3) then
                call langevin_forces_new(mymd)
                do i=1,MCTRL_natom
                    mymd.v(:,i)=1.d0/(1.d0+mymd.gamma(i)*mymd.dtMD*0.5d0) * &
                               ( mymd.v(:,i)- ( mymd.f(:,i)-mymd.f_s(:,i) )/mymd.m(i)*mymd.dtMD*0.5d0 )
                enddo
            else if(mymd.method==4) then
                !
                call langevin_forces_new(mymd)
                call langevin_forces_lattice_new(mymd)
                call iterative_update_v_LVPR()
                ! 
            else if(mymd.method==5) then
                !do i=1,MCTRL_natom
                !    mymd.v(:,i) = ( mymd.v(:,i) - 0.5d0 * mymd.dtMD * mymd.f(:,i)/ mymd.m(i))&
                !                  / ( 1.d0 + 0.5d0 * mymd.PS)
                !enddo
                !call get_energy_kinetic(mymd)
                !mymd.PS = mymd.PS + 1.d0 * mymd.dtMD / mymd.Q * &
                !        ( 2.d0*mymd.Ek - mymd.num_degree * Hdt * mymd.desiredT )
                call iterative_update_v_NHPR()
            endif
            mymd.v=mymd.v*mymd.imov_at
            !
            contains
                subroutine iterative_update_v_LVPR()
                        real*8 :: v_0(3,matom_1)
                        real*8 :: v_tmp(3,matom_1)
                        real*8 :: v_tmp_old(3,matom_1)
                        real*8 :: pg_0(3,3)
                        real*8 :: pg_tmp(3,3)
                        real*8 :: pg_tmp_old(3,3)
                        real*8 :: v_tmp_err,pg_tmp_err
                        real*8 :: num(3)
                        real*8 :: denum(3,3),denum_inv(3,3)
                        integer i,j
                        v_0=mymd.v
                        pg_0=mymd.pg
                        !do i=1,MCTRL_natom
                        !    v_tmp(:,i)=mymd.v(:,i) + 0.5d0 * mymd.dtMD * vdot(mymd,i)
                        !enddo
                        !pg_tmp(:,:)=mymd.pg(:,:) + 0.5d0 * mymd.dtMD * pgdot(mymd)
                        v_tmp=v_0
                        pg_tmp=pg_0
                        do j=1,100
                            v_tmp_old=v_tmp
                            pg_tmp_old=pg_tmp
                            denum=eye(3)+0.5d0*mymd.dtMD*(pg_tmp/mymd.Wg+trace(pg_tmp)/(mymd.num_degree*mymd.Wg)*eye(3)+mymd.gamma(1)*eye(3))
                            call get_ALI(denum,denum_inv) 
                            do i=1,MCTRL_natom
                                num(:)=v_0(:,i)+0.5d0*mymd.dtMD*(-mymd.f(:,i)+mymd.f_s(:,i))/mymd.m(i)
                                v_tmp(:,i)=matmul(transpose(denum_inv),num)
                            enddo
                            mymd.v=v_tmp
                            call get_energy_kinetic(mymd)
                            !pg_tmp=(pg_0+0.5d0*mymd.dtMD*(det(mymd.h)*(Pinter(mymd)-mymd.Pext*eye(3))+ 2.d0*mymd.Ek/mymd.num_degree*eye(3)+mymd.fL_s))/(1.d0+0.5d0*mymd.dtMD*mymd.gammaL)
                            pg_tmp=(pg_0+0.5d0*mymd.dtMD*(det(mymd.h)*(Pinter(mymd)-to_diag(mymd.Pext_xyz,3))+ 2.d0*mymd.Ek/mymd.num_degree*eye(3)+mymd.fL_s))/(1.d0+0.5d0*mymd.dtMD*mymd.gammaL)
                            pg_tmp=pg_tmp*mymd.stress_mask
                            !
                            v_tmp_err=maxval(abs(v_tmp_old-v_tmp))
                            pg_tmp_err=maxval(abs(pg_tmp_old-pg_tmp))
                            !if(inode .eq. 1) then
                            !    write(*,*) "errs=",v_tmp_err,pg_tmp_err
                            !endif
                            if(v_tmp_err<1.d-5 .and. pg_tmp_err<1.d-5) then
                                exit
                            endif                   
                        enddo
                        !if(inode .eq. 1) then
                        !    write(*,*) "diff=",sum(abs(v_0-v_tmp)),sum(abs(pg_0-pg_tmp))
                        !    write(*,*) "pg_0=",pg_0
                        !    write(*,*) "pg_tmp=",pg_tmp
                        !endif
                        mymd.v=v_tmp
                        mymd.pg=pg_tmp
                endsubroutine iterative_update_v_LVPR
                subroutine iterative_update_v_NHPR()
                        real*8 :: v_0(3,matom_1)
                        real*8 :: v_tmp(3,matom_1)
                        real*8 :: v_tmp_old(3,matom_1)
                        real*8 :: pg_0(3,3)
                        real*8 :: ps_0
                        real*8 :: ps_tmp
                        real*8 :: ps_tmp_old
                        real*8 :: pg_tmp(3,3)
                        real*8 :: pg_tmp_old(3,3)
                        real*8 :: ps_cell_0,ps_cell_tmp,ps_cell_tmp_old,ps_cell_tmp_err
                        real*8 :: v_tmp_err,pg_tmp_err,ps_tmp_err
                        real*8 :: num(3)
                        real*8 :: denum(3,3),denum_inv(3,3)
                        integer i,j
                        v_0=mymd.v
                        pg_0=mymd.pg
                        ps_0=mymd.ps
                        ps_cell_0=mymd.ps_cell
                        do i=1,MCTRL_natom
                            v_tmp(:,i)=mymd.v(:,i) + 0.5d0 * mymd.dtMD * vdot(mymd,i)
                        enddo
                        mymd.v=v_tmp
                        call get_energy_kinetic(mymd)
                        !
                        pg_tmp=mymd.pg(:,:) + 0.5d0 * mymd.dtMD * pgdot(mymd)
                        pg_tmp=pg_tmp*mymd.stress_mask
                        mymd.pg=pg_tmp
                        !
                        ps_tmp=ps_0 + 0.5d0 * mymd.dtMD * psdot(mymd)
                        mymd.ps=ps_tmp
                        !
                        ps_cell_tmp=ps_cell_0 + 0.5d0 * mymd.dtMD * pscelldot(mymd)
                        mymd.ps_cell=ps_cell_tmp
                        do j=1,100
                            !
                            v_tmp_old=v_tmp
                            pg_tmp_old=pg_tmp
                            ps_tmp_old=ps_tmp
                            ps_cell_tmp_old=ps_cell_tmp
                            !
                            do i=1,MCTRL_natom
                                v_tmp(:,i)=v_0(:,i) + 0.5d0 * mymd.dtMD * vdot(mymd,i)
                            enddo
                            mymd.v=v_tmp
                            call get_energy_kinetic(mymd)
                            !
                            pg_tmp=pg_0 + 0.5d0 * mymd.dtMD * pgdot(mymd)
                            pg_tmp=pg_tmp*mymd.stress_mask
                            mymd.pg=pg_tmp
                            !
                            ps_tmp=ps_0 + 0.5d0 * mymd.dtMD * psdot(mymd)
                            mymd.ps=ps_tmp
                            !
                            ps_cell_tmp=ps_cell_0 + 0.5d0 * mymd.dtMD * pscelldot(mymd)
                            mymd.ps_cell=ps_cell_tmp
                            !
                            v_tmp_err=maxval(abs(v_tmp_old-v_tmp))
                            pg_tmp_err=maxval(abs(pg_tmp_old-pg_tmp))
                            ps_tmp_err=abs(ps_tmp_old-ps_tmp)
                            ps_cell_tmp_err=abs(ps_cell_tmp_old-ps_cell_tmp)
                            
                            !if(inode .eq. 1) then
                            !    write(*,*) "errs=",v_tmp_err,pg_tmp_err,ps_tmp_err
                            !endif
                            !
                            if(v_tmp_err<1.d-5 .and. pg_tmp_err<1.d-5 .and. ps_tmp_err<1.d-5 .and. ps_cell_tmp_err <1.d-5) then
                                exit
                            endif                   
                        enddo
                endsubroutine iterative_update_v_NHPR
    end subroutine update_v
    subroutine get_cell_info(mymd)
            implicit none
            type(type_md) :: mymd
            real*8 h11,h22,h33,a1,a2,a3
            real*8 norm1,norm2,norm3
            h11=mymd.h(1,1)
            h22=mymd.h(2,2)
            h33=mymd.h(3,3)
            norm1=sqrt(dot_product(mymd.h(:,1),mymd.h(:,1)))
            norm2=sqrt(dot_product(mymd.h(:,2),mymd.h(:,2)))
            norm3=sqrt(dot_product(mymd.h(:,3),mymd.h(:,3)))
            a1=acos(dot_product(mymd.h(:,1),mymd.h(:,2))/norm1/norm2)
            a2=acos(dot_product(mymd.h(:,1),mymd.h(:,3))/norm1/norm3)
            a3=acos(dot_product(mymd.h(:,2),mymd.h(:,3))/norm2/norm3)

            if(inode.eq.1) then
                open(1009,file='cell_info',access='append')
                write(1009,'(4(1x,E13.5),3(1x,F13.5))') mymd.curtime,h11,h22,h33,a1,a2,a3
                close(1009)
            endif
    end subroutine get_cell_info

    subroutine get_diffusion_coeff(mymd)

            implicit none
            type(type_md) :: mymd
            real*8 istep
            real*8 sum
            integer i
            istep=mymd.curtime/mymd.dtMD
            sum=0.d0
            do i=1,MCTRL_natom
                sum=sum+dot_product(mymd.drtot(:,i),mymd.drtot(:,i))
            enddo
            mymd.diffusion_coeff=mymd.diffusion_coeff+sum/(6.d0*mymd.curtime)/istep ! unit bohr^2/fs
    end subroutine get_diffusion_coeff

    subroutine get_average_temperature(mymd)

            implicit none
            type(type_md) :: mymd
            real*8 istep
            real*8 sum
            integer i
            istep=mymd.curtime/mymd.dtMD
            mymd.totT=mymd.totT+mymd.curT
            if(mymd.curtime.le.mymd.tau) then
                mymd.averageT=mymd.totT/dble(istep)
            else
                mymd.averageT=mymd.averageT*(1.d0-mymd.dtMD/mymd.tau)+mymd.curT*mymd.dtMD/mymd.tau
            endif
            !mymd.averageT=totT/dble(istep)
    end subroutine get_average_temperature
    
    subroutine get_average_pressure(mymd)

            implicit none
            type(type_md) :: mymd
            real*8 istep
            real*8 sum
            integer i
            real*8 tmp(3,3)
            tmp=Pinter(mymd)
            if(mymd.is_MSST) then
                mymd.curpress=tmp(mymd.MSST_dir+1,mymd.MSST_dir+1)
            else
                mymd.curpress=(tmp(1,1)+tmp(2,2)+tmp(3,3))/3.d0
            endif
            istep=mymd.curtime/mymd.dtMD
            mymd.totP=mymd.totP+mymd.curpress
            if(mymd.curtime.le.mymd.tau) then
                mymd.averageP=mymd.totP/dble(istep)
            else
                mymd.averageP=mymd.averageP*(1.d0-mymd.dtMD/mymd.tau)+mymd.curpress*mymd.dtMD/mymd.tau
            endif
    end subroutine get_average_pressure


    subroutine periodic_boundary(r)

            implicit none
            real*8 r(3,matom_1)
            r=mod(r+1.d0,1.d0)
            !r=r-int(r)
    end subroutine periodic_boundary


!    subroutine get_energy_kinetic(mymd,vin)
!
!            implicit none
!            type(type_md) :: mymd
!            real*8,optional :: vin(3,matom_1) 
!            real*8 v(3,matom_1)
!            integer i
!            if(present(vin)) then
!                v=vin
!            else
!                v=mymd.v
!            endif
!            mymd.Ek=0.d0
!            do i=1,MCTRL_natom
!                mymd.Ek = mymd.Ek + mymd.m(i)*dot_product(v(:,i),v(:,i))
!            enddo
!            mymd.Ek=0.5d0*mymd.Ek
!    end subroutine get_energy_kinetic
    subroutine get_energy_kinetic(mymd)

            implicit none
            type(type_md) :: mymd
            integer i
            mymd.Ek=0.d0
            mymd.v=mymd.v*mymd.imov_at
            do i=1,MCTRL_natom
                mymd.Ek = mymd.Ek + mymd.m(i)*dot_product(mymd.v(:,i),mymd.v(:,i))
            enddo
            mymd.Ek=0.5d0*mymd.Ek
    end subroutine get_energy_kinetic


    subroutine get_temperature(mymd)

            implicit none
            type(type_md) :: mymd
            integer i
            mymd.curT=2.d0*mymd.Ek/dble(mymd.num_degree*Hdt)
    end subroutine get_temperature

    subroutine exchange_data_scf2md(mymd,f,Etot,stress)

            use mod_data, only: e_atom

            implicit none
            type(type_md) :: mymd
            real*8 :: f(3,matom_1)
            real*8 :: Etot
            real*8 :: stress(3,3)
            integer i
            mymd.Etot_old=mymd.Etot
            mymd.f_old=mymd.f
            mymd.f_scf_old=mymd.f_scf
            mymd.atomic_energy = e_atom 
            ! 
            mymd.Etot = Etot
            !mymd.stress=stress*mymd.stress_mask
            mymd.stress=stress

            do i=1,MCTRL_natom
                mymd.f(:,i)=f(:,i)*mymd.imov_at(:,i)
            enddo
            
            mymd.f_scf=f
            
    end subroutine exchange_data_scf2md

    subroutine exchange_data_md2scf(mymd)

            implicit none
            type(type_md) :: mymd
            MCTRL_xatom=mymd.r
            MCTRL_AL=mymd.h
            MCTRL_ALI=mymd.hinv
    end subroutine exchange_data_md2scf

    subroutine post_check(mymd)

            implicit none
            type(type_md) :: mymd
            real*8 dE_sum,dL
            real*8 dx(3),dxf(3)
            integer i,j
            dE_sum=0.d0
            dL=0.d0
            do i=1,MCTRL_natom
                dx=mymd.dr(:,i)
                dL=dL+dot_product(dx,dx)
                dE_sum=dE_sum+sum(dx(:)*(mymd.f_scf_old(:,i)+mymd.f_scf(:,i))) 
            enddo
            mymd.dL=dsqrt(dL/MCTRL_natom)
            if(abs(mymd.Etot-mymd.Etot_old).gt.1E-20) then
                dE_sum=dE_sum/2.d0
                mymd.Fcheck=dE_sum/(mymd.Etot-mymd.Etot_old)
            else
                mymd.Fcheck=0.d0
            endif
    end subroutine post_check

    subroutine write_diffusion_coeff(mymd)

            implicit none
            type(type_md) :: mymd
            if(inode.eq.1) then
                open(1009,file='MD_DCOEFF',access='append')
                write(1009,*) mymd.curtime,mymd.diffusion_coeff
                close(1009)
            endif
    end subroutine write_diffusion_coeff
    
    subroutine write_average_temperature(mymd)

            implicit none
            type(type_md) :: mymd
            if(inode.eq.1) then
                open(1009,file='MD_AVET',access='append')
                write(1009,*) mymd.curtime,mymd.averageT
                close(1009)
            endif
    end subroutine write_average_temperature

    subroutine write_MDSTEPS(mymd)

            implicit none
            type(type_md) :: mymd

            if(inode.eq.1) then
                open(224, file="MDSTEPS",position="append")
                write(224, 1116) mymd.curtime,(mymd.Etot+mymd.Ek)*Hartree_ev,mymd.Etot*Hartree_ev, mymd.Ek*Hartree_ev,mymd.curT,mymd.averageT,mymd.dL,mymd.Fcheck

 1116   format("Iter(fs)= ",E13.6,1x,"Etot,Ep,Ek(eV)= ",3(E19.10),1x,"Temp(K)= ",(F15.5),1x,"aveTemp(K)= ",(F15.5),1x," dL= ", E9.2, " Fcheck= ",E10.3)

                close(224)
            endif
    end subroutine write_MDSTEPS
    
    subroutine write_MOVEMENT(mymd,istep,totstep)
            implicit none 
            type(type_md) :: mymd
            integer :: istep,totstep

            if(inode.eq.1) then
                open(223, file="MOVEMENT", position="append")
                write(223, 1111) MCTRL_natom,mymd.curtime,(mymd.Etot+mymd.Ek)*Hartree_ev,mymd.Etot*Hartree_ev,mymd.Ek*Hartree_ev
                1111   format(i8,1x,"atoms,Iteration (fs) = ",E19.10,", Etot,Ep,Ek (eV) = ",3(E19.10),", SCF = ",(I5))
                call write_information(223,mymd)
                call write_lattice(223,mymd.stress)
                
                if(MCTRL_istress_cal.eq.1) then
                    call write_internal_pressure(223,Pinter(mymd))
                endif
                call write_position(223,mymd.r)
                call write_force(223,mymd.f_scf)
                call write_velocity(223,mymd.v)
                
                ! wlj delete condition, 2022.10.8
                !if(mymd.method==100.or.mymd.method==101) then
                call write_atomic_energy(223,mymd)
                !end if
                
                ! if(iflag_energydecomp.eq.1) then
                !     call write_atomic_energy(223,mymd)
                ! endif
                if(istep==totstep) then
                    write(223,*) '----------------------------------------------END'
                else
                    write(223,*) '-------------------------------------------------'
                endif
                close(223)

            endif
    end subroutine write_MOVEMENT

    subroutine write_information(fileunit,mymd)
            implicit none
            type(type_md) :: mymd
            integer :: fileunit
            character(len=200) message
            real*8 :: tmp(3,3)
             write(fileunit,'(5x,A)') "MD_INFO: METHOD(1-VV,2-NH,3-LV,4-LVPR,5-NHRP) TIME(fs) TEMP(K) DESIRED_TEMP(K) AVE_TEMP(K) TIME_INTERVAL(fs) TOT_TEMP(K)"
            write(message,'(I,1x,E19.10,1x,E13.5,1x,E13.5,1x,E13.5,1x,E13.5,1x,E13.5)') mymd.method,mymd.curtime,mymd.curT,mymd.desiredT,mymd.averageT,mymd.tau,mymd.totT
            write(fileunit,'(5x,9x,A)') adjustL(trim(message))
            if(mymd.method==1) then
                write(fileunit,'(5x,A)') "MD_VV_INFO: Basic Velocity Verlet Dynamics (NVE), Initialized total energy(Hartree)"
                write(message,'(E19.10)') mymd.totalE_ini
                write(fileunit,'(5x,9x,A)') adjustL(trim(message))
            endif
            if(mymd.method==2 .or. mymd.method==5) then
                write(fileunit,'(5x,A)') "MD_NH_INFO: Nose-Hoover Dynamics (NVT), IONS' THERMOSTAT VELOCITY(1/fs)"
                write(message,'(E19.10)') mymd.PS
                write(fileunit,'(5x,9x,A)') adjustL(trim(message))
            endif
            if(mymd.method==3) then
                write(fileunit,'(5x,A)') "MD_LV_INFO: Langevin Dynamics (NVT)"
            endif
            if(mymd.method==4) then
                write(fileunit,'(5x,A)') "MD_NPT_INFO: Langevin + PR Dynamics (NPT), PRESSURE(Hartree/bohr^3) AVE_PRESSURE(Hartree/bohr^3) TOT_PRESSURE(Hartree/bohr^3) LATTICE_VELOCITY"
                write(message,'(1x,E13.5,1x,E13.5,1x,E13.5)') mymd.curpress,mymd.averageP,mymd.totP
                write(fileunit,'(5x,9x,A)') adjustL(trim(message))
                write(message,'(9(1x,E13.5))') mymd.pg
                write(fileunit,'(5x,9x,A)') adjustL(trim(message))
            endif
            if(mymd.method==5) then
                write(fileunit,'(5x,A)') "MD_NPT_INFO: Nose-Hoover + PR Dynamics (NPT), CELL'S THERMOSTAT VELOCITY(1/fs) PRESSURE(Hartree/bohr^3) AVE_PRESSURE(Hartree/bohr^3) TOT_PRESSURE(Hartree/bohr^3) LATTICE_VELOCITY"
                write(message,'(1x,E13.5,1x,E13.5,1x,E13.5,1x,E13.5)') mymd.PS_cell,mymd.curpress,mymd.averageP,mymd.totP
                write(fileunit,'(5x,9x,A)') adjustL(trim(message))
                write(message,'(9(1x,E13.5))') mymd.pg
                write(fileunit,'(5x,9x,A)') adjustL(trim(message))
                if(mymd.is_MSST) then
                write(fileunit,'(5x,A)') "MD_MSST_INFO: vs(bohr/fs), e0(Hartree), v0(bohr^3), p0(Hartree/bohr^3), direction(0--x, 1--y, 2--z)"
                write(message,'(1x,E13.5,1x,E13.5,1x,E13.5,1x,E13.5,1x,I)') mymd.MSST_vs, mymd.MSST_e0, mymd.MSST_v0, mymd.MSST_p0, mymd.MSST_dir
                write(fileunit,'(5x,9x,A)') adjustL(trim(message))
                write(fileunit,'(5x,A)') "e(Hartree), p(Hartree/bohr^3), u(bohr/fs)"
                tmp=MSST_PEXT(mymd)
                mymd.MSST_p=tmp(mymd.MSST_dir+1,mymd.MSST_dir+1)
                mymd.MSST_u=mymd.MSST_vs *(1.d0-det(mymd.h)/mymd.MSST_v0)
                mymd.MSST_e=mymd.MSST_e0 + mymd.MSST_p0*(mymd.MSST_v0-det(mymd.h)) + 0.5d0 * mymd.MSST_vs**2 *(1.d0-det(mymd.h)/mymd.MSST_v0)**2
                write(message,'(1x,E13.5,1x,E13.5,1x,E13.5)') mymd.MSST_e, mymd.MSST_p, mymd.MSST_u
                write(fileunit,'(5x,9x,A)') adjustL(trim(message))
                endif
            endif
            if(mymd.method==7) then
                write(fileunit,'(5x,A)') "MD_NPT_INFO BERENDSEN (NPT), PRESSURE(Hartree/bohr^3) "
                write(message,'(9(1x,E13.5))') Pinter(mymd)
                write(fileunit,'(5x,9x,A)') adjustL(trim(message))
            endif
    end subroutine write_information

    subroutine write_lattice(fileunit,stress)

            implicit none
            integer fileunit
            real*8 stress(3,3)
            write(fileunit, *) "Lattice vector (Angstrom)"
            if(MCTRL_istress_cal.eq.0) then
                write(fileunit,'(3(E19.10, 1x))') A_AU_1*MCTRL_AL(1,1),A_AU_1*MCTRL_AL(2,1),A_AU_1*MCTRL_AL(3,1)
                write(fileunit,'(3(E19.10, 1x))') A_AU_1*MCTRL_AL(1,2),A_AU_1*MCTRL_AL(2,2),A_AU_1*MCTRL_AL(3,2)
                write(fileunit,'(3(E19.10, 1x))') A_AU_1*MCTRL_AL(1,3),A_AU_1*MCTRL_AL(2,3),A_AU_1*MCTRL_AL(3,3)
            else
                write(fileunit,1122) A_AU_1*MCTRL_AL(1,1),A_AU_1*MCTRL_AL(2,1),A_AU_1*MCTRL_AL(3,1),stress(1,1)*hartree_ev,stress(2,1)*hartree_ev,stress(3,1)*hartree_ev
                write(fileunit,1122) A_AU_1*MCTRL_AL(1,2),A_AU_1*MCTRL_AL(2,2),A_AU_1*MCTRL_AL(3,2),stress(1,2)*hartree_ev,stress(2,2)*hartree_ev,stress(3,2)*hartree_ev
                write(fileunit,1122) A_AU_1*MCTRL_AL(1,3),A_AU_1*MCTRL_AL(2,3),A_AU_1*MCTRL_AL(3,3),stress(1,3)*hartree_ev,stress(2,3)*hartree_ev,stress(3,3)*hartree_ev
            endif
            1122   format(3(E19.10, 1x), "    stress (eV): ", 3(E13.6,1x))
    end subroutine write_lattice

    subroutine write_norm_lattice(fileunit,AL,istress_cal,stress)

            implicit none
            integer fileunit
            real*8 stress(3,3)
            real*8 AL(3,3)
            integer :: istress_cal
            write(fileunit, *) "Lattice vector (Angstrom)"
            if(istress_cal.eq.0) then
                write(fileunit,'(3(E19.10, 1x))') A_AU_1*AL(1,1),A_AU_1*AL(2,1),A_AU_1*AL(3,1)
                write(fileunit,'(3(E19.10, 1x))') A_AU_1*AL(1,2),A_AU_1*AL(2,2),A_AU_1*AL(3,2)
                write(fileunit,'(3(E19.10, 1x))') A_AU_1*AL(1,3),A_AU_1*AL(2,3),A_AU_1*AL(3,3)
            else
                write(fileunit,1122) A_AU_1*AL(1,1),A_AU_1*AL(2,1),A_AU_1*AL(3,1),stress(1,1)*hartree_ev,stress(2,1)*hartree_ev,stress(3,1)*hartree_ev
                write(fileunit,1122) A_AU_1*AL(1,2),A_AU_1*AL(2,2),A_AU_1*AL(3,2),stress(1,2)*hartree_ev,stress(2,2)*hartree_ev,stress(3,2)*hartree_ev
                write(fileunit,1122) A_AU_1*AL(1,3),A_AU_1*AL(2,3),A_AU_1*AL(3,3),stress(1,3)*hartree_ev,stress(2,3)*hartree_ev,stress(3,3)*hartree_ev
            endif
            1122   format(3(E19.10, 1x), "    stress (eV): ", 3(E13.6,1x))
    end subroutine write_norm_lattice


    subroutine write_internal_pressure(fileunit,stress)

            implicit none
            integer fileunit
            real*8 stress(3,3)
            write(fileunit, *) "Pressure Internal(Hartree/bohr^3)"
            write(fileunit,'(3(E19.10, 1x))') stress(1,1),stress(2,1),stress(3,1)
            write(fileunit,'(3(E19.10, 1x))') stress(1,2),stress(2,2),stress(3,2)
            write(fileunit,'(3(E19.10, 1x))') stress(1,3),stress(2,3),stress(3,3)
    end subroutine write_internal_pressure


    subroutine write_position(fileunit,r)

            implicit none
            real*8 r(3,matom_1)
            integer fileunit
            integer ia
            write(fileunit,*) "Position (normalized), move_x, move_y, move_z"
            do ia = 1, MCTRL_natom
                write(fileunit,"(i4, 1x, 3(f25.15,1x),4x, 3(i1, 2x))") MCTRL_iatom(ia), r(1, ia), r(2,ia), r(3,ia), MCTRL_imov_at(1,ia),MCTRL_imov_at(2,ia),MCTRL_imov_at(3,ia)
            enddo
    end subroutine write_position

    subroutine write_norm_position(fileunit,natom,r,imov_at)

            implicit none
            integer natom
            real*8 r(3,natom)
            integer imov_at(3,natom)
            integer fileunit
            integer ia
            write(fileunit,*) "Position (normalized), move_x, move_y, move_z"
            do ia = 1, natom
                write(fileunit,"(i4, 1x, 3(f25.15,1x),4x, 3(i1, 2x))") MCTRL_iatom(ia), r(1, ia), r(2,ia), r(3,ia), imov_at(1,ia),imov_at(2,ia),imov_at(3,ia)
            enddo
    end subroutine write_norm_position


    subroutine write_force(fileunit,f)

            implicit none
            real*8 f(3,matom_1)
            integer fileunit
            integer ia
            write(fileunit,*) "Force (-force, eV/Angstrom)"
            do ia = 1, MCTRL_natom
                write(fileunit,"(i4, 1x, 3(f25.15,1x))") MCTRL_iatom(ia),f(1,ia)*Hartree_ev/A_AU_1,f(2,ia)*Hartree_ev/A_AU_1,f(3,ia)*Hartree_ev/A_AU_1
            enddo
    end subroutine write_force

    subroutine write_norm_force(fileunit,natom,f)

            implicit none
            integer :: natom
            real*8 f(3,natom)
            integer fileunit
            integer ia
            write(fileunit,*) "Force (eV/Angstrom)"
            do ia = 1, natom
                write(fileunit,"(i4, 1x, 3(f25.15,1x))") MCTRL_iatom(ia),f(1,ia)*Hartree_ev/A_AU_1,f(2,ia)*Hartree_ev/A_AU_1,f(3,ia)*Hartree_ev/A_AU_1
            enddo
    end subroutine write_norm_force
    
    subroutine write_velocity(fileunit,v)

            implicit none
            integer fileunit
            real*8 v(3,matom_1)
            integer ia
            write(fileunit,*) "Velocity (bohr/fs)"
            do ia = 1, MCTRL_natom
                write(fileunit,"(i4, 1x, 3(f25.15,1x))") MCTRL_iatom(ia),v(1,ia),v(2,ia),v(3,ia) 
            enddo
    end subroutine write_velocity
    
    subroutine write_atomic_energy(fileunit,mymd)
            implicit none
            type(type_md) :: mymd
            integer fileunit
            integer ia,ii,i
            real*8 sum_E
            sum_E=0.d0
            
            !            do ia=1,MCTRL_natom
            !            sum_E=sum_E+Etotdiv(ia)
            !            enddo

            !write(fileunit,"(a,2x,E17.10)") "Atomic-Energy, Etot(eV),E_nonloc(eV),Q_atom:dE(eV)=",mymd.Etot*Hartree_ev-sum_E
            write(fileunit,"(a,2x,E17.10)") "Atomic-Energy, Etot(eV),E_nonloc(eV),Q_atom:dE(eV)=", 0.0
            do ia = 1, MCTRL_natom
                write(fileunit, "(i4, 1x, 3(E18.10, 2x))") MCTRL_iatom(ia), mymd.atomic_energy(ia), 0.0, 0.0
            end do
            
            !           if(iflag_energydecomp_type.eq.1.or.iflag_energydecomp_type.eq.2) then
            !           do ia=1, MCTRL_natom
            !               write(fileunit,"(i4,1x,3(E18.10,2x))") MCTRL_iatom(ia),Etotdiv(ia),E_nonloc_div(ia),Qtot_atom(ia)
            !           enddo
            !           endif   
            
    end subroutine write_atomic_energy


    subroutine write_finalconfig(mymd)

            implicit none
            type(type_md) :: mymd

            if(inode.eq.1) then
                open (unit = 2200, file = 'final.config')
                write(2200, 1111) MCTRL_natom,mymd.curtime,(mymd.Etot+mymd.Ek)*Hartree_ev,mymd.Etot*Hartree_ev,mymd.Ek*Hartree_ev
                1111   format(i6,1x,"atoms,Iteration (fs) = ",E19.10,", Etot,Ep,Ek (eV) = ",3(E19.10),", SCF = ",(I5))
                call write_information(2200,mymd)
                call write_lattice(2200,mymd.stress)
                call write_position(2200,mymd.r)
                call write_force(2200,mymd.f_scf)
                call write_velocity(2200,mymd.v)
                write(223,*) '----------------------------------------------END'
                close(2200)
            endif
    end subroutine write_finalconfig

    subroutine write_norm_finalconfig(natom,Etot,Ek,stress,xatom,fatom,istress_cal,AL,imov_at)

            implicit none
            integer :: natom
            real*8 :: Etot,Ek
            real*8 :: stress(3,3)
            real*8 :: xatom(3,natom)
            real*8 :: fatom(3,natom)
            integer :: istress_cal
            real*8 :: AL(3,3)
            integer :: imov_at(3,natom)
            if(inode.eq.1) then
                open (unit = 2200, file = 'OUT.MLMD')
                write(2200, 1111) natom,-1.0,(Etot+Ek)*Hartree_ev,Etot*Hartree_ev,Ek*Hartree_ev
                1111   format(i6,1x,"atoms,Iteration (fs) = ",E19.10,", Etot,Ep,Ek (eV) = ",3(E19.10),", SCF = ",(I5))
                call write_norm_lattice(2200,AL,istress_cal,stress)
                call write_norm_position(2200,natom,xatom,imov_at)
                call write_norm_force(2200,natom,fatom)
                write(2200,*) '----------------------------------------------END'
                close(2200)
            endif
    end subroutine write_norm_finalconfig

    subroutine push_r(r,istep,r_his)

            implicit none
            real*8,intent(inout) :: r_his(3,matom_1,0:2)
            real*8,intent(in) :: r(3,matom_1)
            integer,intent(in) :: istep
            r_his(:,:,mod((istep-1),3))=r(:,:)
    end subroutine push_r


    subroutine get_interpolation_coeff(istep,Aalpha,Abeta,r,r_his,lp,llp,lllp)

            implicit none
            integer istep
            real*8 Aalpha,Abeta
            real*8 r(3,matom_1)
            real*8 r_his(3,matom_1,0:2)
            integer lp,llp,lllp
            !
            integer ia,ixyz
            real*8  dr_lp_llp(3)
            real*8  dr_llp_lllp(3)
            real*8  dr_cp_lp(3)
            real*8  rr(3,MCTRL_natom)
            real*8 a11,a12,a22,a21,b1,b2,det_A
            integer i
            ! interpolation
            ! calculate a11, a12, a21, a22, b1, b2, det_A,Aalpha,Abeta
            if(istep>3) then
            
                a11=0.d0
                a12=0.d0
                a22=0.d0
                b1=0.d0
                b2=0.d0
                do ia=1,MCTRL_natom
                    dr_lp_llp  =get_dr(r_his(:,ia,lp),r_his(:,ia,llp))
                    dr_llp_lllp=get_dr(r_his(:,ia,llp),r_his(:,ia,lllp))
                    dr_cp_lp   =get_dr(r(:,ia),r_his(:,ia,lp))
                    a11 = a11+dot_product(dr_lp_llp  ,dr_lp_llp)
                    a12 = a12+dot_product(dr_lp_llp  ,dr_llp_lllp) 
                    a22 = a22+dot_product(dr_llp_lllp,dr_llp_lllp) 
                    b1  = b1 +dot_product(dr_cp_lp   ,dr_lp_llp  )
                    b2  = b2 +dot_product(dr_cp_lp   ,dr_llp_lllp)
                enddo
                a21=a12
                det_A=a11*a22-a12*a21
                if(abs(det_A).gt.1.D-10) then
                    Aalpha=(b1*a22-b2*a12)/det_A
                    Abeta =(b2*a11-b1*a21)/det_A
                else
                    Aalpha=2.d0
                    ! this is the formula for equal distance quadratic interpolation
                    Abeta=-1.d0
                endif
            endif
            contains
                function get_dr(r1,r2) result (dr)
                        implicit none
                        real*8 r1(3),r2(3)
                        real*8 dr(3)
                        integer i
                        dr=r1-r2
                        do i=1,3
                            if(abs(dr(i)+1).lt.abs(dr(i))) dr(i)=dr(i)+1
                            if(abs(dr(i)-1).lt.abs(dr(i))) dr(i)=dr(i)-1
                        enddo
                        dr=matmul(MCTRL_AL,dr)
                end function get_dr
    end subroutine get_interpolation_coeff

    subroutine energy_scaling(mymd,istep)

            implicit none
            integer istep
            type(type_md) :: mymd
            real*8 Enki
            real*8 scaling
            Enki=mymd.totalE_ini-mymd.Etot
            if(Enki.lt.0.d0) Enki=0.d0
            scaling=sqrt(dabs(Enki/mymd.Ek))

            if(istep.eq.2) then
            mymd.totalE_ini=mymd.Etot+mymd.Ek
            endif

            !
            if(mod(istep,MCTRL_nstep_temp_VVMD).eq.0) then
                if(mymd.method==1) then


                    mymd.v(:,1:MCTRL_natom)=mymd.v(:,1:MCTRL_natom)*scaling
                    mymd.Ek=mymd.Ek*scaling**2
                    mymd.curT=2.d0*mymd.Ek/dble(mymd.num_degree*Hdt)
                endif
            endif

            if(mymd.method==6.or.mymd.method==7) then
            scaling=sqrt(abs(1+(mymd.desiredT/mymd.curT-1)*mymd.dtMD/mymd.Berendsen_tau))
            if(scaling.gt.1.2) scaling=1.1
            if(scaling.lt.0.9) scaling=0.9
                    mymd.v(:,1:MCTRL_natom)=mymd.v(:,1:MCTRL_natom)*scaling
                    mymd.Ek=mymd.Ek*scaling**2
                    mymd.curT=2.d0*mymd.Ek/dble(mymd.num_degree*Hdt)
            endif


    end subroutine energy_scaling

    subroutine update_time(mymd)

            implicit none
            type(type_md) mymd
            mymd.curtime=mymd.curtime+mymd.dtMD
    end subroutine update_time

    subroutine read_information(mymd)

            implicit none
            type(type_md) :: mymd
            logical :: scanit
            integer ierr,method_tmp
            real*8 curT_tmp
            if(inode.eq.1) then
                open(10,file=MCTRL_XATOM_FILE,status='old',action='read')
                rewind(10)
                call scan_key_words (10,"MD_INFO", len("MD_INFO"), scanit)
                if(scanit) then
                    read(10,*) method_tmp,mymd.curtime,curT_tmp,mymd.desiredT,mymd.averageT,mymd.tau,mymd.totT 
                else
                    mymd.curtime=0.d0
                    mymd.desiredT=mymd.T1
                    mymd.averageT=mymd.T1
                    mymd.tau=100.d0*mymd.dtMD
                    mymd.totT=0.d0
                endif
                if(mymd.method==1) then
                    rewind(10)
                    call scan_key_words (10,"MD_VV_INFO", len("MD_VV_INFO"), scanit)
                    if(scanit) then
                        read(10,*) mymd.totalE_ini 
                    else
                        mymd.totalE_ini=0.d0
                        MCTRL_nstep_temp_VVMD=MCTRL_MDstep+100
                    endif
                endif
                if(mymd.method==2 .or. mymd.method==5) then
                    rewind(10)
                    call scan_key_words (10,"MD_NH_INFO", len("MD_NH_INFO"), scanit)
                    if(scanit) then
                        read(10,*) mymd.PS 
                    else
                        mymd.PS=0.d0
                    endif
                endif
                if(mymd.method==4) then
                    rewind(10)
                    call scan_key_words (10,"MD_NPT_INFO", len("MD_NPT_INFO"), scanit)
                    if(scanit) then
                        read(10,*) mymd.curpress,mymd.averageP,mymd.totP
                        read(10,*) mymd.pg
                    else
                        mymd.curpress=0.d0
                        mymd.averageP=0.d0
                        mymd.pg=0.d0
                        mymd.totP=0.d0
                    endif
                endif
                if(mymd.method==5) then
                    rewind(10)
                    call scan_key_words (10,"MD_NPT_INFO", len("MD_NPT_INFO"), scanit)
                    if(scanit) then
                        read(10,*) mymd.PS_cell,mymd.curpress,mymd.averageP,mymd.totP
                        read(10,*) mymd.pg
                    else
                        mymd.PS_cell=0.d0
                        mymd.curpress=0.d0
                        mymd.averageP=0.d0
                        mymd.pg=0.d0
                        mymd.totP=0.d0
                    endif
                    if(mymd.is_MSST) then
                        call scan_key_words (10,"MD_MSST_INFO", len("MD_MSST_INFO"), scanit)
                        if(scanit) then
                            read(10,*) mymd.MSST_vs, mymd.MSST_e0, mymd.MSST_v0, mymd.MSST_p0, mymd.MSST_DIR
                        endif
                    endif
                endif
                close(10)
            endif
            !write(*,*)"info=",inode,mymd.curtime,mymd.desiredT,mymd.totalE_ini,mymd.PS
            call mpi_bcast(mymd.curtime,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.desiredT,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.averageT,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.tau,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.totalE_ini,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.PS,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.PS_cell,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.curpress,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.averageP,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.pg,9,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.totT,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.totP,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.MSST_vs,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.MSST_v0,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.MSST_e0,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.MSST_p0,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(mymd.MSST_dir,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
    end subroutine read_information
    subroutine read_mdopt(mydtMD,temperature1,temperature2,imov_at,natom,AL)
            use mod_data, only: stress_mask,langevin_factT,langevin_factG
            implicit none
            integer :: ierr
            character*200 :: message
            character(len=200) :: right, temp_right
            real*8 :: mydtMD,temperature1,temperature2
            integer imov_at(3,matom_1)
            integer natom
            real*8 :: AL(3,3)
            if(inode .eq. 1) then
                open(92,file='OUT.MDOPT',iostat=ierr) 
                open(91,file='IN.MDOPT',status='old',action='read',iostat=ierr) 

                if(ierr.ne.0) then
                    message="file ***IN.MDOPT*** does not exist, stop"
                    write(*,*) message
                    stop
                endif

                rewind(91)  
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                CALL read_key_words ( 91, 'MD_CELL_TAU', LEN('MD_CELL_TAU'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_MD_CELL_TAU
                else
                    MCTRL_MD_CELL_TAU=400*mydtMD
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,E,5x,A)') "MD_CELL_TAU=", MCTRL_MD_CELL_TAU , "#characteristic time for cell oscillations (fs)"
                endif
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                CALL read_key_words ( 91, 'MD_ION_TAU', LEN('MD_ION_TAU'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_MD_ION_TAU
                else
                    MCTRL_MD_ION_TAU=40*mydtMD
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,E,5x,A)') "MD_ION_TAU=", MCTRL_MD_ION_TAU, "#characteristic time for particles oscillations (fs)"
                endif
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                CALL read_key_words ( 91, 'MD_BERENDSEN_TAU', LEN('MD_BERENDSEN_TAU'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_MD_BERENDSEN_TAU
                else
                    MCTRL_MD_BERENDSEN_TAU=100*mydtMD
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,E,5x,A)') "MD_BERENDSEN_TAU=", MCTRL_MD_BERENDSEN_TAU, "#characteristic time for Berendsen velocity scaling (fs)"
                endif
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                CALL read_key_words ( 91, 'MD_BERENDSEN_TAUP', LEN('MD_BERENDSEN_TAUP'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_MD_BERENDSEN_TAUP
                else
                    MCTRL_MD_BERENDSEN_TAUP=100*mydtMD
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,E,5x,A)') "MD_BERENDSEN_TAUP=", MCTRL_MD_BERENDSEN_TAUP, "#characteristic time for Berendsen cell scaling (fs)"
                endif
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                CALL read_key_words ( 91, 'MD_BERENDSEN_CELL_STEPS', LEN('MD_BERENDSEN_CELL_STEPS'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_MD_BERENDSEN_CELL_STEPS
                else
                    MCTRL_MD_BERENDSEN_CELL_STEPS=1
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,I,5x,A)') "MD_BERENDSEN_CELL_STEPS=", MCTRL_MD_BERENDSEN_CELL_STEPS , "#time steps to update  cell"
                endif
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                CALL read_key_words ( 91, 'MD_KINSTRESS_FLAG', LEN('MD_KINSTRESS_FLAG'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_MD_KINSTRESS_FLAG
                else
                    MCTRL_MD_KINSTRESS_FLAG=0
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,I,5x,A)') "MD_KINSTRESS_FLAG=", MCTRL_MD_KINSTRESS_FLAG , "#flag to include kinetic stress"
                endif
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                CALL read_key_words ( 91, 'MD_LV_GAMMA', LEN('MD_LV_GAMMA'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_GAMMA_LVMD
                else
                    MCTRL_GAMMA_LVMD=0.01d0
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,E,5x,A)') "MD_LV_GAMMA=", MCTRL_GAMMA_LVMD, "#friction coefficient for particles (fs^-1)"
                endif
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                !CALL read_key_words ( 91, 'MD_NH_Q', LEN('MD_NH_Q'), right, IERR )
                !if(ierr.eq.0) then
                !    READ ( right, * )  MCTRL_Q_NHMD
                !else
                !    MCTRL_Q_NHMD = -1.d0
                !endif
                !call init_Q(Temperature1,Temperature2,mydtMD,imov_at,natom)
                !if(inode.eq.1) then
                !    write(92,*) "MD_NH_Q=", MCTRL_Q_NHMD
                !endif
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                CALL read_key_words ( 91, 'MD_SEED', LEN('MD_SEED'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_SEED_MD
                else
                    MCTRL_SEED_MD=12345
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,I,5x,A)') "MD_SEED=", MCTRL_SEED_MD,"#random seed for initializing the velocities" 
                endif
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                CALL read_key_words ( 91, 'MD_AVET_TIMEINTERVAL', LEN('MD_AVET_TIMEINTERVAL'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_MD_AVET_TIMEINTERVAL
                else
                    MCTRL_MD_AVET_TIMEINTERVAL=100.d0*mydtMD
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,E,5x,A)') "MD_AVET_TIMEINTERVAL=", MCTRL_MD_AVET_TIMEINTERVAL,"#time interval to calculate average temperature and pressure (fs)"
                endif
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                CALL read_key_words ( 91, 'MD_NPT_GAMMA', LEN('MD_NPT_GAMMA'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_MD_NPT_GAMMA
                else
                    MCTRL_MD_NPT_GAMMA=0.01d0
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,E,5x,A)') "MD_NPT_GAMMA=", MCTRL_MD_NPT_GAMMA,"#friction coefficient for cell (fs^-1)"
                endif
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                !CALL read_key_words ( 91, 'MD_NPT_LMASS', LEN('MD_NPT_LMASS'), right, IERR )
                !if(ierr.eq.0) then
                !    READ ( right, * )  MCTRL_MD_NPT_LMASS
                !else
                !    MCTRL_MD_NPT_LMASS=-1.d0
                !endif
                !call init_lmass(Temperature1,Temperature2,mydtMD,imov_at,natom)
                !if(inode.eq.1) then
                !    write(92,*) "MD_NPT_LMASS=", MCTRL_MD_NPT_LMASS
                !endif
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                CALL read_key_words ( 91, 'MD_NPT_PEXT', LEN('MD_NPT_PEXT'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_MD_NPT_PEXT
                ! Input pressure is in GPascal, change it to Hartree/Bohr**3
                else
           !         MCTRL_MD_NPT_PEXT=1.d0/det(AL)
                    MCTRL_MD_NPT_PEXT=0.d0
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,E,5x,A)') "MD_NPT_PEXT=", MCTRL_MD_NPT_PEXT,"#external hydrastatic pressure (GPa)"
                endif
                    MCTRL_MD_NPT_PEXT=MCTRL_MD_NPT_PEXT/(160.21766208*27.211396d0/0.529177**3)
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                CALL read_key_words ( 91, 'MD_NPT_PEXT_XYZ', LEN('MD_NPT_PEXT_XYZ'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * ) &
                    MCTRL_MD_NPT_PEXT_XYZ(1),MCTRL_MD_NPT_PEXT_XYZ(2),MCTRL_MD_NPT_PEXT_XYZ(3)
                else
                    MCTRL_MD_NPT_PEXT_XYZ(:)=MCTRL_MD_NPT_PEXT*(160.21766208*27.211396d0/0.529177**3)
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,3(E,1x),5x,A)') "MD_NPT_PEXT_XYZ=", MCTRL_MD_NPT_PEXT_XYZ,"#external xyz pressure (GPa)"
                endif
                    MCTRL_MD_NPT_PEXT_XYZ=MCTRL_MD_NPT_PEXT_XYZ/(160.21766208*27.211396d0/0.529177**3)

                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                CALL read_key_words ( 91, 'MD_NPT_ISOSCALEV', LEN('MD_NPT_ISOSCALEV'), right, IERR )
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_MD_NPT_ISOSCALEV
                else
                    MCTRL_MD_NPT_ISOSCALEV=0
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,I,5x,A)') "MD_NPT_ISOSCALEV=", MCTRL_MD_NPT_ISOSCALEV,"#1--overall scaling of the box; default=0"
                endif
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                !CALL read_key_words ( 91, 'MD_NPT_Q_CELL', LEN('MD_NPT_Q_CELL'), right, IERR )
                !if(ierr.eq.0) then
                !    READ ( right, * )  MCTRL_MD_NPT_Q_CELL
                !else
                !    MCTRL_MD_NPT_Q_CELL = -1.d0
                !endif
                !call init_Q_cell(Temperature1,Temperature2,mydtMD,9)
                !if(inode.eq.1) then
                !    write(92,*) "MD_NPT_Q_CELL=", MCTRL_MD_NPT_Q_CELL
                !endif
                MCTRL_Q_NHMD = -1.d0
                MCTRL_MD_NPT_LMASS=-1.d0
                MCTRL_MD_NPT_Q_CELL = -1.d0
                call init_Q(Temperature1,Temperature2,mydtMD,imov_at,natom)
                call init_Q_cell(Temperature1,Temperature2,mydtMD,int(sum(stress_mask)))
                call init_lmass(Temperature1,Temperature2,mydtMD,imov_at,natom)
                !
                CALL read_key_words ( 91, 'NSTEP_OUTPUT_RHO', LEN('NSTEP_OUTPUT_RHO'), right, IERR)
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_NSTEP_OUTPUT_RHO
                else
                    MCTRL_NSTEP_OUTPUT_RHO=100
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,I,5x,A)') "NSTEP_OUTPUT_RHO=",MCTRL_NSTEP_OUTPUT_RHO,"# step interval to output the charge density"
                endif


                !
                CALL read_key_words ( 91, 'MD_MSST_VS', LEN('MD_MSST_VS'), right, IERR)
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_MSST_VS
                else
                    MCTRL_MSST_VS=0.d0
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,E,5x,A)') "MD_MSST_VS=",MCTRL_MSST_VS,"# velocity of shock wave (bohr/fs)"
                endif
                !
                CALL read_key_words ( 91, 'MD_MSST_DIR', LEN('MD_MSST_DIR'), right, IERR)
                if(ierr.eq.0) then
                    READ ( right, * )  MCTRL_MSST_DIR
                else
                    MCTRL_MSST_DIR=0
                endif
                if(inode.eq.1) then
                    write(92,'(A,1x,I,5x,A)') "MD_MSST_DIR=",MCTRL_MSST_DIR,"# direction of shock wave (0--x, 1--y, 2--z)"
                endif
                !

                close(91)
                close(92)
            endif
            !bcast
            call mpi_bcast(MCTRL_MD_CELL_TAU,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_ION_TAU,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_BERENDSEN_TAU,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_BERENDSEN_TAUP,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_BERENDSEN_CELL_STEPS,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_KINSTRESS_FLAG,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_GAMMA_LVMD,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_NPT_GAMMA,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_NPT_PEXT,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_NPT_PEXT_XYZ,3,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_SEED_MD,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_AVET_TIMEINTERVAL,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_Q_NHMD,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_NPT_LMASS,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_NPT_Q_CELL,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_NPT_ISOSCALEV,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_NSTEP_OUTPUT_RHO,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MSST_VS,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MSST_DIR,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
    end subroutine read_mdopt
    subroutine default_mdopt(mydtMD,temperature1,temperature2,imov_at,natom,AL)
            use mod_data,only: stress_mask,langevin_factT,langevin_factG
            implicit none
            integer :: ierr
            character*200 :: message
            character(len=200) :: right, temp_right
            real*8 :: mydtMD,temperature1,temperature2
            integer imov_at(3,matom_1)
            integer natom
            real*8 :: AL(3,3)
            stress_mask=1.d0
            if(inode .eq. 1) then
                open(92,file='OUT.MDOPT',iostat=ierr) 
                rewind(92)  
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                MCTRL_MD_CELL_TAU=400*mydtMD
                write(92,'(A,1x,E,5x,A)') "MD_CELL_TAU=", MCTRL_MD_CELL_TAU , "#characteristic time for cell oscillations (fs)"
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                MCTRL_MD_ION_TAU=40*mydtMD
                write(92,'(A,1x,E,5x,A)') "MD_ION_TAU=", MCTRL_MD_ION_TAU, "#characteristic time for particles oscillations (fs)"
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                MCTRL_MD_BERENDSEN_TAU=500*mydtMD
                write(92,'(A,1x,E,5x,A)') "MD_BERENDSEN_TAU=", MCTRL_MD_BERENDSEN_TAU, "#characteristic time for Berendsen velocity scaling  (fs)"
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                MCTRL_MD_BERENDSEN_TAUP=100*mydtMD
                write(92,'(A,1x,E,5x,A)') "MD_BERENDSEN_TAUP=", MCTRL_MD_BERENDSEN_TAUP, "#characteristic time for Berendsen cell scaling  (fs)"
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                MCTRL_MD_BERENDSEN_CELL_STEPS=1
                write(92,'(A,1x,I,5x,A)') "MD_BERENDSEN_CELL_STEPS=", MCTRL_MD_BERENDSEN_CELL_STEPS , "#characteristic time for cell oscillations (fs)"
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                MCTRL_MD_KINSTRESS_FLAG=0
                write(92,'(A,1x,I,5x,A)') "MD_KINSTRESS_FLAG=", MCTRL_MD_KINSTRESS_FLAG , "#flag to include kinetic stress"
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                MCTRL_GAMMA_LVMD=0.01d0
                write(92,'(A,1x,E,5x,A)') "MD_LV_GAMMA=", MCTRL_GAMMA_LVMD, "#friction coefficient for particles (fs^-1)"
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                MCTRL_MD_NPT_GAMMA=0.01d0
                write(92,'(A,1x,E,5x,A)') "MD_NPT_GAMMA=", MCTRL_MD_NPT_GAMMA,"#friction coefficient for cell (fs^-1)"
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
             !    MCTRL_MD_NPT_PEXT=1.d0/det(AL)
                 MCTRL_MD_NPT_PEXT=0.d0
                write(92,'(A,1x,E,5x,A)') "MD_NPT_PEXT=", MCTRL_MD_NPT_PEXT,"#external hydrastatic pressure (GPa)"
                 MCTRL_MD_NPT_PEXT_XYZ=0.d0
                write(92,'(A,1x,3(E,1x),5x,A)') "MD_NPT_PEXT_XYZ=", MCTRL_MD_NPT_PEXT_XYZ,"#external hydrastatic pressure (GPa)"
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                MCTRL_SEED_MD=12345
                write(92,'(A,1x,I,5x,A)') "MD_SEED=", MCTRL_SEED_MD,"#random seed for initializing the velocities" 
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                MCTRL_MD_AVET_TIMEINTERVAL=100.d0*mydtMD
                write(92,'(A,1x,E,5x,A)') "MD_AVET_TIMEINTERVAL=", MCTRL_MD_AVET_TIMEINTERVAL, "#time interval to calculate average temperature and pressure (fs)"
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                MCTRL_MD_NPT_ISOSCALEV=0
                write(92,'(A,1x,I,5x,A)') "MD_NPT_ISOSCALEV=", MCTRL_MD_NPT_ISOSCALEV,"#1--overall scaling of the box; default=0"
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                MCTRL_Q_NHMD = -1.d0
                MCTRL_MD_NPT_LMASS=-1.d0
                MCTRL_MD_NPT_Q_CELL = -1.d0
                call init_Q(Temperature1,Temperature2,mydtMD,imov_at,natom)
                call init_Q_cell(Temperature1,Temperature2,mydtMD,int(sum(stress_mask)))
                call init_lmass(Temperature1,Temperature2,mydtMD,imov_at,natom)
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                MCTRL_NSTEP_OUTPUT_RHO=100
                if(inode.eq.1) then
                    write(92,'(A,1x,I,5x,A)') "NSTEP_OUTPUT_RHO=",MCTRL_NSTEP_OUTPUT_RHO,"# step interval to output the charge density"
                endif
                !
                MCTRL_MSST_VS=0.d0
                write(92,'(A,1x,E,5x,A)') "MD_MSST_VS=",MCTRL_MSST_VS,"# velocity of shock wave (bohr/fs)"
                !
                MCTRL_MSST_DIR=0
                write(92,'(A,1x,I,5x,A)') "MD_MSST_DIR=",MCTRL_MSST_DIR,"# direction of shock wave (0--x, 1--y, 2--z)"
                !
                !
                close(92)

            endif
            call mpi_bcast(MCTRL_MD_CELL_TAU,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_ION_TAU,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_BERENDSEN_TAU,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_BERENDSEN_TAUP,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_BERENDSEN_CELL_STEPS,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_KINSTRESS_FLAG,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_GAMMA_LVMD,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_NPT_GAMMA,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_NPT_PEXT,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_NPT_PEXT_XYZ,3,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_SEED_MD,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_AVET_TIMEINTERVAL,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_Q_NHMD,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_NPT_LMASS,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_NPT_Q_CELL,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MD_NPT_ISOSCALEV,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_NSTEP_OUTPUT_RHO,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MSST_VS,1,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
            call mpi_bcast(MCTRL_MSST_DIR,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
    end subroutine default_mdopt

    subroutine langevin_forces_new(mymd)
            type(type_md) :: mymd

            real*8 var,vom(3)
            real*8 fx(3,matom_1)
            integer i,ixyz
            real*8 tmpf(3)
           
            !Get gaussian distribution (unit variance)
            call gauss(MCTRL_natom,fx(1,:),fx(2,:),fx(3,:))
            do i=1,MCTRL_natom
!                var=sqrt(2.d0*mymd.gamma(i)*mymd.m(i)*mymd.desiredT*Hdt/mymd.dtMD)
                var=sqrt(2.d0*mymd.gamma(i)*mymd.m(i)*mymd.desiredT_atom(i)*Hdt/mymd.dtMD)
                mymd.f_s(:,i)=fx(:,i)*var*mymd.imov_at(:,i)
            enddo
            !do i=1,MCTRL_natom
            !    var=sqrt(2.d0*mymd.gamma(i)*MCTRL_iMDatom(i)*mymd.desiredT*Hdt/mymd.dtMD)
            !    do ixyz=1,3
            !        !mymd.f_s(ixyz,i)=gauss_distribution(0.d0,var)
            !        !mymd.f_s(ixyz,i)=boltzmann_distribution(0.d0,var)
            !    enddo
            !enddo
    end subroutine langevin_forces_new

    subroutine langevin_forces_lattice_new(mymd)
            type(type_md) :: mymd

            real*8 var,vom(3)
            real*8 fx(3,3)
            integer i,j,ixyz
            real*8 tmpf(3)
           
            !Get gaussian distribution (unit variance)
            call gauss(3,fx(1,:),fx(2,:),fx(3,:))
            var=sqrt(2.d0*mymd.gammaL*mymd.Wg*mymd.desiredT*Hdt/mymd.dtMD)
            !no cell constrains
            mymd.fL_s=fx*var*mymd.stress_mask
    end subroutine langevin_forces_lattice_new

    subroutine update_T(mymd,istep)
            use mod_data,only: langevin_factT,langevin_factG
            implicit none
            type(type_md) :: mymd
            integer,intent(in) :: istep
            real*8,save:: max_MDtime
            if(istep==1) then
                max_MDtime=mymd.curtime+MCTRL_MDstep*mymd.dtMD
            endif
            mymd.desiredT=mymd.T1+(mymd.T2-mymd.T1)*(mymd.curtime/max_MDtime)
            mymd.desiredT_atom(:)=mymd.desiredT*langevin_factT(:)
    end subroutine update_T

    function trace(a)
            implicit none
            real*8,dimension(:,:) :: a
            real*8 :: trace
            integer r,i
            r=size(a,1)
            trace=0.d0
            do i=1,r
                trace=trace+a(i,i)
            enddo
    endfunction trace
    
    function vdot(mymd,i)
            implicit none
            type(type_md) :: mymd
            integer i ! num atom
            real*8 vdot(3)
            if(mymd.method.eq.4) then
                vdot(:)=(-mymd.f(:,i)+mymd.f_s(:,i))/mymd.m(i)-mymd.gamma(i)*mymd.v(:,i)- &
                        1.d0/mymd.Wg*matmul(mymd.pg,mymd.v(:,i))- &
                        1.d0/(mymd.Wg*mymd.num_degree)*trace(mymd.pg)*mymd.v(:,i)
            endif
            if(mymd.method.eq.5) then
                if(mymd.is_MSST) then
                vdot(:)=-mymd.f(:,i)/mymd.m(i)-mymd.PS*mymd.v(:,i)- &
                        1.d0/mymd.Wg*matmul(mymd.pg,mymd.v(:,i))
                else
                vdot(:)=-mymd.f(:,i)/mymd.m(i)-mymd.PS*mymd.v(:,i)- &
                        1.d0/mymd.Wg*matmul(mymd.pg,mymd.v(:,i))- &
                        1.d0/(mymd.Wg*mymd.num_degree)*trace(mymd.pg)*mymd.v(:,i)
                endif
            endif
    endfunction vdot
    function pgdot(mymd)
            implicit none
            type(type_md) :: mymd
            real*8 pgdot(3,3) 
            real*8 tmp
            if(mymd.method.eq.4) then
                !pgdot=det(mymd.h)*(Pinter(mymd)-mymd.Pext*eye(3))+2.d0*mymd.Ek/mymd.num_degree*eye(3)-mymd.gammaL*mymd.pg+mymd.fL_s
                pgdot=det(mymd.h)*(Pinter(mymd)-to_diag(mymd.Pext_xyz,3))+2.d0*mymd.Ek/mymd.num_degree*eye(3)-mymd.gammaL*mymd.pg+mymd.fL_s
            endif
            if(mymd.method.eq.5) then
                if(mymd.is_MSST) then
                    pgdot=det(mymd.h)*(Pinter(mymd)-MSST_PEXT(mymd))
                    tmp=pgdot(1,1)
                    if(det(mymd.h)>mymd.MSST_v0 .and. tmp>0) then
                        pgdot=-pgdot
                    endif
                else
                    !pgdot=det(mymd.h)*(Pinter(mymd)-mymd.Pext*eye(3))+2.d0*mymd.Ek/mymd.num_degree*eye(3)-mymd.PS_cell*mymd.pg
                    pgdot=det(mymd.h)*(Pinter(mymd)-to_diag(mymd.Pext_xyz,3))+2.d0*mymd.Ek/mymd.num_degree*eye(3)-mymd.PS_cell*mymd.pg
                endif
            endif

    endfunction pgdot
    function psdot(mymd)
            implicit none
            type(type_md) :: mymd
            real*8 psdot
            !if(mymd.method.eq.2) then
                psdot=2.d0*mymd.Ek-mymd.num_degree*Hdt*mymd.desiredT
                psdot=psdot/mymd.Q
            !endif
            !if(mymd.method.eq.5) then
                psdot=2.d0*mymd.Ek-mymd.num_degree*Hdt*mymd.desiredT
                psdot=psdot/mymd.Q
            !endif
    endfunction psdot
    function pscelldot(mymd)
            implicit none
            type(type_md) :: mymd
            real*8 pscelldot
            if(mymd.method.eq.5) then
                !psdot=2.d0*mymd.Ek-mymd.num_degree*Hdt*mymd.desiredT
                pscelldot=-mymd.num_degree_cell*Hdt*mymd.desiredT+trace(matmul(transpose(mymd.pg),mymd.pg))/mymd.wg
                pscelldot=pscelldot/mymd.Q_cell
            endif
    endfunction pscelldot

    function det(h)
            implicit none
            real*8,dimension(:,:) :: h
            real*8 det
            integer r,i,j
            r=size(h,1)
            det=h(1,1)*(h(2,2)*h(3,3)-h(3,2)*h(2,3)) &
                    -h(1,2)*(h(2,1)*h(3,3)-h(3,1)*h(2,3)) &
                    +h(1,3)*(h(2,1)*h(3,2)-h(3,1)*h(2,2))
            det=dabs(det)
    endfunction det
!    function Pinter(mymd,vin)
!            implicit none
!            type(type_md) :: mymd
!            real*8 Pinter(3,3)
!            integer i,j
!            real*8,optional :: vin(3,matom_1) 
!            if(present(vin)) then
!                Pinter=(-matmul(mymd.stress,transpose(mymd.h))+kinetic_stress(mymd,vin))/det(mymd.h)
!            else
!                Pinter=(-matmul(mymd.stress,transpose(mymd.h))+kinetic_stress(mymd))/det(mymd.h)
!            endif
!    endfunction Pinter
    function Pinter(mymd)
            implicit none
            type(type_md) :: mymd
            real*8 Pinter(3,3)
            real*8 tmp(3,3)
            integer i,j
            !! NOT USED: Pinter=(-matmul(mymd.stress,transpose(mymd.h))+kinetic_stress(mymd))/det(mymd.h)
            !Pinter=(-mymd.stress+kinetic_stress(mymd))/det(mymd.h)
            Pinter=(-mymd.stress+mymd.kinstress_flag*kinetic_stress(mymd))/det(mymd.h)

            !eliminate cell rotation
            tmp=transpose(Pinter)+Pinter
            Pinter=0.5d0*tmp
            tmp=0.d0
            tmp(1,1)=Pinter(1,1)
            Pinter=tmp

    endfunction Pinter

    function kinetic_stress(mymd)
            implicit none
            type(type_md) :: mymd
            real*8 :: kinetic_stress(3,3)
            real*8 :: hs(3)
            integer i,j,k
            real*8 tmp
            do i=1,3
                do j=1,3
                    tmp=0.d0
                    do k=1,MCTRL_natom
                        !hs=matmul(mymd.h,mymd.r(:,k))
                        !tmp=tmp+mymd.m(k)*mymd.v(i,k)*mymd.v(j,k)-hs(i)*mymd.f(j,k)
                        tmp=tmp+mymd.m(k)*mymd.v(i,k)*mymd.v(j,k)
                    enddo
                    kinetic_stress(i,j)=tmp
                enddo
            enddo
    endfunction kinetic_stress



    function eye(r)
            implicit none
            integer i,r
            real*8 eye(r,r) 
            eye=0.d0
            do i=1,r
                eye(i,i)=1.d0
            enddo
    endfunction eye   
    ! if use the following two subroutines, need to do:
    ! 
    !  CALL RANDOM_SEED(SIZE = K_SEED)
    !  ALLOCATE(SEED(K_SEED))
    !  SEED(:)=0
    !  CALL SYSTEM_CLOCK(COUNT=CLOCK)
    !  SEED = CLOCK + 37 * (/ (i - 1, i = 1, K_SEED) /)
    !  CALL RANDOM_SEED (PUT = SEED (1 : K_SEED))
    !  mpi_bcast(SEED)
    !
    ! in a global field.
    FUNCTION boltzmann_distribution(RNULL,WIDTH)
            REAL(8) :: num1,num2
            REAL(8) :: boltzmann_distribution,WIDTH,RNULL
            !REAL(q),PARAMETER :: TWOPI = 6.283185307179586_q 

            CALL RANDOM_NUMBER(num1)
            num2=0.d0
            DO
                CALL RANDOM_NUMBER(num2)
                num2=ABS(num2)
                IF (num2 .GT. 1E-08) EXIT
            ENDDO
            IF (num2 .GT. 1.d0) num2=1.d0

            boltzmann_distribution= COS( 2.d0*3.1415926d0*num1 ) * SQRT( 2.d0*ABS(LOG(num2)) )
            boltzmann_distribution = WIDTH * boltzmann_distribution  +  RNULL
    END FUNCTION boltzmann_distribution
    
    function gauss_distribution(mu,sigma)
            implicit none
            real*8 mu,sigma
            real*8 gauss_distribution
            real*8 ran0,ran1,ran2
            ran0=1.d0
            Do While (ran0 >= 1.0d0 .or. ran0 <=1.d-10)
                call random_number(ran1)
                call random_number(ran2)

                ran1=2.0d0*ran1-1.d0
                ran2=2.0d0*ran2-1.d0
                ran0=ran1**2+ran2**2
            enddo
            ran0=Sqrt(-2.0d0*Log(ran0)/ran0)
            gauss_distribution=mu+ran0*ran1*sigma
    end function gauss_distribution

    function MSST_PEXT(mymd)
        type(type_md) :: mymd
        real*8 :: v
        real*8 :: MSST_PEXT(3,3)
        integer :: i
        real*8 :: ss(3,3)
        !
        v=det(mymd.h)
        MSST_PEXT=0.d0
        i=mymd.MSST_DIR+1
        MSST_PEXT(i,i)=mymd.MSST_M * mymd.MSST_vs**2 / mymd.MSST_v0**2 * (mymd.MSST_v0 - v) + mymd.MSST_p0 
        !ss=Pinter(mymd)
        !if(inode.eq.1) then
        !    write(*,*) "TESTMSST_M=",mymd.MSST_M
        !    write(*,*) "TESTGGYT=",mymd.MSST_vs**2 / mymd.MSST_v0**2 
        !endif
    end function MSST_PEXT
    function to_diag(sigma,r)
            implicit none
            integer i,r
            real*8 sigma(r)
            real*8 to_diag(r,r) 
            to_diag=0.d0
            do i=1,r
                to_diag(i,i)=sigma(i)
            enddo
    endfunction to_diag
end module mod_md
