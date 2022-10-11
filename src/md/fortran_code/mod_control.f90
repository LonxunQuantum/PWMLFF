module mod_control
!    use data_variable_1, only: matom_1,mscf_1,mtype_1
    !w_line control flag
    use mod_parameter
    logical :: smooth_vloc                 ! default .true.
    logical :: pwscf_integration_method    ! .true. use Simpson method and 10 cutoff (get the same result as PWscf)
    ! .false. use smooth cutoff (smaller rcut) and current integration method.
    ! default .false.
    !About the pwscf_integration_method 
    ! the cause for the difference between Pwmat and PWscf. As we suspected, it is
    ! caused by the dufferent ways to do vq(iq). I am not so sure which method of
    ! integration is better! we can do the following: for option of PWscf style
    ! integration, use the PWscf routine to do the integration, but still first
    ! calculate a vq(iq) array like what we have, then do a third order
    ! interpolation base on vq(iq), so we don't need to change those parts.  All we
    ! need is to call the PWscf routine to calculate our vq(iq), and use rcut_p=10.
    ! The purpose is to get the same result as the PWscf, so they will have
    ! confidence for our code. We will have another option, actually the default
    ! option, for which we can keep our integration scheme, and a smooth cutoff,
    ! using rcut_p like 3.5, or input from the pp file. I think this should be a
    ! better option, more sensible. We can also discuss which integration scheme is
    ! more accurate.----LWW
    !
    logical :: MCTRL_use_plumed
    !
    logical :: check_version
    !
    logical :: mpn123_nosymm
    logical :: exist_stress_ext
    logical :: exist_pstress_ext

    logical :: exist_velocity
    !
    integer :: gen_from_mp
    !
    logical :: MCTRL_is_MSST
    real*8 :: MCTRL_MSST_VS
    integer :: MCTRL_MSST_DIR
    !
    logical :: MCTRL_OUT_MLMD
    !
    !
    ! input(global) data 
    
    !
    logical :: MCTRL_FSM_LASTSTEP=.false.
    logical :: MCTRL_FSM_LASTSTEP_DONE=.false.
    logical :: MCTRL_FSM_DOS_DONE=.false.
    logical :: MCTRL_FSM_ALLBAND=.false.
    !
    integer ipart_DOS
    integer ivr_in
    !
    integer :: MCTRL_NSTEP_OUTPUT_RHO
    character*200 MCTRL_vwr_atom(mtype_1)
    !
    real*8 :: MCTRL_PSTRESS_EXT
    !
    integer :: MCTRL_NMAP_MAX
    !
    real*8 :: MCTRL_QMFIRE_DT ! for QM & FIRE 
    real*8 :: MCTRL_RELAX_MAXMOVE
    integer :: MCTRL_LBFGS_MEMORY
    integer :: MCTRL_RHOWG_INTER_TYPE
    !
    real*8 :: MCTRL_MD_CELL_TAU 
    real*8 :: MCTRL_MD_ION_TAU 
    real*8 :: MCTRL_MD_BERENDSEN_TAU 
    real*8 :: MCTRL_MD_BERENDSEN_TAUP 
    integer :: MCTRL_MD_BERENDSEN_CELL_STEPS
    real*8 :: MCTRL_MD_NPT_Q_CELL
    real*8 :: MCTRL_MD_NPT_GAMMA
    real*8 :: MCTRL_MD_NPT_LMASS
    real*8 :: MCTRL_MD_NPT_PEXT
    real*8 :: MCTRL_MD_NPT_PEXT_xyz(3)
    real*8 :: MCTRL_MD_AVET_TIMEINTERVAL
    integer :: MCTRL_MD_NPT_ISOSCALEV
    real*8 :: MCTRL_GAMMA_LVMD
    real*8 :: MCTRL_Q_NHMD
    integer :: MCTRL_stress
    integer :: MCTRL_SEED_MD
    character(len=20) :: MCTRL_XATOM_FILE
    character(len=200) :: MCTRL_md100_movement           ! liuliping, md100
    integer :: MCTRL_MP_SYMM

    !global data
    real*8 MCTRL_AL(3,3) ! lattice vector, in bohr unit
    real*8 MCTRL_ALI(3,3) 
    real*8 MCTRL_w_cg
    real*8 MCTRL_w_scf ! SCF convergence relative stop criteria based on rho and force
    real*8 MCTRL_tatom(matom_1) ! atom's  type
    real*8 MCTRL_xatom(3,matom_1) ! fractional coordinates of atom's position
    real*8 MCTRL_iMDatom(matom_1) ! atom's mass
    integer MCTRL_iatom(matom_1) 
    !MD data
    real*8 MCTRL_dtMD ! time step
    integer MCTRL_iMD  ! MD method
    integer MCTRL_MDstep ! total num of MD loop
    real*8 MCTRL_errNH
    real*8 MCTRL_temperature1
    real*8 MCTRL_temperature2 ! temperature
    integer MCTRL_MD_kinstress_flag   ! flag for whether to include kinetic stress in the stress definition
    integer MCTRL_iscale_temp_VVMD ! not used
    integer MCTRL_nstep_temp_VVMD ! delta step to scale the total energy when vvmd 
    integer MCTRL_imov_at(3,matom_1) ! whether move the atom
    integer MCTRL_output_nstep     ! the interval to output MOVEMENT

    !global control

    !scf control
    integer MCTRL_ido_stop
    integer MCTRL_ido_ns
    integer MCTRL_istress_cal
    integer MCTRL_iforce_cal ! whether calculate forces as output

    integer MCTRL_natom

    integer MCTRL_nskip_begin_AL  ! for iMD=100, number of line from begin to AL,read from IN.MOVEMENT
    integer MCTRL_nskip_AL_x  ! for iMD=100, number of line from AL to x
    integer MCTRL_nskip_x_end   !for iMD=100, number of line from x to end
    integer MCTRL_jump100   !for iMD=100, number of step jumps to calculate,jump100=1, no jump, calculate all step
    
    !

contains

    subroutine init_global(AL,xatom,iMDatom,dtmd,iMD,MDstep,errNH,temperature1,temperature2,iscale_temp_VVMD,nstep_temp_VVMD,imov_at,istress_cal,natom,ALI,iatom)
            implicit none
            !should remove the subroutine in the future
            !global data
            real*8 AL(3,3) ! lattice vector, in bohr unit
            real*8 ALI(3,3) 
            real*8 xatom(3,matom_1) ! fractional coordinates of atom's position
            integer ntype ! type num of atoms
            real*8 iMDatom(matom_1) ! atom's mass
            integer iatom(matom_1)
            !MD data
            real*8 dtMD ! time step
            integer iMD  ! MD method
            integer MDstep ! total num of MD loop
            real*8 errNH
            real*8 temperature1
            real*8 temperature2 ! temperature
            integer iscale_temp_VVMD ! not used
            integer nstep_temp_VVMD ! delta step to scale the total energy when vvmd 
            integer imov_at(3,matom_1) ! whether move the atom

            !global control

            !scf control
            integer istress_cal
            integer natom

            MCTRL_AL=AL
            MCTRL_ALI=ALI

            MCTRL_xatom(:,1:natom)=xatom(:,1:natom)
            MCTRL_iMDatom(1:natom)=iMDatom(1:natom)
            MCTRL_iatom(1:natom)=iatom(1:natom)
            MCTRL_dtMD=dtMD
            MCTRL_iMD=iMD
            MCTRL_MDstep=MDstep
            MCTRL_errNH=errNH
            MCTRL_temperature1=temperature1
            MCTRL_temperature2=temperature2
            MCTRL_iscale_temp_VVMD=iscale_temp_VVMD
            MCTRL_nstep_temp_VVMD=nstep_temp_VVMD 
            MCTRL_imov_at(:,1:natom)=imov_at(:,1:natom)

            MCTRL_istress_cal=istress_cal
            MCTRL_natom=natom
            !
    end subroutine init_global
end module mod_control
