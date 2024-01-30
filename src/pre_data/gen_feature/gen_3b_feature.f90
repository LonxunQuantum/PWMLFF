PROGRAM gen_3b_feature
   USE LeastSquaresSolver
   USE CountsATOMS
   IMPLICIT NONE
   INTEGER :: ierr
   integer :: move_file=1101
   real(8) AL(3,3),Etotp,Ep
   real(8),allocatable,dimension (:,:) :: xatom,fatom
   real(8),allocatable,dimension (:,:) :: xatom0
   real(8),allocatable,dimension (:) :: Eatom
   real(8),allocatable,dimension (:) :: Ep_tmp
   integer,allocatable,dimension (:) :: iatom
   logical :: nextline,found_atomic_energy
   character(len=200) :: the_line
   integer num_step, natom, i, j
   integer num_step0,num_step1,natom0,max_neigh,max_step
   real(8) Etotp_ave,E_tolerance
   character(len=50) char_tmp(20)
   character(len=200) trainSetFileDir(5000)
   character(len=200) trainSetDir
   character(len=200) MOVEMENTDir,dfeatDir,infoDir,trainDataDir,inquirepos2
   integer(8) inp
   integer sys_num,sys,recalc_grid

   integer,allocatable,dimension (:,:,:) :: list_neigh,iat_neigh,iat_neigh_M
   integer,allocatable,dimension (:,:) :: num_neigh
   real*8,allocatable,dimension (:,:,:,:) :: dR_neigh
   real*8,allocatable,dimension (:,:) :: grid31,grid32
   real*8,allocatable,dimension (:,:,:) :: grid31_2,grid32_2
   integer n2b_t,n3b1_t,n3b2_t,it

   real*8 Rc_M
   integer n3b1m,n3b2m,kkk,ii

   real*8,allocatable,dimension (:,:) :: feat
   real*8,allocatable,dimension (:,:) :: feat1
   real*8,allocatable,dimension (:,:,:,:) :: dfeat
   real*8,allocatable,dimension (:,:,:,:) :: dfeat0

   integer,allocatable,dimension (:,:) :: list_neigh_alltype
   integer,allocatable,dimension (:) :: num_neigh_alltype

   integer,allocatable,dimension (:,:) :: list_neigh_alltypeM
   integer,allocatable,dimension (:) :: num_neigh_alltypeM
   integer,allocatable,dimension (:,:) :: map2neigh_alltypeM
   integer,allocatable,dimension (:,:) :: list_tmp
   integer,allocatable,dimension (:) :: nfeat_atom
   integer,allocatable,dimension (:) :: itype_atom
   integer,allocatable,dimension (:,:,:) :: map2neigh_M
   integer,allocatable,dimension (:,:,:) :: list_neigh_M
   integer,allocatable,dimension (:,:) :: num_neigh_M

   integer(4) alive

   real*8 sum1,diff

   integer m_neigh,num,itype1,itype2,itype
   integer iat1,max_neigh_M,num_M

   integer ntype,n2b,n3b1,n3b2,nfeat0m,nfeat0(100)
   integer ntype1,ntype2,k1,k2,k12,ii_f,iat,ixyz
   integer iat_type(100)
   real*8 Rc, Rc2,Rm
   real*8 alpha31,alpha32

   real*8, allocatable, dimension (:,:) :: dfeat_tmp
   integer,allocatable, dimension (:) :: iat_tmp,jneigh_tmp,ifeat_tmp
   integer ii_tmp,jj_tmp,iat2,num_tmp,num_tot,i_tmp,jjj,jj
   integer iflag_grid,iflag_ftype
   real*8 fact_grid,dR_grid1,dR_grid2

   real*8 Rc_type(100), Rc2_type(100), Rm_type(100),fact_grid_type(100),dR_grid1_type(100),dR_grid2_type(100)
   integer iflag_grid_type(100),n3b1_type(100),n3b2_type(100)

   integer,allocatable,dimension (:) :: iatom_type_num

   integer, parameter :: num_elems = 63  ! Total number of elements

   type dictionary_type
      integer :: order
      real(8) :: atomic_E
      real(8) :: atomic_normalization  ! normalization some properties for each element
   end type dictionary_type

   type(dictionary_type), dimension(num_elems) :: dictionary
   integer :: o      ! order of the element
   real(8),allocatable,dimension (:) :: normalization_coeff

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
   INTERFACE
      SUBROUTINE scan_title (io_file, title, title_line, if_find)
         CHARACTER(LEN=200), OPTIONAL :: title_line
         LOGICAL, OPTIONAL :: if_find
         INTEGER :: io_file
         CHARACTER(LEN=*) :: title
      END SUBROUTINE scan_title
   END INTERFACE
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

   !!> this part is for the least square solver
   DOUBLE PRECISION, ALLOCATABLE :: A(:,:), B(:), X(:)
   INTEGER :: M, N, NRHS, INFO
   !!!> above is for the least square solver

   open(10,file="input/gen_3b_feature.in",status="old",action="read")
   rewind(10)
   read(10,*) Rc_M,m_neigh
   read(10,*) ntype

   !!>>
   M = 1       ! A 的行数
   N = ntype       ! A 的列数
   NRHS = 1    ! B 的列数
   allocate(A(M,N), B(M), X(N))
   !!!>>

   do i=1,ntype
      read(10,*) iat_type(i)
      read(10,*) Rc_type(i),Rc2_type(i),Rm_type(i),iflag_grid_type(i),fact_grid_type(i),dR_grid1_type(i),dR_grid2_type(i)
      read(10,*) n3b1_type(i),n3b2_type(i)

      if(Rc_type(i).gt.Rc_M) then
         write(6,*) "Rc_type must be smaller than Rc_M, gen_3b_feature.in",i,Rc_type(i),Rc_M
         stop
      endif

      if(Rc2_type(i).gt.2*Rc_type(i)) then
         write(6,*) "Rc2_type must be smaller than 2*Rc_type, gen_3b_feature.in",i,Rc_type(i),Rc2_type(i)
         stop
      endif

   enddo
   read(10,*) E_tolerance
   read(10,*) iflag_ftype
   read(10,*) recalc_grid
   close(10)

   open(13,file="input/location")
   rewind(13)
   read(13,*) sys_num  !,trainSetDir
   read(13,'(a200)') trainSetDir
   ! allocate(trainSetFileDir(sys_num))
   do i=1,sys_num
      read(13,'(a200)') trainSetFileDir(i)
      write(*,*) i, trainSetFileDir(i)
   enddo
   close(13)
   trainDataDir=trim(trainSetDir)//"/trainData.txt.Ftype2"
   inquirepos2=trim(trainSetDir)//"/inquirepos2.txt"
   !cccccccccccccccccccccccccccccccccccccccc

   do i=1,ntype
      if(iflag_ftype.eq.3.and.iflag_grid_type(i).ne.3) then
         write(6,*) "if iflag_ftype.eq.3, iflag_grid must equal 3, stop"
         stop
      endif
   enddo

   n3b1m=0
   n3b2m=0
   do i=1,ntype
      if(n3b1_type(i).gt.n3b1m) n3b1m=n3b1_type(i)
      if(n3b2_type(i).gt.n3b2m) n3b2m=n3b2_type(i)
   enddo

!cccccccccccccccccccccccccccccccccccccccccccccccc
   num=0
   do itype2=1,ntype
      do itype1=1,itype2
         do k1=1,n3b1m
            do k2=1,n3b1m
               do k12=1,n3b2m
                  ii_f=0
                  if(itype1.ne.itype2) ii_f=1
                  if(itype1.eq.itype2.and.k1.le.k2) ii_f=1
                  if(ii_f.gt.0) then
                     num=num+1
                  endif
               enddo
            enddo
         enddo
      enddo
   enddo
   nfeat0m=num
   write(6,*) "max,nfeat0m=",nfeat0m

   do itype=1,ntype
      num=0
      do itype2=1,ntype
         do itype1=1,itype2
            do k1=1,n3b1_type(itype)
               do k2=1,n3b1_type(itype)
                  do k12=1,n3b2_type(itype)
                     ii_f=0
                     if(itype1.ne.itype2) ii_f=1
                     if(itype1.eq.itype2.and.k1.le.k2) ii_f=1
                     if(ii_f.gt.0) then
                        num=num+1
                     endif
                  enddo
               enddo
            enddo
         enddo
      enddo
      nfeat0(itype)=num
   enddo
   write(6,*) "itype,nfeat0=",(nfeat0(itype),itype=1,ntype)


!cccccccccccccccccccccccccccccccccccccccccccccccccccc


!cccccccccccccccccccccccccccccccccccccccccccccccccccc
   allocate(grid31(0:n3b1m+1,ntype))
   allocate(grid32(0:n3b2m+1,ntype))
   allocate(grid31_2(2,n3b1m,ntype))
   allocate(grid32_2(2,n3b2m,ntype))
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
   do kkk=1,ntype    ! center atom


      Rc=Rc_type(kkk)
      Rc2=Rc2_type(kkk)
      Rm=Rm_type(kkk)
      iflag_grid=iflag_grid_type(kkk)
      fact_grid=fact_grid_type(kkk)
      dR_grid1=dR_grid1_type(kkk)
      dR_grid2=dR_grid2_type(kkk)
      n3b1=n3b1_type(kkk)
      n3b2=n3b2_type(kkk)

!cccccccccccccccccccccccccccccccccccccccc

      if(iflag_grid.eq.1.or.iflag_grid.eq.2) then

         if (recalc_grid.eq.1) then
            if(iflag_grid.eq.1) then
               call get_grid3b_type1(grid31(0,kkk),grid32(0,kkk),Rc,Rc2,Rm,n3b1,n3b2)
            elseif(iflag_grid.eq.2) then
               call get_grid3b_type2(trainSetFileDir,sys_num,grid31(0,kkk),grid32(0,kkk),Rc,Rc2,Rm,n3b1,n3b2, &
                  m_neigh,ntype,iat_type,fact_grid,dR_grid1,dR_grid2,kkk)
            endif
            open(10,file="output/grid3b_cb12_type12."//char(kkk+48))
            rewind(10)
            do i=0,n3b1+1
               write(10,*) grid31(i,kkk),0
               write(10,*) grid31(i,kkk),1
               write(10,*) grid31(i,kkk),0
            enddo
            close(10)

            open(10,file="output/grid3b_b1b2_type12."//char(kkk+48))
            rewind(10)
            do i=0,n3b2+1
               write(10,*) grid32(i,kkk),0
               write(10,*) grid32(i,kkk),1
               write(10,*) grid32(i,kkk),0
            enddo
            close(10)
         else
            open(10,file="output/grid3b_cb12_type12."//char(kkk+48))
            rewind(10)
            do i=0,n3b1+1
               read(10,*) grid31(i,kkk)
               read(10,*) grid31(i,kkk)
               read(10,*) grid31(i,kkk)
            enddo
            close(10)

            open(10,file="output/grid3b_b1b2_type12."//char(kkk+48))
            rewind(10)
            do i=0,n3b2+1
               read(10,*) grid32(i,kkk)
               read(10,*) grid32(i,kkk)
               read(10,*) grid32(i,kkk)
            enddo
            close(10)
         endif      ! recalc_grid.eq.1

      endif   ! iflag_grid.eq.1,2

      !cccccccccccccccccccccccccccccccccccccccccccc
      if(iflag_grid.eq.3) then
         ! for iflag_grid.eq.3, the grid is just read in.
         ! Its format is different from above grid31, grid32.
         ! For each point, it just have two numbers, r1,r2, indicating the region of the sin peak function.

         open(13,file="output/grid3b_cb12_type3."//char(kkk+48))
         rewind(13)
         read(13,*) n3b1_t
         if(n3b1_t.ne.n3b1) then
            write(6,*) "n3b1_t.ne.n3b1,in grid31_type3", n3b1_t,n3b1
            stop
         endif
         do i=1,n3b1
            read(13,*) it,grid31_2(1,i,kkk),grid31_2(2,i,kkk)
            if(grid31_2(2,i,kkk).gt.Rc_type(kkk)) write(6,*) "grid31_2.gt.Rc",grid31_2(2,i,kkk),Rc_type(kkk)
         enddo
         close(13)

         open(13,file="output/grid3b_b1b2_type3."//char(kkk+48))
         rewind(13)
         read(13,*) n3b2_t
         if(n3b2_t.ne.n3b2) then
            write(6,*) "n3b2_t.ne.n3b2,in grid32_type3", n3b2_t,n3b2
            stop
         endif
         do i=1,n3b2
            read(13,*) it,grid32_2(1,i,kkk),grid32_2(2,i,kkk)
            if(grid32_2(2,i,kkk).gt.Rc2_type(kkk)) write(6,*) "grid32_2.gt.Rc",grid32_2(2,i,kkk),Rc2_type(kkk)
         enddo
         close(13)
      endif

   enddo     ! kkk=1,ntype

   !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

   !cccccccccccccccccccccccccccccccccccccccccccccccccccc
   !  FInish the initial grid treatment
   !cccccccccccccccccccccccccccccccccccccccccccccccccccc


   do 2333 sys=1,sys_num
      MOVEMENTDir=trim(trainSetFileDir(sys))//"/MOVEMENT"
      dfeatDir=trim(trainSetFileDir(sys))//"/dfeat.fbin.Ftype2"
      infoDir=trim(trainSetFileDir(sys))//"/info.txt.Ftype2"

      write(*,*) "current mvt path:",MOVEMENTDir

      inquire(file=MOVEMENTDir,exist=alive)
      if (alive.ne..true.) then
         write(*,*) MOVEMENTDir, " not found. Terminate."
         stop
      endif


      open (move_file,file=MOVEMENTDir,status="old",action="read")
      write(*,*) "mvt opened:", MOVEMENTDir
      rewind(move_file)

      num_step0=0
      Etotp_ave=0.d0

      ! First, check if "ATOMIC-ENERGY" exists anywhere in the file
      CALL scan_title(move_file, "ATOMIC-ENERGY", if_find=found_atomic_energy)
      rewind(move_file)

1001  continue
      call scan_title (move_file,"ITERATION",if_find=nextline)
      if(.not.nextline) goto 1002
      num_step0=num_step0+1
      backspace(move_file)
      read(move_file,*) natom0
      if(num_step0.gt.1.and.natom.ne.natom0) then
         write(6,*) "The natom cannot change within one MOVEMENT FILE", &
            num_step0,natom0
      endif
      natom=natom0

      if (found_atomic_energy) then
         CALL scan_title (move_file, "ATOMIC-ENERGY",if_find=nextline)
         if(nextline) then
            backspace(move_file)
            read(move_file,*) char_tmp(1:4),Etotp
            Etotp_ave=Etotp_ave+Etotp
         endif
      endif

      goto 1001
1002  continue
      close(move_file)

      if (found_atomic_energy) then
         Etotp_ave=Etotp_ave/num_step0
         write(6,*) "num_step,natom,Etotp_ave=",num_step0,natom,Etotp_ave
      else
         write(6,*) "Atomic-energy not found, skipping"
         write(6,*) "num_step,natom=",num_step0,natom
      endif

      !cccccccccccccccccccccccccccccccccccccccccccccccccccc
      ALLOCATE (iatom(natom),xatom(3,natom),fatom(3,natom),Eatom(natom))
      !cccccccccccccccccccccccccccccccccccccccccccccccccccc
      OPEN (move_file,file=MOVEMENTDir,status="old",action="read")
      rewind(move_file)

      num_step1=0

      ! First, check if "ATOMIC-ENERGY" exists anywhere in the file
      CALL scan_title(move_file, "ATOMIC-ENERGY", if_find=found_atomic_energy)
      rewind(move_file)

1003  continue
      call scan_title (move_file,"ITERATION",if_find=nextline)
      if(.not.nextline) goto 1004
      num_step1=num_step1+1

      CALL scan_title (move_file, "POSITION")
      DO j = 1, natom
         READ(move_file, *) iatom(j),xatom(1,j),xatom(2,j),xatom(3,j)
      ENDDO

      if (found_atomic_energy) then
         CALL scan_title (move_file, "ATOMIC-ENERGY",if_find=nextline)
         if(nextline) then
            backspace(move_file)
            read(move_file,*) char_tmp(1:4),Etotp
            if(abs(Etotp-Etotp_ave).gt.E_tolerance) then
               write(6,*) "Etotp-Etotp_ave > E_tolerance, stop", num_step1
               stop
            endif
         endif
      endif

      goto 1003

1004  continue
      close(move_file)

      write(6,*) "nstep0,nstep1(used)",num_step0,num_step1
      max_step = max(num_step0,num_step1)

      allocate(Ep_tmp(max_step))
      !!> we need to counts the elements with atom type first
      allocate(iatom_type_num(ntype))
      call num_elem_type(natom,ntype,iatom,max_step,iat_type,iatom_type_num)
      write(6,*) "iatom_type_num=",iatom_type_num  ! for example, 10 atom EC --> return 3 3 4
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
      open(333,file=infoDir)
      rewind(333)
      write(333,"(i4,3x,i3,2x,10(i4,1x))") nfeat0M,ntype,(nfeat0(ii),ii=1,ntype)
      write(333,*) natom
      ! write(333,*) iatom

      num_tot=0

      open(25,file=dfeatDir,form="unformatted",access='stream')
      rewind(25)
      write(25) num_step1,natom,nfeat0m,m_neigh
      write(25) ntype,(nfeat0(ii),ii=1,ntype)
      write(25) iatom
      DEALLOCATE (iatom,xatom,fatom,Eatom)

      write(333,*) num_step0

      !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc    !
      OPEN (move_file,file=MOVEMENTDir,status="old",action="read")
      rewind(move_file)

      max_neigh=-1
      num_step=0
      num_step1=0

      ! First, check if "ATOMIC-ENERGY" exists anywhere in the file
      CALL scan_title(move_file, "ATOMIC-ENERGY", if_find=found_atomic_energy)
      rewind(move_file)

1000  continue

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      call scan_title (move_file,"ITERATION",if_find=nextline)

      if(.not.nextline) goto 2000
      num_step=num_step+1

      backspace(move_file)
      read(move_file, *) natom,char_tmp(2:12),Ep
      if (num_step.gt.0) then
         Ep_tmp(num_step)=Ep
      endif
      ! open(2233, file='./PWdata/Etot.dat', access='append')
      ! write(2233,"(E18.10)") Ep
      ! close(2233)
      ALLOCATE (iatom(natom),xatom(3,natom),fatom(3,natom),Eatom(natom))

      CALL scan_title (move_file, "LATTICE")
      DO j = 1, 3
         READ (move_file,*) AL(1:3,j)
      ENDDO

      CALL scan_title (move_file, "POSITION")
      DO j = 1, natom
         READ(move_file, *) iatom(j),xatom(1,j),xatom(2,j),xatom(3,j)
      ENDDO

      CALL scan_title (move_file, "FORCE", if_find=nextline)
      if(.not.nextline) then
         write(6,*) "force not found, stop", num_step
         stop
      endif
      DO j = 1, natom
         READ(move_file, *) iatom(j),fatom(1,j),fatom(2,j),fatom(3,j)
      ENDDO

      !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      ! Initializes the element's Ei dictionary
      data dictionary%order /1, 3, 4, 5, 6, 7, 8, 9, &
         11, 12, 13, 14, 15, 16, 17, &
         19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, &
         37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, &
         55, 56, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83/
      data dictionary%atomic_E /-45.140551665, -210.0485218888889, -321.1987119, -146.63024691666666, -399.0110205833333, -502.070125, -879.0771215, -1091.0652775, &
         -1275.295054, -2131.9724644444445, -2412.581311, -787.3439924999999, -1215.4995769047619, -1705.5754946875, -557.9141695, &
         -1544.3553605, -1105.0024515, -1420.574128, -1970.9374273333333, -2274.598644, -2331.976294, -2762.3960913793107, -3298.6401545, -3637.624857, -4140.3502, -5133.970898611111, -5498.13054, -2073.70436625, -2013.83114375, -463.783827, -658.83885375, -495.05260075, &
         -782.22601375, -1136.1897344444444, -1567.6510633333335, -2136.8407, -2568.946113, -2845.9228975, -3149.6645705, -3640.458547, -4080.81555, -4952.347355, -5073.703895555555, -4879.3604305, -2082.8865266666667, -2051.94076125, -2380.010715, -2983.2449, -3478.003375, &
         -1096.984396724138, -969.538106, -2433.925215, -2419.015324, -2872.458516, -4684.01374, -5170.37679, -4678.720765, -5133.04942, -5055.7201, -5791.21431, -1412.194369, -2018.85905225, -2440.8732966666666/
      data dictionary%atomic_normalization /0.0972155006, 0.1662967034, 0.1933328204, 0.2683159806, 0.3785987377, 0.1466022127, 0.2164212908, 0.2894568911, &
         0.2048294409, 0.1904067795, 0.2294265320, 0.3035709500, 0.1980542376, 0.2635331168, 0.3278584232, &
         0.2747696192, 0.2494581247, 0.2921985569, 0.2928696308, 0.3177585601, 0.3102721842, 0.2510199148, 0.2950537757, 0.3132558335, 0.3309821868, 0.3253499880, 0.2143201786, 0.2510186465, 0.3371936516, 0.2735732594, 0.3209931710, 0.3700832491, &
         0.3402616140, 0.3118128684, 0.3618887692, 0.3818009070, 0.4264250058, 0.4450683467, 0.4004432178, 0.4315776394, 0.4206211344, 0.3743982209, 0.3689389468, 0.2696836269, 0.3133029021, 0.3606206066, 0.3601795933, 0.3799354317, 0.4214285718, &
         0.4221036950, 0.3808102206, 0.4581242958, 0.5129592570, 0.5727363443, 0.5264437577, 0.5527034453, 0.5425899611, 0.5328131649, 0.5092874991, 0.3374220280, 0.3949182334, 0.4334024473, 0.4329462221 /
      !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      !!> here start the least square solver

      ! A(1,1) = 1.D0
      ! A(1,2) = 2.D0
      ! A(1,3) = 6.D0
      ! B(1) = -0.8409734170E+03

      B(1) = Ep_tmp(num_step)
      DO i = 1, ntype
         A(1,i) = iatom_type_num(i)
      END DO
      CALL SolveLeastSquares(A, B, X, M, N, NRHS, INFO)

      IF (INFO == 0) THEN
         ! Do not display output
         ! PRINT *, "Ei, calcualted by SolveLeastSquares: ", X
      ELSE
         PRINT *, "Error in dgels, INFO = ", INFO
      END IF
      !!!> above is for the least square solver

      if (found_atomic_energy) then
         CALL scan_title (move_file, "ATOMIC-ENERGY",if_find=nextline)
         if(nextline) then
            backspace(move_file)
            read(move_file,*) char_tmp(1:4),Etotp

            DO j = 1, natom
               READ(move_file, *) iatom(j),Eatom(j)
               do o = 1, num_elems
                  if (iatom(j) == dictionary(o)%order) then
                     if (Eatom(j) > 0) then
                        Eatom(j) = dictionary(o)%atomic_E + Eatom(j)
                     else
                        exit
                     endif
                  endif
               enddo
            ENDDO

            write(6,"('num_step',2(i4,1x),2(E15.7,1x),i5)") num_step,natom,Etotp,Etotp-Etotp_ave,max_neigh
            if(abs(Etotp-Etotp_ave).gt.E_tolerance) then
               write(6,*) "escape this step, dE too large"
               write(333,*) num_step
               deallocate(iatom,xatom,fatom,Eatom)
               goto 1000
            endif
         endif
      else
         Etotp = 0.d0
         do i = 1, natom
            do j = 1, ntype
               if (iatom(i) == iat_type(j)) then
                  Eatom(i) = X(j)
               endif
            enddo
            Etotp = Etotp + Eatom(i)
         enddo
         write(6,"('num_step,natom',2(i4,1x),1(E18.10,1x))") num_step,natom,Etotp
      endif

      num_step1=num_step1+1

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Finished readin the movement file.
! fetermined the num_step1
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


      allocate(list_neigh(m_neigh,ntype,natom))
      allocate(map2neigh_M(m_neigh,ntype,natom)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
      allocate(list_neigh_M(m_neigh,ntype,natom)) ! the neigh list of Rc_M
      allocate(num_neigh_M(ntype,natom))
      allocate(iat_neigh(m_neigh,ntype,natom))
      allocate(dR_neigh(3,m_neigh,ntype,natom))   ! d(neighbore)-d(center) in xyz
      allocate(num_neigh(ntype,natom))
      allocate(list_neigh_alltype(m_neigh,natom))
      allocate(num_neigh_alltype(natom))

      allocate(iat_neigh_M(m_neigh,ntype,natom))
      allocate(list_neigh_alltypeM(m_neigh,natom))
      allocate(num_neigh_alltypeM(natom))
      allocate(map2neigh_alltypeM(m_neigh,natom)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
      allocate(list_tmp(m_neigh,ntype))
      allocate(itype_atom(natom))
      allocate(nfeat_atom(natom))

      allocate(feat(nfeat0m,natom))
      allocate(dfeat(nfeat0m,natom,m_neigh,3))

      allocate(normalization_coeff(ntype))
!ccccccccccccccccccccccccccccccccccccccccccc
      itype_atom=0
      do i=1,natom
         do j=1,ntype
            if(iatom(i).eq.iat_type(j)) then
               itype_atom(i)=j
               !!> this part just used to subrutine: find_feature_2b_type3_type_embedding
               do o = 1, num_elems
                  if (iatom(i) == dictionary(o)%order) then
                     normalization_coeff(j) = dictionary(o)%atomic_normalization
                  endif
               enddo
               !!< above done
            endif
         enddo
         if(itype_atom(i).eq.0) then
            write(6,*) "this atom type didn't found", itype_atom(i)
            stop
         endif
      enddo
      !ccccccccccccccccccccccccccccccccccccccccccc


      call find_neighbore(iatom,natom,xatom,AL,Rc_type,num_neigh,list_neigh, &
         dR_neigh,iat_neigh,ntype,iat_type,m_neigh,Rc_M,map2neigh_M,list_neigh_M, &
         num_neigh_M,iat_neigh_M)

      !ccccccccccccccccccccccccccccccccc
      !ccccccccccccccccccccccccccccccccc

      max_neigh=-1
      num_neigh_alltype=0
      max_neigh_M=-1
      num_neigh_alltypeM=0
      do iat=1,natom
         list_neigh_alltype(1,iat)=iat
         list_neigh_alltypeM(1,iat)=iat


         num_M=1
         do itype=1,ntype
            do j=1,num_neigh_M(itype,iat)
               num_M=num_M+1
               if(num_M.gt.m_neigh) then
                  !write(6,*) "Error! maxNeighborNum too small",m_neigh
                  write(6,*) "ERROR! Max neighbor number is too small. Assign a larger value in parameters.py"
                  stop
               endif
               list_neigh_alltypeM(num_M,iat)=list_neigh_M(j,itype,iat)
               list_tmp(j,itype)=num_M
            enddo
         enddo


         num=1
         map2neigh_alltypeM(1,iat)=1
         do  itype=1,ntype
            do   j=1,num_neigh(itype,iat)
               num=num+1
               list_neigh_alltype(num,iat)=list_neigh(j,itype,iat)
               map2neigh_alltypeM(num,iat)=list_tmp(map2neigh_M(j,itype,iat),itype)
               ! map2neigh_M(j,itype,iat), maps the jth neigh in list_neigh(Rc) to jth' neigh in list_neigh_M(Rc_M)
            enddo
         enddo


         num_neigh_alltype(iat)=num
         num_neigh_alltypeM(iat)=num_M
         if(num.gt.max_neigh) max_neigh=num
         if(num_M.gt.max_neigh_M) max_neigh_M=num_M
      enddo

      !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      ! This num_neigh_alltype(iat) include itself !
      dfeat=0.d0
      feat=0.d0
      if(iflag_ftype.eq.1) then
         ! iflag_ftype.eq.1, the sin peak span over two grid points
         call find_feature_3b_type1(natom,itype_atom,Rc_type,Rc2_type,n3b1_type,n3b2_type,num_neigh,  &
            list_neigh,dR_neigh,iat_neigh,ntype,grid31,grid32, &
            feat,dfeat,nfeat0m,m_neigh,n3b1m,n3b2m,nfeat_atom)
      endif
      if(iflag_ftype.eq.2) then
         !  iflag_ftype.eq.2, the sin peak span over three grid points
         call find_feature_3b_type2(natom,itype_atom,Rc_type,Rc2_type,n3b1_type,n3b2_type,num_neigh,  &
            list_neigh,dR_neigh,iat_neigh,ntype,grid31,grid32, &
            feat,dfeat,nfeat0m,m_neigh,n3b1m,n3b2m,nfeat_atom)
      endif
      if(iflag_ftype.eq.3) then
         !  iflag_ftype.eq.3, the sin peak span over the two ends specified by grid31_2,grid32_2
         !  So, there could be many overlaps between different sin peaks
         call find_feature_3b_type3(natom,itype_atom,Rc_type,Rc2_type,n3b1_type,n3b2_type,num_neigh,  &
            list_neigh,dR_neigh,iat_neigh,ntype,grid31_2,grid32_2, &
            feat,dfeat,nfeat0m,m_neigh,n3b1m,n3b2m,nfeat_atom)
         ! call find_feature_3b_type3_type_embedding(natom,itype_atom,Rc_type,Rc2_type,n3b1_type,n3b2_type,num_neigh,  &
         !    list_neigh,dR_neigh,iat_neigh,ntype,grid31_2,grid32_2, &
         !    feat,dfeat,nfeat0m,m_neigh,n3b1m,n3b2m,nfeat_atom, normalization_coeff)
      endif
      !cccccccccccccccccccccccccccccccccccccccccccccccccccc
      !cccccccccccccccccccccccccccccccccccccccccccccccccccc

      num_tot=num_tot+natom

      open(44,file=inquirepos2,position="append")
      Inquire(25,pos=inp)
      write(44,"(A,',',I5,',',I20)") dfeatDir, num_step, inp
      close(44)

      write(25) Eatom
      write(25) fatom
      write(25) feat
!    write(25) num_neigh_alltype
!    write(25) list_neigh_alltype
      write(25) num_neigh_alltypeM    ! the num of neighbor using Rc_M
      write(25) list_neigh_alltypeM   ! The list of neighor using Rc_M
!    write(25) map2neigh_alltypeM    ! the neighbore atom, from list_neigh_alltype list to list_neigh_alltypeM list
      write(25) nfeat_atom  ! The number of feature for this atom

!cccccccccccccccccccccccccccccccccccccccccccccchhhhhh
!  Only output the nonzero points for dfeat
      num_tmp=0
      do jj_tmp=1,m_neigh
         do iat2=1,natom
            do ii_tmp=1,nfeat0M
               if(abs(dfeat(ii_tmp,iat2,jj_tmp,1))+abs(dfeat(ii_tmp,iat2,jj_tmp,2))+ &
                  abs(dfeat(ii_tmp,iat2,jj_tmp,3)).gt.1.E-22) then
                  num_tmp=num_tmp+1
               endif
            enddo
         enddo
      enddo
      allocate(dfeat_tmp(3,num_tmp))
      allocate(iat_tmp(num_tmp))
      allocate(jneigh_tmp(num_tmp))
      allocate(ifeat_tmp(num_tmp))

      num_tmp=0
      do jj_tmp=1,m_neigh
         do iat2=1,natom
            do ii_tmp=1,nfeat0M
               if(abs(dfeat(ii_tmp,iat2,jj_tmp,1))+abs(dfeat(ii_tmp,iat2,jj_tmp,2))+ &
                  abs(dfeat(ii_tmp,iat2,jj_tmp,3)).gt.1.E-22) then
                  num_tmp=num_tmp+1
                  dfeat_tmp(:,num_tmp)=dfeat(ii_tmp,iat2,jj_tmp,:)
                  iat_tmp(num_tmp)=iat2

!    jneigh_tmp(num_tmp)=jj_tmp


! store the max neigh list position
                  jneigh_tmp(num_tmp)=map2neigh_alltypeM(jj_tmp,iat2)


                  ifeat_tmp(num_tmp)=ii_tmp
               endif
            enddo
         enddo
      enddo
!TODO:
      ! write(25) dfeat
      write(25) num_tmp
      write(25) iat_tmp
      write(25) jneigh_tmp
      write(25) ifeat_tmp
      write(25) dfeat_tmp
      write(25) xatom
      write(25) AL


      open(55,file=trainDataDir,position="append")
      do i=1,natom
         !write(55,"(i5,',',i3,',',f12.7,',', i3,<nfeat0m>(',',f15.10))")  &
         !   i,iatom(i),Eatom(i),nfeat_atom(i),(feat(j,i),j=1,nfeat_atom(i))
         write(55,"(i5,',',i3,',',E18.10,',', i3, ',',<nfeat0m>(E22.10))")  &
            i,iatom(i),Eatom(i),nfeat_atom(i),(feat(j,i),j=1,nfeat_atom(i))
      enddo
      close(55)


      deallocate(iat_tmp)
      deallocate(jneigh_tmp)
      deallocate(ifeat_tmp)
      deallocate(dfeat_tmp)
!cccccccccccccccccccccccccccccccccccccccccccccchhhhhh


      deallocate(list_neigh)
      deallocate(iat_neigh)
      deallocate(dR_neigh)
      deallocate(num_neigh)
      deallocate(feat)
      deallocate(dfeat)
      deallocate(list_neigh_alltype)
      deallocate(num_neigh_alltype)


      deallocate(list_neigh_M)
      deallocate(num_neigh_M)
      deallocate(map2neigh_M)
      deallocate(iat_neigh_M)
      deallocate(list_neigh_alltypeM)
      deallocate(num_neigh_alltypeM)
      deallocate(map2neigh_alltypeM)
      deallocate(list_tmp)
      deallocate(itype_atom)
      deallocate(nfeat_atom)
      deallocate(normalization_coeff)


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      DEALLOCATE (iatom,xatom,fatom,Eatom)
!--------------------------------------------------------
      goto 1000

      deallocate(A, B, X)
      
2000  continue
      close(move_file)
      !   write(25) num_step1,num_step0
      !   write(333,*) "num_step1,num_step0",num_step1,num_step0
      close(333)
      close(25)
      deallocate(iatom_type_num)
      deallocate(Ep_tmp)

2333 continue

   stop
end
