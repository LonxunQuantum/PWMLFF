PROGRAM write_Egroup
   IMPLICIT NONE
   integer :: move_file=1101
   real*8 AL(3,3),Etotp
   real*8,allocatable,dimension (:,:) :: xatom,fatom
   real*8,allocatable,dimension (:,:) :: xatom0
   real*8,allocatable,dimension (:) :: Eatom
   integer,allocatable,dimension (:) :: iatom
   logical nextline
   integer num_step, natom, i, j
   integer num_step0,num_step1,natom0,max_neigh
   real*8 Etotp_ave,E_tolerance
   character(len=50) char_tmp(20)
   character(len=200) trainSetFileDir(5000)
   character(len=200) trainSetDir
   character(len=200) MOVEMENTDir,trainDataDir
   integer sys_num,sys

   real*8 Rc_M, Esum, sum, dx1, dx2, dx3, dx, dy, dz, dd

   integer,allocatable,dimension (:) :: itype_atom

   real*8,allocatable,dimension (:,:) :: fact
   real*8,allocatable,dimension (:) :: energy_group
   real*8,allocatable,dimension (:) :: divider

   integer m_neigh,itype
   integer iat1,iat2

   integer ntype
   integer iat_type(100)
   integer num_tot

   real*8 Rc_type(100),Rm_type(100)

   logical*2::alive
   real*8 dwidth,ddcut
   integer max_natom

   integer, parameter :: num_elems = 63  ! Total number of elements

   type dictionary_type
      integer :: order
      real :: atomic_E
   end type dictionary_type

   type(dictionary_type), dimension(num_elems) :: dictionary
   integer :: o


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
   ! write(6,*) "input dwidth"
   ! read(5,*) dwidth
   ! open(10,file="input/egroup.in",status="old",action="read")
   ! rewind(10)
   ! read(10,*) dwidth
   ! read(10,*) ntype
   ! allocate(E_init(ntype))
   ! do i=1,ntype
   ! read(10,*) E_init(i)
   ! enddo
   ! close(10)

   open(10,file="input/gen_dR_feature.in",status="old",action="read")
   rewind(10)
   read(10,*) Rc_M,m_neigh
   read(10,*) ntype
   do i=1,ntype
      read(10,*) iat_type(i)
      read(10,*) Rc_type(i),Rm_type(i)

      if(Rc_type(i).gt.Rc_M) then
         write(6,*) "Rc_type must be smaller than Rc_M, gen_3b_feature.in",i,Rc_type(i),Rc_M
         stop
      endif

   enddo
   read(10,*) E_tolerance
   close(10)

   open(13,file="input/location")
   rewind(13)
   read(13,*) sys_num  !,trainSetDir
   read(13,'(a200)') trainSetDir
   do i=1,sys_num
      read(13,'(a200)') trainSetFileDir(i)
   enddo

   close(13)
   trainDataDir=trim(trainSetDir)//"/Egroup_weight.dat"

   !cccccccccccccccccccccccccccccccccccccccc
   inquire(file=trainDataDir,exist=alive)
   if (alive) then
      open(10,file=trainDataDir)
      close(10,status='delete')
   endif
   max_natom = 0
   !cccccccccccccccccccccccccccccccccccccccccccccccccccc
   !  FInish the initial grid treatment

   !cccccccccccccccccccccccccccccccccccccccccccccccccccc

   do 2333 sys=1,sys_num
      MOVEMENTDir=trim(trainSetFileDir(sys))//"/MOVEMENT"

      !cccccccccccccccccccccccccccccccccccccccccccccccccccc
      OPEN (move_file,file=MOVEMENTDir,status="old",action="read")
      rewind(move_file)

      num_step0=0
      Etotp_ave=0.d0
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

      if (natom.gt.max_natom) max_natom = natom

      CALL scan_title (move_file, "ATOMIC-ENERGY",if_find=nextline)
      if(.not.nextline) then
         write(6,*) "Atomic-energy not found, stop",num_step0
         stop
      endif

      backspace(move_file)
      read(move_file,*) char_tmp(1:4),Etotp
      Etotp_ave=Etotp_ave+Etotp
      goto 1001
1002  continue
      close(move_file)

      Etotp_ave=Etotp_ave/num_step0
      write(6,*) "num_step,natom,Etotp_ave=",num_step0,natom,Etotp_ave
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
      ALLOCATE (iatom(natom),xatom(3,natom),fatom(3,natom),Eatom(natom))
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
      OPEN (move_file,file=MOVEMENTDir,status="old",action="read")
      rewind(move_file)

      num_step1=0
1003  continue
      call scan_title (move_file,"ITERATION",if_find=nextline)
      if(.not.nextline) goto 1004

      CALL scan_title (move_file, "POSITION")
      DO j = 1, natom
         READ(move_file, *) iatom(j),xatom(1,j),xatom(2,j),xatom(3,j)
      ENDDO

      CALL scan_title (move_file, "ATOMIC-ENERGY",if_find=nextline)

      backspace(move_file)
      read(move_file,*) char_tmp(1:4),Etotp

      if(abs(Etotp-Etotp_ave).le.E_tolerance) then
         num_step1=num_step1+1
      endif
      goto 1003
1004  continue
      close(move_file)

      write(6,*) "nstep0,nstep1(used)",num_step0,num_step1

!cccccccccccccccccccccccccccccccccccccccccccccccccc

      num_tot=0

      DEALLOCATE (iatom,xatom,fatom,Eatom)

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc    !
      OPEN (move_file,file=MOVEMENTDir,status="old",action="read")
      rewind(move_file)

      max_neigh=-1
      num_step=0
      num_step1=0
1000  continue

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      call scan_title (move_file,"ITERATION",if_find=nextline)

      if(.not.nextline) goto 2000
      num_step=num_step+1

      backspace(move_file)
      read(move_file, *) natom
      ALLOCATE (iatom(natom),xatom(3,natom),fatom(3,natom),Eatom(natom))

      CALL scan_title (move_file, "(ANGSTROM)", if_find=nextline)
      if(.not.nextline) then
         write(6,*) "LATTICE not found, stop",num_step
         stop
      endif
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

      !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      CALL scan_title (move_file, "ATOMIC-ENERGY",if_find=nextline)
      if(.not.nextline) then
         write(6,*) "Atomic-energy not found, stop",num_step
         stop
      endif

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
         deallocate(iatom,xatom,fatom,Eatom)
         goto 1000
      endif

      num_step1=num_step1+1

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Finished readin the movement file.
! fetermined the num_step1
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      allocate(itype_atom(natom))

!ccccccccccccccccccccccccccccccccccccccccccc
      itype_atom=0
      do i=1,natom
         do j=1,ntype
            if(iatom(i).eq.iat_type(j)) then
               itype_atom(i)=j
            endif
         enddo
         if(itype_atom(i).eq.0) then
            write(6,*) "this atom type didn't found", itype_atom(i)
            stop
         endif
      enddo

      allocate(divider(natom))
      allocate(energy_group(natom))
      allocate(fact(natom,natom))
      fact=0.d0
      energy_group=0.d0
      divider=0.d0
      dwidth = sqrt(-Rc_M**2/log(0.01)) ! log(0.01)=-4.60517018598809 = ln(0.01)
      !   ddcut=-dwidth**2*log(0.01)

      do iat1=1,natom   ! center position (not even call it atom)
         Esum=0.d0
         sum=0.d0
         do iat2=1,natom
            itype=itype_atom(iat2)

            dx1=xatom(1,iat2)-xatom(1,iat1)
            dx2=xatom(2,iat2)-xatom(2,iat1)
            dx3=xatom(3,iat2)-xatom(3,iat1)
            if(abs(dx1+1).lt.abs(dx1)) dx1=dx1+1
            if(abs(dx1-1).lt.abs(dx1)) dx1=dx1-1
            if(abs(dx2+1).lt.abs(dx2)) dx2=dx2+1
            if(abs(dx2-1).lt.abs(dx2)) dx2=dx2-1
            if(abs(dx3+1).lt.abs(dx3)) dx3=dx3+1
            if(abs(dx3-1).lt.abs(dx3)) dx3=dx3-1
            dx=AL(1,1)*dx1+AL(1,2)*dx2+AL(1,3)*dx3
            dy=AL(2,1)*dx1+AL(2,2)*dx2+AL(2,3)*dx3
            dz=AL(3,1)*dx1+AL(3,2)*dx2+AL(3,3)*dx3
            dd=dx**2+dy**2+dz**2

            ! if(dd.lt.ddcut) then
            if(dd.lt.Rc_M) then
               fact(iat1,iat2)=exp(-dd/dwidth**2)
               Esum=Esum+(Eatom(iat2))*fact(iat1,iat2)
               sum=sum+fact(iat1,iat2)

            endif

         enddo

         divider(iat1) = sum
         energy_group(iat1)=Esum/sum

      enddo

      open(55,file=trainDataDir,position="append")
      do i=1,natom
         write(55,"(f20.7)") energy_group(i)
      enddo
      close(55)

      num_tot=num_tot+natom

      deallocate(itype_atom)

      deallocate(fact)
      deallocate(divider)
      deallocate(energy_group)


      ! ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      DEALLOCATE (iatom,xatom,fatom,Eatom)
      ! --------------------------------------------------------
      goto 1000
2000  continue
      close(move_file)

2333 continue

   open(11,file="output/max_natom")
   rewind(11)
   write(11,*) max_natom
   close(11)

   stop
end