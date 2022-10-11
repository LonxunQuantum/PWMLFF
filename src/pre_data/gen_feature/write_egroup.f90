PROGRAM write_Egroup
    IMPLICIT NONE
    INTEGER :: ierr
    integer :: move_file=1101
    real*8 AL(3,3),Etotp
    real*8,allocatable,dimension (:,:) :: xatom,fatom
    real*8,allocatable,dimension (:,:) :: xatom0
    real*8,allocatable,dimension (:) :: Eatom
    integer,allocatable,dimension (:) :: iatom
    logical nextline
    character(len=200) :: the_line
    integer num_step, natom, i, j
    integer num_step0,num_step1,natom0,max_neigh
    real*8 Etotp_ave,E_tolerance
    character(len=50) char_tmp(20)
    character(len=200) trainSetFileDir(200)
    character(len=200) trainSetDir
    character(len=200) MOVEMENTDir,dfeatDir,infoDir,trainDataDir,inquirepos1
    integer(8) inp
    integer sys_num,sys,recalc_grid

    integer,allocatable,dimension (:,:,:) :: list_neigh,iat_neigh,iat_neigh_M
    integer,allocatable,dimension (:,:) :: num_neigh
    real*8,allocatable,dimension (:,:,:,:) :: dR_neigh
    ! real*8,allocatable,dimension (:,:) :: grid2
    ! real*8,allocatable,dimension (:,:,:) :: grid2_2
    ! integer n2b_t,n3b1_t,n3b2_t,it
    ! integer n2b_type(100),n2bm

    real*8 Rc_M, Esum, sum, dx1, dx2, dx3, dx, dy, dz, dd
    integer n3b1m,n3b2m,kkk,ii


    integer,allocatable,dimension (:,:) :: list_neigh_alltype
    integer,allocatable,dimension (:) :: num_neigh_alltype

    integer,allocatable,dimension (:,:) :: list_neigh_alltypeM
    integer,allocatable,dimension (:) :: num_neigh_alltypeM
    integer,allocatable,dimension (:,:) :: map2neigh_alltypeM
    integer,allocatable,dimension (:,:) :: list_tmp
    ! integer,allocatable,dimension (:) :: nfeat_atom
    integer,allocatable,dimension (:) :: itype_atom
    integer,allocatable,dimension (:,:,:) :: map2neigh_M
    integer,allocatable,dimension (:,:,:) :: list_neigh_M
    integer,allocatable,dimension (:,:) :: num_neigh_M
    
    real*8,allocatable,dimension (:,:) :: fact 
    real*8,allocatable,dimension (:) :: energy_group 
    real*8,allocatable,dimension (:) :: divider 
    real*8,allocatable,dimension (:) :: E_init

    real*8 sum1,diff

    integer m_neigh,num,itype1,itype2,itype
    integer iat1,max_neigh_M,num_M
    
    integer ntype,n2b,n3b1,n3b2,nfeat0m,nfeat0(100)
    integer ntype1,ntype2,k1,k2,k12,ii_f,iat,ixyz
    integer iat_type(100)
    real*8 Rc, Rc2,Rm
    real*8 alpha31,alpha32

    integer ii_tmp,jj_tmp,iat2,num_tmp,num_tot,i_tmp,jjj,jj
    ! integer iflag_grid,iflag_ftype
    ! real*8 fact_grid,dR_grid1,dR_grid2

    real*8 Rc_type(100), Rc2_type(100), Rm_type(100),fact_grid_type(100),dR_grid1_type(100),dR_grid2_type(100)
    ! integer iflag_grid_type(100),n3b1_type(100),n3b2_type(100)

    logical*2::alive
    real*8 dwidth,ddcut
    integer max_natom
    

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
    open(10,file="input/egroup.in",status="old",action="read")
    rewind(10)
    read(10,*) dwidth
    read(10,*) ntype
    allocate(E_init(ntype))
    do i=1,ntype
    read(10,*) E_init(i)
    enddo
    close(10)



    open(10,file="input/gen_2b_feature.in",status="old",action="read")
    rewind(10)
    read(10,*) Rc_M,m_neigh
    read(10,*) ntype
    do i=1,ntype
    read(10,*) iat_type(i)
    read(10,*) Rc_type(i),Rm_type(i)
    read(10,*) 

     if(Rc_type(i).gt.Rc_M) then
      write(6,*) "Rc_type must be smaller than Rc_M, gen_3b_feature.in",i,Rc_type(i),Rc_M
      stop
     endif

    enddo
    read(10,*) E_tolerance
    read(10,*) 
    read(10,*) recalc_grid
    close(10)

    open(13,file="input/location")
    rewind(13)
    read(13,*) sys_num  !,trainSetDir
    read(13,'(a200)') trainSetDir
    ! allocate(trainSetFileDir(sys_num))
    do i=1,sys_num
    read(13,'(a200)') trainSetFileDir(i)    
    enddo
    close(13)
    trainDataDir=trim(trainSetDir)//"/Egroup_weight"
    ! inquirepos1=trim(trainSetDir)//"/inquirepos1.txt"
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
        ! dfeatDir=trim(trainSetFileDir(sys))//"/dfeat.fbin.Ftype1"
        infoDir=trim(trainSetFileDir(sys))//"/info.txt.Ftype1"
    

!cccccccccccccccccccccccccccccccccccccccccccccccccccc
    OPEN (move_file,file=MOVEMENTDir,status="old",action="read") 
    rewind(move_file)

      num_step0=0
      Etotp_ave=0.d0
1001 continue
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
1003 continue
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

!cccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
    !  open(333,file=infoDir)
    !  rewind(333)
    !  write(333,"(i4,2x,i2,3x,10(i4,1x))") nfeat0M,ntype,(nfeat0(ii),ii=1,ntype)
    !  write(333,*) natom
    ! write(333,*) iatom



      num_tot=0

    DEALLOCATE (iatom,xatom,fatom,Eatom)
     
    !  write(333,*) num_step0

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

    CALL scan_title (move_file, "ATOMIC-ENERGY",if_find=nextline)
       if(.not.nextline) then
         write(6,*) "Atomic-energy not found, stop",num_step
         stop
        endif

        backspace(move_file)
        read(move_file,*) char_tmp(1:4),Etotp

        DO j = 1, natom
            READ(move_file, *) iatom(j),Eatom(j)
        ENDDO

        write(6,"('num_step',2(i4,1x),2(E15.7,1x),i5)") num_step,natom,Etotp,Etotp-Etotp_ave,max_neigh

        if(abs(Etotp-Etotp_ave).gt.E_tolerance) then
        write(6,*) "escape this step, dE too large"
        ! write(333,*) num_step
        deallocate(iatom,xatom,fatom,Eatom)
        goto 1000
        endif

        num_step1=num_step1+1

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Finished readin the movement file.  
! fetermined the num_step1
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        ! write(6,*) '111'

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
    
    ! allocate()
    ! write(6,*) '112'

    ! allocate(nfeat_atom(natom))

    ! allocate(feat(nfeat0m,natom))
    ! allocate(dfeat(nfeat0m,natom,m_neigh,3))

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
!ccccccccccccccccccccccccccccccccccccccccccc
    !  write(6,*) '113'
    !    do i=1,natom
    !     iitype=0
    !     do itype=1,ntype
    !     if(itype_atom(itype).eq.iatom(i)) then
    !     iitype=itype
    !     endif
    !     enddo
    !     if(iitype.eq.0) then
    !     write(6,*) "this type not found", iatom(i)
    !     endif
    !     iatom_type(i)=iitype
    !   enddo   


!     call find_neighbore(iatom,natom,xatom,AL,Rc_type,num_neigh,list_neigh, &
!        dR_neigh,iat_neigh,ntype,iat_type,m_neigh,Rc_M,map2neigh_M,list_neigh_M, &
!        num_neigh_M,iat_neigh_M)
!     !    write(6,*) '114'
! !ccccccccccccccccccccccccccccccccc
! !ccccccccccccccccccccccccccccccccc

!       max_neigh=-1
!       num_neigh_alltype=0
!       max_neigh_M=-1
!       num_neigh_alltypeM=0
!       list_neigh_alltypeM=0

!       do iat=1,natom
!       list_neigh_alltype(1,iat)=iat
!       list_neigh_alltypeM(1,iat)=iat


!       num_M=1
!       do itype=1,ntype
!       do j=1,num_neigh_M(itype,iat)
!       num_M=num_M+1
!       if(num_M.gt.m_neigh) then
!       write(6,*) "Error! maxNeighborNum too small",m_neigh
!       stop
!       endif
!       list_neigh_alltypeM(num_M,iat)=list_neigh_M(j,itype,iat)
!       list_tmp(j,itype)=num_M
!       enddo
!       enddo


!       num=1
!       map2neigh_alltypeM(1,iat)=1
!       do  itype=1,ntype
!       do   j=1,num_neigh(itype,iat)
!       num=num+1
!       list_neigh_alltype(num,iat)=list_neigh(j,itype,iat)
!       map2neigh_alltypeM(num,iat)=list_tmp(map2neigh_M(j,itype,iat),itype)
! ! map2neigh_M(j,itype,iat), maps the jth neigh in list_neigh(Rc) to jth' neigh in list_neigh_M(Rc_M) 
!       enddo
!       enddo

! !ccccccccccccccccccccccccccccccccccccccc


!       num_neigh_alltype(iat)=num
!       num_neigh_alltypeM(iat)=num_M
!       if(num.gt.max_neigh) max_neigh=num
!       if(num_M.gt.max_neigh_M) max_neigh_M=num_M
!       enddo  ! iat


    !   write(6,*) '115'

      allocate(divider(natom))
      allocate(energy_group(natom))
      allocate(fact(natom,natom))
      fact=0.d0
      energy_group=0.d0
      divider=0.d0
      ddcut=-dwidth**2*log(0.01)
      do iat1=1,natom   ! center position (not even call it atom)
        Esum=0.d0
        ! num=0
        sum=0.d0
        do iat2=1,natom
        ! iat2=list_neigh_alltypeM(i,iat1)
        ! write(6,*) 'iat2 ',iat2
        ! if (iat2.gt.0) then 
        ! do iat2=1,natom
        itype=itype_atom(iat2)
 
        ! num(itype)=num(itype)+1
            dx1=xatom(1,iat2)-xatom(1,iat1)
            dx2=xatom(2,iat2)-xatom(2,iat1)
            dx3=xatom(3,iat2)-xatom(3,iat1)
            ! write(6,*) 'dx ',dx1,dx2,dx3
            if(abs(dx1+1).lt.abs(dx1)) dx1=dx1+1
            if(abs(dx1-1).lt.abs(dx1)) dx1=dx1-1
            if(abs(dx2+1).lt.abs(dx2)) dx2=dx2+1
            if(abs(dx2-1).lt.abs(dx2)) dx2=dx2-1
            if(abs(dx3+1).lt.abs(dx3)) dx3=dx3+1
            if(abs(dx3-1).lt.abs(dx3)) dx3=dx3-1
            ! write(6,*) 'dx2 ',dx1,dx2,dx3
            dx=AL(1,1)*dx1+AL(1,2)*dx2+AL(1,3)*dx3
            dy=AL(2,1)*dx1+AL(2,2)*dx2+AL(2,3)*dx3
            dz=AL(3,1)*dx1+AL(3,2)*dx2+AL(3,3)*dx3
            dd=dx**2+dy**2+dz**2
            if(dd.lt.ddcut) then
                fact(iat1,iat2)=exp(-dd/dwidth**2)
                Esum=Esum+(Eatom(iat2))*fact(iat1,iat2)
                sum=sum+fact(iat1,iat2)

            endif

        enddo

        divider(iat1) = sum
        energy_group(iat1)=Esum/sum
        ! write(6,*) 'divider ',divider(iat1)
        ! write(6,*) 'Eg ',energy_group(iat1)

        enddo
        ! write(6,*) '116'

    open(55,file=trainDataDir,position="append")
    do i=1,natom
    write(55,"(f12.7,',', f12.7,<natom>(',',f15.10))")  &
       energy_group(i),divider(i),(fact(i,j),j=1,natom)
    enddo
    close(55)


    num_tot=num_tot+natom

    deallocate(list_neigh)
    deallocate(iat_neigh)
    deallocate(dR_neigh)
    deallocate(num_neigh)
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

    deallocate(fact)
    deallocate(divider)
    deallocate(energy_group)

    ! deallocate(nfeat_atom)


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      DEALLOCATE (iatom,xatom,fatom,Eatom)
!--------------------------------------------------------
       goto 1000     
2000   continue    
      close(move_file)
    !   write(25) num_step1,num_step0
    !   write(333,*) "num_step1,num_step0",num_step1,num_step0
    !   close(333)
    !   close(25)


2333   continue

        open(11,file="output/max_natom")
        rewind(11)
        write(11,*) max_natom
        close(11)

       stop
       end
