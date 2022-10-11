PROGRAM gen_3bcos_feature
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
    integer n3bm,kkk,ii

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
    real*8 fact_grid,dR_grid1,dR_grid2

    real*8 Rc_type(50)
    integer n3b_type(50)
    real*8 eta_type(100,50),w_type(100,50),alamda_type(100,50)

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

    open(10,file="input/gen_3bcos_feature.in",status="old",action="read")
    rewind(10)
    read(10,*) Rc_M,m_neigh
    read(10,*) ntype
    do i=1,ntype
    read(10,*) iat_type(i)
    read(10,*) Rc_type(i)
       if(Rc_type(i).gt.Rc_M) then
       write(6,*) "Rc_type must be smaller than Rc_M, gen_3b_feature.in",i,Rc_type(i),Rc_M
       stop
       endif
    read(10,*) n3b_type(i)
      do j=1,n3b_type(i)
      read(10,*) eta_type(j,i),w_type(j,i),alamda_type(j,i)
      enddo
    enddo
    read(10,*) E_tolerance
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
    trainDataDir=trim(trainSetDir)//"/trainData.txt.Ftype4"
    inquirepos2=trim(trainSetDir)//"/inquirepos4.txt"
!cccccccccccccccccccccccccccccccccccccccc

     n3bm=0
     do i=1,ntype
     if(n3b_type(i).gt.n3bm) n3bm=n3b_type(i)
     enddo

     nfeat0m=0
     do itype=1,ntype
     num=0
     do itype2=1,ntype
     do itype1=1,itype2
     num=num+1
     enddo
     enddo
     nfeat0(itype)=num*n3b_type(itype)
     if(nfeat0(itype).gt.nfeat0m) nfeat0m=nfeat0(itype)
     enddo

     write(6,*) "itype,nfeat0=",(nfeat0(itype),itype=1,ntype)
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccc

    do 2333 sys=1,sys_num
        MOVEMENTDir=trim(trainSetFileDir(sys))//"/MOVEMENT"
        dfeatDir=trim(trainSetFileDir(sys))//"/dfeat.fbin.Ftype4"
        infoDir=trim(trainSetFileDir(sys))//"/info.txt.Ftype4"
    

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
        write(333,*) num_step
        deallocate(iatom,xatom,fatom,Eatom)
        goto 1000
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
      write(6,*) "Error! maxNeighborNum too small",m_neigh
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
    call find_feature_3bcos(natom,itype_atom,Rc_type,n3b_type,eta_type,w_type,alamda_type,num_neigh,  &
       list_neigh,dR_neigh,iat_neigh,ntype, &
       feat,dfeat,nfeat0m,m_neigh,n3bm,nfeat_atom)
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
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
!    write(25) nfeat_atom  ! The number of feature for this atom 

!cccccccccccccccccccccccccccccccccccccccccccccchhhhhh
!  Only output the nonzero points for dfeat
    num_tmp=0
    do jj_tmp=1,m_neigh
    do iat2=1,natom
    do ii_tmp=1,nfeat0M
    if(abs(dfeat(ii_tmp,iat2,jj_tmp,1))+abs(dfeat(ii_tmp,iat2,jj_tmp,2))+ &
         abs(dfeat(ii_tmp,iat2,jj_tmp,3)).gt.1.E-7) then
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
          abs(dfeat(ii_tmp,iat2,jj_tmp,3)).gt.1.E-7) then
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
    write(55,"(i5,',',i3,',',f12.7,',', i3,<nfeat0m>(',',E23.16))")  &
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


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      DEALLOCATE (iatom,xatom,fatom,Eatom)
!--------------------------------------------------------
       goto 1000     
2000   continue    
      close(move_file)
    !   write(25) num_step1,num_step0
    !   write(333,*) "num_step1,num_step0",num_step1,num_step0
      close(333)
      close(25)


2333   continue

       stop
       end
