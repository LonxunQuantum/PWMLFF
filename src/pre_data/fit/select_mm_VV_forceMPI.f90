program select_mm_VV_forceMPI
       implicit double precision (a-h,o-z)
       include 'mpif.h'
       integer status(MPI_STATUS_SIZE)



       integer lwork
       integer,allocatable,dimension(:) :: iatom,iatom_type,itype_atom
       real*8,allocatable,dimension(:) :: Energy,Energyt,energy_pred
       real*8,allocatable,dimension(:,:) :: force_pred
       real*8,allocatable,dimension(:,:) :: feat,feat2
       real*8,allocatable,dimension(:,:,:) :: feat_type,feat2_type
       integer,allocatable,dimension(:) :: num_neigh,num,num_atomtype
       integer,allocatable,dimension(:) :: num_neight
       integer,allocatable,dimension(:,:) :: list_neigh,ind_type

       real*8,allocatable,dimension(:,:,:,:) :: dfeat,dfeat2
       real*8,allocatable,dimension(:,:,:,:) :: dfeat_new
       real*8,allocatable,dimension(:,:,:) :: dfeat_type,dfeat2_type

       real*8,allocatable,dimension(:,:) :: AA
       real*8,allocatable,dimension(:) :: BB

       real*8,allocatable,dimension(:,:,:) :: Gfeat_type
       real*8,allocatable,dimension(:,:) :: Gfeat_tmp

       real*8,allocatable,dimension(:,:,:) :: AA_type
       real*8,allocatable,dimension(:,:) :: BB_type

       real*8,allocatable,dimension(:,:) :: SS_tmp,SS_tmp2


       real*8,allocatable,dimension(:,:,:) :: feat2_ref

       real*8,allocatable,dimension(:,:,:) :: PV
       real*8,allocatable,dimension(:,:) :: feat2_shift,feat2_scale

       integer, allocatable,dimension(:,:,:) :: idd


       real*8,allocatable,dimension(:,:) :: WW,VV,QQ
       real*8,allocatable,dimension(:,:,:,:) :: SS

       real*8,allocatable,dimension(:,:) :: Gfeat2,dGfeat2

       real*8,allocatable,dimension(:,:) :: force

     
       real*8,allocatable,dimension(:,:) :: xatom
       real*8,allocatable,dimension(:) :: rad_atom,E_ave_vdw
       real*8 AL(3,3),pi,dE,dFx,dFy,dFz,AL_tmp(3,3)

       real*8,allocatable,dimension(:,:) :: xatom_tmp

       real*8,allocatable,dimension(:,:) :: feat_new
       real*8,allocatable,dimension(:,:) :: feat_new_group,feat_new_tot
       real*8,allocatable,dimension(:,:,:) :: feat_new_type
       real*8,allocatable,dimension(:,:,:) :: feat_ext1,feat_ext2, feat_ext3,dfeat_ext1,dfeat_ext2

!       real*8 xp(5,100),xp1(10,100)
       real*8,allocatable,dimension(:,:,:,:) :: xp,xp1

 

       integer,allocatable,dimension(:) :: nfeat1,nfeat2
       integer,allocatable,dimension(:,:) :: nfeat,ipos_feat

       real*8, allocatable, dimension (:,:) :: dfeat_tmp
       real*8, allocatable, dimension (:,:) :: feat_ftype
       integer,allocatable, dimension (:) :: iat_tmp,jneigh_tmp, ifeat_tmp
       integer num_tmp,jj
       ! character(len=200) dfeat_n(400)
       character(len=200) trainSetFileDir(400)
       character(len=200) trainSetDir
       character(len=200) MOVEMENTDir,dfeatDir,infoDir,trainDataDir, MOVEMENTallDir
       integer sys_num,sys
       integer nfeat1tm(100),ifeat_type(100),nfeat1t(100)
       integer mm(100),num_ref(100),nkkk0(100),mm0(100)
       integer mm_ii(100)


       integer, allocatable, dimension (:,:) :: iflag_selected, ind_kkk_kkk0

       integer, allocatable, dimension (:,:,:) :: idd0,idd2,idd3

       integer, allocatable, dimension (:,:) :: kkk0_st,kkk0_st2
       real*8, allocatable, dimension (:,:,:) :: S1,S1_tmp
       real*8, allocatable, dimension (:,:,:,:,:) :: S2,S2_tmp
       real*8, allocatable, dimension (:,:) :: dE_term,fE_term,dE_tmp
       real*8, allocatable, dimension (:,:) :: dE_tot,fE_tot
       real*8, allocatable, dimension (:,:,:) :: dF_term,fF_term,dF_tmp
       real*8, allocatable, dimension (:,:) :: dtot_term,dtot_sum
       real*8, allocatable, dimension (:) :: dE_prev
       real*8, allocatable, dimension (:,:) :: dF_prev
       integer, allocatable, dimension (:,:) :: index,ind_select
       integer, allocatable, dimension (:,:) :: index2
       real*8, allocatable, dimension (:,:,:,:) :: S
       integer*8 inp(100)
! liuliping for relative path
       integer tmp_i
       character(len=200) fitModelDir
       character(len=:), allocatable :: fread_dfeat

       ! this file should be create by prepare.py
       open(1314, file="input/info_dir")
       rewind(1314)
       read(1314,"(A200)") fitModelDir
       close(1314)
       tmp_i = len(trim(adjustl(fitModelDir)))
       allocate(character(len=tmp_i) :: fread_dfeat)
       fread_dfeat = trim(adjustl(fitModelDir))
       write(*,*) "liuliping, fread_dfeat: ", fread_dfeat
! liuliping, end, all .r .x file should be invoke out of fread_dfeat

       
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       call mpi_init(ierr)
       call mpi_comm_rank(MPI_COMM_WORLD,inode,ierr)
       call mpi_comm_size(MPI_COMM_WORLD,nnodes,ierr)
       inode=inode+1
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       include3=0
       nloop=5
       mm3=2000/20
       mm4=mm3*10

       open(10,file=fread_dfeat//"select_VV.input")
       rewind(10) 
       read(10,*) nloop
       read(10,*) mm_tot2
       read(10,*) nimage_jump
       read(10,*) include3
       read(10,*) ndim1,ndim
       close(10)
       mm3=mm_tot2/nloop
       mm4=mm3*10

       open(10,file=fread_dfeat//"fit_linearMM.input")
       rewind(10)
       read(10,*) ntype,m_neigh
       allocate(itype_atom(ntype))
       allocate(nfeat1(ntype))
       allocate(nfeat2(ntype))
       allocate(rad_atom(ntype))
       allocate(E_ave_vdw(ntype))
       do i=1,ntype
       read(10,*) itype_atom(i)
       enddo
       read(10,*) weight_E,weight_E0,weight_F,delta
       read(10,*) dwidth
       close(10)

       open(12,file=fread_dfeat//"OUT.xp",form="unformatted")
       rewind(12)
       read(12) mfeat2,ntype_t,ndim_t,ndim1_t
       if(ntype_t.ne.ntype.or.ndim_t.ne.ndim.or.ndim1_t.ne.ndim1) then
       write(6,*) "Inconsistent ntype,ndim1,ndim from OUT.xp", ntype_t,ndim1_t,ndim_t
       stop
       endif
       read(12) nfeat2
       allocate(xp(2,ndim,mfeat2,ntype))
       allocate(xp1(2,ndim1,mfeat2,ntype))
       read(12) xp
       read(12) xp1
       close(12)
  

       

       open(10,file=fread_dfeat//"feat.info")
       rewind(10)
       read(10,*) iflag_PCA   ! this can be used to turn off degmm part
       read(10,*) nfeat_type
       do kkk=1,nfeat_type
         read(10,*) ifeat_type(kkk)   ! the index (1,2,3) of the feature type
       enddo
       read(10,*) ntype_tmp
        if(ntype_tmp.ne.ntype) then
          write(6,*) "ntype of atom not same, fit_linearMM.input, feat.info, stop"
          write(6,*) ntype,ntype_tmp
          stop
         endif
       allocate(nfeat(ntype,nfeat_type))
       allocate(ipos_feat(ntype,nfeat_type))
        do i=1,ntype
         read(10,*) iatom_tmp,nfeat1(i),nfeat2(i)   ! these nfeat1,nfeat2 include all ftype
          if(iatom_tmp.ne.itype_atom(i)) then
          write(6,*) "iatom not same, fit_linearMM.input, feat.info"
          write(6,*) iatom_tmp,itype_atom(i)
          stop
          endif
        enddo
        
        do ii=1,ntype
        read(10,*) (nfeat(ii,kkk),kkk=1,nfeat_type)
        enddo
        close(10)

!   nfeat1(ii) the total (all iftype) num of feature for iatom type ii (sum_kk nfeat(ii,kk))
!   nfeat2(ii) the total num of PCA feature for iatom type ii
        
        do ii=1,ntype
        ipos_feat(ii,1)=0
        do kkk=2,nfeat_type
        ipos_feat(ii,kkk)=ipos_feat(ii,kkk-1)+nfeat(ii,kkk-1)
        enddo
        enddo

!ccccccc Right now, nfeat1,nfeat2,for different types
!ccccccc must be the same. We will change that later, allow them 
!ccccccc to be different
       nfeat1m=0   ! the original feature
       nfeat2m=0   ! the new PCA, PV feature
       do i=1,ntype
       if(nfeat1(i).gt.nfeat1m) nfeat1m=nfeat1(i)
       if(nfeat2(i).gt.nfeat2m) nfeat2m=nfeat2(i)
       enddo


!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       allocate(PV(nfeat1m,nfeat2m,ntype))
       allocate(feat2_shift(nfeat2m,ntype))
       allocate(feat2_scale(nfeat2m,ntype))
       do itype=1,ntype
       open(11,file=fread_dfeat//"feat_PV."//char(itype+48),form="unformatted")
       rewind(11)
       read(11) nfeat1_tmp,nfeat2_tmp
       if(nfeat2_tmp.ne.nfeat2(itype)) then
       write(6,*) "nfeat2.not.same,feat2_ref",itype,nfeat2_tmp, nfeat2(itype)
       stop
       endif
       if(nfeat1_tmp.ne.nfeat1(itype)) then
       write(6,*) "nfeat1.not.same,feat2_ref",itype,nfeat1_tmp, nfeat1(itype)
       stop
       endif
       read(11) PV(1:nfeat1(itype),1:nfeat2(itype),itype)
!       read(11) ((PV(i1,i2,itype),i1=1,nfeat1(itype)),
!     &               i2=1,nfeat2(itype))
       read(11) feat2_shift(1:nfeat2(itype),itype)
       read(11) feat2_scale(1:nfeat2(itype),itype)
       close(11)
       enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccc  ind_kkk_kkk0(kkk)=kkk0
       mkkk0=0
       do itype=1,ntype
       kkk0=0
       do j1=1,nfeat2(itype)
       do id1=1,ndim1
       kkk0=kkk0+1
       enddo
       enddo

       do j1=1,nfeat2(itype)
       ndimt1=ndim
       if(j1.eq.nfeat2(itype)) ndimt1=1
       do id1=1,ndimt1
       do j2=1,j1
       ndimt2=ndim
       if(j2.eq.nfeat2(itype)) ndimt2=1
       if(j2.eq.j1) ndimt2=id1
       do id2=1,ndimt2
       kkk0=kkk0+1
       enddo
       enddo
       enddo
       enddo

       if(include3.eq.1) then
       do j1=1,nfeat2(itype)
       do j2=1,j1
       do j3=1,j2
       kkk0=kkk0+1
       enddo
       enddo
       enddo
       endif
       nkkk0(itype)=kkk0
       if(kkk0.gt.mkkk0) mkkk0=kkk0
       enddo   ! itype

       allocate(iflag_selected(mkkk0,ntype))
       allocate(ind_kkk_kkk0(mkkk0,ntype))

       allocate(num_atomtype(ntype))
       allocate(num(ntype))   ! only use this as a temp array

       ! sys_num=400
       open(13,file=fread_dfeat//"location")
       rewind(13)
       read(13,*) sys_num  !,trainSetDir
       read(13,'(a200)') trainSetDir
       ! allocate(trainSetFileDir(sys_num))
       do i=1,sys_num
       read(13,'(a200)') trainSetFileDir(i)    
       enddo
       close(13)

       ind_kkk_kkk0=0

       iflag_selected=0


       allocate(index(mm4,ntype))
       allocate(index2(mm3,ntype))

       if(inode.eq.1) then
       do itype=1,ntype
       open(12,file=fread_dfeat//"OUT.VV_index."//char(itype+48))
       rewind(12)
       write(12,*) 0
       close(12)
       enddo

       call fit_VV_force_sub(nimage_jump)
       call calc_VV_force_sub(nimage_jump,0)
       endif

       call mpi_barrier(MPI_COMM_WORLD,ierr)

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       do 5000 iloop=1,nloop

!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
        do itype=1,ntype
        open(12,file=fread_dfeat//"OUT.VV_index."//char(itype+48))
        rewind(12)
        read(12,*) mm0(itype)   ! the number of new features
        close(12)
        enddo

        mm_max=0
        do itype=1,ntype
        if(mm0(itype).gt.mm_max) mm_max=mm0(itype)
        enddo

        if(mm_max.eq.0) then
        allocate(idd0(0:4,1,ntype))
        allocate(kkk0_st(1,ntype))
        else
        allocate(idd0(0:4,mm_max,ntype))
        allocate(kkk0_st(mm_max,ntype))
        endif

        do itype=1,ntype
        open(12,file=fread_dfeat//"OUT.VV_index."//char(itype+48))
        rewind(12)
        read(12,*) mm0(itype)   ! the number of new features
        do ii=1,mm0(itype)
        read(12,*) (idd0(jj,ii,itype),jj=0,4),kkk0_st(ii,itype)
        iflag_selected(kkk0_st(ii,itype),itype)=1   ! this kkk0 has already been used
        enddo
        close(12)
        enddo

        do itype=1,ntype
        mm(itype)=mm0(itype)
        enddo


       allocate(idd(0:4,mkkk0,ntype))    ! this is for this time
       idd=0
!cccccc  ind_kkk_kkk0(kkk)=kkk0
!cccccc  ind_kkk_kkk0(kkk)=kkk0
       do itype=1,ntype
       kkk0=0
       kkk=0
       do j1=1,nfeat2(itype)
       do id1=1,ndim1
       kkk0=kkk0+1       ! this is the original index
       if(iflag_selected(kkk0,itype).eq.0) then   ! This one has not been used yet  
       kkk=kkk+1
       ind_kkk_kkk0(kkk,itype)=kkk0
       idd(0,kkk,itype)=1
       idd(1,kkk,itype)=j1
       idd(2,kkk,itype)=id1
       endif
       enddo
       enddo

       do j1=1,nfeat2(itype)
       ndimt1=ndim
       if(j1.eq.nfeat2(itype)) ndimt1=1
       do id1=1,ndimt1
       do j2=1,j1
       ndimt2=ndim
       if(j2.eq.nfeat2(itype)) ndimt2=1
       if(j2.eq.j1) ndimt2=id1
       do id2=1,ndimt2
       kkk0=kkk0+1
       if(iflag_selected(kkk0,itype).eq.0) then
       kkk=kkk+1
       ind_kkk_kkk0(kkk,itype)=kkk0
       idd(0,kkk,itype)=2
       idd(1,kkk,itype)=j1
       idd(2,kkk,itype)=id1
       idd(3,kkk,itype)=j2
       idd(4,kkk,itype)=id2
       endif
       enddo
       enddo
       enddo
       enddo

       if(include3.eq.1) then
       do j1=1,nfeat2(itype)
       do j2=1,j1
       do j3=1,j2
       kkk0=kkk0+1
       if(iflag_selected(kkk0,itype).eq.0) then
       kkk=kkk+1
       ind_kkk_kkk0(kkk,itype)=kkk0
       idd(0,kkk,itype)=3
       idd(1,kkk,itype)=j1
       idd(2,kkk,itype)=j2
       idd(3,kkk,itype)=j3
       endif
       enddo
       enddo
       enddo
       endif
       mm(itype)=kkk    ! the number of possibke new features for this iloop iteration
       enddo !itype
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cc In the following, we will select mm3 feature from mm(itype)


!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!CC note, the mm(itype) will change from iloop2=1, to iloop2=2
!c for iloop2=2, mm already changed to mm4, and idd has been changed as the selected mm4 features

       do 7000 iloop2=1,2   

!ccccccccccccccccccccccccccccccccccccccccc
       open(70,file=fread_dfeat//"energyVV.pred.tot")
       rewind(70)

       open(90,file=fread_dfeat//"energyVV.pred.group")  ! fitting result from last iteration
       rewind(90)
 
       do itype=1,ntype
       open(40+itype,file=fread_dfeat//"forceVV.pred."//char(itype+48))
       rewind(40+itype)
       enddo

! first calculate dE_term, second, calculate SS (on a much smaller number of mm(itype))
! iloop2=1 is a very large mm(itype)
! iloop2=2, is a smaller mm4 feature, we will calculate SS for iloop2

        mm_max=0
        do itype=1,ntype
        if(mm(itype).gt.mm_max) mm_max=mm(itype)
        enddo

        mm_tot=0
        mm_ii=0
        do itype=1,ntype
        mm_tot=mm_tot+mm(itype)
        enddo

        do itype=2,ntype
        mm_ii(itype)=mm_ii(itype-1)+mm(itype-1)
        enddo
       

!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccc
        num_refm=0
        do itype=1,ntype
        num_ref(itype)=mm(itype)
!cccc in this code, num_ref is just mm, the number of feature
        if(num_ref(itype).gt.num_refm) num_refm=num_ref(itype)
        enddo

        write(6,*) "nfeat_type",(num_ref(itype),itype=1,ntype)

        if(iloop2.eq.2) then
        allocate(S1(mm_max,mm_max,ntype))
        allocate(S2(mm_max,mm_max,3,ntype,ntype))
        S1=0.d0
        S2=0.d0
        endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccc
!   The above are headers
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc


       ! MOVEMENTallDir=trim(trainSetDir)//"/MOVEMENT"
       ! trainDataDir=trim(trainSetDir)//"/trainData.txt"
!ccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       allocate(dE_term(num_refm,ntype))
       allocate(dE_tot(num_refm,ntype))
       allocate(dF_term(num_refm,3,ntype))
       allocate(fE_term(num_refm,ntype))
       allocate(fE_tot(num_refm,ntype))
       allocate(fF_term(num_refm,3,ntype))
       allocate(dtot_term(num_refm,ntype))
       allocate(dtot_sum(num_refm,ntype))

       dE_term=0.d0   ! the major quantity
       dF_term=0.d0
       dE_tot=0.d0   ! the major quantity
       fE_tot=0.d0
       fE_term=0.d0
       fF_term=0.d0
       dtot_term=0.d0
       dtot_sum=0.d0
!cccccccccccccccccccccccccccccccccccccccccccccc
     

       do 900 sys=1,sys_num

       do 777 kkk=1,nfeat_type
       ! MOVEMENTDir=trim(trainSetFileDir(sys))//"/MOVEMENT"
       dfeatDir=trim(trainSetFileDir(sys))//"/dfeat.fbin.Ftype" //char(ifeat_type(kkk)+48)
       open(1000+kkk,file=dfeatDir,action="read",access="stream", form="unformatted")
       rewind(1000+kkk)
       read(1000+kkk) nimaget,natomt,nfeat1tm(kkk), m_neight

!      nfeat1tm(kkk) is the max(nfeat(ii,kkk)) for all ii(iatype)
  
       if(kkk.eq.1) then
       nimage=nimaget
       natom=natomt
       m_neigh=m_neight
       else
       if(nimaget.ne.nimage.or.natomt.ne.natom.or. m_neight.ne.m_neigh) then
       write(6,*) "param changed in diff ifeat_type"
       write(6,*) nimage,natom,m_neigh
       write(6,*) nimaget,natomt,m_neight
       stop
       endif
       endif


       read(1000+kkk) ntype_tmp,(nfeat1t(ii),ii=1,ntype_tmp)
!    This is one etra line, perhaps we don't need it

!! for this kkk_ftype, for each atom type, ii, the num of feature is nfeat1t(ii)
!cccccccccccccccccccccccccccccccccccccccccccccccc
       if(ntype_tmp.ne.ntype) then
       write(6,*) "ntype_tmp.ne.ntype,dfeat.fbin,stop"
       write(6,*) ntype_tmp,ntype
       stop
       endif

        do ii=1,ntype
        if(nfeat1t(ii).ne.nfeat(ii,kkk)) then   ! the num of feat for ii_th iatype, and kkk_th feat type
        write(6,*) "nfeat1t not the same, dfeat.fbin,stop"
        write(6,*) nfeat1t(ii),nfeat(ii,kkk),ii,kkk
        stop
        endif
        enddo

       if(kkk.eq.1) then
         if(sys.ne.1.or.iloop.gt.1.or.iloop2.gt.1) then
         deallocate(iatom)
         endif
       allocate(iatom(natom))
       endif
       read(1000+kkk) iatom      ! The same for different kkk

777    continue
!ccccccccccccccccccccccccccccccccccccccccccccccccc
       
           
       if (sys.ne.1.or.iloop.gt.1.or.iloop2.gt.1) then      

       deallocate(iatom_type)
       deallocate(Energy)
       deallocate(Energyt)
       deallocate(num_neight)
       deallocate(feat)
       deallocate(feat2)
       deallocate(feat_type)
       deallocate(feat2_type)
       deallocate(num_neigh)
       deallocate(list_neigh)
       deallocate(ind_type)
       deallocate(dfeat)
       deallocate(dfeat_type)
       deallocate(dfeat2_type)
       deallocate(dfeat2)
       deallocate(xatom)

       deallocate(force)
       deallocate(SS)

       deallocate(energy_pred)
       deallocate(force_pred)

       endif

!cccccccccccccccccccccccccccccccccccccccccccccccc

       allocate(iatom_type(natom))
       allocate(Energy(natom))
       allocate(Energyt(natom))
       allocate(num_neight(natom))
       allocate(feat(nfeat1m,natom))  
! nfeat1m is the max(nfeat1(ii)) for ii(iatype), nfeat1(ii)=sum_kkk nfeat(ii,kkk)
! nfeat1m is the max num of total feature (sum over all feature type)
       allocate(feat2(nfeat2m,natom))
       allocate(feat_type(nfeat1m,natom,ntype))
       allocate(feat2_type(nfeat2m,natom,ntype))
       allocate(num_neigh(natom))
       allocate(list_neigh(m_neigh,natom))
       allocate(ind_type(natom,ntype))
       allocate(dfeat(nfeat1m,natom,m_neigh,3))
       allocate(dfeat_type(nfeat1m,natom*m_neigh*3,ntype))
       allocate(dfeat2_type(nfeat2m,natom*m_neigh*3,ntype))
       allocate(dfeat2(nfeat2m,natom,m_neigh,3))
       allocate(xatom(3,natom))
       allocate(energy_pred(natom))
       allocate(force_pred(3,natom))

       dfeat2=0.d0

       dfeat=0.d0
       dfeat_type=0.d0

       allocate(force(3,natom))
       allocate(SS(num_refm,natom,3,ntype))



       pi=4*datan(1.d0)


       do i=1,natom
        iitype=0
        do itype=1,ntype
        if(itype_atom(itype).eq.iatom(i)) then
        iitype=itype
        endif
        enddo
        if(iitype.eq.0) then
        write(6,*) "this type not found", iatom(i)
        endif
        iatom_type(i)=iitype
       enddo


       num_atomtype=0
       do i=1,natom
       itype=iatom_type(i)
       num_atomtype(itype)=num_atomtype(itype)+1
       enddo



!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       allocate(dE_prev(natom))
       allocate(dF_prev(natom,3))


       do 3000 image=1,nimage/nimage_jump

! the stream access
       do kkk=1,nfeat_type
       inquire(1000+kkk,pos=inp(kkk))
       enddo
!  This inp(kkk) alway at the position before the next energy

     


       if(mod(image-1,nnodes).eq.inode-1) then
       write(6,"('iloop,iloop2,sys,image=',4(i6,1x))") iloop,iloop2,sys,image
       endif

       read(70,*) energy_tot,energy_tot_pred

       dE_prev_tot=energy_tot-energy_tot_pred

       do i=1,natom
        itype=iatom_type(i)
!        read(20+itype,*) energyt1,energy_predt1
!        dE_prev(i)=energyt1-energy_predt1
        read(90,*) energyt1,energy_predt1
        dE_prev(i)=energyt1-energy_predt1    ! this is group energy, not atom energy
        read(40+itype,*) forcet,force_predt
        dF_prev(i,1)=forcet-force_predt
        read(40+itype,*) forcet,force_predt
        dF_prev(i,2)=forcet-force_predt
        read(40+itype,*) forcet,force_predt
        dF_prev(i,3)=forcet-force_predt
       enddo
!-----------------------------------------------------------------
!-----------------------------------------------------------------
!---- read in the feature from different kkk, and put them together
!-----------------------------------------------------------------
       dfeat(:,:,:,:)=0.0
       feat(:,:)=0.0


       do jump=1,nimage_jump-1
       do kkk=1,nfeat_type

       inp(kkk)=inp(kkk)+(4*natom+nfeat1tm(kkk)*natom)*8+ (natom+m_neigh*natom)*4    ! jump from eenrgy to num_tmp
       read(1000+kkk,pos=inp(kkk)) num_tmp
       inp(kkk)=inp(kkk)+4+3*num_tmp*4+3*num_tmp*8+3*natom*8+9*8

!       read(1000+kkk)  ! energy   ! repeated
!       read(1000+kkk)  ! force    ! repeated
!       read(1000+kkk)  ! feat_ftype
!       read(1000+kkk)  ! num_neigh     ! this is actually the num_neighM (of Rc_M)
!       read(1000+kkk)  ! list_neigh    ! this is actually the list_neighM (of Rc_M)
!       read(1000+kkk)  ! num_tmp
!       read(1000+kkk)  ! iat_tmp
!       read(1000+kkk)  ! jneigh_tmp
!       read(1000+kkk)  ! ifeat_tmp
!       read(1000+kkk)  ! dfeat_tmp
!       read(1000+kkk)  ! xatom    ! xatom(3,natom), repeated for diff kkk
!       read(1000+kkk)  ! AL       ! AL(3,3), repeated for diff kkk
       enddo
       enddo






       do 778 kkk=1,nfeat_type

       allocate(feat_ftype(nfeat1tm(kkk),natom))
       read(1000+kkk,pos=inp(kkk)) energy   ! repeated
       read(1000+kkk) force    ! repeated
       read(1000+kkk) feat_ftype

       if(kkk.eq.1) then
       energyt=energy
       else
        diff=0.d0
        do ii=1,natom
        diff=diff+abs(energyt(ii)-energy(ii))
        enddo
        if(diff.gt.1.E-9) then
        write(6,*) "energy Ei not the same for diff ifeature type, stop"
        stop
        endif
       endif


       do iat=1,natom
       itype=iatom_type(iat)
       do ii=1,nfeat(itype,kkk)
       feat(ii+ipos_feat(itype,kkk),iat)=feat_ftype(ii,iat)   ! put different kkk together
       enddo
       enddo
       deallocate(feat_ftype)
       
       read(1000+kkk) num_neigh     ! this is actually the num_neighM (of Rc_M)
       read(1000+kkk) list_neigh    ! this is actually the list_neighM (of Rc_M)
!    the above should be the same for different kkk. 
!    Perhaps we should check it later. Here we proceed without checking 
        if(kkk.eq.1) then
        num_neight=num_neigh
        else
         diff=0.d0
         do ii=1,natom
         diff=diff+abs(num_neight(ii)-num_neigh(ii))
         enddo
         if(diff.gt.1.E-9) then
         write(6,*) "num_neigh not the same for diff ifeature type,stop"
         stop
         endif
        endif


!TODO:
       ! read(10) dfeat
       read(1000+kkk) num_tmp
       allocate(dfeat_tmp(3,num_tmp))
       allocate(iat_tmp(num_tmp))
       allocate(jneigh_tmp(num_tmp))
       allocate(ifeat_tmp(num_tmp))
       read(1000+kkk) iat_tmp
       read(1000+kkk) jneigh_tmp
       read(1000+kkk) ifeat_tmp
       read(1000+kkk) dfeat_tmp
       
       read(1000+kkk) xatom    ! xatom(3,natom), repeated for diff kkk
       read(1000+kkk) AL       ! AL(3,3), repeated for diff kkk

       do jj=1,num_tmp


       itype2=iatom_type(list_neigh(jneigh_tmp(jj),iat_tmp(jj))) ! itype2: the type of the neighbor
       dfeat(ifeat_tmp(jj)+ipos_feat(itype2,kkk), iat_tmp(jj),jneigh_tmp(jj),:) =dfeat_tmp(:,jj)
!  Place dfeat from different iftype into the same dfeat
       enddo
       deallocate(dfeat_tmp)
       deallocate(iat_tmp)
       deallocate(jneigh_tmp)
       deallocate(ifeat_tmp)

778    continue
!ccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccc  Big laber division
       if(mod(image-1,nnodes).ne.inode-1) goto 4444
!ccccccccccccccccccccccccccccccccccccccccccccccccc
       
       num=0
       do i=1,natom
       itype=iatom_type(i)
       num(itype)=num(itype)+1
       ind_type(num(itype),itype)=i
       feat_type(:,num(itype),itype)=feat(:,i)
!  we have to seperate the feature into different iatype, since they have different PV
!  The num of total feature for different iatype is nfeat1(iatype)
       enddo
!cccccccccccccccccccccccccccccccccccccccccc


       do itype=1,ntype
       call dgemm('T','N',nfeat2(itype),num_atomtype(itype), nfeat1(itype),1.d0,PV(1,1,itype), nfeat1m,feat_type(1,1,itype),nfeat1m,0.d0,feat2_type(1,1,itype), nfeat2m)
       enddo


       do itype=1,ntype
       do i=1,num_atomtype(itype)
       do j=1,nfeat2(itype)-1
       feat2_type(j,i,itype)=(feat2_type(j,i,itype)- feat2_shift(j,itype))*feat2_scale(j,itype)
       enddo
       feat2_type(nfeat2(itype),i,itype)=1.d0   ! the special value 1 component 
       enddo
       enddo  ! itype

       num=0
       do i=1,natom
       itype=iatom_type(i)
       num(itype)=num(itype)+1
       feat2(:,i)=feat2_type(:,num(itype),itype)
!  Here, we collect different iatype back, into feat2(:,i)
!  But actually, different atom (different iatype) will have different number of features
!  But all stored within nfeat2m
       enddo


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cc dfeat(nfeat1,natom,j_neigh,3): dfeat(j,i,jj,3)= d/dR_i(feat(j,list_neigh(jj,i))
!cccccccccccc


       num=0

       do i=1,natom
       do jj=1,num_neigh(i)

       itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
       num(itype)=num(itype)+1
       dfeat_type(:,num(itype),itype)=dfeat(:,i,jj,1)
       num(itype)=num(itype)+1
       dfeat_type(:,num(itype),itype)=dfeat(:,i,jj,2)
       num(itype)=num(itype)+1
       dfeat_type(:,num(itype),itype)=dfeat(:,i,jj,3)
       enddo
       enddo
!cccccccc Note: num(itype) is rather large, in the scane of natom*num_neigh
    

!cccccccccccccccccccccccccccccccccccccccc


       do itype=1,ntype
! bug bug bug (check all the other code!)
!       call dgemm('T','N',nfeat2(itype),num_atomtype(itype),
       call dgemm('T','N',nfeat2(itype),num(itype), nfeat1(itype),1.d0,PV(1,1,itype), nfeat1m,dfeat_type(1,1,itype),nfeat1m,0.d0, dfeat2_type(1,1,itype), nfeat2m)
       enddo
!cccccccccccccccccccccccccccccccccccccccc




       num=0
       do i=1,natom
       do jj=1,num_neigh(i)
       itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
       num(itype)=num(itype)+1
       do j=1,nfeat2(itype)-1
       dfeat2(j,i,jj,1)=dfeat2_type(j,num(itype),itype)* feat2_scale(j,itype)
       enddo
       dfeat2(nfeat2(itype),i,jj,1)=0.d0
       num(itype)=num(itype)+1
       do j=1,nfeat2(itype)-1
       dfeat2(j,i,jj,2)=dfeat2_type(j,num(itype),itype)* feat2_scale(j,itype)
       enddo
       dfeat2(nfeat2(itype),i,jj,2)=0.d0
       num(itype)=num(itype)+1
       do j=1,nfeat2(itype)-1
       dfeat2(j,i,jj,3)=dfeat2_type(j,num(itype),itype)* feat2_scale(j,itype)
       enddo
       dfeat2(nfeat2(itype),i,jj,3)=0.d0
       enddo
       enddo



!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc

!ccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cc  Now, we have yield the new feature: feat2_type(nfeat2,num(itype),itype)
!ccc The last feature, nfeat2, =1
!ccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccc We will generate the mm VV feature from these features
!ccccccccccccccccccccccccccccccccccccccccccccccc

       num_refm=mm_max


       allocate(feat_new(num_refm,natom))
       allocate(feat_new_tot(num_refm,ntype))
       allocate(feat_new_group(mm_tot,natom))
       allocate(feat_new_type(num_refm,natom,ntype))
       allocate(feat_ext1(natom,nfeat2m,ndim1))
       allocate(dfeat_ext1(natom,nfeat2m,ndim1))
       allocate(feat_ext2(natom,nfeat2m,ndim))
       allocate(dfeat_ext2(natom,nfeat2m,ndim))
       allocate(feat_ext3(natom,nfeat2m,1))
       allocate(dfeat_new(num_refm,natom,m_neigh,3))


!ccccccccccccccccccccccccccccccccccccccccccc
!ccc The old one
!       xp(1,1)=-3.9
!       xp(2,1)=2.6
!       xp(1,2)=-1.3
!       xp(2,2)=2.6
!       xp(1,3)=1.3
!       xp(2,3)=2.6
!       xp(1,4)=3.9
!       xp(2,4)=2.6
!       do id1=1,ndim1
!       xp1(1,id1)=-(id1-ndim1/2)*3.0/ndim1
!       xp1(2,id1)=3.d0/ndim1
!       enddo
!cccccccccccccccccccccccccccccccc

        do i=1,natom
        itype=iatom_type(i)
        do id=1,ndim1
        do j=1,nfeat2m
        feat_ext1(i,j,id)=exp(-((feat2(j,i)-xp1(1,id,j,itype)) /xp1(2,id,j,itype))**2)
        dfeat_ext1(i,j,id)=-feat_ext1(i,j,id)*2* (feat2(j,i)-xp1(1,id,j,itype)) /xp1(2,id,j,itype)**2 
        enddo
        enddo
        enddo

        do i=1,natom
        itype=iatom_type(i)
        do id=1,ndim
        do j=1,nfeat2m
        feat_ext2(i,j,id)=exp(-((feat2(j,i)-xp(1,id,j,itype)) /xp(2,id,j,itype))**2)
        dfeat_ext2(i,j,id)=-feat_ext2(i,j,id)*2* (feat2(j,i)-xp(1,id,j,itype))/xp(2,id,j,itype)**2

        enddo
        enddo
        enddo

        do i=1,natom
        do j=1,nfeat2m
        feat_ext3(i,j,1)=feat2(j,i)
        enddo
        enddo
!cccccccccccccccccccccccccccccccc

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!       do i=1,natom
!       itype=iatom_type(i)
!       do iii=1,nfeat2(itype)
!       feat_new(iii,i)=feat2(iii,i)
!       enddo
!       enddo

       do i=1,natom
       itype=iatom_type(i)
       do kkk=1,mm(itype)

       if(idd(0,kkk,itype).eq.1) then
       j1=idd(1,kkk,itype)
       id1=idd(2,kkk,itype)
!       feat_new(kkk+nfeat2(itype),i)=feat_ext1(i,j1,id1)
       feat_new(kkk,i)=feat_ext1(i,j1,id1)   ! same below STOPPED
       elseif(idd(0,kkk,itype).eq.2) then
       j1=idd(1,kkk,itype)
       id1=idd(2,kkk,itype)
       j2=idd(3,kkk,itype)
       id2=idd(4,kkk,itype)
!       feat_new(kkk+nfeat2(itype),i)=feat_ext2(i,j1,id1)*
       feat_new(kkk,i)=feat_ext2(i,j1,id1)* feat_ext2(i,j2,id2)
       elseif(idd(0,kkk,itype).eq.3) then
       j1=idd(1,kkk,itype)
       j2=idd(2,kkk,itype)
       j3=idd(3,kkk,itype)
!       feat_new(kkk+nfeat2(itype),i)=feat_ext3(i,j1,1)*
       feat_new(kkk,i)=feat_ext3(i,j1,1)* feat_ext3(i,j2,1)*feat_ext3(i,j3,1)
       endif
       enddo
       enddo
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       do itype=1,ntype
       num_ref(itype)=mm(itype)
       mfeat=num_ref(itype)
       num(itype)=0
       do i=1,natom
       if(itype.eq.iatom_type(i)) then
       num(itype)=num(itype)+1
       feat_new_type(1:mfeat,num(itype),itype)=feat_new(1:mfeat,i)
       endif
       enddo
       enddo   ! itype

        ddcut=-dwidth**2*log(0.01)

        feat_new_group=0.d0
        do iat1=1,natom   ! center position (not even call it atom)
        num=0
        sum=0.d0
        do iat2=1,natom
        itype=iatom_type(iat2)
        iii=mm_ii(itype)

        num(itype)=num(itype)+1
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
        if(dd.lt.ddcut) then
        fact=exp(-dd/dwidth**2)
!ccccccccc Might want to store the fact(iat1,iat2), then do a dgmm to improve the 
! speed of this step. 
        sum=sum+fact
        do j=1,mm(itype)
        feat_new_group(j+iii,iat1)=feat_new_group(j+iii,iat1)+ feat_new(j,iat2)*fact
        enddo
        endif

        enddo
        feat_new_group(:,iat1)=feat_new_group(:,iat1)/sum
        enddo
!ccccccccccccccccccccccccccccccccccccccccccccc
        
        feat_new_tot=0.d0
        do iat1=1,natom
        itype=iatom_type(iat1)
        do j=1,mm(itype)
        feat_new_tot(j,itype)=feat_new_tot(j,itype)+feat_new(j,iat1)
        enddo
        enddo

        

       do itype=1,ntype
       iii=mm_ii(itype)
       do j=1,num_ref(itype)
       sum=0.d0
       sum1=0.d0
       do i=1,natom    ! this is actually group
       sum=sum+dE_prev(i)*feat_new_group(j+iii,i)
       sum1=sum1+feat_new_group(j+iii,i)**2
       enddo
       dE_term(j,itype)=dE_term(j,itype)+sum
       fE_term(j,itype)=fE_term(j,itype)+sum1
       dE_tot(j,itype)=dE_tot(j,itype)+dE_prev_tot*feat_new_tot(j,itype)
       dE_tot(j,itype)=dE_tot(j,itype)+feat_new_tot(j,itype)**2
       enddo
       enddo
       


!       do i=1,natom
!       itype=iatom_type(i)
!       do j=1,num_ref(itype)
!       dE_term(j,itype)=dE_term(j,itype)+dE_prev(i)*feat_new(j,i)
!       fE_term(j,itype)=fE_term(j,itype)+feat_new(j,i)**2
!       enddo
!       enddo


       if(iloop2.eq.2) then

       do itype=1,ntype
       call dgemm('N','T',mm_max,mm_max,num_atomtype(itype),1.d0, feat_new_type(1,1,itype),mm_max,feat_new_type(1,1,itype), mm_max,1.d0, S1(1,1,itype),mm_max)
       enddo

       endif


!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccc Now, we have finished the energy part. In the following, we will 
!ccc include the force part. Which is more complicated. 
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cc  Now, dfeat2 is: 
!cc dfeat2(nfeat2,natom,j_neigh,3): dfeat2(j,i,jj,3)= d/dR_i(feat2(j,list_neigh(jj,i))
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       do i=1,natom
       do jj=1,num_neigh(i)
       itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
       ii=list_neigh(jj,i)  ! the atom index of the neigh

! Note, list_neigh is for the Rc_m, will there be any problem? !
! ----------------------------
! I will assume, beyond the actual neigh, everything is zero

!       do iii=1,nfeat2(itype)
!       dfeat_new(iii,i,jj,:)=dfeat2(iii,i,jj,:)
!       enddo

       do kkk=1,mm(itype)
       if(idd(0,kkk,itype).eq.1) then
       j1=idd(1,kkk,itype)
       id1=idd(2,kkk,itype)
!       dfeat_new(kkk+nfeat2(itype),i,jj,:)=dfeat_ext1(ii,j1,id1)*
       dfeat_new(kkk,i,jj,:)=dfeat_ext1(ii,j1,id1)* dfeat2(j1,i,jj,:)

       elseif(idd(0,kkk,itype).eq.2) then
       j1=idd(1,kkk,itype)
       id1=idd(2,kkk,itype)
       j2=idd(3,kkk,itype)
       id2=idd(4,kkk,itype)
!       dfeat_new(kkk+nfeat2(itype),i,jj,:)=dfeat_ext2(ii,j1,id1)*
       dfeat_new(kkk,i,jj,:)=dfeat_ext2(ii,j1,id1)* feat_ext2(ii,j2,id2)*dfeat2(j1,i,jj,:)+feat_ext2(ii,j1,id1) *dfeat_ext2(ii,j2,id2)*dfeat2(j2,i,jj,:)


       elseif(idd(0,kkk,itype).eq.3) then
       j1=idd(1,kkk,itype)
       j2=idd(2,kkk,itype)
       j3=idd(3,kkk,itype)
!       dfeat_new(kkk+nfeat2(itype),i,jj,:)=dfeat2(j1,i,jj,:)*
       dfeat_new(kkk,i,jj,:)=dfeat2(j1,i,jj,:)* feat_ext3(ii,j2,1)*feat_ext3(ii,j3,1)+ feat_ext3(ii,j1,1)*dfeat2(j2,i,jj,:)*feat_ext3(ii,j3,1)+ feat_ext3(ii,j1,1)*feat_ext3(ii,j2,1)*dfeat(j3,i,jj,:)

       endif
       enddo
       enddo
       enddo

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       SS=0.d0

       do i=1,natom
       do jj=1,num_neigh(i)
       itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
       do j=1,num_ref(itype)

       SS(j,i,1,itype)=SS(j,i,1,itype)+dfeat_new(j,i,jj,1)
       SS(j,i,2,itype)=SS(j,i,2,itype)+dfeat_new(j,i,jj,2)
       SS(j,i,3,itype)=SS(j,i,3,itype)+dfeat_new(j,i,jj,3)
       enddo
       enddo
       enddo
     
       do ixyz=1,3
       do itype=1,ntype
       do j=1,num_ref(itype)
       do i=1,natom
       dF_term(j,ixyz,itype)=dF_term(j,ixyz,itype)+dF_prev(i,ixyz)* SS(j,i,ixyz,itype)
       fF_term(j,ixyz,itype)=fF_term(j,ixyz,itype)+SS(j,i,ixyz,itype)**2
       enddo
       enddo
       enddo
       enddo

       if(iloop2.eq.2) then
! S2 is not correct,  need to use NfeatNtot !
       do itype1=1,ntype
       do itype2=1,ntype
       do ixyz=1,3
       call dgemm('N','T',mm_max,mm_max,natom,1.d0, SS(1,1,ixyz,itype1),mm_max,SS(1,1,ixyz,itype2),mm_max,1.d0, S2(1,1,ixyz,itype1,itype2),mm_max)
       enddo
       enddo
       enddo
       endif
       

       deallocate(feat_new)
       deallocate(feat_new_type)
       deallocate(feat_new_tot)
       deallocate(feat_new_group)
       deallocate(feat_ext1)
       deallocate(dfeat_ext1)
       deallocate(feat_ext2)
       deallocate(dfeat_ext2)
       deallocate(feat_ext3)
       deallocate(dfeat_new)


4444   continue
3000   continue   ! image
       deallocate(dE_prev)
       deallocate(dF_prev)


       do kkk=1,nfeat_type
       close(1000+kkk)     ! dfeat.fbin.Ftype
       enddo

900    continue   ! system

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccc Now, select mm4 from the mm(itype)
       call mpi_barrier(MPI_COMM_WORLD,ierr)
       allocate(dE_tmp(num_refm,ntype))
       allocate(dF_tmp(num_refm,3,ntype))
        call mpi_allreduce(dE_term,dE_tmp,num_refm*ntype,MPI_REAL8, MPI_SUM,MPI_COMM_WORLD,ierr)
        dE_term=dE_tmp
        call mpi_allreduce(fE_term,dE_tmp,num_refm*ntype,MPI_REAL8, MPI_SUM,MPI_COMM_WORLD,ierr)
        fE_term=dE_tmp
        call mpi_allreduce(dE_tot,dE_tmp,num_refm*ntype,MPI_REAL8, MPI_SUM,MPI_COMM_WORLD,ierr)
        dE_tot=dE_tmp
        call mpi_allreduce(fE_tot,dE_tmp,num_refm*ntype,MPI_REAL8, MPI_SUM,MPI_COMM_WORLD,ierr)
        fE_tot=dE_tmp
        call mpi_allreduce(dF_term,dF_tmp,3*num_refm*ntype,MPI_REAL8, MPI_SUM,MPI_COMM_WORLD,ierr)
        dF_term=dF_tmp
        call mpi_allreduce(fF_term,dF_tmp,3*num_refm*ntype,MPI_REAL8, MPI_SUM,MPI_COMM_WORLD,ierr)
        fF_term=dF_tmp
       deallocate(dE_tmp)
       deallocate(dF_tmp)


       do itype=1,ntype

       do kkk=1,mm(itype)
!       (kkk,itype)=dE_term(kkk,itype)**2/fE_term(kkk,itype)
!     &      *weight_E+(dF_term(kkk,1,itype)**2+dF_term(kkk,2,itype)**2
!     &     +dF_term(kkk,3,itype)**2)/fF_term(kkk,itype)*weight_F
       dtot_term(kkk,itype)=dE_term(kkk,itype)*weight_E+ (dF_term(kkk,1,itype)+dF_term(kkk,2,itype) +dF_term(kkk,3,itype))*weight_F +dE_tot(kkk,itype)*weight_E0
       dtot_sum(kkk,itype)=fE_term(kkk,itype)*weight_E+ +(fF_term(kkk,1,itype)+fF_term(kkk,2,itype)+ +fF_term(kkk,3,itype))*weight_F+ +fE_tot(kkk,itype)*weight_E0
       dtot_sum(kkk,itype)=abs(dtot_term(kkk,itype)**2/ dtot_sum(kkk,itype))
       enddo
       enddo

      
        if(iloop2.eq.1) then 
!    select mm4 from the large number of features
       do itype=1,ntype        
        do iii=1,mm4    ! preselect mm4 features, index index(iii)
        dE_max=0.d0
        do kkk=1,mm(itype)
        if(dtot_sum(kkk,itype).gt.dE_max) then
        dE_max=dtot_sum(kkk,itype)
        kkkm=kkk
        endif
        enddo
        index(iii,itype)=kkkm
        dtot_sum(kkkm,itype)=0.d0
        enddo
       enddo    ! itype

       allocate(idd2(0:4,mkkk0,ntype))
       idd2=idd  ! idd2 is the original (large,80,000) mkkk0 index
       idd=0
       do itype=1,ntype
       do iii=1,mm4
       kkk=index(iii,itype)
       idd(:,iii,itype)=idd2(:,kkk,itype)   ! now, idd is kkk0
       enddo
       mm(itype)=mm4
       enddo

       endif   ! iloop2.eq.1
!cccccccccccccccccccccccccccccccccccccccccccccccj

       if(iloop2.eq.2) then
!ccccc select mm3 from mm4 features
!  use S matrix

       call mpi_barrier(MPI_COMM_WORLD,ierr)
        allocate(S1_tmp(mm_max,mm_max,ntype))
        allocate(S2_tmp(mm_max,mm_max,3,ntype,ntype))
        call mpi_allreduce(S1,S1_tmp,mm_max**2*ntype,MPI_REAL8, MPI_SUM,MPI_COMM_WORLD,ierr)
         S1=S1_tmp
        call mpi_allreduce(S2,S2_tmp,3*mm_max**2*ntype**2,MPI_REAL8, MPI_SUM,MPI_COMM_WORLD,ierr)
         S2=S2_tmp
         deallocate(S1_tmp)
         deallocate(S2_tmp)


       allocate(S(mm4,mm4,ntype,ntype))
       allocate(ind_select(mm3,ntype))
       allocate(idd3(0:4,mm_max+mm3,ntype))
       allocate(kkk0_st2(mm_max+mm3,ntype))

       S=0.d0

       do itype=1,ntype
       S(:,:,itype,itype)=S1(:,:,itype)*weight_E
       enddo

       do itype1=1,ntype
       do itype2=1,ntype
       S(:,:,itype1,itype2)=S(:,:,itype1,itype2)+ (S2(:,:,1,itype1,itype2)+S2(:,:,2,itype1,itype2)+ S2(:,:,3,itype1,itype2))*weight_F
       enddo
       enddo


       do jjj=1,mm3     ! select mm3 biggest component features
! We choose the feature, alternatively, from different ntype, one at a time

       do itype=1,ntype
!        S(:,:)=S1(:,:,itype)*weight_E+(S2(:,:,1,itype)+
!     &         S2(:,:,2,itype)+S2(:,:,3,itype))*weight_F

    
 
         dE_max=0.d0
         iii_max=1
         do iii=1,mm4
         if(dtot_term(iii,itype)**2/S(iii,iii,itype,itype) .gt.dE_max) then
         dE_max=dtot_term(iii,itype)**2/S(iii,iii,itype,itype)
         iii_max=iii
         endif
         enddo

         if(dE_max.lt.1.E-20) then
         write(6,*) "something wrong", dE_max
         stop
         endif


         index2(jjj,itype)=iii_max   ! this iii is selected as this jjj
         bc=dtot_term(iii_max,itype)/S(iii_max,iii_max,itype,itype)

         do itype2=1,ntype
         do iii=1,mm4
         dtot_term(iii,itype2)=dtot_term(iii,itype2)- S(iii,iii_max,itype2,itype)*bc    ! the change of dE_sum due to the substract of iii_max
!      This is only an approximation, avoid to choose some very linearly correlated features
         enddo
         enddo

         do iii=1,jjj   ! the already selected ones
         dtot_term(index2(iii,itype),itype)=0.d0
         enddo
      enddo   ! ityoe
      enddo  ! jjj
!cccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccc
!  Now, the selected index in kkk

       do itype=1,ntype
       do jjj=1,mm3
       iii=index2(jjj,itype)    ! from jjj [1,mm3] to iii [1,mm4]
       kkk=index(iii,itype)     ! from iii [1,mm4] to kkk [1,nkkk]
       ind_select(jjj,itype)=kkk
       enddo
       enddo  
!cccccccccccccccccccccccccccccccccccccccccccccccc



       do itype=1,ntype
       do ii=1,mm0(itype)
       idd3(:,ii,itype)=idd0(:,ii,itype)
       kkk0_st2(ii,itype)=kkk0_st(ii,itype)
       enddo
       enddo

       do itype=1,ntype
       do jjj=1,mm3
       kkk=ind_select(jjj,itype)  ! in the original large mm(k) index
       idd3(:,jjj+mm0(itype),itype)=idd2(:,kkk,itype)  ! this is the original large mm(k)
       kkk0_st2(jjj+mm0(Itype),itype)=ind_kkk_kkk0(kkk,itype)   ! kkk0
       enddo
       enddo
       
       if(inode.eq.1) then
       do itype=1,ntype
       open(12,file=fread_dfeat//"OUT.VV_index."//char(itype+48))
       rewind(12)
       write(12,*) mm0(itype)+mm3   ! the number of new features
       do ii=1,mm0(itype)+mm3
       write(12,"(5(i5,1x),2x,i8)") (idd3(jj,ii,itype),jj=0,4),kkk0_st2(ii,itype)
       enddo
       close(12)
       enddo
       endif


       deallocate(S)
       deallocate(ind_select)
       deallocate(idd3)
       deallocate(kkk0_st2)
       deallocate(idd2)

       endif  ! iloop2.eq.2


       close(70)
       do itype=1,ntype
       close(20+itype)
       close(40+itype)
       enddo
!ccccccccccccccccccccccccccccccccccccccccccccchhhhhhhh
       deallocate(dE_term)
       deallocate(fE_term)
       deallocate(dE_tot)
       deallocate(fE_tot)
       deallocate(dF_term)
       deallocate(fF_term)
       deallocate(dtot_term)
       deallocate(dtot_sum)

7000   continue   ! iloop2

       deallocate(idd0)
       deallocate(idd)
       deallocate(S1)
       deallocate(S2)
       deallocate(kkk0_st)


       if(inode.eq.1) then
       call fit_VV_force_sub(nimage_jump)
       call calc_VV_force_sub(nimage_jump,1)
       endif

       call MPI_barrier(MPI_COMM_WORLD,ierr)
5000   continue

    


       call MPI_barrier(MPI_COMM_WORLD,ierr)
       call MPI_finalize(ierr)
       
       
    deallocate(fread_dfeat)
       stop
       end

       
