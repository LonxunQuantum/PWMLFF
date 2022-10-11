subroutine calc_VV_force_sub(nimage_jump,iflag)
       implicit double precision (a-h,o-z)


       integer lwork
       integer,allocatable,dimension(:) :: iatom,iatom_type,itype_atom
       real*8,allocatable,dimension(:) :: Energy,Energyt,energy_pred
       real*8,allocatable,dimension(:) :: energy_group,energy_group_pred
       real*8,allocatable,dimension(:,:) :: force_pred
       real*8,allocatable,dimension(:,:) :: feat,feat2,feat22_type
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

       integer,allocatable,dimension(:) :: ipiv

       real*8,allocatable,dimension(:,:) :: w_feat
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
       real*8,allocatable,dimension(:,:,:) :: wp_atom
       real*8 AL(3,3),pi,dE,dFx,dFy,dFz,AL_tmp(3,3)

       real*8,allocatable,dimension(:,:) :: xatom_tmp

       real*8,allocatable,dimension(:,:) :: feat_new
       real*8,allocatable,dimension(:,:,:) :: feat_new_type
       real*8,allocatable,dimension(:,:,:) :: feat_ext1,feat_ext2, &
     &   feat_ext3,dfeat_ext1,dfeat_ext2


       real*8,allocatable,dimension(:,:,:,:) :: xp,xp1
 
       integer,allocatable,dimension(:) :: num_inv
       integer,allocatable,dimension(:,:) :: index_inv,index_inv2

       integer,allocatable,dimension(:) :: nfeat1,nfeat2,nfeatNi
       integer,allocatable,dimension(:,:) :: nfeat,ipos_feat

       real*8, allocatable, dimension (:,:) :: dfeat_tmp
       real*8, allocatable, dimension (:,:) :: feat_ftype
       integer,allocatable, dimension (:) :: iat_tmp,jneigh_tmp, ifeat_tmp
       integer num_tmp,jj
       ! character(len=200) dfeat_n(400)
       character(len=200) trainSetFileDir(400)
       character(len=200) trainSetDir
       character(len=200) MOVEMENTDir,dfeatDir,infoDir,trainDataDir,MOVEMENTallDir
       integer sys_num,sys
       integer nfeat1tm(100),ifeat_type(100),nfeat1t(100)
       integer mm(100),num_ref(100)
       integer*8 inp(100)
       real*8 Eatom_0(100)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! liuliping for relative path
       integer tmp_i
       character(len=200) fitModelDir
       character(len=:), allocatable :: fread_dfeat

       ! this file should be create by prepare.py
       open(1314, file="input/info_dir")
       rewind(10)
       read(10,"(A200)") fitModelDir
       close(1314)
       tmp_i = len(trim(adjustl(fitModelDir)))
       allocate(character(len=tmp_i) :: fread_dfeat)
       fread_dfeat = trim(adjustl(fitModelDir))
       write(*,*) "liuliping, fread_dfeat: ", fread_dfeat
! liuliping, end, all .r .x file should be invoke out of fread_dfeat



       open(10,file=fread_dfeat//"fit_linearMM.input")
       rewind(1314)
       read(1314,*) ntype,m_neigh
       allocate(itype_atom(ntype))
       allocate(nfeat1(ntype))
       allocate(nfeat2(ntype))
       allocate(nfeatNi(ntype))
       allocate(rad_atom(ntype))
       allocate(E_ave_vdw(ntype))
       allocate(wp_atom(ntype,ntype,2))
       wp_atom=0.d0
       do i=1,ntype
       read(10,*) itype_atom(i)
       enddo
       read(10,*) weight_E,weight_E0,weight_F,delta
       read(10,*) dwidth
       close(10)

        open(12,file=fread_dfeat//"OUT.xp",form="unformatted")
        rewind(12)
        read(12) mfeat2,ntype_t,ndim,ndim1
        if(ntype_t.ne.ntype) then
        write(6,*) "Inconsistent ntype from OUT.xp", ntype_t
        stop
        endif
        read(12) nfeat2
        allocate(xp(2,ndim,mfeat2,ntype))
        allocate(xp1(2,ndim1,mfeat2,ntype))
        read(12) xp
        read(12) xp1
        close(12)




       
       open(10,file=fread_dfeat//"vdw_fitB.ntype")
       rewind(10)
       read(10,*) ntype_t,nterm
       if(nterm.gt.2) then
       write(6,*) "nterm.gt.2,stop"
       stop
       endif
       if(ntype_t.ne.ntype) then
       write(6,*) "ntype not same in vwd_fitB.ntype,something wrong"
       stop
       endif
        do itype1=1,ntype
        read(10,*) itype_t,rad_atom(itype1),E_ave_vdw(itype1),((wp_atom(i,itype1,j1),i=1,ntype),j1=1,nterm)
       enddo
       close(10)


       open(10,file=fread_dfeat//"feat.info")
       rewind(10)
       read(10,*) iflag_PCA   ! this can be used to turn off degmm part
       read(10,*) nfeat_type
       do kkk=1,nfeat_type
         read(10,*) ifeat_type(kkk)   ! the index (1,2,3) of the feature type
       enddo
       read(10,*) ntype_tmp
        if(ntype_tmp.ne.ntype) then
          write(6,*)  "ntype of atom not same, fit_linearMM.input, feat.info, stop"
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


       allocate(w_feat(nfeat2m,ntype))
       do itype=1,ntype
       open(10,file=fread_dfeat//"weight_feat."//char(itype+48))
       rewind(10)
       do j=1,nfeat2(itype)
       read(10,*) j1,w_feat(j,itype)
       w_feat(j,itype)=w_feat(j,itype)**2
       enddo
       close(10)
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
       write(6,*) "nfeat2.not.same,feat2_ref",itype,nfeat2_tmp,nfeat2(itype)
       stop
       endif
       if(nfeat1_tmp.ne.nfeat1(itype)) then
       write(6,*) "nfeat1.not.same,feat2_ref",itype,nfeat1_tmp,nfeat1(itype)
       stop
       endif
       read(11) PV(1:nfeat1(itype),1:nfeat2(itype),itype)
       read(11) feat2_shift(1:nfeat2(itype),itype)
       read(11) feat2_scale(1:nfeat2(itype),itype)
       close(11)
       enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
        do itype=1,ntype
        open(12,file=fread_dfeat//"OUT.VV_index."//char(itype+48))
        rewind(12)
        read(12,*) mm(itype)   ! the number of new features
        close(12)
        enddo

        mm_max=0
        do itype=1,ntype
        if(mm(itype).gt.mm_max) mm_max=mm(itype)
        enddo

        if(mm_max.eq.0) then
        allocate(idd(0:4,1,ntype))
        else
        allocate(idd(0:4,mm_max,ntype))
        endif

        do itype=1,ntype
        open(12,file=fread_dfeat//"OUT.VV_index."//char(itype+48))
        rewind(12)
        read(12,*) mm(itype)   ! the number of new features
        do ii=1,mm(itype)
        read(12,*) (idd(jj,ii,itype),jj=0,4)
        enddo
        close(12)
        enddo

        nfeatNtot=0 ! tht total feature of diff atom type
        num_refm=0
        nfeatNi=0
        nfeatNi(1)=0
        do itype=1,ntype
        num_ref(itype)=nfeat2(itype)+mm(itype)
        if(num_ref(itype).gt.num_refm) num_refm=num_ref(itype)
        nfeatNtot=nfeatNtot+num_ref(itype)
        if(itype.gt.1) then
        nfeatNi(itype)=nfeatNi(itype-1)+num_ref(itype-1)
        endif
        enddo

        write(6,*) "nfeatNtot",nfeatNtot
        write(6,*) "nfeat_type",(num_ref(itype),itype=1,ntype)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccc
!   The above are headers
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc

       allocate(num(ntype))
       allocate(num_atomtype(ntype))
!       allocate(AA(nfeatNtot,nfeatNtot))
       allocate(BB(nfeatNtot))
!       allocate(AA_type(num_refm,num_refm,ntype))  ! num_refm not defibed yet
!       allocate(BB_type(num_refm,ntype))

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
       ! MOVEMENTallDir=trim(trainSetDir)//"/MOVEMENT"
       ! trainDataDir=trim(trainSetDir)//"/trainData.txt"

       open(10,file=fread_dfeat//"linear_VV_fitB.ntype")
       rewind(10)
       read(10,*) nfeatNtot_t
       if(nfeatNtot_t.ne.nfeatNtot) then
        write(6,*) "nfeatNtot changed, stop"
        stop
       endif
       do i=1,nfeatNtot
       read(10,*) it, BB(i)
       enddo
       close(10)

       do itype=1,ntype
       ii=nfeatNi(itype)+nfeat2(itype)
       Eatom_0(itype)=BB(ii)  ! the constant energy
       enddo

!ccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccc

       open(70,file=fread_dfeat//"energyVV.pred.tot")
       rewind(70)

       open(90,file=fread_dfeat//"energyVV.pred.group")
       rewind(90)
 
       do itype=1,ntype
       open(20+itype,file=fread_dfeat//"energyVV.pred."//char(itype+48))
       rewind(20+itype)
       open(40+itype,file=fread_dfeat//"forceVV.pred."//char(itype+48))
       rewind(40+itype)
       enddo

       if(iflag.eq.0) then

       open(110,file=fread_dfeat//"energyVV.pred.tot0")
       rewind(110)

       open(100,file=fread_dfeat//"energyVV.pred.group0")
       rewind(100)

       do itype=1,ntype
       open(60+itype,file=fread_dfeat//"energyVV.pred0."//char(itype+48))
       rewind(60+itype)
       open(80+itype,file=fread_dfeat//"forceVV.pred0."//char(itype+48))
       rewind(80+itype)
       enddo
       endif

!cccccccccccccccccccccccccccccccccccccccccccccc

       E_lost=0.d0

       do 900 sys=1,sys_num

       do 777 kkk=1,nfeat_type
       ! MOVEMENTDir=trim(trainSetFileDir(sys))//"/MOVEMENT"
       dfeatDir=trim(trainSetFileDir(sys))//"/dfeat.fbin.Ftype"//char(ifeat_type(kkk)+48)
       open(1000+kkk,file=dfeatDir,action="read",access="stream",form="unformatted")
       rewind(1000+kkk)
       read(1000+kkk) nimaget,natomt,nfeat1tm(kkk),m_neight

!      nfeat1tm(kkk) is the max(nfeat(ii,kkk)) for all ii(iatype)
  
       if(kkk.eq.1) then
       nimage=nimaget
       natom=natomt
       m_neigh=m_neight
       else
       if(nimaget.ne.nimage.or.natomt.ne.natom.or.m_neight.ne.m_neigh) then
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
         if(sys.ne.1) then
         deallocate(iatom)
         endif
       allocate(iatom(natom))
       endif
       read(1000+kkk) iatom      ! The same for different kkk

777    continue
!ccccccccccccccccccccccccccccccccccccccccccccccccc
       
           
       if (sys.ne.1) then      

       deallocate(iatom_type)
       deallocate(Energy)
       deallocate(Energyt)
       deallocate(num_neight)
       deallocate(feat)
       deallocate(feat2)
       deallocate(feat_type)
       deallocate(feat2_type)
       deallocate(feat22_type)
       deallocate(num_neigh)
       deallocate(list_neigh)
       deallocate(ind_type)
       deallocate(dfeat)
       deallocate(dfeat_type)
       deallocate(dfeat2_type)
       deallocate(dfeat2)
       deallocate(xatom)

       deallocate(ipiv)
       deallocate(num_inv)
       deallocate(index_inv)
       deallocate(index_inv2)
       deallocate(force)
       deallocate(VV)
       deallocate(SS)

       deallocate(energy_pred)
       deallocate(energy_group)
       deallocate(energy_group_pred)
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
       allocate(feat22_type(num_refm,ntype))
       allocate(num_neigh(natom))
       allocate(list_neigh(m_neigh,natom))
       allocate(ind_type(natom,ntype))
       allocate(dfeat(nfeat1m,natom,m_neigh,3))
       allocate(dfeat_type(nfeat1m,natom*m_neigh*3,ntype))
       allocate(dfeat2_type(nfeat2m,natom*m_neigh*3,ntype))
       allocate(dfeat2(nfeat2m,natom,m_neigh,3))
       allocate(xatom(3,natom))
       allocate(energy_pred(natom))
       allocate(energy_group(natom))
       allocate(energy_group_pred(natom))
       allocate(force_pred(3,natom))

       dfeat=0.d0
       dfeat_type=0.d0

       allocate(ipiv(nfeatNtot))
       allocate(num_inv(natom))
       allocate(index_inv(3*m_neigh,natom))
       allocate(index_inv2(3*m_neigh,natom))
       allocate(force(3,natom))
       allocate(VV(nfeatNtot,3*natom))
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


       num_tmp=0


       do 3000 image=1,nimage/nimage_jump

       write(6,"('calc,sys,image=',2(i6,1x))") sys,image

! the stream access
        do kkk=1,nfeat_type
        inquire(1000+kkk,pos=inp(kkk))
        enddo
!  This inp(kkk) alway at the position before the next energy



       do jump=1,nimage_jump-1
       do kkk=1,nfeat_type

       inp(kkk)=inp(kkk)+(4*natom+nfeat1tm(kkk)*natom)*8+(natom+m_neigh*natom)*4    ! jump from eenrgy to num_tmp
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


!-----------------------------------------------------------------
!-----------------------------------------------------------------
!---- read in the feature from different kkk, and put them together
!-----------------------------------------------------------------
       dfeat(:,:,:,:)=0.0
       feat(:,:)=0.0
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
       dfeat(ifeat_tmp(jj)+ipos_feat(itype2,kkk),iat_tmp(jj),jneigh_tmp(jj),:)=dfeat_tmp(:,jj)
!  Place dfeat from different iftype into the same dfeat
       enddo
       deallocate(dfeat_tmp)
       deallocate(iat_tmp)
       deallocate(jneigh_tmp)
       deallocate(ifeat_tmp)

778    continue
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
       do i=1,natom
       rad1=rad_atom(iatom_type(i))
       dE=0.d0
       dFx=0.d0
       dFy=0.d0
       dFz=0.d0
       do jj=1,num_neigh(i)
       j=list_neigh(jj,i)
       if(i.ne.j) then
       rad2=rad_atom(iatom_type(j))
       rad=rad1+rad2
       dx1=mod(xatom(1,j)-xatom(1,i)+100.d0,1.d0)
       if(abs(dx1-1).lt.abs(dx1)) dx1=dx1-1
       dx2=mod(xatom(2,j)-xatom(2,i)+100.d0,1.d0)
       if(abs(dx2-1).lt.abs(dx2)) dx2=dx2-1
       dx3=mod(xatom(3,j)-xatom(3,i)+100.d0,1.d0)
       if(abs(dx3-1).lt.abs(dx3)) dx3=dx3-1
       dx=AL(1,1)*dx1+AL(1,2)*dx2+AL(1,3)*dx3
       dy=AL(2,1)*dx1+AL(2,2)*dx2+AL(2,3)*dx3
       dz=AL(3,1)*dx1+AL(3,2)*dx2+AL(3,3)*dx3
       dd=dsqrt(dx**2+dy**2+dz**2)
       if(dd.lt.2*rad) then

       w22_1=wp_atom(iatom_type(j),iatom_type(i),1)
       w22_2=wp_atom(iatom_type(j),iatom_type(i),2)
       w22F_1=(wp_atom(iatom_type(j),iatom_type(i),1)+wp_atom(iatom_type(i),iatom_type(j),1))/2     ! take the average for force calc. 
       w22F_2=(wp_atom(iatom_type(j),iatom_type(i),2)+wp_atom(iatom_type(i),iatom_type(j),2))/2     ! take the average for force calc. 
       yy=pi*dd/(4*rad)
!       dE=dE+0.5*w22*exp((1-dd/rad)*4.0)*cos(yy)**2
!       dEdd=w22*exp((1-dd/rad)*4.d0)*((-4/rad)*cos(yy)**2 -(pi/(2*rad))*cos(yy)*sin(yy))

       dE=dE+0.5*4*(w22_1*(rad/dd)**12*cos(yy)**2+w22_2*(rad/dd)**6*cos(yy)**2)
       dEdd=4*(w22F_1*(-12*(rad/dd)**12/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12) &
     &   +W22F_2*(-6*(rad/dd)**6/dd*cos(yy)**2 -(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**6))

       dFx=dFx-dEdd*dx/dd       ! note, -sign, because dx=d(j)-x(i)
       dFy=dFy-dEdd*dy/dd
       dFz=dFz-dEdd*dz/dd
       endif
       endif
       enddo
!       write(6,*) "dE,dFx",dE,dFx
       energy(i)=energy(i)-dE
       force(1,i)=force(1,i)-dFx   ! Note, assume force=dE/dx, no minus sign
       force(2,i)=force(2,i)-dFy
       force(3,i)=force(3,i)-dFz
       enddo
!cccccccccccccccccccccccccccccccccccccccccc


       do itype=1,ntype
       call dgemm('T','N',nfeat2(itype),num(itype),nfeat1(itype),1.d0,PV(1,1,itype),&
     & nfeat1m,feat_type(1,1,itype),nfeat1m,0.d0,feat2_type(1,1,itype),nfeat2m)
       enddo

       do itype=1,ntype
       do i=1,num(itype)
       do j=1,nfeat2(itype)-1
       feat2_type(j,i,itype)=(feat2_type(j,i,itype)-feat2_shift(j,itype))*feat2_scale(j,itype)
       enddo
       feat2_type(nfeat2(itype),i,itype)=1.d0   ! the special value 1 component 
       enddo
       enddo

       num=0
       do i=1,natom
       itype=iatom_type(i)
       num(itype)=num(itype)+1
       feat2(:,i)=feat2_type(:,num(itype),itype)
!  Here, we collect different iatype back, into feat2(:,i)
!  But actually, different atom (different iatype) will have different number of features
!  But all stored within nfeat2m
       enddo


        Etot=0.d0
        do i=1,natom
        Etot=Etot+energy(i)
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

       do itype=1,ntype
       call dgemm('T','N',nfeat2(itype),num(itype),nfeat1(itype),1.d0,PV(1,1,itype), &
     & nfeat1m,dfeat_type(1,1,itype),nfeat1m,0.d0,dfeat2_type(1,1,itype), nfeat2m)
       enddo
!cccccccccccccccccccccccccccccccccccccccc




       num=0
       do i=1,natom
       do jj=1,num_neigh(i)
       itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
       num(itype)=num(itype)+1
       do j=1,nfeat2(itype)-1
       dfeat2(j,i,jj,1)=dfeat2_type(j,num(itype),itype)*feat2_scale(j,itype)
       enddo
       dfeat2(nfeat2(itype),i,jj,1)=0.d0
       num(itype)=num(itype)+1
       do j=1,nfeat2(itype)-1
       dfeat2(j,i,jj,2)=dfeat2_type(j,num(itype),itype)*feat2_scale(j,itype)
       enddo
       dfeat2(nfeat2(itype),i,jj,2)=0.d0
       num(itype)=num(itype)+1
       do j=1,nfeat2(itype)-1
       dfeat2(j,i,jj,3)=dfeat2_type(j,num(itype),itype)*feat2_scale(j,itype)
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


       num_refm=nfeat2m+mm_max

       allocate(feat_new(num_refm,natom))
       allocate(feat_new_type(num_refm,natom,ntype))
       allocate(feat_ext1(natom,nfeat2m,ndim1))
       allocate(dfeat_ext1(natom,nfeat2m,ndim1))
       allocate(feat_ext2(natom,nfeat2m,ndim))
       allocate(dfeat_ext2(natom,nfeat2m,ndim))
       allocate(feat_ext3(natom,nfeat2m,1))
       allocate(dfeat_new(num_refm,natom,m_neigh,3))


!ccccccccccccccccccccccccccccccccccccc
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
        feat_ext1(i,j,id)=exp(-((feat2(j,i)-xp1(1,id,j,itype))/xp1(2,id,j,itype))**2)
        dfeat_ext1(i,j,id)=-feat_ext1(i,j,id)*2*(feat2(j,i)-xp1(1,id,j,itype)) /xp1(2,id,j,itype)**2
        enddo
        enddo
        enddo

        do i=1,natom
        itype=iatom_type(i)
        do id=1,ndim
        do j=1,nfeat2m
        feat_ext2(i,j,id)=exp(-((feat2(j,i)-xp(1,id,j,itype))/xp(2,id,j,itype))**2)
        dfeat_ext2(i,j,id)=-feat_ext2(i,j,id)*2*(feat2(j,i)-xp(1,id,j,itype))/xp(2,id,j,itype)**2
        enddo
        enddo
        enddo


        do i=1,natom
        do j=1,nfeat2m
        feat_ext3(i,j,1)=feat2(j,i)
        enddo
        enddo
!cccccccccccccccccccccccccccccccc

       do i=1,natom
       itype=iatom_type(i)
       do iii=1,nfeat2(itype)
       feat_new(iii,i)=feat2(iii,i)
       enddo
       enddo

       do i=1,natom
       itype=iatom_type(i)
       do kkk=1,mm(itype)

       if(idd(0,kkk,itype).eq.1) then
       j1=idd(1,kkk,itype)
       id1=idd(2,kkk,itype)
       feat_new(kkk+nfeat2(itype),i)=feat_ext1(i,j1,id1)
       elseif(idd(0,kkk,itype).eq.2) then
       j1=idd(1,kkk,itype)
       id1=idd(2,kkk,itype)
       j2=idd(3,kkk,itype)
       id2=idd(4,kkk,itype)
       feat_new(kkk+nfeat2(itype),i)=feat_ext2(i,j1,id1)*feat_ext2(i,j2,id2)
       elseif(idd(0,kkk,itype).eq.3) then
       j1=idd(1,kkk,itype)
       j2=idd(2,kkk,itype)
       j3=idd(3,kkk,itype)
       feat_new(kkk+nfeat2(itype),i)=feat_ext3(i,j1,1)*feat_ext3(i,j2,1)*feat_ext3(i,j3,1)
       endif
       enddo
       enddo

   
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       do itype=1,ntype
       num_ref(itype)=nfeat2(itype)+mm(itype)
       nfeatN=num_ref(itype)
       num(itype)=0
       do i=1,natom
       if(itype.eq.iatom_type(i)) then
       num(itype)=num(itype)+1
       feat_new_type(1:nfeatN,num(itype),itype)=feat_new(1:nfeatN,i)
       endif
       enddo
       enddo   ! itype

       do i=1,natom
       sum=0.d0
       itype=iatom_type(i)
       do j=1,num_ref(itype)
       sum=sum+BB(j+nfeatNi(itype))*feat_new(j,i)
       enddo
       energy_pred(i)=sum
       enddo

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccc Now, we have finished the energy part. In the following, we will 
!ccc include the force part. Which is more complicated. 
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccc  
        do itype=1,ntype
        do j=1,num_ref(itype)
        sum=0.d0
        do i=1,num(itype)   ! num atom of this type
        sum=sum+feat_new_type(j,i,itype)
        enddo
        feat22_type(j,itype)=sum   ! sum over all the atoms
        enddo
        enddo


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

       do iii=1,nfeat2(itype)
       dfeat_new(iii,i,jj,:)=dfeat2(iii,i,jj,:)
       enddo

       do kkk=1,mm(itype)
       if(idd(0,kkk,itype).eq.1) then
       j1=idd(1,kkk,itype)
       id1=idd(2,kkk,itype)
       dfeat_new(kkk+nfeat2(itype),i,jj,:)=dfeat_ext1(ii,j1,id1)*dfeat2(j1,i,jj,:)
       elseif(idd(0,kkk,itype).eq.2) then
       j1=idd(1,kkk,itype)
       id1=idd(2,kkk,itype)
       j2=idd(3,kkk,itype)
       id2=idd(4,kkk,itype)
       dfeat_new(kkk+nfeat2(itype),i,jj,:)=dfeat_ext2(ii,j1,id1)* &
     & feat_ext2(ii,j2,id2)*dfeat2(j1,i,jj,:)+feat_ext2(ii,j1,id1) &
     &  *dfeat_ext2(ii,j2,id2)*dfeat2(j2,i,jj,:)
       elseif(idd(0,kkk,itype).eq.3) then
       j1=idd(1,kkk,itype)
       j2=idd(2,kkk,itype)
       j3=idd(3,kkk,itype)
       dfeat_new(kkk+nfeat2(itype),i,jj,:)=dfeat2(j1,i,jj,:)*feat_ext3(ii,j2,1)*feat_ext3(ii,j3,1)+ &
     & feat_ext3(ii,j1,1)*dfeat2(j2,i,jj,:)*feat_ext3(ii,j3,1)+ feat_ext3(ii,j1,1)*feat_ext3(ii,j2,1)*dfeat(j3,i,jj,:)
       endif
       enddo
       enddo
       enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


!cccc Now, we have the new features, we need to calculate the distance to reference state

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


       do i=1,natom
       do ixyz=1,3
       sum=0.d0
       do itype=1,ntype
       do j=1,num_ref(itype)
       sum=sum+SS(j,i,ixyz,itype)*BB(j+nfeatNi(itype))
       enddo
       enddo
       force_pred(ixyz,i)=sum
       enddo
       enddo

       Etot=0.d0
       Etot_pred=0.d0
       do i=1,natom
       Etot=Etot+energy(i)
       Etot_pred=Etot_pred+energy_pred(i)
       enddo
       write(70,*) Etot, Etot_pred
       if(iflag.eq.0) then
       write(110,*) Etot, Etot_pred
       endif




!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        ddcut=-dwidth**2*log(0.01)
        do iat1=1,natom   ! center position (not even call it atom)
        Esum1=0.d0
        Esum2=0.d0
        sum=0.d0
        do iat2=1,natom
        itype=iatom_type(iat2)
 
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
        Esum1=Esum1+(energy(iat2)-Eatom_0(itype))*fact
        Esum2=Esum2+(energy_pred(iat2)-Eatom_0(itype))*fact
        sum=sum+fact
        endif
        enddo
        energy_group(iat1)=Esum1/sum
        energy_group_pred(iat1)=Esum2/sum
        enddo
!ccccccccccccccccccccccccccccccccccccccccccc

       
       do i=1,natom
        write(90,*) energy_group(i),energy_group_pred(i)
        itype=iatom_type(i)
        write(20+itype,*) energy(i),energy_pred(i)
        write(40+itype,*) force(1,i),force_pred(1,i)
        write(40+itype,*) force(2,i),force_pred(2,i)
        write(40+itype,*) force(3,i),force_pred(3,i)
       E_lost=E_lost+(energy(i)-energy_pred(i))**2*weight_E+ &
     & ((force(1,i)-force_pred(1,i))**2+ (force(2,i)-force_pred(2,i))**2+ &
     &    (force(3,i)-force_pred(3,i))**2)*weight_F
       enddo
      
       if(iflag.eq.0) then
       do i=1,natom
        write(100,*) energy_group(i),energy_group_pred(i)
        itype=iatom_type(i)
        write(60+itype,*) energy(i),energy_pred(i)
        write(80+itype,*) force(1,i),force_pred(1,i)
        write(80+itype,*) force(2,i),force_pred(2,i)
        write(80+itype,*) force(3,i),force_pred(3,i)
       enddo
       endif




       deallocate(feat_new)
       deallocate(feat_new_type)
       deallocate(feat_ext1)
       deallocate(dfeat_ext1)
       deallocate(feat_ext2)
       deallocate(dfeat_ext2)
       deallocate(feat_ext3)
       deallocate(dfeat_new)

       

3000   continue


900    continue

       do kkk=1,nfeat_type
       close(1000+kkk)
       enddo

       close(90)
       close(70)
       do itype=1,ntype
       close(20+itype)
       close(40+itype)
       enddo

       if(iflag.eq.0) then
       close(100)
       close(110)
       do itype=1,ntype
       close(60+itype)
       close(80+itype)
       enddo
       endif

       if(iflag.eq.0) then
       open(23,file=fread_dfeat//"loop.inter")
       rewind(23)
       write(23,*) E_lost,iflag
       close(23)
       else
       open(23,file=fread_dfeat//"loop.inter",position="append")
       write(23,*) E_lost
       close(23)
       endif



!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       deallocate(iatom_type)
       deallocate(Energy)
       deallocate(Energyt)
       deallocate(num_neight)
       deallocate(feat)
       deallocate(feat2)
       deallocate(feat_type)
       deallocate(feat2_type)
       deallocate(feat22_type)
       deallocate(num_neigh)
       deallocate(list_neigh)
       deallocate(ind_type)
       deallocate(dfeat)
       deallocate(dfeat_type)
       deallocate(dfeat2_type)
       deallocate(dfeat2)
       deallocate(xatom)

       deallocate(ipiv)
       deallocate(num_inv)
       deallocate(index_inv)
       deallocate(index_inv2)
       deallocate(force)
       deallocate(VV)
       deallocate(SS)

       deallocate(energy_pred)
       deallocate(energy_group)
       deallocate(energy_group_pred)
       deallocate(force_pred)


       deallocate(fread_dfeat)
       return
       end

       
