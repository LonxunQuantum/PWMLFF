program linear_forceMM
     use data_ewald
     implicit double precision (a-h,o-z)

     integer lwork
     integer,allocatable,dimension(:) :: iatom,iatom_type,itype_atom
     real*8,allocatable,dimension(:) :: Energy,energy_pred
     real*8,allocatable,dimension(:) :: Energy_group,energy_group_pred
     real*8,allocatable,dimension(:) :: Energyt
     real*8,allocatable,dimension(:,:) :: feat,feat2
     real*8,allocatable,dimension(:,:,:) :: feat_type,feat2_type
     integer,allocatable,dimension(:) :: num_neigh,num,num_atomtype
     integer,allocatable,dimension(:) :: num_neight
     integer,allocatable,dimension(:,:) :: list_neigh,ind_type

     real*8,allocatable,dimension(:,:,:,:) :: dfeat,dfeat2
     real*8,allocatable,dimension(:,:,:) :: dfeat_type,dfeat2_type

     real*8,allocatable,dimension(:,:) :: AA
     real*8,allocatable,dimension(:) :: BB

     real*8,allocatable,dimension(:,:,:) :: Gfeat_type
     real*8,allocatable,dimension(:,:) :: Gfeat_tmp

     real*8,allocatable,dimension(:,:,:) :: AA_type
     real*8,allocatable,dimension(:,:) :: BB_type,BB_type0

     real*8,allocatable,dimension(:,:) :: SS_tmp,SS_tmp2

     integer,allocatable,dimension(:) :: ipiv

     real*8,allocatable,dimension(:,:) :: w_feat
     real*8,allocatable,dimension(:,:,:) :: feat2_ref

     real*8,allocatable,dimension(:,:,:) :: PV
     real*8,allocatable,dimension(:,:) :: feat2_shift,feat2_scale
     real*8,allocatable,dimension(:,:) :: feat_ftype


     real*8,allocatable,dimension(:,:) :: WW,VV,QQ
     real*8,allocatable,dimension(:,:,:,:) :: SS

     real*8,allocatable,dimension(:,:) :: Gfeat2,dGfeat2

     real*8,allocatable,dimension(:,:) :: force,force_pred

 
     integer,allocatable,dimension(:) :: num_inv
     integer,allocatable,dimension(:,:) :: index_inv,index_inv2
     ! character(len=200) dfeat_n(400)
     character(len=200) trainSetFileDir(400)
     character(len=200) trainSetDir
     character(len=200) MOVEMENTDir,dfeatDir,infoDir,trainDataDir
     integer sys_num,sys
     
     real*8, allocatable, dimension (:,:) :: dfeat_tmp
     integer,allocatable, dimension (:) :: iat_tmp,jneigh_tmp, ifeat_tmp
     integer num_tmp,jj

     integer,allocatable,dimension(:) :: nfeat1,nfeat2,num_ref, nfeat2i

     integer,allocatable,dimension(:,:) :: nfeat,ipos_feat

     real*8,allocatable,dimension(:,:) :: xatom,xatom_tmp
     real*8,allocatable,dimension(:) :: rad_atom,E_ave_vdw
     real*8,allocatable,dimension(:,:,:) :: wp_atom
     real*8 AL(3,3),pi,dE,dFx,dFy,dFz,AL_tmp(3,3)

     integer nfeat1tm(100),ifeat_type(100),nfeat1t(100)
     real*8 Eatom_0(100)
! liuliping for relative path
     integer tmp_i
     character(len=200) fitModelDir
     character(len=:), allocatable :: fread_dfeat

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
     ! this file should be create by prepare.py
     open(1314, file="input/info_dir")
     rewind(1314)
     read(1314,"(A200)") fitModelDir
     close(1314)
     tmp_i = len(trim(adjustl(fitModelDir)))
     allocate(character(len=tmp_i) :: fread_dfeat)
     fread_dfeat = trim(adjustl(fitModelDir))
     !write(*,*) "liuliping, fread_dfeat: ", fread_dfeat
! liuliping, end, all .r .x file should be invoke out of fread_dfeat



!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
     pi=4*datan(1.d0)

     open(10,file=fread_dfeat//"fit_linearMM.input")
     rewind(10)
     read(10,*) ntype,m_neigh
      allocate(itype_atom(ntype))
      allocate(nfeat1(ntype))
      allocate(nfeat2(ntype))
      allocate(nfeat2i(ntype))
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
      read(10,*) itype_t,rad_atom(itype1),E_ave_vdw(itype1), &
    &             ((wp_atom(i,itype1,j1),i=1,ntype),j1=1,nterm)
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
        read(10,*) iatom_tmp,nfeat1(i),nfeat2(i)   ! these nfeat1,nfeat2 include all     ftype
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
! nfeat1(iatype): the num of feature including all type
! nfeat1(iatyo2): the num of PCA feature including all type

       do ii=1,ntype
       ipos_feat(ii,1)=0
       do kkk=2,nfeat_type
       ipos_feat(ii,kkk)=ipos_feat(ii,kkk-1)+nfeat(ii,kkk-1)
       enddo
       enddo




      nfeat1m=0
      nfeat2m=0
      nfeat2tot=0
      nfeat2i=0
      nfeat2i(1)=0
      do i=1,ntype
      if(nfeat1(i).gt.nfeat1m) nfeat1m=nfeat1(i)
      if(nfeat2(i).gt.nfeat2m) nfeat2m=nfeat2(i)
      nfeat2tot=nfeat2tot+nfeat2(i)
      if(i.gt.1) then
      nfeat2i(i)=nfeat2i(i-1)+nfeat2(i-1)
      endif

      enddo



     allocate(BB(nfeat2tot))
     allocate(BB_type(nfeat2m,ntype))
     allocate(BB_type0(nfeat2m,ntype))

     open(12,file=fread_dfeat//"linear_fitB.ntype")
         rewind(12)
         read(12,*) ntmp
       if(ntmp.ne.nfeat2tot) then
      write(6,*) "ntmp.not.right,linear_fitB.ntype",ntmp,nfeat2tot
        stop
       endif
        do i=1,nfeat2tot
       read(12,*) itmp, BB(i)
        enddo
       close(12)
        do itype=1,ntype
        do k=1,nfeat2(itype)
        BB_type0(k,itype)=BB(k+nfeat2i(itype))
        enddo
        enddo

        do itype=1,ntype
        Eatom_0(itype)=BB(nfeat2i(itype)+nfeat2(itype))
        enddo


!cccccccc Right now, nfeat1,nfeat2,num_ref for different types
!cccccccc must be the same. We will change that later, allow them 
!cccccccc to be different

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

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
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
     read(11) feat2_shift(1:nfeat2(itype),itype)
     read(11) feat2_scale(1:nfeat2(itype),itype)
     close(11)
     enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccc

     allocate(num(ntype))
     allocate(num_atomtype(ntype))
     allocate(AA(nfeat2tot,nfeat2tot))
     allocate(AA_type(nfeat2m,nfeat2m,ntype))

     open(13,file=fread_dfeat//"location")
     rewind(13)
     read(13,*) sys_num  !,trainSetDir
     read(13,'(a200)') trainSetDir
     ! allocate(trainSetFileDir(sys_num))
     do i=1,sys_num
     read(13,'(a200)') trainSetFileDir(i)    
     enddo
     close(13)

     AA=0.d0

     open(70,file=fread_dfeat//"energyL.pred.tot") 
     rewind(70)
     open(90,file=fread_dfeat//"energyL.pred.group") 
     rewind(90)
 
     do itype=1,ntype
     open(20+itype,file=fread_dfeat//"energyL.pred."//char(itype+48))
     rewind(20+itype)
     open(40+itype,file=fread_dfeat//"forceL.pred."//char(itype+48))
     rewind(40+itype)
     enddo

     AEM_Etot=0
     AEM_Eatom=0
     AEM_F=0
     AEM_Egroup=0
     num_AEM_Etot=0
     num_AEM_Eatom=0
     num_AEM_F=0
     num_AEM_Egroup=0


     do 900 sys=1,sys_num

     do 777 kkk=1,nfeat_type

      dfeatDir=trim(trainSetFileDir(sys))//"/dfeat.fbin.Ftype"//char(ifeat_type(kkk)+48)
      open(1000+kkk,file=dfeatDir,action="read",access="stream", form="unformatted")
      rewind(1000+kkk)
      read(1000+kkk) nimaget,natomt,nfeat1tm(kkk), m_neight

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
       if(nfeat1t(ii).ne.nfeat(ii,kkk)) then   ! the num of feat for ii_th iatype, a    nd kkk_th feat type
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
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc



     allocate(iatom_type(natom))
     allocate(Energy(natom))
     allocate(Energyt(natom))
     allocate(num_neight(natom))
     allocate(Energy_pred(natom))
     allocate(Energy_group(natom))
     allocate(Energy_group_pred(natom))
     allocate(feat(nfeat1m,natom))
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

     allocate(ipiv(nfeat2tot))
     allocate(num_inv(natom))
     allocate(index_inv(3*m_neigh,natom))
     allocate(index_inv2(3*m_neigh,natom))
     allocate(force(3,natom))
     allocate(force_pred(3,natom))
     allocate(VV(nfeat2tot,3*natom))
     allocate(SS(nfeat2m,natom,3,ntype))


     pi=4*datan(1.d0)

!       read(10) iatom      ! 1,2,3,...,ntype

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



      ! liuliping, output energy forces in sequence of atoms in images.
      open(1314, file=fread_dfeat//"energyL.pred.all")
      open(1315, file=fread_dfeat//"forceL.pred.all")
      !
     do 3000 image=1,nimage

     write(6,*) "image=",image,sys

     AA_type=0.d0
     BB_type=0.d0

!-----------------------------------------------------------------
!-----------------------------------------------------------------
!---- read in the feature from different kkk, and put them together
!-----------------------------------------------------------------
      dfeat(:,:,:,:)=0.0
      feat(:,:)=0.0
      do 778 kkk=1,nfeat_type

      allocate(feat_ftype(nfeat1tm(kkk),natom))
      read(1000+kkk) energy   ! repeated
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
      feat(ii+ipos_feat(itype,kkk),iat)=feat_ftype(ii,iat)   ! put different kkk tog    ether
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
      dfeat(ifeat_tmp(jj)+ipos_feat(itype2,kkk),iat_tmp(jj),jneigh_tmp(jj),:)& 
       & =dfeat_tmp(:,jj)
!  Place dfeat from different iftype into the same dfeat
      enddo
      deallocate(dfeat_tmp)
      deallocate(iat_tmp)
      deallocate(jneigh_tmp)
      deallocate(ifeat_tmp)
778     continue

!cccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccc

     
     num=0
     do i=1,natom
     itype=iatom_type(i)
     num(itype)=num(itype)+1
     ind_type(num(itype),itype)=i
     feat_type(:,num(itype),itype)=feat(:,i)
     enddo


     do itype=1,ntype
     call dgemm('T','N',nfeat2(itype),num(itype),nfeat1(itype), 1.d0,PV(1,1,itype), &
    & nfeat1m,feat_type(1,1,itype),nfeat1m,0.d0,feat2_type(1,1,itype), nfeat2m)
     enddo


     do itype=1,ntype
     do i=1,num(itype)
     do j=1,nfeat2(itype)-1
     feat2_type(j,i,itype)=(feat2_type(j,i,itype)- &
     &  feat2_shift(j,itype))*feat2_scale(j,itype)
     enddo
     feat2_type(nfeat2(itype),i,itype)=1.d0
     enddo
     enddo

     num=0
     do i=1,natom
     itype=iatom_type(i)
     num(itype)=num(itype)+1
     feat2(:,i)=feat2_type(:,num(itype),itype)
     enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cc  Now, we have yield the new feature: feat2_type(nfeat2,num(itype),itype)
!ccc The last feature, nfeat2, =1
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

     do i=1,natom
     itype=iatom_type(i)
     sum=0.d0
     do j=1,nfeat2(itype)
     sum=sum+feat2(j,i)*BB_type0(j,itype)
     enddo
     energy_pred(i)=sum
     enddo


!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cc dfeat(nfeat1,natom,j_neigh,3): dfeat(j,i,jj,3)= d/dR_i(feat(j,list_neigh(jj,i))
!ccccccccccccc


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
!ccccccccc Note: num(itype) is rather large, in the scane of natom*num_neigh

     do itype=1,ntype
     call dgemm('T','N',nfeat2(itype),num(itype),nfeat1(itype), &
     & 1.d0,PV(1,1,itype), nfeat1m,dfeat_type(1,1,itype),nfeat1m,0.d0, &
     & dfeat2_type(1,1,itype), nfeat2m)
     enddo


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
    
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cc  Now, dfeat2 is: 
!cc dfeat2(nfeat2,natom,j_neigh,3): dfeat2(j,i,jj,3)= d/dR_i(feat2(j,list_neigh(jj,i))
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccc Now, we have the new features, we need to calculate the distance to reference state

     SS=0.d0

     do i=1,natom
     do jj=1,num_neigh(i)
     itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
     do j=1,nfeat2(itype)

     SS(j,i,1,itype)=SS(j,i,1,itype)+dfeat2(j,i,jj,1)
     SS(j,i,2,itype)=SS(j,i,2,itype)+dfeat2(j,i,jj,2)
     SS(j,i,3,itype)=SS(j,i,3,itype)+dfeat2(j,i,jj,3)
     enddo
     enddo
     enddo


     do i=1,natom
     do ixyz=1,3
     sum=0.d0
     do itype=1,ntype
     do j=1,nfeat2(itype)
     sum=sum+SS(j,i,ixyz,itype)*BB_type0(j,itype)
     enddo
     enddo
     force_pred(ixyz,i)=sum
     enddo
     enddo
    ! liuliping ewald
    if(iflag_born_charge_ewald .eq. 1) then
        !write(*,*) "calc_lin_forceMM ewald"
        allocate(ewald_atom(natom))
        allocate(fatom_ewald(3,natom))
        AL_bohr = AL*Angstrom2Bohr  ! Angstrom2Bohr; to atmoic units
        vol = dabs(AL_bohr(3, 1)*(AL_bohr(1, 2)*AL_bohr(2, 3) - AL_bohr(1, 3)*AL_bohr(2, 2)) &
           + AL_bohr(3, 2)*(AL_bohr(1, 3)*AL_bohr(2, 1) - AL_bohr(1, 1)*AL_bohr(2, 3)) &
           + AL_bohr(3, 3)*(AL_bohr(1, 1)*AL_bohr(2, 2) - AL_bohr(1, 2)*AL_bohr(2, 1)))

        call get_ewald(natom, AL_bohr, iatom, xatom, zatom_ewald, ewald, ewald_atom, fatom_ewald)
        ! forces in get_ewald and fit_lin are both dE/dx
        !write(*,*) "calc lin before ewald(1:3): ", energy_pred(1:3)
        energy_pred(1:natom) = energy_pred(1:natom) + ewald_atom(1:natom)*Hartree2eV
        force_pred(1:3,1:natom) = force_pred(1:3,1:natom) + fatom_ewald(1:3,1:natom)*Hartree2eV*Angstrom2Bohr
        !write(*,*) "calc lin after ewald(1:3): ", energy_pred(1:3)
        deallocate(ewald_atom)
        deallocate(fatom_ewald)
    endif

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

!ccccccccccccccccccccccccccccccccccccccccccc
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
!       w22=dsqrt(wp_atom(iatom_type(i))*wp_atom(iatom_type(j)))
!       w22=dsqrt(wp_atom(iatom_type(i))*wp_atom(iatom_type(j)))
      w22_1=wp_atom(iatom_type(j),iatom_type(i),1)
      w22_2=wp_atom(iatom_type(j),iatom_type(i),2)
      w22F_1=(wp_atom(iatom_type(j),iatom_type(i),1)+ &
     &      wp_atom(iatom_type(i),iatom_type(j),1))/2     ! take the average for force calc.
      w22F_2=(wp_atom(iatom_type(j),iatom_type(i),2)+ &
     &      wp_atom(iatom_type(i),iatom_type(j),2))/2     ! take the average for force calc.

     yy=pi*dd/(4*rad)
!       dE=dE+0.5*w22*exp((1-dd/rad)*4.0)*cos(yy)**2
!       dEdd=w22*exp((1-dd/rad)*4.d0)*((-4/rad)*cos(yy)**2
!     &   -(pi/(2*rad))*cos(yy)*sin(yy))
     dE=dE+0.5*4*(w22_1*(rad/dd)**12*cos(yy)**2+ &
     &     w22_2*(rad/dd)**6*cos(yy)**2)
     dEdd=4*(w22F_1*(-12*(rad/dd)**12/dd*cos(yy)**2 &
     &   -(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)+W22F_2*(-6*(rad/dd)**6/dd*cos(yy)**2 &
     &   -(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**6))

     dFx=dFx-dEdd*dx/dd       ! note, -sign, because dx=d(j)-x(i)
     dFy=dFy-dEdd*dy/dd
     dFz=dFz-dEdd*dz/dd
     endif
     endif
     enddo
     energy(i)=energy(i)-dE
     force(1,i)=force(1,i)-dFx   ! Note, assume force=dE/dx, no minus sign
     force(2,i)=force(2,i)-dFy
     force(3,i)=force(3,i)-dFz
     enddo
!ccccccccccccccccccccccccccccccccccccccccccc
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
     fact=exp(-dd/dwidth**2)
     Esum1=Esum1+(energy(iat2)-Eatom_0(itype))*fact
     Esum2=Esum2+(energy_pred(iat2)-Eatom_0(itype))*fact
     sum=sum+fact
     enddo
     energy_group(iat1)=Esum1/sum
     energy_group_pred(iat1)=Esum2/sum
     num_AEM_Egroup=num_AEM_Egroup+1
     AEM_Egroup=AEM_Egroup+(Esum1/sum-Esum2/sum)**2
     
     enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
     Etot=0.d0
     Etot_pred=0.d0
     do i=1,natom
     Etot=Etot+energy(i)
     Etot_pred=Etot_pred+energy_pred(i)
     enddo
     write(70,*) Etot, Etot_pred

     num_AEM_Etot=num_AEM_Etot+1
     AEM_Etot=AEM_Etot+(Etot-Etot_pred)**2
      do i=1,natom
      itype=iatom_type(i)
      ! liuliping, output energy forces in sequence of atoms in images.
      write(1314,*) energy(i),energy_pred(i)
      write(1315,*) force(1,i),force_pred(1,i)
      write(1315,*) force(2,i),force_pred(2,i)
      write(1315,*) force(3,i),force_pred(3,i)
      ! liuliping, end

      write(20+itype,*) energy(i),energy_pred(i)
      write(90,*) energy_group(i),energy_group_pred(i)
      write(40+itype,*) force(1,i),force_pred(1,i)
      write(40+itype,*) force(2,i),force_pred(2,i)
      write(40+itype,*) force(3,i),force_pred(3,i)
      num_AEM_Eatom=num_AEM_Eatom+1
      AEM_Eatom=AEM_Eatom+(energy(i)-energy_pred(i))**2
      num_AEM_F=num_AEM_F+3
      AEM_F=AEM_F+(force(1,i)-force_pred(1,i))**2+  &
     & (force(2,i)-force_pred(2,i))**2+(force(3,i)-force_pred(3,i))**2
      enddo

3000   continue
      ! 1314 and 1315 contain energy-forces for all images
      close(1314)
      close(1315)

     deallocate(iatom_type)
     deallocate(Energy)
     deallocate(Energyt)
     deallocate(num_neight)
     deallocate(Energy_pred)
     deallocate(Energy_group)
     deallocate(Energy_group_pred)
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

     deallocate(ipiv)
     deallocate(num_inv)
     deallocate(index_inv)
     deallocate(index_inv2)
     deallocate(force)
     deallocate(force_pred)
     deallocate(VV)
     deallocate(SS)


900    continue

     close(70)

     write(6,*) "RMSE Eatom=", dsqrt(AEM_Eatom/num_AEM_Eatom) 
     write(6,*) "RMSE Egroup=", dsqrt(AEM_Egroup/num_AEM_Egroup) 
     write(6,*) "RMSE Etot=", dsqrt(AEM_Etot/num_AEM_Etot) 
     write(6,*) "RMSE F=", dsqrt(AEM_F/num_AEM_F) 

     deallocate(fread_dfeat)
     stop
     end

     
