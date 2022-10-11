program fit_vdW
       implicit double precision (a-h,o-z)


       integer lwork
       integer,allocatable,dimension(:) :: iatom,iatom_type,itype_atom
       real*8,allocatable,dimension(:) :: Energy,Energyt
       real*8,allocatable,dimension(:,:) :: feat,feat2,feat22_type
       real*8,allocatable,dimension(:,:,:) :: feat_type,feat2_type
       integer,allocatable,dimension(:) :: num_neigh,num,num_atomtype
       integer,allocatable,dimension(:) :: num_neight
       integer,allocatable,dimension(:,:) :: list_neigh,ind_type

       real*8,allocatable,dimension(:,:,:,:) :: dfeat,dfeat2
       real*8,allocatable,dimension(:,:,:) :: dfeat_type,dfeat2_type

       real*8,allocatable,dimension(:,:) :: AA,AAt,BBt
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


       real*8,allocatable,dimension(:,:) :: WW,VV,QQ
       real*8,allocatable,dimension(:,:,:,:) :: SS

       real*8,allocatable,dimension(:,:) :: Gfeat2,dGfeat2

       real*8,allocatable,dimension(:,:) :: force

     
       real*8,allocatable,dimension(:,:) :: xatom
       real*8,allocatable,dimension(:) :: rad_atom,wp_atom
       real*8 AL(3,3),pi,dE,dFx,dFy,dFz,AL_tmp(3,3)

       real*8,allocatable,dimension(:,:) :: xatom_tmp

 
       integer,allocatable,dimension(:) :: num_inv
       integer,allocatable,dimension(:,:) :: index_inv,index_inv2

       integer,allocatable,dimension(:) :: nfeat1,nfeat2,nfeat2i
       integer,allocatable,dimension(:,:) :: nfeat,ipos_feat

       real*8, allocatable, dimension (:,:) :: dfeat_tmp
       real*8, allocatable, dimension (:,:) :: feat_ftype
       integer,allocatable, dimension (:) :: iat_tmp,jneigh_tmp,ifeat_tmp
       integer num_tmp,jj
       ! character(len=200) dfeat_n(400)
       character(len=200) trainSetFileDir(400)
       character(len=200) trainSetDir
       character(len=200) MOVEMENTDir,dfeatDir,infoDir,trainDataDir,MOVEMENTallDir
       integer sys_num,sys
       integer nfeat1tm(100),ifeat_type(100),nfeat1t(100)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       real*8,allocatable,dimension (:,:,:,:,:,:) :: AA_vdw
       real*8,allocatable,dimension (:,:,:) :: BB_vdw,BB0_vdw
       real*8,allocatable,dimension (:,:) :: sum_vdw
       real*8,allocatable,dimension (:) :: E_ave_vdw
       real*8,allocatable,dimension (:,:,:) :: sumF_vdw
       real*8,allocatable,dimension (:,:,:) :: ff
       real*8,allocatable,dimension (:,:) :: weight_type,weight_type2
       real*8,allocatable,dimension (:) :: w_vdw_type
       real*8,allocatable,dimension (:,:) :: dd_vdw_type
       integer,allocatable,dimension (:) :: num_ave_vdw
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


!       write(6,*) "input w_f,nterm"
!       read(5,*) w_f,nterm   ! force weight
        w_f=0.2  ! weight for force term
        nterm=1  ! 1/r^12, and 1/r^6


       open(10,file=fread_dfeat//"fit_linearMM.input")
       rewind(10)
       read(10,*) ntype,m_neigh
       allocate(itype_atom(ntype))
       allocate(nfeat1(ntype))
       allocate(nfeat2(ntype))
       allocate(nfeat2i(ntype))
       allocate(rad_atom(ntype))
       allocate(wp_atom(ntype))
       do i=1,ntype
!       read(10,*) itype_atom(i),   !rad_atom(i),wp_atom(i)
       read(10,*) itype_atom(i)     !rad_atom(i)
       rad_atom(i)=1.d0    ! always use 1
       enddo
       read(10,*) weight_E,weight_E0,weight_F,delta
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
       nfeat2tot=0 ! tht total feature of diff atom type
       nfeat2i=0   ! the starting point
       nfeat2i(1)=0
       do i=1,ntype
       if(nfeat1(i).gt.nfeat1m) nfeat1m=nfeat1(i)
       if(nfeat2(i).gt.nfeat2m) nfeat2m=nfeat2(i)
       nfeat2tot=nfeat2tot+nfeat2(i)
       if(i.gt.1) then
       nfeat2i(i)=nfeat2i(i-1)+nfeat2(i-1)
       endif

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

       allocate(num(ntype))
       allocate(num_atomtype(ntype))
       allocate(AA(nfeat2tot,nfeat2tot))
       allocate(BB(nfeat2tot))
       allocate(AA_type(nfeat2m,nfeat2m,ntype))
       allocate(BB_type(nfeat2m,ntype))

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

       AA=0.d0
       BB=0.d0

!       nterm=2

       allocate(AA_vdw(ntype,ntype,nterm,ntype,ntype,nterm))
       allocate(BB_vdw(ntype,ntype,nterm))
       allocate(BB0_vdw(ntype,ntype,nterm))
       allocate(sum_vdw(ntype,4))
       allocate(sumF_vdw(3,ntype,4))
       allocate(E_ave_vdw(ntype))
       allocate(num_ave_vdw(ntype))
       allocate(weight_type(ntype,ntype))
       allocate(weight_type2(ntype,ntype))
       allocate(dd_vdw_type(ntype,ntype))



       do 6000 iloop=1,2

       AA_vdw=0.d0
       BB_vdw=0.d0
       BB0_vdw=0.d0
       E_ave_vdw=0.d0
       num_ave_vdw=0.d0

       if(iloop.eq.1) then
       weight_type=0.d0
       weight_type2=1.d0
       endif

       if(iloop.eq.2) then
       sum=0.d0
       do it2=1,ntype
       do it1=1,ntype
       sum=sum+1.d0/weight_type(it1,it2)
       enddo
       enddo
       do it2=1,ntype
       do it1=1,ntype
       weight_type2(it1,it2)=1.d0/weight_type(it1,it2)/sum
       if(dd_vdw_type(it1,it2).gt.3.2d0) then   ! Angstrom
! We should use ionic radius to judge
       weight_type2(it1,it2)=weight_type2(it1,it2)/20  
       endif
       enddo
       enddo
       weight_type=0.d0
       endif

       dd_vdw_type=1.D+30


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
         if(sys.ne.1.or.iloop.eq.2) then
         deallocate(iatom)
         endif
       allocate(iatom(natom))
       endif
       read(1000+kkk) iatom      ! The same for different kkk

777    continue
!ccccccccccccccccccccccccccccccccccccccccccccccccc
       
           
       if (sys.ne.1.or.iloop.eq.2) then      

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
       allocate(feat22_type(nfeat2m,ntype))
       allocate(num_neigh(natom))
       allocate(list_neigh(m_neigh,natom))
       allocate(ind_type(natom,ntype))
       allocate(dfeat(nfeat1m,natom,m_neigh,3))
       allocate(dfeat_type(nfeat1m,natom*m_neigh*3,ntype))
       allocate(dfeat2_type(nfeat2m,natom*m_neigh*3,ntype))
       allocate(dfeat2(nfeat2m,natom,m_neigh,3))
       allocate(xatom(3,natom))

       dfeat=0.d0
       dfeat_type=0.d0

       allocate(ipiv(nfeat2tot))
       allocate(num_inv(natom))
       allocate(index_inv(3*m_neigh,natom))
       allocate(index_inv2(3*m_neigh,natom))
       allocate(force(3,natom))
       allocate(VV(nfeat2tot,3*natom))
       allocate(SS(nfeat2m,natom,3,ntype))



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
       read(1000+kkk) AL       ! AL(3,3), repeated for diff kkk, this is angstrom

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
       itype1=iatom_type(i)
       rad1=rad_atom(iatom_type(i))
       dE=0.d0
       dFx=0.d0
       dFy=0.d0
       dFz=0.d0
       sum_vdw=0.d0
       sumF_vdw=0.d0
       w_vdw=0.d0
       w_vdw_type=0.d0

       
       do jj=1,num_neigh(i)
       j=list_neigh(jj,i)
       if(i.ne.j) then
       itype2=iatom_type(j)
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

       if(dd.lt.dd_vdw_type(itype1,itype2)) then
       dd_vdw_type(itype1,itype2)=dd
       dd_vdw_type(itype2,itype1)=dd
       endif

       if(dd.lt.2*rad) then
!       write(6,"(2(i4,1x),3(f10.5,1x),2x,f13.6)") i,j,dx1,dx2,dx3,dd
!       w22=dsqrt(wp_atom(iatom_type(i))*wp_atom(iatom_type(j)))
       yy=pi*dd/(4*rad)
!       dE=dE+0.5*w22*exp((1-dd/rad)*4.0)*cos(yy)**2
!       dEdd=w22*exp((1-dd/rad)*4.d0)*((-4/rad)*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy))
       sum_vdw(itype2,1)=sum_vdw(itype2,1)+0.5*4*(rad/dd)**12*cos(yy)**2
       sum_vdw(itype2,2)=sum_vdw(itype2,2)+0.5*4*(rad/dd)**6*cos(yy)**2
       w_vdw=w_vdw+(rad/dd)**12*weight_type2(itype2,itype1)
       weight_type(itype2,itype1)=weight_type(itype2,itype1)+(rad/dd)**12

       dEdd=0.5*4*(-12*(rad/dd)**12/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)
       dEdd2=0.5*4*(-6*(rad/dd)**6/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**6)
       sumF_vdw(1,itype2,1)=sumF_vdw(1,itype2,1)-dEdd*dx/dd
       sumF_vdw(2,itype2,1)=sumF_vdw(2,itype2,1)-dEdd*dy/dd
       sumF_vdw(3,itype2,1)=sumF_vdw(3,itype2,1)-dEdd*dz/dd
       sumF_vdw(1,itype2,2)=sumF_vdw(1,itype2,2)-dEdd2*dx/dd
       sumF_vdw(2,itype2,2)=sumF_vdw(2,itype2,2)-dEdd2*dy/dd
       sumF_vdw(3,itype2,2)=sumF_vdw(3,itype2,2)-dEdd2*dz/dd

!       dE=dE+0.5*4*w22*(rad/dd)**12*cos(yy)**2
!       dEdd=4*w22*(-12*(rad/dd)**12/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)
!       dFx=dFx-dEdd*dx/dd       ! note, -sign, because dx=d(j)-x(i)
!       dFy=dFy-dEdd*dy/dd
!       dFz=dFz-dEdd*dz/dd
       endif
       endif
       enddo
!       write(6,*) "dE,dFx",dE,dFx

!       w_f=0.2
     
       do j1=1,nterm
       do j2=1,nterm
       do it1=1,ntype
       do it2=1,ntype
!       AA_vdw(it1,it2,itype1)=AA_vdw(it1,it2,itype1)+(sum_vdw(it1)*
! itype1 is the center atom
       AA_vdw(it1,itype1,j1,it2,itype1,j2)=AA_vdw(it1,itype1,j1,it2,itype1,j2)+&
     &    (sum_vdw(it1,j1)*sum_vdw(it2,j2)+w_f*sumF_vdw(1,it1,j1)*sumF_vdw(1,it2,j2) &
     &   +w_f*sumF_vdw(2,it1,j1)*sumF_vdw(2,it2,j2) &
     &   +w_f*sumF_vdw(3,it1,j1)*sumF_vdw(3,it2,j2))*w_vdw    ! 0.2 is the weight for force
       AA_vdw(itype1,it1,j1,itype1,it2,j2)=AA_vdw(itype1,it1,j1,itype1,it2,j2) &
     &   +(w_f*sumF_vdw(1,it1,j1)*sumF_vdw(1,it2,j2)+w_f*sumF_vdw(2,it1,j1)*sumF_vdw(2,it2,j2) &
     &   +w_f*sumF_vdw(3,it1,j1)*sumF_vdw(3,it2,j2))*w_vdw      ! 0.2 is the weight for force
       AA_vdw(it1,itype1,j1,itype1,it2,j2)=AA_vdw(it1,itype1,j1,itype1,it2,j2) &
     &   +(w_f*sumF_vdw(1,it1,j1)*sumF_vdw(1,it2,j2)+w_f*sumF_vdw(2,it1,j1)*sumF_vdw(2,it2,j2)&
     &   +w_f*sumF_vdw(3,it1,j1)*sumF_vdw(3,it2,j2))*w_vdw      ! 0.2 is the weight for force
       AA_vdw(itype1,it1,j1,it2,itype1,j2)=AA_vdw(itype1,it1,j1,it2,itype1,j2)&
     &   +(w_f*sumF_vdw(1,it1,j1)*sumF_vdw(1,it2,j2)+w_f*sumF_vdw(2,it1,j1)*sumF_vdw(2,it2,j2)&
     &   +w_f*sumF_vdw(3,it1,j1)*sumF_vdw(3,it2,j2))*w_vdw      ! 0.2 is the weight for force
       enddo
       enddo
       enddo
       enddo

       do j1=1,nterm
       do it1=1,ntype
       BB_vdw(it1,itype1,j1)=BB_vdw(it1,itype1,j1)+(energy(i)*sum_vdw(it1,j1)&
     &       +w_f*force(1,i)*sumF_vdw(1,it1,j1)&
     &       +w_f*force(2,i)*sumF_vdw(2,it1,j1)&
     &       +w_f*force(3,i)*sumF_vdw(3,it1,j1))*w_vdw
       BB_vdw(itype1,it1,j1)=BB_vdw(itype1,it1,j1) &
     &      +(w_f*force(1,i)*sumF_vdw(1,it1,j1) &
     &       +w_f*force(2,i)*sumF_vdw(2,it1,j1) &
     &       +w_f*force(3,i)*sumF_vdw(3,it1,j1))*w_vdw 
       BB0_vdw(it1,itype1,j1)=BB0_vdw(it1,itype1,j1)+sum_vdw(it1,j1)*w_vdw     ! this is only for energy
       enddo
       enddo

       E_ave_vdw(itype1)=E_ave_vdw(itype1)+energy(i)
       num_ave_vdw(itype1)=num_ave_vdw(itype1)+1


       energy(i)=energy(i)-dE
       force(1,i)=force(1,i)-dFx   ! Note, assume force=dE/dx, no minus sign
       force(2,i)=force(2,i)-dFy
       force(3,i)=force(3,i)-dFz
       enddo  ! i=1,natom
!cccccccccccccccccccccccccccccccccccccccccc

3000   continue



       close(10)



900    continue

       do kkk=1,nfeat_type
       close(1000+kkk)
       enddo

       do it1=1,ntype
       E_ave_vdw(it1)=E_ave_vdw(it1)/num_ave_vdw(it1)
       enddo

       do j1=1,nterm
       do itype1=1,ntype
       do it1=1,ntype
       BB_vdw(it1,itype1,j1)=BB_vdw(it1,itype1,j1)-E_ave_vdw(itype1)*BB0_vdw(it1,itype1,j1)
!       write(6,*) "TEST BB_vdw",it1,itype1,j1,BB_vdw(it1,itype1,j1)
       enddo
       enddo
       enddo

!       write(6,*) "AA_vdw"
!       do j1=1,nterm
!       do itype1=1,ntype
!       do it1=1,ntype
!       write(6,"(8(E10.3,1x))") (((AA_vdw(it1,itype1,j1,it2,itype2,j2),
!     &   it2=1,ntype),itype2=1,ntype),j2=1,nterm)
!       enddo
!       enddo
!       enddo

!ccccccccccccccccccccccccccccccccccccccccccccccccc

       do j1=1,nterm
       do itype1=1,ntype
       do it1=1,ntype
       AA_vdw(it1,itype1,j1,it1,itype1,j1)=AA_vdw(it1,itype1,j1,it1,itype1,j1)+1.D-8
       enddo
       enddo
       enddo

       call dgesv(ntype**2*nterm,1,AA_vdw,ntype**2*nterm,ipiv,BB_vdw,ntype**2*nterm,info)

!       do it1=1,ntype
!       do it2=1,ntype
!       if(BB_vdw(it1,it2)+BB_vdw(it2,it1).lt.0.d0) then
!       BB_vdw(it1,it2)=0.d0
!       BB_vdw(it2,it1)=0.d0
!       endif
!       enddo
!       enddo



       if(iloop.eq.1) then
       open(10,file=fread_dfeat//"vdw_fitB.ntype0")
       rewind(10)
       write(10,*) ntype,nterm
       do itype1=1,ntype
       write(10,"(i4,2x,2(f10.5,2x),2x,20(E13.6,1x))") itype_atom(itype1), rad_atom(itype1),E_ave_vdw(itype1),&
     &    ((BB_vdw(it1,itype1,j1),it1=1,ntype),j1=1,nterm)
       enddo
       close(10)
       elseif(iloop.eq.2) then
       open(10,file=fread_dfeat//"vdw_fitB.ntype")
       rewind(10)
       write(10,*) ntype,nterm
       do itype1=1,ntype
       write(10,"(i4,2x,2(f10.5,2x),2x,10(E13.6,1x))") itype_atom(itype1), rad_atom(itype1),E_ave_vdw(itype1),&
     &    ((BB_vdw(it1,itype1,j1),it1=1,ntype),j1=1,nterm)
       enddo
       close(10)
       endif

6000   continue   ! iloop

    deallocate(fread_dfeat)
    stop
end


