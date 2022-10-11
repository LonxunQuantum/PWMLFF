program linear_forceMM
    use data_ewald
    implicit double precision (a-h,o-z)
    integer lwork
    integer,allocatable,dimension(:) :: iatom,iatom_type,itype_atom
    real*8,allocatable,dimension(:) :: Energy,Energyt
    real*8,allocatable,dimension(:,:) :: feat,feat2,feat22_type
    real*8,allocatable,dimension(:,:,:) :: feat_type,feat2_type
    real*8,allocatable,dimension(:,:) :: feat2_group
    real*8,allocatable,dimension(:) :: energy_group
    integer,allocatable,dimension(:) :: num_neigh,num,num_atomtype
    integer,allocatable,dimension(:) :: num_neight
    integer,allocatable,dimension(:,:) :: list_neigh,ind_type

    real*8,allocatable,dimension(:,:,:,:) :: dfeat,dfeat2
    real*8,allocatable,dimension(:,:,:) :: dfeat_type,dfeat2_type

    real*8,allocatable,dimension(:,:) :: AA,AA_tmp
    real*8,allocatable,dimension(:) :: BB,BB_tmp

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
    real*8,allocatable,dimension(:) :: rad_atom
    real*8,allocatable,dimension(:,:,:) :: wp_atom
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
        !write(*,*) "zatom_ewald: ", zatom_ewald(8), zatom_ewald(72)
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



    open(10,file=fread_dfeat//"fit_linearMM.input")
    rewind(10)
    read(10,*) ntype,m_neigh
    allocate(itype_atom(ntype))
    allocate(nfeat1(ntype))
    allocate(nfeat2(ntype))
    allocate(nfeat2i(ntype))
    allocate(rad_atom(ntype))
    allocate(wp_atom(ntype,ntype,2))
    wp_atom=0.d0
    do i=1,ntype
    read(10,*) itype_atom(i)   !rad_atom(i),wp_atom(i)
    enddo
    read(10,*) weight_E,weight_E0,weight_F,delta
    read(10,*) dwidth
    close(10)

    open(10,file=fread_dfeat//"vdw_fitB.ntype")
    rewind(10)
    read(10,*) ntype_t,nterm
    if(ntype_t.ne.ntype) then
    write(6,*) "ntype not same in vwd_fitB.ntype,something wrong"
    stop
    endif
    do itype1=1,ntype
    read(10,*) itype_t,rad_atom(itype1),E_ave_vdw,((wp_atom(i,itype1,j1),i=1,ntype),j1=1,nterm)
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
    allocate(AA_tmp(nfeat2tot,nfeat2tot))
    allocate(BB_tmp(nfeat2tot))
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
    deallocate(feat2_group)
    deallocate(energy_group)

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
    allocate(feat2_group(nfeat2tot,natom))
    allocate(energy_group(natom))

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
!       write(6,"(2(i4,1x),3(f10.5,1x),2x,f13.6)") i,j,dx1,dx2,dx3,dd
!       w22=dsqrt(wp_atom(iatom_type(i))*wp_atom(iatom_type(j)))
    w22_1=wp_atom(iatom_type(j),iatom_type(i),1)
    w22_2=wp_atom(iatom_type(j),iatom_type(i),2)
    w22F_1=(wp_atom(iatom_type(j),iatom_type(i),1)+wp_atom(iatom_type(i),iatom_type(j),1))/2     ! take the average for force calc. 
    w22F_2=(wp_atom(iatom_type(j),iatom_type(i),2)+wp_atom(iatom_type(i),iatom_type(j),2))/2     ! take the average for force calc. 
    yy=pi*dd/(4*rad)
!       dE=dE+0.5*w22*exp((1-dd/rad)*4.0)*cos(yy)**2
!       dEdd=w22*exp((1-dd/rad)*4.d0)*((-4/rad)*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy))

    dE=dE+0.5*4*(w22_1*(rad/dd)**12*cos(yy)**2+w22_2*(rad/dd)**6*cos(yy)**2)
    dEdd=4*(w22F_1*(-12*(rad/dd)**12/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)&
     &   +W22F_2*(-6*(rad/dd)**6/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**6))

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
    ! liuliping ewald
    if(iflag_born_charge_ewald .eq. 1) then
        write(*,*) "MLFF predict with ewald"
        allocate(ewald_atom(natom))
        allocate(fatom_ewald(3,natom))
        AL_bohr = AL*Angstrom2Bohr  ! Angstrom2Bohr; to atmoic units
        vol = dabs(AL_bohr(3, 1)*(AL_bohr(1, 2)*AL_bohr(2, 3) - AL_bohr(1, 3)*AL_bohr(2, 2)) &
           + AL_bohr(3, 2)*(AL_bohr(1, 3)*AL_bohr(2, 1) - AL_bohr(1, 1)*AL_bohr(2, 3)) &
           + AL_bohr(3, 3)*(AL_bohr(1, 1)*AL_bohr(2, 2) - AL_bohr(1, 2)*AL_bohr(2, 1)))

        call get_ewald(natom, AL_bohr, iatom, xatom, zatom_ewald, ewald, ewald_atom, fatom_ewald)
        ! forces in get_ewald and fit_lin are both dE/dx
        !write(*,*) "fit-lin-ewald, before ewald(1:3):", energy(1:3)
        !write(*,*) "fit-lin-ewald, ewald(1:3) hartree:", ewald_atom(1:3)
        !write(*,*) "Hartree2eV: ", Hartree2eV
        !write(*,*) "fit-lin-ewald, ewald(1:3):", ewald_atom(1:3)*Hartree2eV
        energy(1:natom) = energy(1:natom) - ewald_atom(1:natom)*Hartree2eV
        force(1:3,1:natom) = force(1:3,1:natom) - fatom_ewald(1:3,1:natom)*Hartree2eV*Angstrom2Bohr
        !write(*,*) "fit-lin-ewald, after ewald(1:3):", energy(1:3)
        deallocate(ewald_atom)
        deallocate(fatom_ewald)
    endif

    do itype=1,ntype
    call dgemm('T','N',nfeat2(itype),num(itype),nfeat1(itype),1.d0,PV(1,1,itype), &
     & nfeat1m,feat_type(1,1,itype),nfeat1m,0.d0,feat2_type(1,1,itype), nfeat2m)
    enddo

    do itype=1,ntype
    do i=1,num(itype)
    do j=1,nfeat2(itype)-1
    feat2_type(j,i,itype)=(feat2_type(j,i,itype)-feat2_shift(j,itype))*feat2_scale(j,itype)
    enddo
    feat2_type(nfeat2(itype),i,itype)=1.d0
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

!ccccccccccccccccccccccccccccccccccccc  
    do itype=1,ntype
    do j=1,nfeat2(itype)
    sum=0.d0
    do i=1,num(itype)
    sum=sum+feat2_type(j,i,itype)
    enddo
    feat22_type(j,itype)=sum   ! sum over all the atoms
    enddo
    enddo

    Etot=0.d0
    do i=1,natom
    Etot=Etot+energy(i)
    enddo
!ccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccc around each atom, there is a group
!ccc We only fit group energy, instead of each individual atom energy
!ccccc here, the feat in group is the combined feature (j,itype)
    feat2_group=0.d0
    do iat1=1,natom   ! center position (not even call it atom)
    Esum=0.d0
    num=0
    sum=0.d0
    do iat2=1,natom
    itype=iatom_type(iat2)
    iii=nfeat2i(itype)

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
    fact=exp(-dd/dwidth**2)
    Esum=Esum+energy(iat2)*fact
    sum=sum+fact
    do j=1,nfeat2(itype)
    feat2_group(j+iii,iat1)=feat2_group(j+iii,iat1)+feat2_type(j,num(itype),itype)*fact
    enddo
    enddo
    energy_group(iat1)=Esum/sum
    feat2_group(:,iat1)=feat2_group(:,iat1)/sum
    enddo
    
!ccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cc  Now, we have yield the new feature: feat2_type(nfeat2,num(itype),itype)
!ccc The last feature, nfeat2, =1
!ccccccccccccccccccccccccccccccccccccccccccccccccccccc

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    
    call dgemm('N','T',nfeat2tot,nfeat2tot,natom, &
     & 1.d0,feat2_group,nfeat2tot,feat2_group,nfeat2tot,0.d0,AA_tmp,nfeat2tot)

    do j=1,nfeat2tot
    sum=0.d0
    do i=1,natom
    sum=sum+energy_group(i)*feat2_group(j,i)
    enddo
    BB_tmp(j)=sum
    enddo

    AA=AA+AA_tmp*weight_E
    BB=BB+BB_tmp*weight_E
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccc Now, we have finished the energy part. In the following, we will 
!ccc include the force part. Which is more complicated. 
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc

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
    call dgemm('T','N',nfeat2(itype),num(itype),nfeat1(itype), 1.d0,PV(1,1,itype),&
     & nfeat1m,dfeat_type(1,1,itype),nfeat1m,0.d0, dfeat2_type(1,1,itype), nfeat2m)
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
      
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cc  Now, dfeat2 is: 
!cc dfeat2(nfeat2,natom,j_neigh,3): dfeat2(j,i,jj,3)= d/dR_i(feat2(j,list_neigh(jj,i))
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
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
    do itype=1,ntype
    do j=1,nfeat2(itype)
    VV(j+nfeat2i(itype),(i-1)*3+1)=SS(j,i,1,itype) 
    VV(j+nfeat2i(itype),(i-1)*3+2)=SS(j,i,2,itype) 
    VV(j+nfeat2i(itype),(i-1)*3+3)=SS(j,i,3,itype) 
    enddo
    enddo
    enddo
    

    call dgemm('N','T',nfeat2tot,nfeat2tot,3*natom,weight_F,&
     & VV,nfeat2tot,VV,nfeat2tot,1.d0,AA,nfeat2tot)


    do itype=1,ntype
    do j=1,nfeat2(itype)
    sum=0.d0
    do i=1,natom
    do ixyz=1,3
    sum=sum+force(ixyz,i)*VV(j+nfeat2i(itype),(i-1)*3+ixyz)
    enddo
    enddo

    BB(j+nfeat2i(itype))=BB(j+nfeat2i(itype))+sum*weight_F
    enddo
    enddo


!        do itype=1,ntype
!        iii=nfeat2i(itype)
!        do k1=1,nfeat2(itype)
!        do k2=1,nfeat2(itype)
!        AA(k1+iii,k2+iii)=AA(k1+iii,k2+iii)+
!     &              weight_E*AA_type(k1,k2,itype)
!        enddo
!        enddo
!        enddo

!        do itype=1,ntype
!        iii=nfeat2i(itype)
!        do k=1,nfeat2(itype)
!        BB(k+iii)=BB(k+iii)+weight_E*BB_type(k,itype)
!        enddo
!        enddo


    do itype1=1,ntype
    iii1=nfeat2i(itype1)
    do k1=1,nfeat2(itype1)
    do itype2=1,ntype
    iii2=nfeat2i(itype2)
    do k2=1,nfeat2(itype2)
    AA(k1+iii1,k2+iii2)=AA(k1+iii1,k2+iii2)+ &
     &     feat22_type(k1,itype1)*feat22_type(k2,itype2)*weight_E0
    enddo
    enddo
    enddo
    enddo

    do itype=1,ntype
    iii=nfeat2i(itype)
    do k=1,nfeat2(itype)
!        BB(k+iii)=BB(k+iii)+feat22_type(k,itype)*Etot*weight_E0/natom
    BB(k+iii)=BB(k+iii)+feat22_type(k,itype)*Etot*weight_E0
    enddo
    enddo



3000   continue



    do k1=1,nfeat2tot
    AA(k1,k1)=AA(k1,k1)+delta
    enddo

    close(10)



900    continue

    do kkk=1,nfeat_type
    close(1000+kkk)
    enddo

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc

    
    call dgesv(nfeat2tot,1,AA,nfeat2tot,ipiv,BB,nfeat2tot,info)  
!cccccccccccccccccccccccccccccccccccccccccccccccccc
    open(10,file=fread_dfeat//"linear_fitB.ntype")
    rewind(10)
    write(10,*) nfeat2tot
    do i=1,nfeat2tot
    write(10,*) i, BB(i)
    enddo
    close(10)

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    

    deallocate(fread_dfeat)
    stop
    end

    
