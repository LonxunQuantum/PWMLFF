module calc_SNAP_feature
    use mod_mpi

    IMPLICIT NONE
    INTEGER :: ierr

    integer,allocatable,dimension (:) :: iatom
    integer num_step, natom, i, j,natom6,m_neigh6
    integer num_step0,num_step1,natom0,max_neigh
    real*8 Etotp_ave,E_tolerance

    integer n2b_t,n3b1_t,n3b2_t,it
    integer n2b_type(100),n2bm

    real*8 Rc_M

    real*8,allocatable,dimension (:,:) :: feat_M6
    real*8,allocatable,dimension (:,:,:,:) :: dfeat_M6
    integer,allocatable,dimension (:,:) :: list_neigh_alltypeM6
    integer,allocatable,dimension (:) :: num_neigh_alltypeM6
    integer nfeat0M6

    integer m_neigh,num,itype1,itype2,itype
    integer iat1,max_neigh_M,num_M
    
    integer ntype,n2b,n3b1,n3b2,nfeat0m,nfeat0(100)
    integer ntype1,ntype2,k1,k2,k12,ii_f,iat,ixyz
    integer iat_type(100)
    real*8 Rc, Rc2,Rm
    real*8 alpha31,alpha32

    real*8 Rc_type(100), Rm_type(100)
    integer nsnapw_type(50)    ! the number of weight combination
    real*8 snapj_type(50)      !  the J  (can be integer, or hald integer, 1, 1.5)
    real*8 wsnap_type(50,10,50)  ! the weight
    integer nBB(50)
   
    integer k,ii,nBBm,jmm,jm,j1,j2,m1,m2,ms,is
    real*8 prod,prod0

    real*8, external:: factorial 
    real*8,allocatable,dimension (:,:,:,:) :: CC_func
    integer,allocatable,dimension (:,:,:) :: jjj123
    real*8, allocatable, dimension (:,:,:,:,:,:) :: Clebsch_Gordan

    contains
    subroutine load_model_type6()
    

        open(10,file="input/gen_SNAP_feature.in",status="old",action="read")
        rewind(10)
        read(10,*) Rc_M,m_neigh
        m_neigh6=m_neigh
        read(10,*) ntype
        do i=1,ntype
        read(10,*) iat_type(i)
        read(10,*) Rc_type(i)
           if(Rc_type(i).gt.Rc_M) then
           write(6,*) "Rc_type must be smaller than Rc_M, gen_3b_feature.in",i,Rc_type(i),Rc_M
           stop
           endif
          read(10,*) snapj_type(i),nsnapw_type(i) ! the type of snap within this atom type, each type is indicated by one J
          do k=1,nsnapw_type(i)
          read(10,*) (wsnap_type(ii,k,i),ii=1,ntype)
          enddo
        enddo
        read(10,*) E_tolerance
        close(10)
    
        nfeat0m=0
        nBBm=0
        do itype=1,ntype
        jm=snapj_type(itype)*2*1.0001     ! double index 
        num=0
        do j=0,jm     ! jm is the double index, the real j = j/2
        do j1=0,j     ! double index, original: 0,0.5,1,1.5
        do j2=0,j1    ! double index
        if(abs(j1-j2).le.j.and.j.le.j1+j2.and.mod(j1+j2-j+100,2).eq.0) then
        num=num+1       ! num os the index of the feature, and its corresponding j1,j2,j
        endif
        enddo
        enddo
        enddo
        nBB(itype)=num
        nfeat0(itype)=num*nsnapw_type(itype)
        if(nBB(itype).gt.nBBm) nBBm=nBB(itype)
        if(nfeat0(itype).gt.nfeat0m) nfeat0m=nfeat0(itype)
        enddo
   
        allocate(jjj123(3,nBBm,ntype))
   
        jmm=0
        do itype=1,ntype
        jm=snapj_type(itype)*2*1.0001     ! double index 
        if(jm.gt.jmm) jmm=jm
        num=0
        do j=0,jm     ! jm is the double index
        do j1=0,j     ! double index, original: 0,0.5,1,1.5
        do j2=0,j1    ! double index
        if(abs(j1-j2).le.j.and.j.le.j1+j2.and.mod(j1+j2-j+100,2).eq.0) then
        num=num+1       ! num os the index of the feature, and its corresponding j1,j2,j
        jjj123(1,num,itype)=j1
        jjj123(2,num,itype)=j2
        jjj123(3,num,itype)=j
        endif
        enddo
        enddo
        enddo
   
        write(6,*) "itype,num,nBB(itype),nfeat0",itype,nBB(itype),nfeat0(itype)
        enddo
        
        write(6,*) "jmm=",jmm
   
   !-------------------------------------------
        allocate(CC_func(0:jmm,-jmm:jmm,-jmm:jmm,0:jmm))
        CC_func=0.d0
   
        do j=0,jmm   ! double index
        do m2=-j,j,2
        do m1=-j,j,2
        if(m1+m2.ge.0) then
        prod0=factorial((j+m1)/2)*factorial((j-m1)/2)*factorial((j+m2)/2)*factorial((j-m2)/2)
        prod0=dsqrt(1.d0*prod0)
   
        ms=j-m1
        if(j-m2.lt.ms) ms=j-m2
        do is=0,ms/2   
        prod=prod0/(factorial(is)*factorial(is+(m1+m2)/2)*factorial((j-m1)/2-is)*factorial((j-m2)/2-is))
        CC_func(is,m1,m2,j)=prod
        enddo
        endif
        enddo
        enddo
        enddo
   !-------------------------------------------
        allocate(Clebsch_Gordan(-jmm:jmm,-jmm:jmm,-jmm:jmm,0:jmm,0:jmm,0:jmm))
        call calc_Clebsch_Gordan(Clebsch_Gordan,jmm)
     
    !cccccccccccccccccccccccccccccccccccccccc
    
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
!  FInish the initial grid treatment

    end subroutine load_model_type6
    
    subroutine set_image_info_type6(atom_type_list,is_reset,natom_tmp)
        integer(4),dimension(:),intent(in) :: atom_type_list(natom_tmp)
        logical,intent(in) :: is_reset
        integer,intent(in) :: natom_tmp
        
            natom=natom_tmp
            natom6=natom

            if(is_reset) then 
              if(allocated(iatom))then
              deallocate(iatom)
              endif
            allocate(iatom(natom))                   
            iatom(1:natom)=atom_type_list(1:natom)
           endif
        
    end subroutine set_image_info_type6

    subroutine gen_SNAP_feature(AL,xatom)
        integer(4)  :: itype,i,j
        integer ii,jj,jjm,iat
    
        real(8), intent(in) :: AL(3,3)
        real(8),dimension(:,:),intent(in) :: xatom
    
        integer,allocatable,dimension (:,:,:) :: list_neigh,iat_neigh,iat_neigh_M
        integer,allocatable,dimension (:,:) :: num_neigh
        real*8,allocatable,dimension (:,:,:,:) :: dR_neigh
        integer,allocatable,dimension (:,:) :: map2neigh_alltypeM
        integer,allocatable,dimension (:,:) :: list_tmp
        integer,allocatable,dimension (:) :: nfeat_atom
        integer,allocatable,dimension (:) :: itype_atom
        integer,allocatable,dimension (:,:,:) :: map2neigh_M
        integer,allocatable,dimension (:,:,:) :: list_neigh_M
        integer,allocatable,dimension (:,:) :: num_neigh_M
        integer,allocatable,dimension (:,:) :: list_neigh_alltype
        integer,allocatable,dimension (:) :: num_neigh_alltype
        real*8 :: timestart, timefinish
        real*8,allocatable,dimension (:,:) :: feat
        real*8,allocatable,dimension (:,:,:,:) :: dfeat
        real*8 tt2,tt1,tt0,tt3,tt4,tt5,tt6
    !cccccccccccccccccccccccccccccccccccccccccccccccccccc
        if (allocated(dfeat_M6)) then
            deallocate(feat_M6)
            deallocate(dfeat_M6)
            deallocate(list_neigh_alltypeM6)
            deallocate(num_neigh_alltypeM6)
        endif
    
    ! the dimension of these array, should be changed to natom_n
    ! instead of natom. Each process only needs to know its own natom_n
    
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
        allocate(list_neigh_alltypeM6(m_neigh,natom))
        allocate(num_neigh_alltypeM6(natom))
        allocate(map2neigh_alltypeM(m_neigh,natom)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
        allocate(list_tmp(m_neigh,ntype))
        allocate(itype_atom(natom))
        allocate(nfeat_atom(natom))
    
        allocate(feat(nfeat0m,natom_n))         ! each note, only know its own feat
        allocate(dfeat(nfeat0m,natom_n,m_neigh,3))  ! dfeat is the derivative from the neighboring dR, 
        allocate(feat_M6(nfeat0m,natom_n))
        allocate(dfeat_M6(nfeat0m,natom_n,m_neigh,3))
    
        feat = 0.d0
        dfeat = 0.d0
        feat_M6 = 0.d0
        dfeat_M6 = 0.d0
        
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
    !    call cpu_time(timestart)
         call find_neighbore(iatom,natom,xatom,AL,Rc_type,num_neigh,list_neigh, &
         dR_neigh,iat_neigh,ntype,iat_type,m_neigh,Rc_M,map2neigh_M,list_neigh_M, &
         num_neigh_M,iat_neigh_M,inode,nnodes)
    
    !    call cpu_time(timefinish)
    !    write(*,*) 'find_nei time: ', timefinish-timestart
    !ccccccccccccccccccccccccccccccccc
    !ccccccccccccccccccccccccccccccccc
    
          max_neigh=-1
          num_neigh_alltype=0
          max_neigh_M=-1
          num_neigh_alltypeM6=0
          list_neigh_alltypeM6=0
    
          do iat=1,natom
    
          if(mod(iat-1,nnodes).eq.inode-1) then
    
    
          list_neigh_alltype(1,iat)=iat
          list_neigh_alltypeM6(1,iat)=iat
    
    
          num_M=1
          do itype=1,ntype
          do j=1,num_neigh_M(itype,iat)
          num_M=num_M+1
          if(num_M.gt.m_neigh) then
          write(6,*) "total num_neigh.gt.m_neigh,stop",m_neigh
          stop
          endif
          list_neigh_alltypeM6(num_M,iat)=list_neigh_M(j,itype,iat)
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
    
    !ccccccccccccccccccccccccccccccccccccccc
    
    
          num_neigh_alltype(iat)=num
          num_neigh_alltypeM6(iat)=num_M
          if(num.gt.max_neigh) max_neigh=num
          if(num_M.gt.max_neigh_M) max_neigh_M=num_M
    
          endif
    
          enddo  ! iat
    
    !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    ! This num_neigh_alltype(iat) include itself !
    !    dfeat=0.d0
    !    feat=0.d0
    
          call find_feature_snap(natom,itype_atom,Rc_type,nsnapw_type,snapj_type,wsnap_type,num_neigh,  &
          list_neigh,dR_neigh,iat_neigh,ntype, &
          feat,dfeat,nfeat0m,m_neigh,nfeat_atom,nBB,nBBm,jjj123,CC_func,Clebsch_Gordan,jmm)
   
    !cccccccccccccccccccccccccccccccccccccccccccccccccccc
    

        do iat1=1,natom_n
        do ii=1,nfeat0m
        feat_M6(ii,iat1)=feat(ii,iat1)   
        enddo
        enddo
    
        iat1=0
        do iat=1,natom
        if(mod(iat-1,nnodes).eq.inode-1) then
        iat1=iat1+1
        do jj=1,num_neigh_alltype(iat)
        jjm=map2neigh_alltypeM(jj,iat)
        do ii=1,nfeat0m
        dfeat_M6(ii,iat1,jjm,1)=dfeat(ii,iat1,jj,1)  ! this is the feature stored in neigh list of Rc_M
        dfeat_M6(ii,iat1,jjm,2)=dfeat(ii,iat1,jj,2)
        dfeat_M6(ii,iat1,jjm,3)=dfeat(ii,iat1,jj,3)
        enddo
        enddo
        endif
        enddo
    
        nfeat0M6=nfeat0m    ! the number of features for feature type 1
    
    
        deallocate(list_neigh)
        deallocate(iat_neigh)
        deallocate(dR_neigh)
        deallocate(num_neigh)
        ! deallocate(feat)
        ! deallocate(dfeat)
        deallocate(list_neigh_alltype)
        deallocate(num_neigh_alltype)
    
    
        deallocate(list_neigh_M)
        deallocate(num_neigh_M)
        deallocate(map2neigh_M)
        deallocate(iat_neigh_M)
        ! deallocate(list_neigh_alltypeM6)
        ! deallocate(num_neigh_alltypeM6)
        deallocate(map2neigh_alltypeM)
        deallocate(list_tmp)
        deallocate(itype_atom)
        deallocate(nfeat_atom)
        deallocate(feat)
        deallocate(dfeat)
    



    end subroutine gen_SNAP_feature


end module calc_SNAP_feature