module calc_ftype1

     use mod_mpi

    IMPLICIT NONE
    INTEGER :: ierr
    integer,allocatable,dimension (:) :: iatom
    integer num_step, natom, i, j,natom1,m_neigh1
    integer num_step0,num_step1,natom0,max_neigh
    real*8 Etotp_ave,E_tolerance

    integer sys_num,sys,recalc_grid


    real*8,allocatable,dimension (:,:) :: grid2
    real*8,allocatable,dimension (:,:,:) :: grid2_2
    integer n2b_t,n3b1_t,n3b2_t,it
    integer n2b_type(100),n2bm

    real*8 Rc_M
    integer n3b1m,n3b2m,kkk,ii

!ccccccccccccccccccccc  The variables to be used in global feature type
    real*8,allocatable,dimension (:,:) :: feat_M1
    real*8,allocatable,dimension (:,:,:,:) :: dfeat_M1
    integer,allocatable,dimension (:,:) :: list_neigh_alltypeM1
    integer,allocatable,dimension (:) :: num_neigh_alltypeM1
    integer nfeat0M1
!ccccccccccccccccccccc  The variables to be used in global feature type

    real*8 sum1,diff

    integer m_neigh,num,itype1,itype2,itype
    integer iat1,max_neigh_M,num_M
    
    integer ntype,n2b,n3b1,n3b2,nfeat0m,nfeat0(100)
    integer ntype1,ntype2,k1,k2,k12,ii_f,iat,ixyz
    integer iat_type(100)
    real*8 Rc, Rc2,Rm
    real*8 alpha31,alpha32

    ! real*8, allocatable, dimension (:,:) :: dfeat_tmp
    ! integer,allocatable, dimension (:) :: iat_tmp,jneigh_tmp,ifeat_tmp
    integer ii_tmp,jj_tmp,iat2,num_tmp,num_tot,i_tmp,jjj,jj
    integer iflag_grid,iflag_ftype
    real*8 fact_grid,dR_grid1,dR_grid2

    real*8 Rc_type(100), Rc2_type(100), Rm_type(100),fact_grid_type(100),dR_grid1_type(100),dR_grid2_type(100)
    integer iflag_grid_type(100),n3b1_type(100),n3b2_type(100)


!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

contains
    subroutine load_model_type1()
    
        open(10,file="input/gen_2b_feature.in",status="old",action="read")
        rewind(10)
        read(10,*) Rc_M,m_neigh
        read(10,*) ntype
        do i=1,ntype
        read(10,*) iat_type(i)
        read(10,*) Rc_type(i),Rm_type(i),iflag_grid_type(i),fact_grid_type(i),dR_grid1_type(i)
        read(10,*) n2b_type(i)
    
         if(Rc_type(i).gt.Rc_M) then
          write(6,*) "Rc_type must be smaller than Rc_M, gen_3b_feature.in",i,Rc_type(i),Rc_M
          stop
         endif
    
        enddo
        read(10,*) E_tolerance
        read(10,*) iflag_ftype
        read(10,*) recalc_grid
        close(10)
        m_neigh1=m_neigh
    !cccccccccccccccccccccccccccccccccccccccc
    
        do i=1,ntype
        if(iflag_ftype.eq.3.and.iflag_grid_type(i).ne.3) then
        write(6,*) "if iflag_ftype.eq.3, iflag_grid must equal 3, stop"
        stop
        endif
        enddo
    
         n2bm=0
         do i=1,ntype
         if(n2b_type(i).gt.n2bm) n2bm=n2b_type(i)
         enddo
    
    !cccccccccccccccccccccccccccccccccccccccccccccccc
         nfeat0m=ntype*n2bm


    
         do itype=1,ntype
         nfeat0(itype)=n2b_type(itype)*ntype
         enddo

        !if(inode.eq.1) then
        !write(6,*) "ftype1: max_nfeat0m=",nfeat0m
        !write(6,*) "ftype1: nfeat0(at_type)=",(nfeat0(itype),itype=1,ntype)
        !endif
    
    !cccccccccccccccccccccccccccccccccccccccccccccccccccc
    
    !cccccccccccccccccccccccccccccccccccccccccccccccccccc
        !write(*,*) "n2bm, ntype: ", n2bm, ntype
        if (.not. allocated(grid2)) then
            allocate(grid2(0:n2bm+1,ntype))
            allocate(grid2_2(2,n2bm+1,ntype))
        end if
    !cccccccccccccccccccccccccccccccccccccccccccccccccccc
         do kkk=1,ntype    ! center atom
         
         Rc=Rc_type(kkk)
         Rm=Rm_type(kkk)
         iflag_grid=iflag_grid_type(kkk)
         fact_grid=fact_grid_type(kkk)
         dR_grid1=dR_grid1_type(kkk)
         n2b=n2b_type(kkk)
    
    !cccccccccccccccccccccccccccccccccccccccc
    
        if(iflag_grid.eq.1.or.iflag_grid.eq.2) then
    
            open(10,file="output/grid2b_type12."//char(kkk+48))
            rewind(10)
            do i=0,n2b+1
            read(10,*) grid2(i,kkk)
            read(10,*) grid2(i,kkk)
            read(10,*) grid2(i,kkk)
            enddo
            close(10)
    
         endif   ! iflag_grid.eq.1,2
    
    !cccccccccccccccccccccccccccccccccccccccccccc
        if(iflag_grid.eq.3) then  
     ! for iflag_grid.eq.3, the graid is just read in. 
     ! Its format is different from above grid31, grid32. 
     ! For each point, it just have two numbers, r1,r2, indicating the region of the sin peak function.
    
        open(13,file="output/grid2b_type3."//char(kkk+48))
        rewind(13)
        read(13,*) n2b_t
        if(n2b_t.ne.n2b) then
        write(6,*) "n2b_t.ne.n2b,in grid2b_type3", n2b_t,n2b
        stop
        endif
        do i=1,n2b
        read(13,*) it,grid2_2(1,i,kkk),grid2_2(2,i,kkk)
        if(grid2_2(2,i,kkk).gt.Rc_type(kkk)) write(6,*) "grid2_2.gt.Rc",grid2_2(2,i,kkk),Rc_type(kkk)
        enddo
        close(13)
        endif
    
        enddo     ! kkk=1,ntype

!cccccccccccccccccccccccccccccccccccccccccccccccccccc
!  FInish the initial grid treatment

    end subroutine load_model_type1
    
    subroutine set_image_info_type1(atom_type_list,is_reset,natom_tmp)
        integer(4),dimension(:),intent(in) :: atom_type_list(natom_tmp)
        logical,intent(in) :: is_reset
        integer,intent(in) :: natom_tmp
        
            natom=natom_tmp
            natom1=natom

            if(is_reset) then 
                if(allocated(iatom))then
                    deallocate(iatom)
                endif

                allocate(iatom(natom))                   
                
                iatom(1:natom)=atom_type_list(1:natom)
            endif
        
    end subroutine set_image_info_type1
    
    

subroutine gen_feature_type1(AL,xatom)
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
    if (allocated(feat_M1)) then
        deallocate(feat_M1)
    endif
    if (allocated(dfeat_M1)) then
        deallocate(dfeat_M1)
    endif
    if (allocated(list_neigh_alltypeM1)) then
        deallocate(list_neigh_alltypeM1)
    endif
    if (allocated(num_neigh_alltypeM1)) then
        deallocate(num_neigh_alltypeM1)
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
    allocate(list_neigh_alltypeM1(m_neigh,natom))
    allocate(num_neigh_alltypeM1(natom))
    allocate(map2neigh_alltypeM(m_neigh,natom)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
    allocate(list_tmp(m_neigh,ntype))
    allocate(itype_atom(natom))
    allocate(nfeat_atom(natom))

    allocate(feat(nfeat0m,natom_n))         ! each note, only know its own feat
    allocate(dfeat(nfeat0m,natom_n,m_neigh,3))  ! dfeat is the derivative from the neighboring dR, 
    allocate(feat_M1(nfeat0m,natom_n))
    allocate(dfeat_M1(nfeat0m,natom_n,m_neigh,3))

    feat = 0.d0
    dfeat = 0.d0
    feat_M1 = 0.d0
    dfeat_M1 = 0.d0
    
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
    call cpu_time(timestart)

!if(inode .eq. 1) then
!  write(*,*) "before find_neighbor neighborM:", num_neigh_M(1, 1)
!  write(*,*) "before find_neighbor neighbor list:", list_neigh(1:3, 1, 1)
!  write(*,*) "before find_neighbor dR_neigh:", dR_neigh(1:3, 1, 1, 1)
!endif
    call find_neighbore(iatom,natom,xatom,AL,Rc_type,num_neigh,list_neigh, &
       dR_neigh,iat_neigh,ntype,iat_type,m_neigh,Rc_M,map2neigh_M,list_neigh_M, &
       num_neigh_M,iat_neigh_M,inode,nnodes)
!if(inode .eq. 1) then
!  write(*,*) "after find_neighbor neighborM:", num_neigh_M(1, 1)
!  write(*,*) "after find_neighbor neighbor list:", list_neigh(1:3, 1, 1)
!  write(*,*) "after find_neighbor dR_neigh:", dR_neigh(1:3, 1, 1, 1)
!endif

!    call cpu_time(timefinish)
!    write(*,*) 'find_nei time: ', timefinish-timestart
!ccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccc

      max_neigh=-1
      num_neigh_alltype=0
      max_neigh_M=-1
      num_neigh_alltypeM1=0
      list_neigh_alltypeM1=0
  
        do iat=1,natom
  
        if(mod(iat-1,nnodes).eq.inode-1) then
  
  
        list_neigh_alltype(1,iat)=iat
        list_neigh_alltypeM1(1,iat)=iat
  
  
        num_M=1
        do itype=1,ntype
        do j=1,num_neigh_M(itype,iat)
        num_M=num_M+1
        if(num_M.gt.m_neigh) then
        write(6,*) "total num_neigh.gt.m_neigh,stop",m_neigh
        stop
        endif
        list_neigh_alltypeM1(num_M,iat)=list_neigh_M(j,itype,iat)
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
        num_neigh_alltypeM1(iat)=num_M
        if(num.gt.max_neigh) max_neigh=num
        if(num_M.gt.max_neigh_M) max_neigh_M=num_M
  
      endif


      enddo  ! iat

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! This num_neigh_alltype(iat) include itself !
!    dfeat=0.d0
!    feat=0.d0
!   if (inode .eq. 1) then
!       write(*,*) "before type3, dR_neigh:"
!       do i = 1,4
!           write(*,*) dR_neigh(1:3,i,1,1)
!       enddo
!   endif
    if(iflag_ftype.eq.1) then
! iflag_ftype.eq.1, the sin peak span over two grid points
    call find_feature_2b_type1(natom,itype_atom,Rc_type,n2b_type,num_neigh,  &
       list_neigh,dR_neigh,iat_neigh,ntype,grid2, &
       feat,dfeat,nfeat0m,m_neigh,n2bm,nfeat_atom)
    endif
    if(iflag_ftype.eq.2) then
!  iflag_ftype.eq.2, the sin peak span over three grid points
    call find_feature_2b_type2(natom,itype_atom,Rc_type,n2b_type,num_neigh,  &
       list_neigh,dR_neigh,iat_neigh,ntype,grid2, &
       feat,dfeat,nfeat0m,m_neigh,n2bm,nfeat_atom)
    endif
    if(iflag_ftype.eq.3) then
!  iflag_ftype.eq.3, the sin peak span over the two ends specified by grid31_2,grid32_2
!  So, there could be many overlaps between different sin peaks
    call find_feature_2b_type3(natom,itype_atom,Rc_type,n2b_type,num_neigh,  &
       list_neigh,dR_neigh,iat_neigh,ntype,grid2_2, &
       feat,dfeat,nfeat0m,m_neigh,n2bm,nfeat_atom)
    endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccc

    do iat1=1,natom_n
    do ii=1,nfeat0m
    feat_M1(ii,iat1)=feat(ii,iat1)   
    enddo
    enddo

    iat1=0
    do iat=1,natom
    if(mod(iat-1,nnodes).eq.inode-1) then
    iat1=iat1+1
    do jj=1,num_neigh_alltype(iat)
    jjm=map2neigh_alltypeM(jj,iat)
    do ii=1,nfeat0m
    dfeat_M1(ii,iat1,jjm,1)=dfeat(ii,iat1,jj,1)  ! this is the feature stored in neigh list of Rc_M
    dfeat_M1(ii,iat1,jjm,2)=dfeat(ii,iat1,jj,2)
    dfeat_M1(ii,iat1,jjm,3)=dfeat(ii,iat1,jj,3)
    enddo
    enddo
    endif
    enddo

    nfeat0M1=nfeat0m    ! the number of features for feature type 1


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
    ! deallocate(list_neigh_alltypeM1)
    ! deallocate(num_neigh_alltypeM1)
    deallocate(map2neigh_alltypeM)
    deallocate(list_tmp)
    deallocate(itype_atom)
    deallocate(nfeat_atom)
    deallocate(feat)
    deallocate(dfeat)
    ! mem leak
    !deallocate(grid2)
    !deallocate(grid2_2)

!--------------------------------------------------------


end subroutine gen_feature_type1

end module calc_ftype1
