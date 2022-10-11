
module calc_feature
    IMPLICIT NONE
    INTEGER :: ierr

    integer(4),allocatable,dimension (:) :: iatom

    integer(4) natom
    
    real(8),allocatable,dimension (:) :: grid2,grid31,grid32
    real(8),allocatable,dimension (:,:) :: grid2_2,grid31_2,grid32_2
    
    real*8,allocatable,dimension (:,:) :: feat

    integer,allocatable,dimension (:,:) :: list_neigh_alltype
    integer,allocatable,dimension (:) :: num_neigh_alltype

    integer(4) m_neigh,num
    
    integer(4) ntype,n2b,n3b1,n3b2,nfeat0
    
    integer(4) iat_type(100)
    real*8 Rc, Rc2,Rm
    ! real*8 alpha2,alpha31,alpha32,fact_grid2,fact_grid31,fact_grid32
    integer(4) :: num_tmp
    ! real*8, allocatable, dimension (:,:) :: dfeat_tmp
    real*8, allocatable, dimension (:,:,:,:) :: dfeat
    ! integer(4),allocatable, dimension (:) :: iat_tmp,jneigh_tmp,ifeat_tmp
    
    integer(4) :: iflag_grid,iflag_ftype
    

contains

subroutine load_model()
    real(8) :: E_tolerance
    integer(4) :: itype1,itype2,i
    integer(4) :: n2b_t,n3b1_t,n3b2_t,it
    real*8 fact_grid,dR_grid1,dR_grid2
    integer(4) :: k1,k2,k12,ii_f
    open(10,file="input/gen_feature.in",status="old",action="read")
    rewind(10)
    read(10,*) ntype
    do i=1,ntype
    read(10,*) iat_type(i)
    enddo
    read(10,*) Rc, Rc2, Rm,iflag_grid,fact_grid,dR_grid1,dR_grid2
    read(10,*) n2b,n3b1,n3b2
    read(10,*) m_neigh
    read(10,*) E_tolerance
    read(10,*) iflag_ftype
    ! read(10,*) recalc_grid
    close(10)


    if(iflag_ftype.eq.4.and.iflag_grid.ne.3) then
    write(6,*) "if iflag_ftype.eq.4, iflag_grid must equal 3, stop"
    stop
    endif


    num=n2b*ntype
    do itype2=1,ntype
    do itype1=1,itype2
    do k1=1,n3b1
    do k2=1,n3b1
    do k12=1,n3b2
    ii_f=0
    if(itype1.ne.itype2) ii_f=1
    if(itype1.eq.itype2.and.k1.le.k2) ii_f=1
    if(ii_f.gt.0) then
    num=num+1
    endif
    enddo
    enddo
    enddo
    enddo
    enddo
    nfeat0=num
    write(6,*) "nfeat0=",nfeat0
!cccccccccccccccccccccccccccccccccccccccc
    if (allocated(grid2))then
        deallocate(grid2)
        deallocate(grid31)
        deallocate(grid32)
    end if
    if (allocated(grid2_2))then
        deallocate(grid2_2)
        deallocate(grid31_2)
        deallocate(grid32_2)
    end if


    if(iflag_grid.eq.1.or.iflag_grid.eq.2) then
        allocate(grid2(0:n2b+1))
        allocate(grid31(0:n3b1+1))
        allocate(grid32(0:n3b2+1))
        open(10,file="output/grid2")
        rewind(10)
        do i=0,n2b+1
        read(10,*) grid2(i)
        read(10,*) grid2(i)
        read(10,*) grid2(i)
        enddo
        close(10)

        open(10,file="output/grid31")
        rewind(10)
        do i=0,n3b1+1
        read(10,*) grid31(i)
        read(10,*) grid31(i)
        read(10,*) grid31(i)
        enddo
        close(10)

        open(10,file="output/grid32")
        rewind(10)
        do i=0,n3b2+1
        read(10,*) grid32(i)
        read(10,*) grid32(i)
        read(10,*) grid32(i)
        enddo
        close(10)
    endif


    if(iflag_grid.eq.3) then
    allocate(grid2_2(2,n2b))
    allocate(grid31_2(2,n3b1))
    allocate(grid32_2(2,n3b2))
    open(13,file="output/grid2.type2")
    rewind(13)
    read(13,*) n2b_t
    if(n2b_t.ne.n2b) then
    write(6,*) "n2b_t.ne.n2b,in grid2.type2", n2b_t,n2b
    stop
    endif
    do i=1,n2b
    read(13,*) it,grid2_2(1,i),grid2_2(2,i)
    if(grid2_2(2,i).gt.Rc) write(6,*) "grid2_2.gt.Rc",grid2_2(2,i),Rc
    enddo
    close(13)
    open(13,file="output/grid31.type2")
    rewind(13)
    read(13,*) n3b1_t
    if(n3b1_t.ne.n3b1) then
    write(6,*) "n3b1_t.ne.n3b1,in grid31.type2", n3b1_t,n3b1
    stop
    endif
    do i=1,n3b1
    read(13,*) it,grid31_2(1,i),grid31_2(2,i)
    if(grid31_2(2,i).gt.Rc) write(6,*) "grid31_2.gt.Rc",grid31_2(2,i),Rc
    enddo
    close(13)

    open(13,file="output/grid32.type2")
    rewind(13)
    read(13,*) n3b2_t
    if(n3b2_t.ne.n3b2) then
    write(6,*) "n3b2_t.ne.n3b2,in grid32.type2", n3b2_t,n3b2
    stop
    endif
    do i=1,n3b2
    read(13,*) it,grid32_2(1,i),grid32_2(2,i)
    if(grid32_2(2,i).gt.Rc2) write(6,*) "grid32_2.gt.Rc",grid32_2(2,i),Rc
    enddo
    close(13)
    
    endif   
end subroutine load_model

subroutine set_image_info(atom_type_list,is_reset)
    integer(4),dimension(:),intent(in) :: atom_type_list
    logical,intent(in) :: is_reset
    integer(4) :: image_size
    
    image_size=size(atom_type_list)
    if (is_reset .or. (.not. allocated(iatom)) .or. image_size/=natom) then
    
        if (allocated(iatom))then
            if (image_size==natom .and. maxval(abs(atom_type_list-iatom))==0) then
                return
            end if
            deallocate(iatom)
        end if
          
        natom=image_size
        allocate(iatom(natom))                   
        iatom=atom_type_list
          
    end if
    
end subroutine set_image_info


subroutine gen_feature(AL,xatom)
    integer(4)  :: itype,ixyz,i,j,jj,iat
    integer(4) max_neigh
    real(8), intent(in) :: AL(3,3)
    real(8),dimension(:,:),intent(in) :: xatom
    integer(4) :: ii_tmp,jj_tmp,iat2
    integer(4),allocatable,dimension (:,:,:) :: list_neigh,iat_neigh
    integer(4),allocatable,dimension (:,:) :: num_neigh
    real*8,allocatable,dimension (:,:,:,:) :: dR_neigh
    ! real*8,allocatable,dimension (:,:,:,:) :: dfeat

    if (allocated(feat)) then
        deallocate(dfeat)
        ! deallocate(iat_tmp)
        ! deallocate(jneigh_tmp)
        ! deallocate(ifeat_tmp)
        ! deallocate(dfeat_tmp)

        deallocate(feat)
        deallocate(list_neigh_alltype)
        deallocate(num_neigh_alltype)
    endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccc



    allocate(list_neigh(m_neigh,ntype,natom))
    allocate(iat_neigh(m_neigh,ntype,natom))
    allocate(dR_neigh(3,m_neigh,ntype,natom))   ! d(neighbore)-d(center) in xyz
    allocate(num_neigh(ntype,natom))
    allocate(list_neigh_alltype(m_neigh,natom))
    allocate(num_neigh_alltype(natom))
    
    allocate(feat(nfeat0,natom))

    allocate(dfeat(nfeat0,natom,m_neigh,3))


    call find_neighbore(iatom,natom,xatom,AL,Rc,num_neigh,list_neigh, &
       dR_neigh,iat_neigh,ntype,iat_type,m_neigh)

!ccccccccccccccccccccccccccccccccc

      max_neigh=-1
      num_neigh_alltype=0
      do iat=1,natom
      num=1
      list_neigh_alltype(1,iat)=iat
      do  itype=1,ntype
      do   j=1,num_neigh(itype,iat)
      num=num+1
      list_neigh_alltype(num,iat)=list_neigh(j,itype,iat)
      enddo
      enddo
      if(num.gt.m_neigh) then
      write(6,*) "total num_neigh.gt.m_neigh,stop",m_neigh
      stop
      endif
      num_neigh_alltype(iat)=num
      if(num.gt.max_neigh) max_neigh=num

      list_neigh_alltype((num_neigh_alltype(iat)+1):,iat)=0
      enddo

    !   list_neigh_alltype(num,iat)
! This num_neigh_alltype(iat) include itself !
    dfeat=0.d0
    feat=0.d0
    if(iflag_ftype.eq.2) then
    call find_feature(natom,Rc,Rc2,n2b,n3b1,n3b2,num_neigh,  &
       list_neigh,dR_neigh,iat_neigh,ntype,grid2,grid31,grid32, &
       feat,dfeat,nfeat0,m_neigh)
    endif
    if(iflag_ftype.eq.3) then
    call find_feature3(natom,Rc,Rc2,n2b,n3b1,n3b2,num_neigh,  &
       list_neigh,dR_neigh,iat_neigh,ntype,grid2,grid31,grid32, &
       feat,dfeat,nfeat0,m_neigh)
    endif
    if(iflag_ftype.eq.4) then
    call find_feature4(natom,Rc,Rc2,n2b,n3b1,n3b2,num_neigh,  &
       list_neigh,dR_neigh,iat_neigh,ntype,grid2_2,grid31_2,grid32_2, &
       feat,dfeat,nfeat0,m_neigh)
    endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccc

!cccccccccccccccccccccccccccccccccccccccccccccchhhhhh
!  Only output the nonzero points for dfeat
    ! num_tmp=0
    ! do jj_tmp=1,m_neigh
    ! do iat2=1,natom
    ! do ii_tmp=1,nfeat0
    ! if(abs(dfeat(ii_tmp,iat2,jj_tmp,1))+abs(dfeat(ii_tmp,iat2,jj_tmp,2))+ &
    !      abs(dfeat(ii_tmp,iat2,jj_tmp,3)).gt.1.E-7) then
    ! num_tmp=num_tmp+1
    ! endif
    ! enddo
    ! enddo
    ! enddo
    ! allocate(dfeat_tmp(3,num_tmp))
    ! allocate(iat_tmp(num_tmp))
    ! allocate(jneigh_tmp(num_tmp))
    ! allocate(ifeat_tmp(num_tmp))

    ! num_tmp=0
    ! do jj_tmp=1,m_neigh
    ! do iat2=1,natom
    ! do ii_tmp=1,nfeat0
    ! if(abs(dfeat(ii_tmp,iat2,jj_tmp,1))+abs(dfeat(ii_tmp,iat2,jj_tmp,2))+ &
    !       abs(dfeat(ii_tmp,iat2,jj_tmp,3)).gt.1.E-7) then
    ! num_tmp=num_tmp+1
    ! dfeat_tmp(:,num_tmp)=dfeat(ii_tmp,iat2,jj_tmp,:)
    ! iat_tmp(num_tmp)=iat2
    ! jneigh_tmp(num_tmp)=jj_tmp
    ! ifeat_tmp(num_tmp)=ii_tmp
    ! endif
    ! enddo
    ! enddo
    ! enddo



    deallocate(list_neigh)
    deallocate(iat_neigh)
    deallocate(dR_neigh)
    deallocate(num_neigh)
!     deallocate(feat)
    ! deallocate(dfeat)
!     deallocate(list_neigh_alltype)
!     deallocate(num_neigh_alltype)


end subroutine gen_feature

end module calc_feature