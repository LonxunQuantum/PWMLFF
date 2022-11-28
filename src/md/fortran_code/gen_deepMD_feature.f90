module calc_deepMD_f
! This is a strange version, with the ghost neighbore atoms, this is due to 
! original DP bug, and we keep the bug here. 
    use mod_mpi

    IMPLICIT NONE
    integer,allocatable,dimension (:) :: iatom
    integer iat_type(100)
    integer natom,ntype

    integer  i, j,natom1,m_neigh1
    integer num_step0,num_step1,natom0,max_neigh


    !ccccccccccccccccccccc  The variables to be used in global feature type
    integer,allocatable,dimension (:,:) :: list_neigh_alltypeM1
    integer,allocatable,dimension (:,:,:) :: list_neigh
    integer,allocatable,dimension (:) :: num_neigh_alltypeM1
    !ccccccccccccccccccccc  The variables to be used in global feature type
    integer m_neigh,num,itype1,itype2,itype
    integer iat1,max_neigh_M,num_M

    real*8 Rc, Rm

    real*8 Rc_M
    real*8 Rc_type(100), R_cs(100)
    real*8,allocatable,dimension (:,:,:) :: s_neigh
    real*8,allocatable,dimension (:,:,:,:) :: ds_neigh
    real*8,allocatable,dimension (:,:,:,:) :: dR_neigh
    real*8,allocatable,dimension (:,:,:,:) :: dxyz_neigh
    real*8,allocatable,dimension (:,:,:,:,:) :: dxyz_dx_neigh
    integer,allocatable,dimension (:,:) :: num_neigh
    real*8 ave_shift(4,100),ave_norm(4,100)

    integer dp_M2                      ! M2 in dp network. 

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

contains

subroutine load_model_deepMD_f()

     open(10,file="gen_dp.in",status="old",action="read")
        rewind(10)
        read(10,*) Rc_M,m_neigh
        read(10,*) dp_M2 
        read(10,*) ntype
        do i=1,ntype
            read(10,*) iat_type(i)
            read(10,*) Rc_type(i),R_cs(i)

            if(Rc_type(i).gt.Rc_M) then
                write(6,*) "Rc_type must be smaller than Rc_M,gen_3b_feature.in",i,Rc_type(i),Rc_M
                stop
            endif

            read(10,*) ave_shift(1,i),ave_shift(2,i),ave_shift(3,i),ave_shift(4,i)
            read(10,*) ave_norm(1,i),ave_norm(2,i),ave_norm(3,i),ave_norm(4,i)
        enddo

        close(10)
        
       !write(6,*) "TEST load_model_deepMD_f",ntype
    !cccccccccccccccccccccccccccccccccccccccc
end subroutine load_model_deepMD_f

subroutine set_image_info_deepMD_f(atom_type_list,is_reset,natom_tmp)
        
    integer(4),dimension(:),intent(in) :: atom_type_list(natom_tmp)
    logical,intent(in) :: is_reset
    integer,intent(in) :: natom_tmp

    natom = natom_tmp
    
    if(is_reset) then
        if(allocated(iatom))then
            deallocate(iatom)
        endif
        
        allocate(iatom(natom))
            
        iatom(1:natom)=atom_type_list(1:natom)
    endif
    
    if (.not.allocated(s_neigh)) then 
        allocate(s_neigh(m_neigh,ntype,natom))
    endif

    if(.not.allocated(ds_neigh)) then 
        allocate(ds_neigh(3,m_neigh,ntype,natom))
    endif

    if(.not.allocated(dR_neigh)) then
        allocate(dR_neigh(3,m_neigh,ntype,natom))
    endif
    
    if(.not.allocated(dxyz_neigh)) then 
        allocate(dxyz_neigh(4,m_neigh,ntype,natom))
    endif
    
    if(.not.allocated(dxyz_dx_neigh)) then 
        allocate(dxyz_dx_neigh(3,4,m_neigh,ntype,natom))
    endif 
    
    if(.not.allocated(num_neigh)) then
        allocate(num_neigh(ntype,natom))
    endif 
    
end subroutine set_image_info_deepMD_f

subroutine gen_deepMD_feature(AL,xatom)

    integer(4)  :: itype,i,j
    integer ii,jj,jjm,iat

    real(8), intent(in) :: AL(3,3)
    real(8), intent(in) :: xatom(3,natom)
    
    integer,allocatable,dimension (:,:,:) :: iat_neigh,iat_neigh_M
    integer,allocatable,dimension (:,:) :: map2neigh_alltypeM
    integer,allocatable,dimension (:,:) :: list_tmp
    integer,allocatable,dimension (:) :: nfeat_atom
    integer,allocatable,dimension (:) :: itype_atom
    integer,allocatable,dimension (:,:,:) :: map2neigh_M
    integer,allocatable,dimension (:,:,:) :: list_neigh_M
    integer,allocatable,dimension (:,:) :: num_neigh_M
    integer,allocatable,dimension (:,:) :: list_neigh_alltype
    integer,allocatable,dimension (:) :: num_neigh_alltype

    real*8 s,rr,r,pi,x,yy,dyy
    real*8 dr(3),ds(3)
    integer m,m2
    
    !cccccccccccccccccccccccccccccccccccccccccccccccccccc
    ! the dimension of these array, should be changed to natom_n
    ! instead of natom. Each process only needs to know its own natom_n

    if(allocated(list_neigh).eq..false.) allocate(list_neigh(m_neigh,ntype,natom))

    allocate(map2neigh_M(m_neigh,ntype,natom)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
    allocate(list_neigh_M(m_neigh,ntype,natom)) ! the neigh list of Rc_M
    allocate(num_neigh_M(ntype,natom))
    allocate(iat_neigh(m_neigh,ntype,natom))
    allocate(list_neigh_alltype(m_neigh,natom))
    allocate(num_neigh_alltype(natom))

    allocate(iat_neigh_M(m_neigh,ntype,natom))
    allocate(list_neigh_alltypeM1(m_neigh,natom))
    allocate(num_neigh_alltypeM1(natom))
    allocate(map2neigh_alltypeM(m_neigh,natom)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
    allocate(list_tmp(m_neigh,ntype))
    allocate(itype_atom(natom))
    
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

    !if(inode .eq. 1) then
    !  write(*,*) "before find_neighbor neighborM:", num_neigh_M(1, 1)
    !  write(*,*) "before find_neighbor neighbor list:", list_neigh(1:3, 1, 1)
    !  write(*,*) "before find_neighbor dR_neigh:", dR_neigh(1:3, 1, 1, 1)
    !endif
    
    ! wlj altered
    !call find_neighbore(iatom,natom,xatom,AL,Rc_type,num_neigh,list_neigh, &
    !    dR_neigh,iat_neigh,ntype,iat_type,m_neigh,Rc_M,map2neigh_M,list_neigh_M, &
    !    num_neigh_M,iat_neigh_M,inode,nnodes)
    
    call find_neighbore(iatom,natom,xatom,AL,Rc_type,num_neigh,list_neigh, &
        dR_neigh,iat_neigh,ntype,iat_type,m_neigh,Rc_M,map2neigh_M,list_neigh_M, &
        num_neigh_M,iat_neigh_M)
        
    ! dR_neigh(:,neighbor idx, type idx, atom idx)
    write(*,*) "printing dbg info: dR"
    do i=1,100
        write(*,'(F16.12 F16.12 F16.12)',advance='no') dR_neigh(:,i,1,1)    
        write(*,*) " "
    enddo 
    !if(inode .eq. 1) then
    !  write(*,*) "after find_neighbor neighborM:", num_neigh_M(1, 1)
    !  write(*,*) "after find_neighbor neighbor list:", list_neigh(1:3, 1, 1)
    !  write(*,*) "after find_neighbor dR_neigh:", dR_neigh(1:3, 1, 1, 1)
    !endif  
    !    call cpu_time(timefinish)
    !    write(*,*) 'find_nei time: ', timefinish-timestart
    !ccccccccccccccccccccccccccccccccc
    !ccccccccccccccccccccccccccccccccc
     
    pi=4*datan(1.d0)
    dxyz_dx_neigh=0.d0

    ! write(*,*) "dbg: ntype", ntype
    ! iat: index of atom 
    
    !write(*,*) "printing num_neigh cu-cu network"
    !write(*,*) num_neigh(1,1) 
    
    ! in find_neighbor.f90, dR_neigh is a rank-4 tensor with indices
    ! dR_neigh(3,idx of neigh, idx of type, idx of atom)
    
    !write(*,*) "dbg: ntype", ntype
    !write(*,*) "dbg: natom", natom
    !write(*,*) "dbg info"
    !write(*,*) dR_neigh(1,:,1,2)

    do iat=1,natom  
        
        itype1=itype_atom(iat)  ! center atom type
        !write(*,*) itype1 
        do itype=1,ntype
            
            do j=1,num_neigh(itype,iat)

                rr=dR_neigh(1,j,itype,iat)**2+dR_neigh(2,j,itype,iat)**2+dR_neigh(3,j,itype,iat)**2
                r=dsqrt(rr) 

                ! don't forgot, there is also a derivative with respective to the center i atom
                dr(1)=dR_neigh(1,j,itype,iat)/r
                dr(2)=dR_neigh(2,j,itype,iat)/r
                dr(3)=dR_neigh(3,j,itype,iat)/r
                
                if(r.lt.R_cs(itype1)) then
                    
                    s=1/r
                    ds(:)=-dr(:)/r**2

                elseif(r.ge.R_cs(itype1).and.r.le.Rc_type(itype1)) then
                    !cs=1/r*(cos(pi*(r-R_cs(itype))/(Rc_type(itype)-R_cs(itype)))+1)*0.5
                    x=(r-R_cs(itype))/(Rc_type(itype1)-R_cs(itype1))
                    yy=x**3*(-6*x**2+15*x-10)+1
                    dyy=3*x**2*(-6*x**2+15*x-10)+x**3*(-12*x+15)
                    s=1/r*yy
                    ds(:)=-dr(:)/r**2*yy+1/r*dyy*dr(:)/(Rc_type(itype1)-R_cs(itype1))

                elseif(r.gt.Rc_type(itype1)) then
                    s=0.d0
                    ds=0.d0
                endif
                
                dxyz_neigh(1,j,itype,iat)=(s-ave_shift(1,itype1))/ave_norm(1,itype1)
                dxyz_neigh(2,j,itype,iat)=(dR_neigh(1,j,itype,iat)*s/r-ave_shift(2,itype1))/ave_norm(2,itype1)
                dxyz_neigh(3,j,itype,iat)=(dR_neigh(2,j,itype,iat)*s/r-ave_shift(3,itype1))/ave_norm(3,itype1)
                dxyz_neigh(4,j,itype,iat)=(dR_neigh(3,j,itype,iat)*s/r-ave_shift(4,itype1))/ave_norm(4,itype1)
                
                !dxyz_neigh(1,j,itype,iat)=s 
                !dxyz_neigh(2,j,itype,iat)=dR_neigh(1,j,itype,iat)*s/r
                !dxyz_neigh(3,j,itype,iat)=dR_neigh(2,j,itype,iat)*s/r
                !dxyz_neigh(4,j,itype,iat)=dR_neigh(3,j,itype,iat)*s/r

                s_neigh(j,itype,iat)=dxyz_neigh(1,j,itype,iat)
                
                if(j.eq.num_neigh(itype,iat).and.j.lt.m_neigh) then  ! this is to keep with the DP bug

                    dxyz_neigh(1,j+1,itype,iat)=(0.d0-ave_shift(1,itype1))/ave_norm(1,itype1)
                    dxyz_neigh(2,j+1,itype,iat)=(0.d0-ave_shift(2,itype1))/ave_norm(2,itype1)
                    dxyz_neigh(3,j+1,itype,iat)=(0.d0-ave_shift(3,itype1))/ave_norm(3,itype1)
                    dxyz_neigh(4,j+1,itype,iat)=(0.d0-ave_shift(4,itype1))/ave_norm(4,itype1)
                    
                    s_neigh(j+1,itype,iat)=dxyz_neigh(1,j+1,itype,iat)
                
                endif
                
                do m2=2,4
                    dxyz_dx_neigh(m2-1,m2,j,itype,iat)=s/r/ave_norm(m2,itype1)
                enddo

                do m=1,3
                    dxyz_dx_neigh(m,1,j,itype,iat)=dxyz_dx_neigh(m,1,j,itype,iat)+ &
                        ds(m)/ave_norm(1,itype1)

                    ds_neigh(m,j,itype,iat)=dxyz_dx_neigh(m,1,j,itype,iat)
                    
                    do m2=2,4
                        dxyz_dx_neigh(m,m2,j,itype,iat)=dxyz_dx_neigh(m,m2,j,itype,iat)+ &
                        dR_neigh(m2-1,j,itype,iat)*(ds(m)/r-s/r**2*dr(m))/ave_norm(m2,itype1)
                    enddo
                    
                enddo
                
            enddo
        enddo
        ! wlj dbg
        !allocate(dxyz_neigh(4,m_neigh,ntype,natom)) 
        
        !if (iat.eq.1) then
        !   write(*,*) dxyz_neigh(:,1,1,1) 
        !    
        !   write(*,*) "**********************************************"
        !endif 
    enddo 
    ! write(*,*) dxyz_neigh(:,2,1,1) 
    ! Ri matrix in DP model, input for the embedding net 
    ! 4, max neighbor num , types of atom, number of atom    
    write(*,*) "printing dbg info: R_i"

    do i=1,100
        write(*,'(F16.12 F16.12 F16.12 F16.12)',advance='no') dxyz_neigh(:,i,1,1) 
        write(*,*) " "
    enddo 
    
    !write(*,*) "m_neigh", m_neigh
    ! dxyz_neigh(4,m_neigh,ntype,natom)
    !do j=1,num_neigh(itype,iat)
    !do j=1,num_neigh(2,1)+1
    !do j=1,m_neigh
    !   
    !    write(*,'(F16.12 F16.12 F16.12 F16.12)',advance='no') dxyz_neigh(:,j,2,1) 
    !    write(*,*) " "
    !
    !enddo
    !deallocate(list_neigh)
    deallocate(map2neigh_M)
    deallocate(list_neigh_M)
    deallocate(num_neigh_M)
    deallocate(iat_neigh)
    deallocate(list_neigh_alltype)
    deallocate(num_neigh_alltype)

    deallocate(iat_neigh_M)
    deallocate(list_neigh_alltypeM1)
    deallocate(num_neigh_alltypeM1)
    deallocate(map2neigh_alltypeM)
    deallocate(list_tmp)
    deallocate(itype_atom)
    !--------------------------------------------------------


end subroutine gen_deepMD_feature

end module calc_deepMD_f
