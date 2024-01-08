module NeighConst
    
    implicit none

    integer, allocatable, dimension(:,:,:,:) :: list_neigh
    real(8), allocatable, dimension(:,:,:,:,:) :: dR_neigh
    
    contains

    subroutine find_neighbore(images, lattice, position, ntypes, natoms, m_neigh, Rc_M, Rc_type, type_maps)
        implicit none
        integer, intent(in) :: images, ntypes, natoms, m_neigh
        real(8), intent(in) :: Rc_M
        real(8), dimension(:), intent(in) :: Rc_type, type_maps
        real(8), dimension(:,:,:), intent(inout) :: lattice, position

        real(8) :: d, Rc2, dx1, dx2, dx3, dx, dy, dz, dd    ! tmp variables
        integer, dimension(ntypes, natoms) :: num_neigh     ! number of neighbors
        integer, dimension(ntypes) :: num_type, num_type_M  ! number of neighbors of each type

        integer :: n, i, j, k, itype  ! loop index
        integer :: i1, i2, i3       ! periodic image index
        integer :: iflag            ! flag for periodic boundary condition
        integer :: nperiod(3)  ! number of periodic images in each direction

        ! initialize
        allocate(list_neigh(m_neigh, ntypes, natoms, images))
        allocate(dR_neigh(3, m_neigh, ntypes, natoms, images))
        list_neigh = 0
        dR_neigh = 0
        iflag = 0

        ! loop over all images
        do n=1,images
            ! find the number of periodic images
            do i=1,3
                d=dsqrt(lattice(n,i,1)**2+lattice(n,i,2)**2+lattice(n,i,3)**2)
                nperiod(i)=int(Rc_M/d)+1
                if(d.lt.2*Rc_M) iflag=1
            enddo

            do i=1,natoms
                position(n,i,1)=mod(position(n,i,1)+10.d0,1.d0)
                position(n,i,2)=mod(position(n,i,2)+10.d0,1.d0)
                position(n,i,3)=mod(position(n,i,3)+10.d0,1.d0)
            enddo

            Rc2=Rc_M**2

            do i=1,natoms
                if(iflag.eq.1) then
                    num_type=0
                    num_type_M=0
                    do j=1,natoms
                        do i1=-nperiod(1),nperiod(1)
                            do i2=-nperiod(2),nperiod(2)
                                do i3=-nperiod(3),nperiod(3)
                                    dx1=position(n,j,1)-position(n,i,1)+i1
                                    dx2=position(n,j,2)-position(n,i,2)+i2
                                    dx3=position(n,j,3)-position(n,i,3)+i3

                                    dx=lattice(n,1,1)*dx1+lattice(n,2,1)*dx2+lattice(n,3,1)*dx3
                                    dy=lattice(n,1,2)*dx1+lattice(n,2,2)*dx2+lattice(n,3,2)*dx3
                                    dz=lattice(n,1,3)*dx1+lattice(n,2,3)*dx2+lattice(n,3,3)*dx3

                                    dd=dx**2+dy**2+dz**2

                                    if(dd.lt.Rc2.and.dd.gt.1.D-8) then
                                        itype=type_maps(j)
                                        num_type_M(itype)=num_type_M(itype)+1
                                        if(num_type_M(itype).gt.m_neigh) then
                                            write(6,*) "Error! maxNeighborNum too small", m_neigh
                                            stop
                                        endif
                                        if(dd.lt.Rc_type(type_maps(i))**2 .and. dd.gt.1.D-8) then
                                            num_type(itype)=num_type(itype)+1
                                            list_neigh(num_type(itype),itype,i,n)=j
                                            dR_neigh(1,num_type(itype),itype,i,n)=dx
                                            dR_neigh(2,num_type(itype),itype,i,n)=dy
                                            dR_neigh(3,num_type(itype),itype,i,n)=dz
                                        endif
                                    endif
                                enddo
                            enddo
                        enddo
                    enddo
                    num_neigh(:,i)=num_type(:)
                endif

                if(iflag.eq.0) then
                    num_type=0
                    num_type_M=0
                    do j=1,natoms
                        dx1=position(n,j,1)-position(n,i,1)
                        if(abs(dx1+1).lt.abs(dx1)) dx1=dx1+1
                        if(abs(dx1-1).lt.abs(dx1)) dx1=dx1-1

                        dx2=position(n,j,2)-position(n,i,2)
                        if(abs(dx2+1).lt.abs(dx2)) dx2=dx2+1
                        if(abs(dx2-1).lt.abs(dx2)) dx2=dx2-1

                        dx3=position(n,j,3)-position(n,i,3)
                        if(abs(dx3+1).lt.abs(dx3)) dx3=dx3+1
                        if(abs(dx3-1).lt.abs(dx3)) dx3=dx3-1

                        dx=lattice(n,1,1)*dx1+lattice(n,2,1)*dx2+lattice(n,3,1)*dx3
                        dy=lattice(n,1,2)*dx1+lattice(n,2,2)*dx2+lattice(n,3,2)*dx3
                        dz=lattice(n,1,3)*dx1+lattice(n,2,3)*dx2+lattice(n,3,3)*dx3

                        dd=dx**2+dy**2+dz**2

                        if(dd.lt.Rc2.and.dd.gt.1.D-8) then
                            itype=type_maps(j)
                            num_type_M(itype)=num_type_M(itype)+1
                            if(num_type_M(itype).gt.m_neigh) then
                                write(6,*) "Error! maxNeighborNum too small", m_neigh
                                stop
                            endif
                            if(dd.lt.Rc_type(type_maps(i))**2 .and. dd.gt.1.D-8) then
                                num_type(itype)=num_type(itype)+1
                                list_neigh(num_type(itype),itype,i,n)=j
                                dR_neigh(1,num_type(itype),itype,i,n)=dx
                                dR_neigh(2,num_type(itype),itype,i,n)=dy
                                dR_neigh(3,num_type(itype),itype,i,n)=dz
                            endif
                        endif
                    enddo
                    num_neigh(:,i)=num_type(:)
                endif
            enddo

            do i=1,natoms
                do j=1,ntypes
                    do k=1,m_neigh
                        if((abs(dR_neigh(1,k,j,i,n)) + abs(dR_neigh(2,k,j,i,n)) + abs(dR_neigh(3,k,j,i,n))) <= 1.D-8) then
                            list_neigh(k,j,i,n) = 0
                            dR_neigh(1,k,j,i,n) = 0
                            dR_neigh(2,k,j,i,n) = 0
                            dR_neigh(3,k,j,i,n) = 0
                        endif
                    enddo
                enddo
            enddo
        enddo
    end subroutine find_neighbore

    subroutine dealloc()
        deallocate(list_neigh)
        deallocate(dR_neigh)
    end subroutine dealloc
end module NeighConst