subroutine get_grid2b_type2(trainSetFileDir,sys_num,grid2,Rc,Rm,n2b, &
     m_neigh,ntype,iat_type,fact_grid,dR_grid1,itype_center)

! In this subroutine, the pair-distribution charge density is used to 
! define the gred. It is controled by parameter dR_grid1,2, and fact_grid
! Note, different center atom type itype_center will have different grid, 
! However, right now, it does not distinguish different second atom type. 
! So, all the second atom type (in the atom pair) will have the same grid
! grid31 is the center-atom-----atom1/2 line
! grid32 is the atom1----atom2 line
! We have a triangle between ceter-atom ---- atom1 ----atom2
! fact_grid (e.g., 0.1) determine how much average density to be added to the density profile
! So, it will not be so dramatic, fill in the zero part
! dR_grid1 and dR_grid2 are the broadening (like Gaussian broading) of the density profile
! density, density2 for the grid31 and grid32 respectively. They should be compared with 
! Rc, and Rc2  (e.g., equals Rc/10, Rc2/10 et)

    IMPLICIT double precision (a-h,o-z)

    real*8 grid2(0:n2b+1)

    INTEGER :: ierr
    integer :: move_file=1101
    real*8 AL(3,3),Etotp
    real*8,allocatable,dimension (:,:) :: xatom,fatom
    real*8,allocatable,dimension (:) :: Eatom
    integer,allocatable,dimension (:) :: iatom
    logical nextline
    character(len=200) :: the_line
    integer num_step, natom, i, j
    integer num_step0,num_step1,natom0
    real*8 Etotp_ave,E_tolerance
    character*50 char_tmp(20)
    real*8 density(0:1000),density2(0:1000)
    real*8 density_sm(0:1000),density2_sm(0:1000)
    real*8 densityI(0:1000),density2I(0:1000)
    integer iat_type(100)

    character(len=200) trainSetFileDir(200)
    character(len=200) MOVEMENTDir
    integer sys_num

    integer,allocatable,dimension (:,:,:) :: list_neigh,iat_neigh
    integer,allocatable,dimension (:,:) :: num_neigh
    real*8,allocatable,dimension (:,:,:,:) :: dR_neigh


!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    INTERFACE
        SUBROUTINE scan_title (io_file, title, title_line, if_find)
            CHARACTER(LEN=200), OPTIONAL :: title_line
            LOGICAL, OPTIONAL :: if_find
            INTEGER :: io_file
            CHARACTER(LEN=*) :: title
        END SUBROUTINE scan_title
    END INTERFACE
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      density=0.d0
      density12=0.d0
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc    !
    num_step=0
    do 7000 isys=1,sys_num
    MOVEMENTDir=trim(trainSetFileDir(isys))//"/MOVEMENT"

    OPEN (move_file,file=MOVEMENTDir,status="old",action="read") 
    rewind(move_file)

1000  continue

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    call scan_title (move_file,"ITERATION",if_find=nextline)

    if(.not.nextline) goto 5000
    num_step=num_step+1

    backspace(move_file) 
    read(move_file, *) natom

    ALLOCATE (iatom(natom),xatom(3,natom),fatom(3,natom),Eatom(natom))

        CALL scan_title (move_file, "LATTICE")
        DO j = 1, 3
            READ (move_file,*) AL(1:3,j)
        ENDDO

       CALL scan_title (move_file, "POSITION")
        DO j = 1, natom
            READ(move_file, *) iatom(j),xatom(1,j),xatom(2,j),xatom(3,j)
        ENDDO

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    allocate(list_neigh(m_neigh,ntype,natom))
    allocate(iat_neigh(m_neigh,ntype,natom))
    allocate(dR_neigh(3,m_neigh,ntype,natom))   ! d(neighbore)-d(center) in xyz
    allocate(num_neigh(ntype,natom))

    call find_neighbore00(iatom,natom,xatom,AL,Rc,num_neigh,list_neigh, &
       dR_neigh,iat_neigh,ntype,iat_type,m_neigh)

      do iat=1,natom

      if(iatom(iat).eq.iat_type(itype_center)) then

      do  itype=1,ntype
      do   j=1,num_neigh(itype,iat)
      dd=dR_neigh(1,j,itype,iat)**2+dR_neigh(2,j,itype,iat)**2+dR_neigh(3,j,itype,iat)**2
      d=dsqrt(dd)
      k=int((d/Rc)*1000.d0)
      if(k.ge.1000) k=1000-1
      density(k)=density(k)+1    ! density is dist between center_atom--atom1(or atom2)
      enddo
      enddo

      endif
      enddo
!ccccccccccccccccccccccccccccccccccc

    deallocate(list_neigh)
    deallocate(iat_neigh)
    deallocate(dR_neigh)
    deallocate(num_neigh)
    deallocate (iatom,xatom,fatom,Eatom)

      goto 1000
5000  continue
      close(move_file)
7000  continue


!cccccccccccccccccccccccccccccccccccccccccc
!     Now, use density, density2 to determine grid31,grid32
!cccccccccccccccccccccccccccccccccccccccccc
!   We use a density integrated line as the distance measurement, 
!   And use that to define grid2
!    First, add 1/5 of the total charge uniformly, so it is not too crazy
!ccccccccccccccccccccccccccccccccccccccccccccccccccc
     n_sm=dR_grid1/Rc*1000

     do k=0,1000
     density(k)=density(k)/(k+1000/5.d0)**4  ! making large d less important
     enddo

     density_sm=0.d0
     do k=0,1000
     do k2=-n_sm,n_sm
     k22=k+k2
     if(k22.ge.0.and.k22.le.1000) then 
     density_sm(k22)=density_sm(k22)+density(k)
     endif
     enddo
     enddo
     density_sm=density_sm/(2*n_sm)

     sum1=0.d0
     do k=0,1000
     sum1=sum1+density_sm(k)
     enddo
     sum1=sum1/1000
     density_sm=density_sm+fact_grid*sum1

     densityI(0)=0.d0
     do k=1,1000
     densityI(k)=densityI(k-1)+density_sm(k-1)
     enddo

     sum1=densityI(1000)/(n2b+1)-1.E-8
     grid2(0)=0.d0
     num=1
     do k=1,1000
     if(densityI(k).gt.num*sum1) then
     grid2(num)=Rc*k/1000.d0
     num=num+1
     endif
     enddo
     grid2(n2b+1)=Rc

!ccccccccccccccccc smooth things out     

     do k=1,n2b
     if(grid2(k)-grid2(k-1).gt.2*(grid2(k+1)-grid2(k))) then
     grid2(k)=2.d0/3*grid2(k-1)+1.d0/3*grid2(k+1)
     endif
     if(grid2(k+1)-grid2(k).gt.2*(grid2(k)-grid2(k-1))) then
     grid2(k)=1.d0/3*grid2(k-1)+2.d0/3*grid2(k+1)
     endif
     enddo

!cccccccccccccccccccccccccccccccccccccccccccc

    open(11,file="output/density2b."//char(itype_center+48))
    rewind(11)
    do k=0,1000
    write(11,*) Rc/1000.d0*k,density(k),density_sm(k)
    enddo
    close(11)
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
    return
    end subroutine get_grid2b_type2
