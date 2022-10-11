      subroutine find_neighbore(iatom,natom,xatom,AL,Rc_type,num_neigh,list_neigh, &
      dR_neigh,iat_neigh,ntype,iat_type,m_neigh,Rc_M,map2neigh_M,list_neigh_M, &
      num_neigh_M,iat_neigh_M)
     
      implicit none
      integer natom,ntype
      real*8 Rc_type(100),Rc_M
      real*8 xatom(3,natom),AL(3,3),ALI(3,3)
      integer iatom(natom)
      real*8 dR_neigh(3,m_neigh,ntype,natom)
      integer iat_neigh(m_neigh,ntype,natom),list_neigh(m_neigh,ntype,natom)
      integer iat_neigh_M(m_neigh,ntype,natom)
      integer list_neigh_M(m_neigh,ntype,natom),map2neigh_M(m_neigh,ntype,natom)
      integer num_neigh(ntype,natom),num_neigh_M(ntype,natom)
      integer num_type(ntype),num_type_M(ntype)
      integer nperiod(3),ngrid(3)
      integer iflag,i,j,num
      integer i1,i2,i3,itype
      integer nd11,nd12,nd21,nd22,nd31,nd32,jj,ii1,ii2,ii3
      integer iat_type(100)
      integer itype_atom(natom)
      real*8 d,Rc2,dx1,dx2,dx3,dx,dy,dz,dd
      integer m_neigh
      integer, allocatable, dimension (:,:,:) :: num_grid
      integer, allocatable, dimension (:,:,:,:) :: list_grid
      integer, allocatable, dimension (:,:) :: ind

!      iflag=0
!      do i=1,3
!      d=dsqrt(AL(1,i)**2+AL(2,i)**2+AL(3,i)**2)
!      nperiod(i)=int(Rc_M/d)+1
!      if(d.lt.2*Rc_M) iflag=1
!      enddo

      call get_ALI(AL,ALI)

      iflag=0
      do i=1,3
      d=dsqrt(ALI(1,i)**2+ALI(2,i)**2+ALI(3,i)**2)
      d=1.d0/d      ! The distance between two plane
      nperiod(i)=int(Rc_M/d)+1
      ngrid(i)=int(d/Rc_M)
      if(ngrid(i).lt.1) ngrid(i)=1
      if(d.lt.2*Rc_M) iflag=1
      enddo



      do i=1,natom
      xatom(1,i)=mod(xatom(1,i)+10.d0,1.d0)
      xatom(2,i)=mod(xatom(2,i)+10.d0,1.d0)
      xatom(3,i)=mod(xatom(3,i)+10.d0,1.d0)
      enddo

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

      Rc2=Rc_M**2


      if(natom.gt.1000.and.ngrid(1).gt.1.and.ngrid(2).gt.1  &
          .and.ngrid(3).gt.1) goto 2001    ! can adjust 1000
! I didn't consider the very thin case, for large system calculations
! That case can slow down the calculation
! No replication of one atom into two atoms, etc. 
! This is for large system calculations



      do 2000 i=1,natom

      if(iflag.eq.1) then
      num_type=0
      num_type_M=0
      do  j=1,natom
      do i1=-nperiod(1),nperiod(1)
      do i2=-nperiod(2),nperiod(2)
      do i3=-nperiod(3),nperiod(3) 
      dx1=xatom(1,j)-xatom(1,i)+i1
      dx2=xatom(2,j)-xatom(2,i)+i2
      dx3=xatom(3,j)-xatom(3,i)+i3
      dx=AL(1,1)*dx1+AL(1,2)*dx2+AL(1,3)*dx3
      dy=AL(2,1)*dx1+AL(2,2)*dx2+AL(2,3)*dx3
      dz=AL(3,1)*dx1+AL(3,2)*dx2+AL(3,3)*dx3
      dd=dx**2+dy**2+dz**2
      if(dd.lt.Rc2.and.dd.gt.1.D-8) then
      itype=itype_atom(j)
      num_type_M(itype)=num_type_M(itype)+1
        if(num_type_M(itype).gt.m_neigh) then
        write(6,*) "num.gt.m_neigh, stop", m_neigh
        stop
        endif
      list_neigh_M(num_type_M(itype),itype,i)=j
      iat_neigh_M(num_type_M(itype),itype,i)=iatom(j)

      if(dd.lt.Rc_type(itype_atom(i))**2.and.dd.gt.1.D-8) then
      num_type(itype)=num_type(itype)+1
      list_neigh(num_type(itype),itype,i)=j
      iat_neigh(num_type(itype),itype,i)=iatom(j)
      map2neigh_M(num_type(itype),itype,i)=num_type_M(itype)
      dR_neigh(1,num_type(itype),itype,i)=dx
      dR_neigh(2,num_type(itype),itype,i)=dy
      dR_neigh(3,num_type(itype),itype,i)=dz
      endif
      endif
      enddo
      enddo
      enddo
      enddo
      num_neigh(:,i)=num_type(:)
      num_neigh_M(:,i)=num_type_M(:)
      endif


      if(iflag.eq.0) then
      num_type=0
      num_type_M=0
      do  j=1,natom
      dx1=xatom(1,j)-xatom(1,i)
      if(abs(dx1+1).lt.abs(dx1)) dx1=dx1+1
      if(abs(dx1-1).lt.abs(dx1)) dx1=dx1-1
      dx2=xatom(2,j)-xatom(2,i)
      if(abs(dx2+1).lt.abs(dx2)) dx2=dx2+1
      if(abs(dx2-1).lt.abs(dx2)) dx2=dx2-1
      dx3=xatom(3,j)-xatom(3,i)
      if(abs(dx3+1).lt.abs(dx3)) dx3=dx3+1
      if(abs(dx3-1).lt.abs(dx3)) dx3=dx3-1

      dx=AL(1,1)*dx1+AL(1,2)*dx2+AL(1,3)*dx3
      dy=AL(2,1)*dx1+AL(2,2)*dx2+AL(2,3)*dx3
      dz=AL(3,1)*dx1+AL(3,2)*dx2+AL(3,3)*dx3
      dd=dx**2+dy**2+dz**2
      if(dd.lt.Rc2.and.dd.gt.1.D-8) then
      itype=itype_atom(j)
      num_type_M(itype)=num_type_M(itype)+1

        if(num_type_M(itype).gt.m_neigh) then
        write(6,*) "num.gt.m_neigh, stop",m_neigh
        stop
        endif
      list_neigh_M(num_type_M(itype),itype,i)=j
      iat_neigh_M(num_type_M(itype),itype,i)=iatom(j)

      if(dd.lt.Rc_type(itype_atom(i))**2.and.dd.gt.1.D-8) then
      num_type(itype)=num_type(itype)+1

      list_neigh(num_type(itype),itype,i)=j
      iat_neigh(num_type(itype),itype,i)=iatom(j)
      map2neigh_M(num_type(itype),itype,i)=num_type_M(itype)
      dR_neigh(1,num_type(itype),itype,i)=dx
      dR_neigh(2,num_type(itype),itype,i)=dy
      dR_neigh(3,num_type(itype),itype,i)=dz
      endif
      endif
      enddo
      num_neigh(:,i)=num_type(:)
      num_neigh_M(:,i)=num_type_M(:)
      endif

2000  continue
      return


!Icccccccccccccccccccccccccccccccccccccccccccccc
2001  continue


!cccccccccccccccccccccccccccccccccccccccccccccccc
      allocate(list_grid(natom,ngrid(1),ngrid(2),ngrid(3)))   ! we can reduce this!
      allocate(num_grid(ngrid(1),ngrid(2),ngrid(3)))
      allocate(ind(3,natom))

!cccccccccccccccccccccccccccccccccccccccccccccccc
      num_grid=0
      do j=1,natom

      i1=xatom(1,j)*ngrid(1)+1
      if(i1.lt.1) i1=1
      if(i1.gt.ngrid(1)) i1=ngrid(1)
      i2=xatom(2,j)*ngrid(2)+1
      if(i2.lt.1) i2=1
      if(i2.gt.ngrid(2)) i2=ngrid(2)
      i3=xatom(3,j)*ngrid(3)+1
      if(i3.lt.1) i3=1
      if(i3.gt.ngrid(3)) i3=ngrid(3)


      num_grid(i1,i2,i3)=num_grid(i1,i2,i3)+1
      list_grid(num_grid(i1,i2,i3),i1,i2,i3)=j
      ind(1,j)=i1
      ind(2,j)=i2
      ind(3,j)=i3
      enddo
!cccccccccccccccccccccccccccccccccccccccccccccccc
      
      ! num_type=0
      ! num_type_M=0

!     ngrid(1) must be larger than 1 here
      if(ngrid(1).gt.2) then
      nd11=-1
      nd12=1
      else
      nd11=0
      nd12=1
      endif

      if(ngrid(2).gt.2) then
      nd21=-1
      nd22=1
      else
      nd21=0
      nd22=1
      endif

      if(ngrid(3).gt.2) then
      nd31=-1
      nd32=1
      else
      nd31=0
      nd32=1
      endif


      do 4001  i=1,natom
      num_type=0
      num_type_M=0

      do i1=ind(1,i)+nd11,ind(1,i)+nd12
      do i2=ind(2,i)+nd21,ind(2,i)+nd22
      do i3=ind(3,i)+nd31,ind(3,i)+nd32

      ii1=mod(i1-1+2*ngrid(1),ngrid(1))+1
      ii2=mod(i2-1+2*ngrid(2),ngrid(2))+1
      ii3=mod(i3-1+2*ngrid(3),ngrid(3))+1

      do jj=1,num_grid(ii1,ii2,ii3)

      j=list_grid(jj,ii1,ii2,ii3)

      dx1=xatom(1,j)-xatom(1,i)
      if(abs(dx1+1).lt.abs(dx1)) dx1=dx1+1
      if(abs(dx1-1).lt.abs(dx1)) dx1=dx1-1

      dx2=xatom(2,j)-xatom(2,i)
      if(abs(dx2+1).lt.abs(dx2)) dx2=dx2+1
      if(abs(dx2-1).lt.abs(dx2)) dx2=dx2-1

      dx3=xatom(3,j)-xatom(3,i)
      if(abs(dx3+1).lt.abs(dx3)) dx3=dx3+1
      if(abs(dx3-1).lt.abs(dx3)) dx3=dx3-1

      dx=AL(1,1)*dx1+AL(1,2)*dx2+AL(1,3)*dx3
      dy=AL(2,1)*dx1+AL(2,2)*dx2+AL(2,3)*dx3
      dz=AL(3,1)*dx1+AL(3,2)*dx2+AL(3,3)*dx3
      dd=dx**2+dy**2+dz**2
      if(dd.lt.Rc2.and.dd.gt.1.D-8) then
      itype=itype_atom(j)
      num_type_M(itype)=num_type_M(itype)+1
        if(num_type_M(itype).gt.m_neigh) then
        write(6,*) "num.gt.m_neigh, stop", m_neigh
        stop
        endif
      list_neigh_M(num_type_M(itype),itype,i)=j
      iat_neigh_M(num_type_M(itype),itype,i)=iatom(j)

      if(dd.lt.Rc_type(itype_atom(i))**2.and.dd.gt.1.D-8) then
      num_type(itype)=num_type(itype)+1
      list_neigh(num_type(itype),itype,i)=j
      iat_neigh(num_type(itype),itype,i)=iatom(j)
      map2neigh_M(num_type(itype),itype,i)=num_type_M(itype)
      dR_neigh(1,num_type(itype),itype,i)=dx
      dR_neigh(2,num_type(itype),itype,i)=dy
      dR_neigh(3,num_type(itype),itype,i)=dz
      endif
      endif

      enddo
      enddo
      enddo
      enddo

      num_neigh(:,i)=num_type(:)
      num_neigh_M(:,i)=num_type_M(:)

4001  continue
      deallocate(list_grid)
      deallocate(num_grid)
      deallocate(ind)

      return

      end subroutine find_neighbore
      

