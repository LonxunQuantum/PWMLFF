      subroutine calc_loop_ind(ind,num_contract)
      implicit double precision (a-h,o-z)
      integer ind(8,6561,0:8),num_contract(0:8)

! This is a fixed function for num_contract
! It does not depend on any input
!ccccccccccccccccccccccccccccccccccccccccccccccccccccc
      indc=0
      num_contract=1
      ind(1,1,indc)=1
!----------------------------
      indc=1  ! number of contraction
      ii=0
      do i1=1,3
      ii=ii+1
      ind(1,ii,indc)=i1
      enddo
      num_contract(indc)=ii
!--------------
      indc=2
      ii=0
      do i1=1,3
      do i2=1,3
      ii=ii+1
      ind(1,ii,indc)=i1
      ind(2,ii,indc)=i2
      enddo
      enddo
      num_contract(indc)=ii
!--------------
      indc=3
      ii=0
      do i1=1,3
      do i2=1,3
      do i3=1,3
      ii=ii+1
      ind(1,ii,indc)=i1
      ind(2,ii,indc)=i2
      ind(3,ii,indc)=i3
      enddo
      enddo
      enddo
      num_contract(indc)=ii
!--------------
      indc=4
      ii=0
      do i1=1,3
      do i2=1,3
      do i3=1,3
      do i4=1,3
      ii=ii+1
      ind(1,ii,indc)=i1
      ind(2,ii,indc)=i2
      ind(3,ii,indc)=i3
      ind(4,ii,indc)=i4
      enddo
      enddo
      enddo
      enddo
      num_contract(indc)=ii
!--------------
      indc=5
      ii=0
      do i1=1,3
      do i2=1,3
      do i3=1,3
      do i4=1,3
      do i5=1,3
      ii=ii+1
      ind(1,ii,indc)=i1
      ind(2,ii,indc)=i2
      ind(3,ii,indc)=i3
      ind(4,ii,indc)=i4
      ind(5,ii,indc)=i5
      enddo
      enddo
      enddo
      enddo
      enddo
      num_contract(indc)=ii
!--------------
      indc=6
      ii=0
      do i1=1,3
      do i2=1,3
      do i3=1,3
      do i4=1,3
      do i5=1,3
      do i6=1,3
      ii=ii+1
      ind(1,ii,indc)=i1
      ind(2,ii,indc)=i2
      ind(3,ii,indc)=i3
      ind(4,ii,indc)=i4
      ind(5,ii,indc)=i5
      ind(6,ii,indc)=i6
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      num_contract(indc)=ii
!--------------
      indc=7
      ii=0
      do i1=1,3
      do i2=1,3
      do i3=1,3
      do i4=1,3
      do i5=1,3
      do i6=1,3
      do i7=1,3
      ii=ii+1
      ind(1,ii,indc)=i1
      ind(2,ii,indc)=i2
      ind(3,ii,indc)=i3
      ind(4,ii,indc)=i4
      ind(5,ii,indc)=i5
      ind(6,ii,indc)=i6
      ind(7,ii,indc)=i7
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      num_contract(indc)=ii
!--------------
      indc=8
      ii=0
      do i1=1,3
      do i2=1,3
      do i3=1,3
      do i4=1,3
      do i5=1,3
      do i6=1,3
      do i7=1,3
      do i8=1,3
      ii=ii+1
      ind(1,ii,indc)=i1
      ind(2,ii,indc)=i2
      ind(3,ii,indc)=i3
      ind(4,ii,indc)=i4
      ind(5,ii,indc)=i5
      ind(6,ii,indc)=i6
      ind(7,ii,indc)=i7
      ind(8,ii,indc)=i8
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      num_contract(indc)=ii
!--------------------------------------------
      return
      end subroutine calc_loop_ind


     
     


