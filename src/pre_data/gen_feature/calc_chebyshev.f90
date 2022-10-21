subroutine calc_chebyshev(x,mx,poly,dpoly)
      implicit double precision (a-h,o-z)
      
      real*8 poly(100),dpoly(100)

!   poly(m) is actually m-1 chebyshev

      poly(1)=1
      poly(2)=x
      do i=3,mx
      poly(i)=2*x*poly(i-1)-poly(i-2)
      enddo

      dpoly(1)=0
      dpoly(2)=1
      do i=3,mx
      dpoly(i)=2*poly(i-1)+2*x*dpoly(i-1)-dpoly(i-2)
      enddo

      return
end subroutine calc_chebyshev

