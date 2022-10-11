      subroutine calc_polynomial(d,mx,poly,dpoly,Rc,Rm)
      implicit double precision (a-h,o-z)
      
      real*8 poly(0:100),dpoly(0:100)

      x=2*(d-Rm)/(Rc-Rm)-1

      poly(0)=1
      poly(1)=x
      do i=2,mx
      poly(i)=2*x*poly(i-1)-poly(i-2)
      enddo

      dpoly(0)=0
      dpoly(1)=1
      do i=2,mx
      dpoly(i)=2*poly(i-1)+2*x*dpoly(i-1)-dpoly(i-2)
      enddo


      fact=(d-Rc)**2
      do i=0,mx
      dpoly(i)=dpoly(i)*fact*2/(Rc-Rm)+poly(i)*2*(d-Rc)
      poly(i)=poly(i)*fact
      enddo

      return
      end subroutine calc_polynomial

      

