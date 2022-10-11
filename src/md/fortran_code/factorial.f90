      function factorial(n)
      implicit none
      real*8 factorial
      integer n,i

      if(n.lt.0) then
      factorial=0
      return
      endif

      factorial=1
      do i=1,n 
      factorial=factorial*i
      enddo
      
      return
      end function factorial

