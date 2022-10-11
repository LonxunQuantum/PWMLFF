       subroutine calc_U_JM1M2(UJ,nsnapw,ww,jm,theta0,theta,phi,CC_func,jmm)
       implicit double precision (a-h,o-z)
       real*8 ww(10)
       complex*16 UJ(-jm:jm,-jm:jm,0:jm,nsnapw)
       complex*16 cu,cai
       complex*16 cc,cc2
       real*8 CC_func(0:jmm,-jmm:jmm,-jmm:jmm,0:jmm)
! m1,m2,j are all doubled index: 2*m1,2*m3,2*j


       cai=cmplx(0.d0,1.d0)

       v=sin(theta0/2)*sin(theta)
       cu=cos(theta0/2)-cai*sin(theta0/2)*cos(theta)


!  All index has been multiplied by 2

       if(abs(v).gt.0.01) then
       
       do j=0,jm
       do m2=-j,j,2
       do m1=-j,j,2

       if(m1+m2.ge.0)  then   ! big if statement
       ms=j-m1
       if(j-m2.lt.ms) ms=j-m2
       cc2=(-cai*v)**j*(cu/(-cai*v))**((m1+m2)/2)*exp(-cai*(m1-m2)*phi/2)
       sum=0.d0
       do is=0,ms/2   ! is is not double index
       sum=sum+CC_func(is,m1,m2,j)*(1-1.d0/v**2)**is
       enddo
       endif

       if(m1+m2.lt.0) then
       ms=j+m1
       if(j+m2.lt.ms) ms=j+m2
       cc2=(-cai*v)**j*(conjg(cu)/(-cai*v))**(-(m1+m2)/2)*exp(-cai*(m1-m2)*phi/2)
       sum=0.d0
       do is=0,ms/2   ! is is not double index
       sum=sum+CC_func(is,-m1,-m2,j)*(1-1.d0/v**2)**is
       enddo
       endif
       
       do kk=1,nsnapw
       UJ(m1,m2,j,kk)=UJ(m1,m2,j,kk)+ww(kk)*cc2*sum
       enddo
  
       enddo
       enddo
       enddo

       else    ! v.lt.0.01

       do j=0,jm
       do m2=-j,j,2
       do m1=-j,j,2

       if(m1+m2.ge.0)  then   ! big if statement
       ms=j-m1
       if(j-m2.lt.ms) ms=j-m2
!       cc2=(-cai*v)**j*(cu/(-cai*v))**((m1+m2)/2)*exp(-cai*(m1-m2)*phi/2)
       cc2=(-cai)**j*(cu/(-cai))**((m1+m2)/2)*exp(-cai*(m1-m2)*phi/2)
       sum=0.d0
       do is=0,ms/2   ! is is not double index
!       sum=sum+CC_func(is,m1,m2,j)*(1-1.d0/v**2)**is
       sum=sum+CC_func(is,m1,m2,j)*(-1.d0)**is*v**(j-(m1+m2)/2-2*is)
       enddo
       endif

       if(m1+m2.lt.0) then
       ms=j+m1
       if(j+m2.lt.ms) ms=j+m2
!       cc2=(-cai*v)**j*(conjg(cu)/(-cai*v))**(-(m1+m2)/2)*exp(-cai*(m1-m2)*phi/2)
       cc2=(-cai)**j*(conjg(cu)/(-cai))**(-(m1+m2)/2)*exp(-cai*(m1-m2)*phi/2)
       sum=0.d0
       do is=0,ms/2   ! is is not double index
!       sum=sum+CC_func(is,-m1,-m2,j)*(1-1.d0/v**2)**is
       sum=sum+CC_func(is,-m1,-m2,j)*(-1.d0)**is*v**(j+(m1+m2)/2-2*is)
       enddo
       endif
       
       do kk=1,nsnapw
       UJ(m1,m2,j,kk)=UJ(m1,m2,j,kk)+ww(kk)*cc2*sum
       enddo
  
       enddo
       enddo
       enddo


       endif

       

       return
       end subroutine calc_U_JM1M2 
        


       
