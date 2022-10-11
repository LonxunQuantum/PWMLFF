       subroutine calc_U_JM1M2(UJ,dUJ,mm_neigh,dx,d,rc,jjat,nsnapw, &
          ww,dww_dx,jm,CC_func,jmm)
       implicit none
       real*8 ww(10),dww_dx(10,3)
       integer nsnapw,jm,jmm,mm_neigh
       real*8 theta0,theta,phi
       complex*16 UJ(-jm:jm,-jm:jm,0:jm,nsnapw)
       complex*16 dUJ(3*mm_neigh,-jm:jm,-jm:jm,0:jm,nsnapw)
       complex*16 cu,cai
       complex*16 cc
       real*8 CC_func(0:jmm,-jmm:jmm,-jmm:jmm,0:jmm)
       real*8 dv_dtheta0,dv_dtheta
       real*8 dtheta0_dx(3),dphi_dx(3),dtheta_dx(3)
       complex*16 dcu_dtheta0,dcu_dtheta,dcu_dx(3)
       complex*16 cc2,dcc2_dx(3)
       real*8 sum,dsum_dx(3)
       real*8 d,pi,v,dx(3),rc
       integer j,m1,m2,jjat,ms,is,kk
       real*8 dv_dx(3)
       integer i,jj4
      
       
! m1,m2,j are all doubled index: 2*m1,2*m3,2*j


       cai=cmplx(0.d0,1.d0)
       pi=4*datan(1.d0)

       if(d.gt.1.D-10) then

        if(abs(dx(1)).lt.1.D-5) dx(1)=1.D-5
        if(abs(dx(2)).lt.1.D-5) dx(2)=1.D-5
        if(abs(dx(3)).lt.1.D-5) dx(3)=1.D-5

        d=dsqrt(dx(1)**2+dx(2)**2+dx(3)**2)

         phi=atan(dx(2)/dx(1))
!  Phi is defined within [-pi,pi], not [-pi/2,pi/2]
          if(abs(dx(1)).lt.1.D-10) then
          if(dx(2).gt.0) then
           phi=pi/2
           else
           phi=-pi/2
           endif
           else
           if(dx(1).lt.0.d0) then
            if(dx(2).lt.0.d0) phi=phi-pi
            if(dx(2).gt.0.d0) phi=phi+pi
          endif
          endif

         theta=acos(dx(3)/d)
!  Theta is within [0:pi]
         theta0=pi*3.d0/4*d/rc

         

       v=sin(theta0/2)*sin(theta)
       cu=cos(theta0/2)-cai*sin(theta0/2)*cos(theta)

       dv_dtheta0=0.5*cos(theta0/2)*sin(theta)
       dv_dtheta=sin(theta0/2)*cos(theta)
       dcu_dtheta0=-0.5*sin(theta0/2)-cai*0.5*cos(theta0/2)*cos(theta)
       dcu_dtheta=cai*sin(theta0/2)*sin(theta)
       dtheta0_dx(:)=3/4.d0*pi*dx(:)/(rc*d)
       dphi_dx(1)=-dx(2)/dx(1)**2*cos(phi)**2
       dphi_dx(2)=cos(phi)**2/dx(1)
       dphi_dx(3)=0.d0
       dtheta_dx(1)=dx(1)*dx(3)/d**3/sin(theta)
       dtheta_dx(2)=dx(2)*dx(3)/d**3/sin(theta)
       dtheta_dx(3)=-(dx(1)**2+dx(2)**2)/d**3/sin(theta)

       else    ! special

       theta0=1.D-6
       theta=1.D-6
       phi=1.D-6
       v=0.d0
       cu=1.d0

       dv_dtheta0=0.d0
       dv_dtheta=0.d0
       dcu_dtheta0=0.d0
       dcu_dtheta=0.d0
       dtheta0_dx(:)=0.d0
       dphi_dx(:)=0.d0
       dtheta_dx(:)=0.d0
       endif

       dv_dx(:)=dv_dtheta0*dtheta0_dx(:)+dv_dtheta*dtheta_dx(:)
       dcu_dx(:)=dcu_dtheta0*dtheta0_dx(:)+dcu_dtheta*dtheta_dx(:)

!-----------------------------------------------------

!  All index has been multiplied by 2

       
       do j=0,jm
       do m2=-j,j,2
       do m1=-j,j,2

       if(abs(v).gt.0.01) then

       if(m1+m2.ge.0)  then   ! big if statement
       ms=j-m1
       if(j-m2.lt.ms) ms=j-m2
       cc2=(-cai*v)**j*(cu/(-cai*v))**((m1+m2)/2)*exp(-cai*(m1-m2)*phi/2)
       dcc2_dx(:)=cc2*((j-(m1+m2)/2)/v*dv_dx(:)+((m1+m2)/2)/cu*dcu_dx(:)+(-cai*(m1-m2)/2)*dphi_dx(:))
       sum=0.d0
       dsum_dx(:)=0.d0
       do is=0,ms/2   ! is is not double index
       sum=sum+CC_func(is,m1,m2,j)*(1-1.d0/v**2)**is
       dsum_dx(:)=dsum_dx(:)+CC_func(is,m1,m2,j)*is*(1-1.d0/v**2)**(is-1)*2/v**3*dv_dx(:)
       enddo
       endif

       if(m1+m2.lt.0) then
       ms=j+m1
       if(j+m2.lt.ms) ms=j+m2
       cc2=(-cai*v)**j*(conjg(cu)/(-cai*v))**(-(m1+m2)/2)*exp(-cai*(m1-m2)*phi/2)
       dcc2_dx(:)=cc2*((j+(m1+m2)/2)/v*dv_dx(:)+(-(m1+m2)/2)/conjg(cu)*conjg(dcu_dx(:))+(-cai*(m1-m2)/2)*dphi_dx(:))
       sum=0.d0
       dsum_dx(:)=0.d0
       do is=0,ms/2   ! is is not double index
       sum=sum+CC_func(is,-m1,-m2,j)*(1-1.d0/v**2)**is
       dsum_dx(:)=dsum_dx(:)+CC_func(is,-m1,-m2,j)*is*(1-1.d0/v**2)**(is-1)*2/v**3*dv_dx(:)
       enddo
       endif
       
       else    ! v.lt.0.01

       if(m1+m2.ge.0)  then   ! big if statement
       ms=j-m1
       if(j-m2.lt.ms) ms=j-m2
!       cc2=(-cai*v)**j*(cu/(-cai*v))**((m1+m2)/2)*exp(-cai*(m1-m2)*phi/2)
       cc2=(-cai)**j*(cu/(-cai))**((m1+m2)/2)*exp(-cai*(m1-m2)*phi/2)
       dcc2_dx(:)=cc2*(((m1+m2)/2)/cu*dcu_dx(:)+(-cai*(m1-m2)/2)*dphi_dx(:))
       
       sum=0.d0
       dsum_dx(:)=0.d0
       do is=0,ms/2   ! is is not double index
!       sum=sum+CC_func(is,m1,m2,j)*(1-1.d0/v**2)**is
       sum=sum+CC_func(is,m1,m2,j)*(-1.d0)**is*v**(j-(m1+m2)/2-2*is)
       dsum_dx(:)=dsum_dx(:)+CC_func(is,m1,m2,j)*(-1.d0)**is*(j-(m1+m2)/2-2*is)*  &
                           v**(j-(m1+m2)/2-2*is-1)*dv_dx(:)
       enddo
       endif

       if(m1+m2.lt.0) then
       ms=j+m1
       if(j+m2.lt.ms) ms=j+m2
!       cc2=(-cai*v)**j*(conjg(cu)/(-cai*v))**(-(m1+m2)/2)*exp(-cai*(m1-m2)*phi/2)
       cc2=(-cai)**j*(conjg(cu)/(-cai))**(-(m1+m2)/2)*exp(-cai*(m1-m2)*phi/2)
       dcc2_dx(:)=cc2*((-(m1+m2)/2)/conjg(cu)*conjg(dcu_dx(:))+(-cai*(m1-m2)/2)*dphi_dx(:))
       sum=0.d0
       dsum_dx(:)=0.d0
       do is=0,ms/2   ! is is not double index
!       sum=sum+CC_func(is,-m1,-m2,j)*(1-1.d0/v**2)**is
       sum=sum+CC_func(is,-m1,-m2,j)*(-1.d0)**is*v**(j+(m1+m2)/2-2*is)
       dsum_dx(:)=dsum_dx(:)+CC_func(is,-m1,-m2,j)*(-1.d0)**is*(j+(m1+m2)/2-2*is)*  &
                            v**(j+(m1+m2)/2-2*is-1)*dv_dx(:)
       enddo
       endif

       endif    ! v.lt.0.001
!cccccccccccccccccccccccccccccccc
       
       do kk=1,nsnapw
       UJ(m1,m2,j,kk)=UJ(m1,m2,j,kk)+ww(kk)*cc2*sum
       enddo

! not the case fot the origin point
! For jjat=origin point, there is no derivative (derivative equals zero) 
! but for other jjat, there is a derivation regarding to the origin at index 1

       if(jjat.ne.0) then     ! not the case for the original point
       do kk=1,nsnapw
       do i=1,3
       jj4=jjat+(i-1)*mm_neigh
       dUJ(jj4,m1,m2,j,kk)=dUJ(jj4,m1,m2,j,kk)+ww(kk)*cc2*dsum_dx(i)+  &
              ww(kk)*dcc2_dx(i)*sum+dww_dx(kk,i)*cc2*sum

       jj4=1+(i-1)*mm_neigh
       dUJ(jj4,m1,m2,j,kk)=dUJ(jj4,m1,m2,j,kk)-(ww(kk)*cc2*dsum_dx(i)+ &
              ww(kk)*dcc2_dx(i)*sum+dww_dx(kk,i)*cc2*sum)
       enddo
       enddo
       endif
       
  
       enddo
       enddo
       enddo



       return
       end subroutine calc_U_JM1M2 
        


       
