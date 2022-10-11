      subroutine find_feature_3bcos(natom,itype_atom,Rc,n3b,eta,w,alamda, &
        num_neigh,list_neigh, &
        dR_neigh,iat_neigh,ntype,&
        feat_all,dfeat_all,nfeat0m,m_neigh,n3bm,nfeat_atom)
      use mod_mpi
      implicit none
      integer ntype
      integer natom,n2b,n3b(ntype)
      integer nfeat0m,m_neigh,itype0
      integer nfeat_atom(natom)
      integer itype_atom(natom)
      real*8 Rc(ntype)
      real*8 dR_neigh(3,m_neigh,ntype,natom)
      real*8 dR_neigh_alltype(3,m_neigh,natom)
      integer iat_neigh(m_neigh,ntype,natom),list_neigh(m_neigh,ntype,natom)
      integer num_neigh(ntype,natom)
      integer num_neigh_alltype(natom)
      integer nperiod(3)
      integer iflag,i,j,num,iat,itype
      integer i1,i2,i3,itype1,itype2,j1,j2,iat1,iat2
      real*8 d,dx,dy,dz,dd
      real*8 pi,pi2,x,f1
      integer n3bm
      real*8 Rt,dx1(3),dx2(3)
      real*8 dd1,d1,dd2,d2,dd3,d3
      real*8 x1,x2,x3,fc1,fc2,fc3
      real*8 dfc1(3),dfc2(3),dfc3(3)
      real*8 dd_dot,costh,ff1,ff2
      integer kk
      real*8 eta1,w1,alamda1
   

      real*8 feat3(n3bm,ntype*(ntype+1)/2,natom_n)
      real*8 dfeat3(n3bm,ntype*(ntype+1)/2,natom_n,m_neigh,3)

      real*8 eta(200,50),w(100,50),alamda(100,50)

      real*8 feat3_tmp(n3bm,m_neigh,ntype)
      real*8 dfeat3_tmp(n3bm,m_neigh,ntype,3)
      integer ind_f(n3bm,m_neigh,ntype,natom)
      integer num_k(m_neigh,ntype),num12
      integer k,k1,k2,k12,j12,ii_f,jj,jj1,jj2,nneigh,ii
      real*8 y,y2
      integer itype12
      integer ind_all_neigh(m_neigh,ntype,natom),list_neigh_alltype(m_neigh,natom)

      real*8 feat_all(nfeat0m,natom_n)
      real*8 dfeat_all(nfeat0m,natom_n,m_neigh,3)

      real*8 dff1_jj1(3),dff2_jj1(3),dfc123_jj1(3)
      real*8 dff1_jj2(3),dff2_jj2(3),dfc123_jj2(3)
      real*8 dff1_1(3),dff2_1(3),dfc123_1(3)
      
      integer nfeat_atom_tmp(natom)
      integer ierr

      num_neigh_alltype=0
      do iat=1,natom
      if(mod(iat-1,nnodes).eq.inode-1) then
      num=1
      list_neigh_alltype(1,iat)=iat   ! the first neighbore is itself
      dR_neigh_alltype(:,1,iat)=0.d0

      do  itype=1,ntype
      do   j=1,num_neigh(itype,iat)


      num=num+1
        if(num.gt.m_neigh) then
        write(6,*) "total num_neigh.gt.m_neigh,stop",m_neigh
        stop
        endif
      ind_all_neigh(j,itype,iat)=num
      list_neigh_alltype(num,iat)=list_neigh(j,itype,iat)
      dR_neigh_alltype(:,num,iat)=dR_neigh(:,j,itype,iat)
      enddo
      enddo
      num_neigh_alltype(iat)=num

      endif
      enddo

!ccccccccccccccccccccccccccccccccccccccccc

      pi=4*datan(1.d0)
      pi2=2*pi

      feat3=0.d0
      dfeat3=0.d0

      iat1=0
      do 3000 iat=1,natom

      if(mod(iat-1,nnodes).eq.inode-1) then
      iat1=iat1+1

      itype0=itype_atom(iat)
      Rt=Rc(itype0)


      do 2000 itype2=1,ntype
      do 2000 itype1=1,itype2

      itype12=itype1+((itype2-1)*itype2)/2


      do 2000 j2=1,num_neigh(itype2,iat)
      do 2000 j1=1,num_neigh(itype1,iat)

      dx1(:)=dR_neigh(:,j1,itype1,iat)
      dx2(:)=dR_neigh(:,j2,itype2,iat)

!      if(itype1.eq.itype2.and.j1.ge.j2) goto 2000
      if(itype1.eq.itype2.and.j1.eq.j2) goto 2001

      jj1=ind_all_neigh(j1,itype1,iat)
      jj2=ind_all_neigh(j2,itype2,iat)

      dd1=dx1(1)**2+dx1(2)**2+dx1(3)**2
      d1=dsqrt(dd1)
      if(d1.gt.Rt) goto 2001
      x1=pi*d1/Rt
      fc1=0.5*(cos(x1)+1)
      dfc1(:)=-0.5*sin(x1)*pi/Rt/d1*dx1(:)

      dd2=dx2(1)**2+dx2(2)**2+dx2(3)**2
      d2=dsqrt(dd2)
      if(d2.gt.Rt) goto 2001
      x2=pi*d2/Rt
      fc2=0.5*(cos(x2)+1)
      dfc2(:)=-0.5*sin(x2)*pi/Rt/d2*dx2(:)


      dd3=(dR_neigh(1,j1,itype1,iat)-dR_neigh(1,j2,itype2,iat))**2+ &
         (dR_neigh(2,j1,itype1,iat)-dR_neigh(2,j2,itype2,iat))**2+ &
         (dR_neigh(3,j1,itype1,iat)-dR_neigh(3,j2,itype2,iat))**2

      dd3=(dx1(1)-dx2(1))**2+(dx1(2)-dx2(2))**2+(dx1(3)-dx2(3))**2

      d3=dsqrt(dd3)
      if(d3.gt.Rt) goto 2001

      x3=pi*d3/Rt
      fc3=0.5*(cos(x3)+1)
      dfc3(:)=-0.5*sin(x3)*pi/Rt/d3*(dx1(:)-dx2(:))


      dd_dot=dx1(1)*dx2(1)+dx1(2)*dx2(2)+dx1(3)*dx2(3)
      costh=dd_dot/(d1*d2)

      do 1000 kk=1,n3b(itype0) 
      eta1=eta(kk,itype0)
      w1=w(kk,itype0)
      alamda1=alamda(kk,itype0)
    

      ff1=2*(0.5+0.5*alamda1*costh)**eta1

      ff2=exp(-(dd1+dd2+dd3)/w1**2)
      
      
      feat3(kk,itype12,iat1)=feat3(kk,itype12,iat1)+ &
        ff1*ff2*fc1*fc2*fc3

!  dx1=x_j1-x_i,dx2=x_j2-x_i

      dff1_jj1(:)=eta1*alamda1*(0.5+0.5*alamda1*costh)**(eta1-1)*(dx2(:)-dx1(:)*dd_dot/d1**2)/(d1*d2)
      dff2_jj1(:)=-2*ff2*(2*dx1(:)-dx2(:))/w1**2
      dfc123_jj1(:)=dfc1(:)*fc2*fc3+fc1*fc2*dfc3(:)

      dff1_jj2(:)=eta1*alamda1*(0.5+0.5*alamda1*costh)**(eta1-1)*(dx1(:)-dx2(:)*dd_dot/d2**2)/(d1*d2)
      dff2_jj2(:)=-2*ff2*(2*dx2(:)-dx1(:))/w1**2
      dfc123_jj2(:)=fc1*dfc2(:)*fc3-fc1*fc2*dfc3(:)

      dfeat3(kk,itype12,iat1,jj1,:)=dfeat3(kk,itype12,iat1,jj1,:)+  &
       dff1_jj1(:)*ff2*fc1*fc2*fc3+ff1*dff2_jj1(:)*fc1*fc2*fc3+ff1*ff2*dfc123_jj1(:) 

      dfeat3(kk,itype12,iat1,jj2,:)=dfeat3(kk,itype12,iat1,jj2,:)+  &
       dff1_jj2(:)*ff2*fc1*fc2*fc3+ff1*dff2_jj2(:)*fc1*fc2*fc3+ff1*ff2*dfc123_jj2(:) 

      dff1_1(:)=-dff1_jj1(:)-dff1_jj2(:)
      dff2_1(:)=2*ff2*(dx1(:)+dx2(:))/w1**2
      dfc123_1(:)=-dfc1(:)*fc2*fc3-fc1*dfc2(:)*fc3   

      dfeat3(kk,itype12,iat1,1,:)=dfeat3(kk,itype12,iat1,1,:)+  &
       dff1_1(:)*ff2*fc1*fc2*fc3+ff1*dff2_1(:)*fc1*fc2*fc3+ff1*ff2*dfc123_1(:) 

1000   continue

2001   continue
2000   continue

      endif
3000   continue

!cccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccc
!   Now, we collect everything together, collapse the index (kk,itype12)
!   in feat3,dfeat3 into a single feature index 
      nfeat_atom_tmp=0

      iat1=0
      do 5000 iat=1,natom
      if(mod(iat-1,nnodes).eq.inode-1) then
      iat1=iat1+1      

      itype0=itype_atom(iat)
      nneigh=num_neigh_alltype(iat)

      num=0
      do itype2=1,ntype
      do itype1=1,itype2

      itype12=itype1+((itype2-1)*itype2)/2

      do kk=1,n3b(itype0) 

      num=num+1

      feat_all(num,iat1)=feat3(kk,itype12,iat1)
      dfeat_all(num,iat1,1:nneigh,:)=dfeat3(kk,itype12,iat1,1:nneigh,:)
      enddo

      enddo
      enddo

      nfeat_atom_tmp(iat)=num
       if(num.gt.nfeat0m) then
       write(6,*) "num.gt.nfeat0m,stop",num,nfeat0m
       stop
       endif

      endif
5000  continue


      call mpi_allreduce(nfeat_atom_tmp,nfeat_atom,natom,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierr)

!ccccccccccccccccccccccccccccccccccccc

      return
      end subroutine find_feature_3bcos



