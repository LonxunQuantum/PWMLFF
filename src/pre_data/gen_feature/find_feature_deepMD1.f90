      subroutine find_feature_deepMD1(natom,itype_atom,Rc,RC2,Rm,weight_rterm, &
        num_neigh,list_neigh, &
        dR_neigh,iat_neigh,ntype,M_type, &
        feat_all,dfeat_allR,nfeat0m,m_neigh,nfeat_atom)
      implicit none
      integer ntype
      integer natom,n2b(ntype)
      integer m_neigh
      integer itype_atom(natom)
      real*8 Rc(ntype),Rc2(ntype),Rm(ntype)
      integer M_type(ntype)
      real*8 weight_rterm(ntype)
      real*8 dR_neigh(3,m_neigh,ntype,natom)
      real*8 dR_neigh_alltype(3,m_neigh,natom)
      integer iat_neigh(m_neigh,ntype,natom),list_neigh(m_neigh,ntype,natom)
      integer num_neigh(ntype,natom)
      integer num_neigh_alltype(natom)
      integer nperiod(3)
      integer iflag,i,j,num,iat,itype
      integer i1,i2,i3,itype1,itype2,j1,j2,iat1,iat2
      real*8 d,dx1,dx2,dx3,dd
      real*8 grid2(200,50),wgauss(200,50)
      real*8 pi,pi2,x,f1
      real*8 Rt,ff,dt
      integer iflag_grid
      integer itype0,nfeat0m

      integer ind_f(2,m_neigh,ntype,natom)
      real*8 f32(2),df32(2,2,3)
      integer inf_f32(2),k,k1,k2,k12,j12,ii_f,jj,jj1,jj2,nneigh,ii
      real*8 y2
      integer itype12,ind_f32(2)
      integer ind_all_neigh(m_neigh,ntype,natom),list_neigh_alltype(m_neigh,natom)

      real*8 feat_all(nfeat0m,natom),dfeat_allR(nfeat0m,natom,m_neigh,3)
      real*8 dfeat_all(nfeat0m,natom,m_neigh,3)
      integer nfeat_atom(natom)

      integer M1,ii1,ii2
      real*8 dx(3),df2,f2,y,s,sum
      real*8 poly(100),dpoly(100)
      real*8 ww(0:3)

      real*8, allocatable, dimension (:,:) :: tensor
      real*8, allocatable, dimension (:,:,:,:) :: dtensor
      real*8, allocatable, dimension (:,:) :: dsum


      num_neigh_alltype=0
      do iat=1,natom
      num=1
      list_neigh_alltype(1,iat)=iat   ! the first neighbore is itself
      dR_neigh_alltype(:,1,iat)=0.d0

      do  itype=1,ntype
      do   j=1,num_neigh(itype,iat)
      num=num+1
        if(num.gt.m_neigh) then
        write(6,*) "Error! maxNeighborNum too small",m_neigh
        stop
        endif
      ind_all_neigh(j,itype,iat)=num
      list_neigh_alltype(num,iat)=list_neigh(j,itype,iat)
      dR_neigh_alltype(:,num,iat)=dR_neigh(:,j,itype,iat)
      enddo
      enddo
      num_neigh_alltype(iat)=num
      enddo

!ccccccccccccccccccccccccccccccccccccccccc

      pi=4*datan(1.d0)
      pi2=2*pi


      do 3000 iat=1,natom
       itype0=itype_atom(iat)
       M1=M_type(itype0)

       nneigh=num_neigh_alltype(iat)

      allocate(tensor(0:3,M1*ntype))
      allocate(dtensor(0:3,M1*ntype,nneigh,3))
      allocate(dsum(nneigh,3))

      tensor=0.d0
      dtensor=0.d0
      

      do 1000 itype=1,ntype
      do 1000 j=1,num_neigh(itype,iat)

      jj=ind_all_neigh(j,itype,iat)

      dx(1)=dR_neigh(1,j,itype,iat)
      dx(2)=dR_neigh(2,j,itype,iat)
      dx(3)=dR_neigh(3,j,itype,iat)
      dd=dx(1)**2+dx(2)**2+dx(3)**2
      d=dsqrt(dd)

      if(d.gt.Rc(itype0)) goto 1001


      if(d.lt.Rc2(itype0)) then
      f2=1.d0
      df2=0.d0
      else
      x=pi*(d-Rc2(itype))/(Rc(itype0)-Rc2(itype))
      f2=0.5*(cos(x)+1)
      df2=-0.5*sin(x)*pi/(Rc(itype0)-Rc2(itype))   ! need to add dx/d
      endif

      s=f2/d

      y=2*Rm(itype)*s-1.d0
      call calc_chebyshev(y,M1,poly,dpoly)
      do i=1,M1
      dpoly(i)=dpoly(i)*(df2/d-f2/d**2)*2*Rm(itype)  ! need to add dx/d
      enddo

      do i=1,M1
      ii=i+(itype-1)*M1
      tensor(0,ii)=tensor(0,ii)+s*poly(i)
      tensor(1,ii)=tensor(1,ii)+dx(1)*s/d*poly(i)
      tensor(2,ii)=tensor(2,ii)+dx(2)*s/d*poly(i)
      tensor(3,ii)=tensor(3,ii)+dx(3)*s/d*poly(i)
      ff=((df2/d-f2/d**2)*poly(i)+s*dpoly(i))   ! d(s*poly)/d_d
      dtensor(0,ii,jj,:)=dtensor(0,ii,jj,:)+ff*dx(:)/d
      dtensor(1,ii,jj,:)=dtensor(1,ii,jj,:)+(ff-s*poly(i)/d)*dx(1)*dx(:)/d**2    
      dtensor(1,ii,jj,1)=dtensor(1,ii,jj,1)+s/d*poly(i)
      dtensor(2,ii,jj,:)=dtensor(2,ii,jj,:)+(ff-s*poly(i)/d)*dx(2)*dx(:)/d**2    
      dtensor(2,ii,jj,2)=dtensor(2,ii,jj,2)+s/d*poly(i)
      dtensor(3,ii,jj,:)=dtensor(3,ii,jj,:)+(ff-s*poly(i)/d)*dx(3)*dx(:)/d**2    
      dtensor(3,ii,jj,3)=dtensor(3,ii,jj,3)+s/d*poly(i)
      dtensor(0,ii,1,:)=dtensor(0,ii,1,:)-ff*dx(:)/d
      dtensor(1,ii,1,:)=dtensor(1,ii,1,:)-(ff-s*poly(i)/d)*dx(1)*dx(:)/d**2    
      dtensor(1,ii,1,1)=dtensor(1,ii,1,1)-s/d*poly(i)
      dtensor(2,ii,1,:)=dtensor(2,ii,1,:)-(ff-s*poly(i)/d)*dx(2)*dx(:)/d**2    
      dtensor(2,ii,1,2)=dtensor(2,ii,1,2)-s/d*poly(i)
      dtensor(3,ii,1,:)=dtensor(3,ii,1,:)-(ff-s*poly(i)/d)*dx(3)*dx(:)/d**2    
      dtensor(3,ii,1,3)=dtensor(3,ii,1,3)-s/d*poly(i)
      enddo

1001  continue
1000  continue

      ww=1.d0
      ww(0)=weight_rterm(itype0)

      ii=0
      do ii1=1,M1*ntype
      do ii2=1,ii1
      ii=ii+1
      sum=0.d0
      dsum=0.d0
      do k=0,3
      sum=sum+tensor(k,ii1)*tensor(k,ii2)*ww(k)
      dsum(:,1)=dsum(:,1)+(dtensor(k,ii1,:,1)*tensor(k,ii2)+tensor(k,ii1)*dtensor(k,ii2,:,1))*ww(k)
      dsum(:,2)=dsum(:,2)+(dtensor(k,ii1,:,2)*tensor(k,ii2)+tensor(k,ii1)*dtensor(k,ii2,:,2))*ww(k)
      dsum(:,3)=dsum(:,3)+(dtensor(k,ii1,:,3)*tensor(k,ii2)+tensor(k,ii1)*dtensor(k,ii2,:,3))*ww(k)
      enddo

      feat_all(ii,iat)=sum
      dfeat_all(ii,iat,1:nneigh,1)=dsum(1:nneigh,1)
      dfeat_all(ii,iat,1:nneigh,2)=dsum(1:nneigh,2)
      dfeat_all(ii,iat,1:nneigh,3)=dsum(1:nneigh,3)
      enddo
      enddo

      deallocate(tensor)
      deallocate(dtensor)
      deallocate(dsum)
      nfeat_atom(iat)=M1*ntype*(M1*ntype+1)/2

3000  continue


!ccccccccccccccccccccccccccccccccccc
!  Now, we have to redefine the dfeat_all in another way. 
!  dfeat_all(i,iat,jneigh,3) means:
!  d_ith_feat_of_iat/d_R(jth_neigh_of_iat)
!  dfeat_allR(i,iat,jneigh,3) means:
!  d_ith_feat_of_jth_neigh/d_R(iat)
!cccccccccccccccccccccccccccccccccccccc
      dfeat_allR=0.d0

      do iat=1,natom
      do j=1,num_neigh_alltype(iat)
!ccccccccccccccccccc, this include the one which is itself, j=1

      iat2=list_neigh_alltype(j,iat)

      do j2=1,num_neigh_alltype(iat2)
      if(list_neigh_alltype(j2,iat2).eq.iat) then
      dd=(dR_neigh_alltype(1,j,iat)+dR_neigh_alltype(1,j2,iat2))**2+  &
         (dR_neigh_alltype(2,j,iat)+dR_neigh_alltype(2,j2,iat2))**2+  &
         (dR_neigh_alltype(3,j,iat)+dR_neigh_alltype(3,j2,iat2))**2  

      if(dd.lt.1.E-8) then
 
      do ii_f=1,nfeat_atom(iat)
      dfeat_allR(ii_f,iat2,j2,:)=dfeat_all(ii_f,iat,j,:)
!ccc Note, dfeat_allR(i,iat2,j2,3), it can have more i then nfeat_atom(iat2), 
! since it is the nfeat of j2_neighbore
      enddo
      endif
      endif

      enddo

      enddo
      enddo
!ccccccccccccccccccccccccccccccccccccc

      return
      end subroutine find_feature_deepMD1



