      subroutine find_feature_snap(natom,itype_atom,Rc,nsnapw_type,snapj_type,wsnap_type, &
        num_neigh,list_neigh, &
        dR_neigh,iat_neigh,ntype, &
        feat_all,dfeat_allR,nfeat0m,m_neigh,nfeat_atom,nBB,nBBm,jjj123, &
        CC_func,Clebsch_Gordan,jmm)
      implicit none
      integer ntype
      integer natom,n2b(ntype)
      integer m_neigh
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
      real*8 d,dx1,dx2,dx3,dd,y,y2
      real*8 dx(3)
      real*8 pi,pi2,f1
      real*8 Rt,f2,ff,dt
      integer iflag_grid
      integer itype0,nfeat0m
      integer nBBm
      integer jjj123(3,nBBm,ntype)
      integer nBB(50)
      integer nsnapw_type(50)
      real*8 snapj_type(50)
      real*8 Wsnap_type(50,10,50)
      integer jmm,jm,kk,kkk
      real*8 CC_func(0:jmm,-jmm:jmm,-jmm:jmm,0:jmm)    ! jmm is the double index
      real*8 Clebsch_Gordan(-jmm:jmm,-jmm:jmm,-jmm:jmm,0:jmm,0:jmm,0:jmm)
      real*8 sum
      real*8 ww(10),dww_dx(10,3)
      integer m11,m12,m21,m22,m1,m2
      integer mm_neigh


      integer ind_f(2,m_neigh,ntype,natom)
      real*8 f32(2),df32(2,2,3)
      integer inf_f32(2),k,k1,k2,k12,j12,ii_f,jj,jj1,jj2,nneigh,ii
      integer itype12,ind_f32(2)
      integer ind_all_neigh(m_neigh,ntype,natom),list_neigh_alltype(m_neigh,natom)

      real*8 feat_all(nfeat0m,natom),dfeat_allR(nfeat0m,natom,m_neigh,3)
      real*8 dfeat_all(nfeat0m,natom,m_neigh,3)
      integer nfeat_atom(natom)
      complex*16,allocatable,dimension (:,:,:,:) :: UJ
      complex*16,allocatable,dimension (:,:,:,:,:) :: dUJ
      real*8,allocatable,dimension (:) :: dsum
      complex*16 cc
      integer jj3,jj4


      num_neigh_alltype=0
      do iat=1,natom
      num=1    ! including the original
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
      feat_all=0.d0
      dfeat_all=0.d0

      pi=4*datan(1.d0)
      pi2=2*pi


      do 3000 iat=1,natom
       itype0=itype_atom(iat)
       jm=snapj_type(itype0)*2*1.001

       mm_neigh=num_neigh_alltype(iat)   ! include the origin at 1
       allocate(UJ(-jm:jm,-jm:jm,0:jm,nsnapw_type(itype0)))
       allocate(dUJ(3*mm_neigh,-jm:jm,-jm:jm,0:jm,nsnapw_type(itype0)))

       UJ=cmplx(0.d0,0.d0)
       dUJ=cmplx(0.d0,0.d0)


      do kk=1,nsnapw_type(itype0)
      ww(kk)=wsnap_type(itype0,kk,itype0)
      dww_dx(kk,:)=0.d0
      enddo



!cccccccccc Not really so sure whether this is necessary. 
! In any case, in their SNAP paper, they have this, so I will add it here
! Is shows it is the spherical expansion for the whole rho, including 
! the origin point. It might only affect j=0 point
      dx=0.d0   ! origin
      d=0.d0
      
      call calc_U_JM1M2(UJ,dUJ,mm_neigh,dx,d,Rc(itype0),1,nsnapw_type(itype0), &
          ww,dww_dx,jm,CC_func,jmm)

      dUJ=cmplx(0.d0,0.d0)
!cccccccccccccccccccccccccccccccccc
      

      do 1000 itype=1,ntype
      do 1000 k=1,num_neigh(itype,iat)   ! this does not include the original point

      jj=ind_all_neigh(k,itype,iat)
      dx(1)=dR_neigh(1,k,itype,iat)
      dx(2)=dR_neigh(2,k,itype,iat)
      dx(3)=dR_neigh(3,k,itype,iat)
      dd=dx(1)**2+dx(2)**2+dx(3)**2
      d=dsqrt(dd)

      if(d.gt.Rc(itype0)) goto 1001


      y=pi*d/Rc(itype0)
      f2=0.5*(cos(y)+1)
      do kk=1,nsnapw_type(itype0)
      ww(kk)=f2*wsnap_type(itype,kk,itype0)
      dww_dx(kk,:)=-0.5*sin(y)*pi/Rc(itype0)*dx(:)/d*wsnap_type(itype,kk,itype0)
      enddo

      
      call calc_U_JM1M2(UJ,dUJ,mm_neigh,dx,d,Rc(itype0),jj,nsnapw_type(itype0), &
              ww,dww_dx,jm,CC_func,jmm)

1001  continue
1000  continue

      allocate(dsum(mm_neigh*3))

      do 1500 kk=1,nsnapw_type(itype0)
      do 1500 kkk=1,nBB(itype0)

      j1=jjj123(1,kkk,itype0)    ! stored the double index
      j2=jjj123(2,kkk,itype0)
      j=jjj123(3,kkk,itype0)

!  Note: |j1-j2|.le.j, j.le.j1+j2, and mod(j1+j2-j,2).eq.0 (i.e, the original j1+j2-j is an integer 
 
      sum=0.d0
      dsum=0.d0
      do m11=-j1,j1,2
      do m12=-j1,j1,2
      do m21=-j2,j2,2
      do m22=-j2,j2,2
!      do m1=-j,j,2
!      do m2=-j,j,2
!      if(m11+m21.eq.m1.and.m12+m22.eq.m2) then
      m1=m11+m21
      m2=m12+m22
      if(abs(m1).le.j.and.abs(m2).le.j) then
      y=Clebsch_Gordan(m1,m11,m21,j,j1,j2)*Clebsch_Gordan(m2,m12,m22,j,j1,j2)
      sum=sum+y*conjg(UJ(m1,m2,j,kk))*UJ(m11,m12,j1,kk)*UJ(m21,m22,j2,kk)
            
      dsum(:)=dsum(:)+y*  &
       (conjg(DUJ(:,m1,m2,j,kk))*UJ(m11,m12,j1,kk)*UJ(m21,m22,j2,kk)+  &
        conjg(UJ(m1,m2,j,kk))*DUJ(:,m11,m12,j1,kk)*UJ(m21,m22,j2,kk)+  &
        conjg(UJ(m1,m2,j,kk))*UJ(m11,m12,j1,kk)*DUJ(:,m21,m22,j2,kk))

!   checked, indeed, after all the sum, only real part remain
      endif
!      enddo
!      enddo
      enddo
      enddo
      enddo
      enddo
      feat_all(kkk+(kk-1)*nBB(itype0),iat)=sum

      do jj3=1,3
      do jj=1,mm_neigh
      jj4=jj+(jj3-1)*mm_neigh
      dfeat_all(kkk+(kk-1)*nBB(itype0),iat,jj,jj3)=dsum(jj4)
      enddo
      enddo

1500  continue

       nfeat_atom(iat)=nBB(itype0)*nsnapw_type(itype0)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       deallocate(UJ)
       deallocate(dUJ)
       deallocate(dsum)
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
      end subroutine find_feature_snap



