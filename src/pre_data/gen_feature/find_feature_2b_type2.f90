      subroutine find_feature_2b_type2(natom,itype_atom,Rc,n2b, &
        num_neigh,list_neigh, &
        dR_neigh,iat_neigh,ntype,grid2, &
        feat_all,dfeat_allR,nfeat0m,m_neigh,n2bm,nfeat_atom)
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
      real*8 d,dx1,dx2,dx3,dx,dy,dz,dd
      real*8 grid2(0:n2bm+1,ntype)
      real*8 pi,pi2,x,f1
      integer iflag_grid
      integer itype0,nfeat0m,n2bm

      real*8 feat2(n2bm,ntype,natom)
      real*8 dfeat2(n2bm,ntype,natom,m_neigh,3)

      integer ind_f(2,m_neigh,ntype,natom)
      real*8 f32(2),df32(2,2,3)
      integer inf_f32(2),k,k1,k2,k12,j12,ii_f,jj,jj1,jj2,nneigh,ii
      real*8 y,y2
      integer itype12,ind_f32(2)
      integer ind_all_neigh(m_neigh,ntype,natom),list_neigh_alltype(m_neigh,natom)

      real*8 feat_all(nfeat0m,natom),dfeat_allR(nfeat0m,natom,m_neigh,3)
      real*8 dfeat_all(nfeat0m,natom,m_neigh,3)
      integer nfeat_atom(natom)


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

      feat2=0.d0
      dfeat2=0.d0


      do 3000 iat=1,natom
       itype0=itype_atom(iat)

      do 1000 itype=1,ntype
      do 1000 j=1,num_neigh(itype,iat)

      jj=ind_all_neigh(j,itype,iat)

      dd=dR_neigh(1,j,itype,iat)**2+dR_neigh(2,j,itype,iat)**2+dR_neigh(3,j,itype,iat)**2
      d=dsqrt(dd)


      do k=1,n2b(itype0)+1
      if(grid2(k,itype0).ge.d) exit
      enddo
      k=k-1


      if(k.gt.n2b(itype0))  k=n2b(itype0)
!  This point will be in the intervalL [grid2(k),grid2(k+1)]
!  It will have feature contribution:
!  feat2(k,itype,iat),feat2(k+1,itype,iat)
       if(k.le.n2b(itype0)-2) then   ! k+2 feature
       x=(d-grid2(k,itype0))/(grid2(k+3,itype0)-grid2(k,itype0))
       y=(x-0.5d0)*pi2
       f1=0.5d0*(cos(y)+1)
       feat2(k+2,itype,iat)=feat2(k+2,itype,iat)+f1
       y2=-pi*sin(y)/(d*(grid2(k+3,itype0)-grid2(k,itype0)))
       dfeat2(k+2,itype,iat,jj,:)=dfeat2(k+2,itype,iat,jj,:)+y2*dR_neigh(:,j,itype,iat)
       dfeat2(k+2,itype,iat,1,:)=dfeat2(k+2,itype,iat,1,:)-y2*dR_neigh(:,j,itype,iat)
! Note, (k+1,itype) is the feature inde
       endif

       if(k.le.n2b(itype0)-1) then   ! k+1 feature
       x=(d-grid2(k-1,itype0))/(grid2(k+2,itype0)-grid2(k-1,itype0))
       y=(x-0.5d0)*pi2
       f1=0.5d0*(cos(y)+1)
       feat2(k+1,itype,iat)=feat2(k+1,itype,iat)+f1
       y2=-pi*sin(y)/(d*(grid2(k+2,itype0)-grid2(k-1,itype0)))
       dfeat2(k+1,itype,iat,jj,:)=dfeat2(k+1,itype,iat,jj,:)+y2*dR_neigh(:,j,itype,iat)
       dfeat2(k+1,itype,iat,1,:)=dfeat2(k+1,itype,iat,1,:)-y2*dR_neigh(:,j,itype,iat)
! Note, (k+1,itype) is the feature inde
       endif
       
       if(k.gt.0.and.k.le.n2b(itype0)) then   ! k feature
       x=(d-grid2(k-2,itype0))/(grid2(k+1,itype0)-grid2(k-2,itype0))
       y=(x-0.5d0)*pi2
       f1=0.5d0*(cos(y)+1)
       feat2(k,itype,iat)=feat2(k,itype,iat)+f1
       y2=-pi*sin(y)/(d*(grid2(k+1,itype0)-grid2(k-2,itype0)))
       dfeat2(k,itype,iat,jj,:)=dfeat2(k,itype,iat,jj,:)+y2*dR_neigh(:,j,itype,iat)
       dfeat2(k,itype,iat,1,:)=dfeat2(k,itype,iat,1,:)-y2*dR_neigh(:,j,itype,iat)
! Note, (k+1,itype) is the feature inde
       endif


!cccccccccccc So, one Rij will always have two features k, k+1  (1,2)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
1000  continue
3000  continue


!   Now, the three body feature
!ccccccccccccccccccccccccccccccccccccc


!cccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccc
!   Now, we collect everything together, collapse the index (k,itype)
!   feat2, into a single feature. 

      do 5000 iat=1,natom
      itype0=itype_atom(iat)
      nneigh=num_neigh_alltype(iat)

      num=0
      do itype=1,ntype
      do k=1,n2b(itype0)
      num=num+1
      feat_all(num,iat)=feat2(k,itype,iat)
      dfeat_all(num,iat,1:nneigh,:)=dfeat2(k,itype,iat,1:nneigh,:)
      enddo
      enddo

      nfeat_atom(iat)=num
      if(num.gt.nfeat0m) then
      write(6,*) "num.gt.nfeat0m,stop",num,nfeat0m
      stop
      endif

5000  continue


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
      end subroutine find_feature_2b_type2



