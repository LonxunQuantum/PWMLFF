subroutine find_feature_MTP(natom,itype_atom,Rc,Rm, &
        num_neigh,list_neigh,dR_neigh,iat_neigh, &
        feat_all,dfeat_allR,nfeat0m,m_neigh,nfeat_atom, &
        numCC_type,numT_all,mu_all,rank_all,jmu_b_all,itype_b_all,indi_all,indj_all,ntype)
      implicit none
      integer ntype
      integer natom,n2b(ntype)
      integer m_neigh
      integer itype_atom(natom)
      real*8 Rc(ntype),Rm(ntype)
      real*8 dR_neigh(3,m_neigh,ntype,natom)
      real*8 dR_neigh_alltype(3,m_neigh,natom)
      integer iat_neigh(m_neigh,ntype,natom),list_neigh(m_neigh,ntype,natom)
      integer num_neigh(ntype,natom)
      integer num_neigh_alltype(natom)
      integer nperiod(3)
      integer iflag,i,j,num,iat,itype
      integer itype1,itype2,j1,j2,iat1,iat2
      real*8 d,dd
      real*8 grid2(200,50),wgauss(200,50)
      real*8 pi,pi2,x,f1
      real*8 Rt,f2,ff,dt
      integer iflag_grid
      integer itype0,nfeat0m


      integer ind_f(2,m_neigh,ntype,natom)
      real*8 f32(2),df32(2,2,3)
      integer inf_f32(2),k,k1,k2,k12,j12,ii_f,jj,jj1,jj2,nneigh,ii
      real*8 y,y2
      integer itype12,ind_f32(2)
      integer ind_all_neigh(m_neigh,ntype,natom),list_neigh_alltype(m_neigh,natom)

      real*8 feat_all(nfeat0m,natom),dfeat_allR(nfeat0m,natom,m_neigh,3)
      real*8 dfeat_all(nfeat0m,natom,m_neigh,3)
      integer nfeat_atom(natom)
      integer numCC_type(10),numT_all(20000,10)
      integer rank_all(4,20000,10),mu_all(4,20000,10),jmu_b_all(4,20000,10),itype_b_all(4,20000,10)
      integer indi_all(5,4,20000,10),indj_all(5,4,20000,10)
      integer ind(8,6561,0:8),num_contract(0:8)
      integer rankm_type(10),mum_type(10)
      integer kk,rankmx,mumx
      real*8, allocatable, dimension (:,:,:,:) :: tensor
      real*8, allocatable, dimension (:,:,:,:,:,:) :: dtensor
      real*8 poly(0:100),dpoly(0:100)
      real*8 dx(3)
      integer imu,i1,i2,i3,i4,num_tot,indc,iii,id,rankT
      integer indi_tmp(5,4),indj_tmp(5,4),iloop(5,4)
      real*8 sum,prod
      real*8 dprod(3,m_neigh),dprod_sum(3,m_neigh)


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
      do itype=1,ntype
      rankm_type(itype)=0
      mum_type(itype)=0
      do kk=1,numCC_type(itype)  ! the basis (contraction)
      do i=1,numT_all(kk,itype)  ! the tensor index, numT_all is the num of tensor to be contracted in this term
      if(rank_all(i,kk,itype).gt.rankm_type(itype)) rankm_type(itype)=rank_all(i,kk,itype)
      if(mu_all(i,kk,itype).gt.mum_type(itype)) mum_type(itype)=mu_all(i,kk,itype)
      enddo
      enddo
      if(rankm_type(itype).gt.4) then
      write(6,*) "tensor rankm.gt.4,stop", rankm_type(itype)
      stop
      endif
      enddo
!ccccccccccccccccccccccccccccccccccccccccc


      pi=4*datan(1.d0)
      pi2=2*pi



      do 3000 iat=1,natom

       itype0=itype_atom(iat)
       rankmx=rankm_type(itype0)
       mumx=mum_type(itype0)
       nneigh=num_neigh_alltype(iat)    ! nneigh=< m_neigh

       allocate(tensor(81,0:mumx,ntype,0:4))
       allocate(dtensor(3,m_neigh,81,0:mumx,ntype,0:4))
       tensor=0.d0
       dtensor=0.d0

 

      do 1000 itype=1,ntype
      do 1000 j=1,num_neigh(itype,iat)

      jj=ind_all_neigh(j,itype,iat)    ! jj.belong.[1:m_neigh]

      dd=dR_neigh(1,j,itype,iat)**2+dR_neigh(2,j,itype,iat)**2+dR_neigh(3,j,itype,iat)**2
      d=dsqrt(dd)

      if(d.gt.Rc(itype0)) goto 1001

      call calc_polynomial(d,mumx,poly,dpoly,Rc(itype0),Rm(itype0))


      dx(1)=dR_neigh(1,j,itype,iat)
      dx(2)=dR_neigh(2,j,itype,iat)
      dx(3)=dR_neigh(3,j,itype,iat)

      if(rankmx.ge.0) then
! rank 0 tensor 
      do imu=0,mumx
      tensor(1,imu,itype,0)=tensor(1,imu,itype,0)+poly(imu)
      dtensor(:,jj,1,imu,itype,0)=dtensor(:,jj,1,imu,itype,0)+dpoly(imu)*dx(:)/d
      dtensor(:,1,1,imu,itype,0)=dtensor(:,1,1,imu,itype,0)-dpoly(imu)*dx(:)/d
      enddo
      endif

      if(rankmx.ge.1) then
! rank 1 tensor 
      do imu=0,mumx
      do i=1,3
      tensor(i,imu,itype,1)=tensor(i,imu,itype,1)+poly(imu)*dx(i)
      dtensor(:,jj,i,imu,itype,1)=dtensor(:,jj,i,imu,itype,1)+dpoly(imu)*dx(i)*dx(:)/d   ! original line bug
      dtensor(:,1,i,imu,itype,1)=dtensor(:,1,i,imu,itype,1)-dpoly(imu)*dx(i)*dx(:)/d

       dtensor(i,jj,i,imu,itype,1)=dtensor(i,jj,i,imu,itype,1)+poly(imu)
       dtensor(i,1,i,imu,itype,1)=dtensor(i,1,i,imu,itype,1)-poly(imu)
      enddo
      enddo
      endif

      if(rankmx.ge.2) then
! rank 2 tensor 
      do imu=0,mumx
      do i1=1,3
      do i2=1,3
      ii=(i2-1)*3+i1
      tensor(ii,imu,itype,2)=tensor(ii,imu,itype,2)+poly(imu)*dx(i1)*dx(i2)
      dtensor(:,jj,ii,imu,itype,2)=dtensor(:,jj,ii,imu,itype,2)+dpoly(imu)*dx(i1)*dx(i2)*dx(:)/d
      dtensor(:,1,ii,imu,itype,2)=dtensor(:,1,ii,imu,itype,2)-dpoly(imu)*dx(i1)*dx(i2)*dx(:)/d

       dtensor(i1,jj,ii,imu,itype,2)=dtensor(i1,jj,ii,imu,itype,2)+poly(imu)*dx(i2)
       dtensor(i1,1,ii,imu,itype,2)=dtensor(i1,1,ii,imu,itype,2)-poly(imu)*dx(i2)

       dtensor(i2,jj,ii,imu,itype,2)=dtensor(i2,jj,ii,imu,itype,2)+poly(imu)*dx(i1)
       dtensor(i2,1,ii,imu,itype,2)=dtensor(i2,1,ii,imu,itype,2)-poly(imu)*dx(i1)
        
      enddo
      enddo
      enddo

      endif

      if(rankmx.ge.3) then
! rank 3 tensor 
      do imu=0,mumx
      do i1=1,3
      do i2=1,3
      do i3=1,3
      ii=(i3-1)*9+(i2-1)*3+i1
      tensor(ii,imu,itype,3)=tensor(ii,imu,itype,3)+poly(imu)*dx(i1)*dx(i2)*dx(i3)
      dtensor(:,jj,ii,imu,itype,3)=dtensor(:,jj,ii,imu,itype,3)+dpoly(imu)*dx(i1)*dx(i2)*dx(i3)*dx(:)/d
      dtensor(:,1,ii,imu,itype,3)=dtensor(:,1,ii,imu,itype,3)-dpoly(imu)*dx(i1)*dx(i2)*dx(i3)*dx(:)/d

       dtensor(i1,jj,ii,imu,itype,3)=dtensor(i1,jj,ii,imu,itype,3)+poly(imu)*dx(i2)*dx(i3)
       dtensor(i1,1,ii,imu,itype,3)=dtensor(i1,1,ii,imu,itype,3)-poly(imu)*dx(i2)*dx(i3)

       dtensor(i2,jj,ii,imu,itype,3)=dtensor(i2,jj,ii,imu,itype,3)+poly(imu)*dx(i1)*dx(i3)
       dtensor(i2,1,ii,imu,itype,3)=dtensor(i2,1,ii,imu,itype,3)-poly(imu)*dx(i1)*dx(i3)

       dtensor(i3,jj,ii,imu,itype,3)=dtensor(i3,jj,ii,imu,itype,3)+poly(imu)*dx(i1)*dx(i2)
       dtensor(i3,1,ii,imu,itype,3)=dtensor(i3,1,ii,imu,itype,3)-poly(imu)*dx(i1)*dx(i2)

      enddo
      enddo
      enddo
      enddo
      endif

      if(rankmx.ge.4) then
! rank 4 tensor 
      do imu=0,mumx
      do i1=1,3
      do i2=1,3
      do i3=1,3
      do i4=1,3
      ii=(i4-1)*27+(i3-1)*9+(i2-1)*3+i1
      tensor(ii,imu,itype,4)=tensor(ii,imu,itype,4)+poly(imu)*dx(i1)*dx(i2)*dx(i3)*dx(i4)
      dtensor(:,jj,ii,imu,itype,4)=dtensor(:,jj,ii,imu,itype,4)+dpoly(imu)*dx(i1)*dx(i2)*dx(i3)*dx(i4)*dx(:)/d
      dtensor(:,1,ii,imu,itype,4)=dtensor(:,1,ii,imu,itype,4)-dpoly(imu)*dx(i1)*dx(i2)*dx(i3)*dx(i4)*dx(:)/d

       dtensor(i1,jj,ii,imu,itype,4)=dtensor(i1,jj,ii,imu,itype,4)+poly(imu)*dx(i2)*dx(i3)*dx(i4)
       dtensor(i1,1,ii,imu,itype,4)=dtensor(i1,1,ii,imu,itype,4)-poly(imu)*dx(i2)*dx(i3)*dx(i4)

       dtensor(i2,jj,ii,imu,itype,4)=dtensor(i2,jj,ii,imu,itype,4)+poly(imu)*dx(i1)*dx(i3)*dx(i4)
       dtensor(i2,1,ii,imu,itype,4)=dtensor(i2,1,ii,imu,itype,4)-poly(imu)*dx(i1)*dx(i3)*dx(i4)

       dtensor(i3,jj,ii,imu,itype,4)=dtensor(i3,jj,ii,imu,itype,4)+poly(imu)*dx(i1)*dx(i2)*dx(i4)
       dtensor(i3,1,ii,imu,itype,4)=dtensor(i3,1,ii,imu,itype,4)-poly(imu)*dx(i1)*dx(i2)*dx(i4)

       dtensor(i4,jj,ii,imu,itype,4)=dtensor(i4,jj,ii,imu,itype,4)+poly(imu)*dx(i1)*dx(i2)*dx(i3)
       dtensor(i4,1,ii,imu,itype,4)=dtensor(i4,1,ii,imu,itype,4)-poly(imu)*dx(i1)*dx(i2)*dx(i3)

      enddo
      enddo
      enddo
      enddo
      enddo
      endif

1001  continue
1000  continue


!ccccccccccccccccccccccccccccccccccccccccccccccccccccc
      call calc_loop_ind(ind,num_contract)


      do 2000 kk=1,numCC_type(itype0)   ! kk is the ind of the TMP(feature)
!1 Now, we will do contraction, change tensor to scalar
! itype0 is the original atom type
! itype is the atom (neighbore) type for this feature
! The itype is within numCC_type

      do i=1,numT_all(kk,itype0)
      do j=1,rank_all(i,kk,itype0)
      indi_tmp(j,i)=indi_all(j,i,kk,itype0)
      indj_tmp(j,i)=indj_all(j,i,kk,itype0)
      enddo
      enddo

      num_tot=0
      indc=0
      do i=1,numT_all(kk,itype0)
      num_tot=num_tot+rank_all(i,kk,itype0)

      do j=1,rank_all(i,kk,itype0)
      if(indi_tmp(j,i).ne.-1) then   ! not contracted yet, add one contraction do loop (indc)
      indc=indc+1                 ! indc is the indc_th contraction, or say the do loop: i=1,3
      iloop(j,i)=indc             ! iloop(j,i) is the j_rank,i_tensor, belong to indc_th contraction line
      iloop(indj_tmp(j,i),indi_tmp(j,i))=indc   ! contraction come in pair
      indi_tmp(indj_tmp(j,i),indi_tmp(j,i))=-1   ! (j,i):  the i_th tensor, j_th rank of the i_th tensor
      indi_tmp(j,i)=-1
      endif
      enddo
      enddo
! iloop(j,i)=indc, the indc is the accumulated number of contraction index. E.g., T1(i1,i2)*T2(i1,i2), then total indc is 2. 
! The indc=1 is for the contraction at the first index, indc=2 is the contraction for the second index
! iloop(j,i)=indc, means for tensor i_th, and its index j_th, it will be contracted by the indc_th contraction. 


      if(indc*2.ne.num_tot) then
      write(6,*) "somehow the contract do number is not correct, stop",indc*2,num_tot
      write(6,*) "numT_all", numT_all(kk,itype0),kk,itype0
      write(6,*) "rank_all", rank_all(1:numT_all(kk,itype0),kk,itype0)
      do i=1,numT_all(kk,itype0)
      write(6,*) "--indi_all",indi_all(1:rank_all(i,kk,itype0),i,kk,itype0)
      write(6,*) "**indj_all",indj_all(1:rank_all(i,kk,itype0),i,kk,itype0)
      enddo
      stop
      endif
      if(indc.gt.8) then
      write(6,*) "the number of contraction ind is too large,stop", indc
      stop
      endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

!-------------------------
!-------------------------
! This is the core contraction part
      sum=0.d0
      dprod_sum=0.d0
      do iii=1,num_contract(indc)  ! the contraction do loop: do i1=1,3,i2=1,3 etc,actual sum, when total contraction line number is indc
                                   ! indc is the total number of contraction line
                                   ! so 2*indc equal the sum of all tensor rank
                                   ! num_contract is the actual sum of all index, 
                                   ! num_contract(indc)=3**indc
                                   ! ind(iloop(j,i),iii,indc) tell us for i_th tensor, j_th rank, 
                                   ! for the iii_th's sum, what is the actual index in the tensor

      dprod=0.d0
      prod=1.d0
      do i=1,numT_all(kk,itype0)  ! number of tensor for this contraction(term), kk the index of term (feature)
      imu=jmu_b_all(i,kk,itype0)
      itype=itype_b_all(i,kk,itype0)
      rankT=rank_all(i,kk,itype0)
        ii=1
        do j=1,rankT  ! the rank can be zero, for M0
        ii=ii+(ind(iloop(j,i),iii,indc)-1)*3**(j-1)    ! ind=i1=1,2,3 in this case
! iloop(j,i) is the contract line indc'  for the j_th rank, i_th tensor, 
! indc here is the total contraction line
! ind(indc',iii,indc)=the 1,2,3, of the indc' contraction line, at iii_th contraction sum, when total contraction line number is indc
        enddo
! ii is the index position in the tensor(imu), for: [ind(iloop(1,i)),ind(iloop(2,i)),...,ind(iloop(rankT,i))]
      dprod(:,1:nneigh)=prod*dtensor(:,1:nneigh,ii,imu,itype,rankT)+dprod(:,1:nneigh)*tensor(ii,imu,itype,rankT)
      prod=prod*tensor(ii,imu,itype,rankT)    ! I am worry about the performance
      enddo
      sum=sum+prod
      dprod_sum=dprod_sum+dprod
      enddo
      feat_all(kk,iat)=sum
      dfeat_all(kk,iat,1:nneigh,1)=dprod_sum(1,1:nneigh)
      dfeat_all(kk,iat,1:nneigh,2)=dprod_sum(2,1:nneigh)
      dfeat_all(kk,iat,1:nneigh,3)=dprod_sum(3,1:nneigh)
! This is the core contraction part
!      write(6,"('feat_all ', i5,2x,i3,2x,E12.5,4x,i4,20(i2,1x))")  &
!        kk,iat,feat_all(kk,iat),numT_all(kk,itype0),jmu_b_all(1:numT_all(kk,itype0),kk,itype0), &
!        itype_b_all(1:numT_all(kk,itype0),kk,itype0)
!     write(6,"('feat_all ', i5,2x,i3,2x,E12.5,1x,3(E10.3,1x),4x,i4,2x,20(i2,1x))")  &
!        kk,iat,feat_all(kk,iat),dfeat_all(kk,iat,1:3,2),numT_all(kk,itype0),jmu_b_all(1:numT_all(kk,itype0),kk,itype0), &
!        rank_all(1:numT_all(kk,itype0),kk,itype0),itype_b_all(1:numT_all(kk,itype0),kk,itype0)
!-------------------------

2000  continue

      nfeat_atom(iat)=numCC_type(itype0)
      deallocate(tensor)
      deallocate(dtensor)

3000  continue

!cccccccccccccccccccccccccccccccccccccccccccccccc
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
      end subroutine find_feature_MTP



   
