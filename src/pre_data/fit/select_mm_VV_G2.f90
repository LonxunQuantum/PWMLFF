program linear_VV_G2
       implicit double precision (a-h,o-z)

       real*8,allocatable,dimension(:,:) :: feat_case0,feat_case
       real*8,allocatable,dimension(:,:) :: feat_tmp,feat_test
       real*8,allocatable,dimension(:) :: Ei_tmp,Ei_test
       real*8,allocatable,dimension(:,:) :: dd_all
       real*8,allocatable,dimension(:) :: Ei_case,Ei_ref,Ei_case0
       real*8,allocatable,dimension(:) :: dEi_case
       real*8,allocatable,dimension(:,:) :: Gfeat_case
       integer,allocatable,dimension(:) :: ind_ref
       real*8,allocatable,dimension(:) :: dist_ref
       real*8,allocatable,dimension(:,:) :: S


       real*8,allocatable,dimension(:) :: work,BB
       real*8,allocatable,dimension(:) :: E_fit,W
       integer,allocatable,dimension(:) :: ipiv
       integer,allocatable,dimension(:) :: index,index2
       integer lwork
       integer iatom_type(10)

       real*8,allocatable,dimension(:,:) :: featCR,featC
       real*8,allocatable,dimension(:,:,:) :: feat_ext2,feat_ext1, feat_ext3
       real*8,allocatable,dimension(:,:,:) :: feat_ext2T,feat_ext1T, feat_ext3T
       real*8,allocatable,dimension(:,:) :: feat_new
       real*8,allocatable,dimension(:) :: dE_term,dE_term0
       real*8,allocatable,dimension(:) :: dE_sum
       integer,allocatable,dimension(:,:) :: idd
       integer,allocatable,dimension(:) :: iselect
       integer,allocatable,dimension(:) :: iflag_selected,ind_kkk_kkk0
       integer,allocatable,dimension(:) :: ind_select
      
       real*8 xp(5,100),xp1(10,100),xp3(10,100)
! liuliping for relative path
       integer tmp_i
       character(len=200) fitModelDir
       character(len=:), allocatable :: fread_dfeat

       ! this file should be create by prepare.py
       open(1314, file="input/info_dir")
       rewind(1314)
       read(1314,"(A200)") fitModelDir
       close(1314)
       tmp_i = len(trim(adjustl(fitModelDir)))
       allocate(character(len=tmp_i) :: fread_dfeat)
       fread_dfeat = trim(adjustl(fitModelDir))
       write(*,*) "liuliping, fread_dfeat: ", fread_dfeat
! liuliping, end, all .r .x file should be invoke out of fread_dfeat
       

        
        write(6,*) "input itype,iseed(negative)"
        read(5,*) itype,iseed
        write(6,*) "include feat**3, (0:no; 1:yes)"
        read(5,*) include3
        write(6,*) "input mm (num of feat,1000-2000), and nloop(20)"
        read(5,*) mm2,nloop


       open(10,file=fread_dfeat//"feat_new_stored."//char(48+itype), form="unformatted")
       rewind(10)
       read(10) num_case,nfeat0
       write(6,*) "num_case,nfeat", num_case,nfeat0
       allocate(Ei_case0(num_case))
       allocate(feat_case0(nfeat0,num_case))
       do ii=1,num_case
       read(10) jj,Ei_case0(ii),feat_case0(:,ii)
       enddo
       close(10)
!cccccccccccccccccccccccccccccccccccccccccccc
!cccccc remove the zero features
        allocate(feat_tmp(nfeat0,num_case))
        allocate(index(nfeat0))
         jj1=0
         do jj=1,nfeat0
         sum=0.d0
         do ii=1,num_case
          sum=sum+abs(feat_case0(jj,ii))  ! bad access approach
          enddo
         if(sum.gt.1.D-8) then    ! remove the zero feat
         jj1=jj1+1
         index(jj1)=jj
         feat_tmp(jj1,:)=feat_case0(jj,:)
         endif
         enddo
         write(6,*) "original nfeat=",nfeat0
         nfeat=jj1
         write(6,*) "reduced nfeat=",nfeat
         deallocate(feat_case0)

         allocate(feat_case0(nfeat,num_case))
          do ii=1,num_case
          do jj=1,nfeat
          feat_case0(jj,ii)=feat_tmp(jj,ii)
          enddo
          enddo
         deallocate(feat_tmp)
!cccccccccccccccccccccccccccccccccccccccccccc

       allocate(iselect(num_case))

!ccccccccccccccc  two different ways to select the test cases
!ccc roughly, 1/3 of the total data
       iselect=0
       do jj=1,5
       do ii=1,num_case/5/4
       iselect((jj-1)*num_case/5+ii)=1
       enddo
       enddo

       do i=1,num_case
       xx=ran1(iseed)
       if(xx.lt.0.2) then
       iselect(i)=1
       endif
       enddo
!ccccccccccccccc  two different ways to select the test cases

       ntest=0
       do i=1,num_case
       ntest=ntest+iselect(i)
       enddo
       write(6,*) "ntrain,ntest=", num_case-ntest,ntest

       ncase=num_case-ntest
       allocate(feat_case(nfeat,ncase))
       allocate(Ei_case(ncase))
       allocate(feat_test(nfeat,ntest))
       allocate(Ei_test(ntest))
      
       itest=0
       itrain=0
       do i=1,num_case
       if(iselect(i).eq.1) then
       itest=itest+1
       feat_test(:,itest)=feat_case0(:,i)
       Ei_test(itest)=Ei_case0(i)
       else
       itrain=itrain+1
       feat_case(:,itrain)=feat_case0(:,i)
       Ei_case(itrain)=Ei_case0(i)
       endif
       enddo
 
       deallocate(feat_case0)
       deallocate(Ei_case0)


       ! ncase: training
       ! ntest: testing


        num_ref=nfeat

!cccccccccccccccccccccccccccccccccccccccccccc
       allocate(S(num_ref,num_ref))
       allocate(BB(num_ref))
       allocate(ipiv(num_ref))
       allocate(E_fit(ncase))

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       S=0.d0
       call dgemm('N','T',num_ref,num_ref,ncase,1.d0, feat_case,nfeat, feat_case,nfeat,0.d0,S,num_ref)         ! feat_case is (nfeat0,ncase)


       do j=1,num_ref
       S(j,j)=S(j,j)+0.0001
       enddo

       do j=1,num_ref
       sum=0.d0
       do i=1,ncase
       sum=sum+Ei_case(i)*feat_case(j,i)
       enddo
       BB(j)=sum
       enddo
!cccccccccccc

       call dgesv(num_ref,1,S,num_ref,ipiv,BB,num_ref,info)  
!cccccccccccccccccccccccccccccccccccccccccccccccccc

       allocate(dEi_case(ncase))


       diff1=0.d0
       diff2=0.d0
       diff4=0.d0
       do i=1,ncase
       sum=0.d0
       do j=1,num_ref
       sum=sum+BB(j)*feat_case(j,i)
       enddo
       diff1=diff1+abs(Ei_case(i)-sum)
       diff2=diff2+(Ei_case(i)-sum)**2
       diff4=diff4+(Ei_case(i)-sum)**4
       dEi_case(i)=Ei_case(i)-sum
       enddo
       diff1a=(diff1/ncase)
       diff2a=dsqrt(diff2/ncase)
       diff4a=dsqrt(dsqrt(diff4/ncase))
!       write(6,*) "iloop",iloop1,iloop2
       write(6,"('diff1,2,4(Lin)     =',3(E14.7,1x))") diff1a,diff2a,diff4a


       diff1=0.d0
       diff2=0.d0
       diff4=0.d0
       do i=1,ntest
       sum=0.d0
       do j=1,num_ref
       sum=sum+BB(j)*feat_test(j,i)
       enddo
       diff1=diff1+abs(Ei_test(i)-sum)
       diff2=diff2+(Ei_test(i)-sum)**2
       diff4=diff4+(Ei_test(i)-sum)**4
       enddo
       diff1t=(diff1/ntest)
       diff2t=dsqrt(diff2/ntest)
       diff4t=dsqrt(dsqrt(diff4/ntest))
!       write(6,*) "iloop",iloop1,iloop2
       write(6,"('diff1,2,4(Lin,test)=',3(E14.7,1x))") diff1t,diff2t,diff4t




       deallocate(S)
       deallocate(BB)
       deallocate(ipiv)
       deallocate(E_fit)
       deallocate(index)
!cccccccccccccccccccccccccccccccccccccccccccccccccc


!ccccccccccccccccccccccccccccccccccccccccccccccccc

       ndim=4
       ndim1=20
       ndim3=3
       allocate(feat_ext2(ncase,nfeat,ndim))
       allocate(feat_ext1(ncase,nfeat,ndim1))
       allocate(feat_ext3(ncase,nfeat,1))

       allocate(feat_ext2T(ntest,nfeat,ndim))
       allocate(feat_ext1T(ntest,nfeat,ndim1))
       allocate(feat_ext3T(ntest,nfeat,1))

       nstore=ndim**2*nfeat*(nfeat+1)/2+nfeat*ndim1
       if(include3.eq.1) then
       nstore=nstore+nfeat**2*(nfeat+1)/2
       endif

       allocate(dE_term(nstore))
       allocate(dE_term0(nstore))
       allocate(idd(0:4,nstore)) ! the type and index

       idd=0
!       do 3000 iloop1=0,6
!       do 3000 iloop2=0,6

!       iloop1=5  ! optimized value
!       iloop2=4  ! optimized value

!       xpp=0.02*2.3**iloop1
!       xdd=0.1*2.3**iloop2


       xp(1,1)=-3.9
       xp(2,1)=2.6
       xp(1,2)=-1.3
       xp(2,2)=2.6
       xp(1,3)=1.3
       xp(2,3)=2.6
       xp(1,4)=3.9
       xp(2,4)=2.6

       do id1=1,ndim1
       xp1(1,id1)=-(id1-ndim1/2)*3.0/ndim1
       xp1(2,id1)=3.d0/ndim1
       enddo

!cccccccccccccccccccccccccccccccc

       do id=1,ndim1
       do j=1,nfeat
       do i=1,ncase
       feat_ext1(i,j,id)=exp(-((feat_case(j,i)-xp1(1,id))/xp1(2,id))**2)
       enddo
       enddo
       enddo


       do id=1,ndim
       do j=1,nfeat
       do i=1,ncase
       feat_ext2(i,j,id)=exp(-((feat_case(j,i)-xp(1,id))/xp(2,id))**2)
       enddo
       enddo
       enddo

       do j=1,nfeat
       do i=1,ncase
       feat_ext3(i,j,1)=feat_case(j,i)
       enddo
       enddo
!cccccccccccccccccccccccccccccccc
       do id=1,ndim1
       do j=1,nfeat
       do i=1,ntest
       feat_ext1T(i,j,id)= exp(-((feat_test(j,i)-xp1(1,id))/xp1(2,id))**2)
       enddo
       enddo
       enddo

       do id=1,ndim
       do j=1,nfeat
       do i=1,ntest
       feat_ext2T(i,j,id)=exp(-((feat_test(j,i)-xp(1,id))/xp(2,id))**2)
       enddo
       enddo
       enddo

       do j=1,nfeat
       do i=1,ntest
       feat_ext3T(i,j,1)=feat_test(j,i)
       enddo
       enddo


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!       mm2=2000
!       nloop=20
       mm3=mm2/nloop  ! the number of new feat to be selected
       mm2=mm3*nloop
       mm4=mm3*10     ! the number of minimum value of feature to consider, and from which to select mm3 
        

!cccccc  ind_kkk_kkk0(kkk)=kkk0
       kkk0=0
       do j1=1,nfeat
       do id1=1,ndim1
       kkk0=kkk0+1
       enddo
       enddo

       do j1=1,nfeat
       ndimt1=ndim
       if(j1.eq.nfeat) ndimt1=1
       do id1=1,ndimt1
       do j2=1,j1
       ndimt2=ndim
       if(j2.eq.nfeat) ndimt2=1
       if(j2.eq.j1) ndimt2=id1
       do id2=1,ndimt2
       kkk0=kkk0+1
       enddo
       enddo
       enddo
       enddo

       if(include3.eq.1) then
       do j1=1,nfeat
       do j2=1,j1
       do j3=1,j2
       kkk0=kkk0+1
       enddo
       enddo
       enddo
       endif
       nkkk0=kkk0

       write(6,*) "The total num of possible feat",nkkk0


       allocate(iflag_selected(nkkk0))
       allocate(ind_kkk_kkk0(nkkk0))
       allocate(ind_select(mm2))
       

       ind_kkk_kkk0=0
       iflag_selected=0

       open(33,file=fread_dfeat//"VV_select_conv.plot")
       rewind(33)
       loop=0
       write(33,"(2(i6,1x),2(E14.7,1x))") loop,nfeat,diff2a,diff2t 

       do 5000  loop=1,nloop
!cccccccccccccccccccccccccccccccc
!ccc  At this state, iflag_selected(kkk0).eq.1, selected
       kkk0=0
       kkk=0

       do j1=1,nfeat
       do id1=1,ndim1
       kkk0=kkk0+1
       idd(0,kkk0)=1
       idd(1,kkk0)=j1
       idd(2,kkk0)=id1
       if(iflag_selected(kkk0).eq.0) then   ! not selected yet
       kkk=kkk+1
       ind_kkk_kkk0(kkk)=kkk0
       sum1=0.d0
       sum2=0.d0
       do i=1,ncase
       tmp=feat_ext1(i,j1,id1)
       sum1=sum1+dEi_case(i)*tmp
       sum2=sum2+tmp**2
       enddo
       dE_term(kkk)=abs(sum1**2/sum2/ncase)
       endif
       enddo
       enddo
       
!cccccccccccccccccccccccccccccccc
     
       do 2000 j1=1,nfeat
       ndimt1=ndim
       if(j1.eq.nfeat) ndimt1=1
       do 2000 id1=1,ndimt1

!        write(6,*) "loop", j1,id1

       do 2000 j2=1,j1
       ndimt2=ndim
       if(j2.eq.nfeat) ndimt2=1
       if(j2.eq.j1) ndimt2=id1
       do 2000 id2=1,ndimt2
       kkk0=kkk0+1
       idd(0,kkk0)=2
       idd(1,kkk0)=j1
       idd(2,kkk0)=id1
       idd(3,kkk0)=j2
       idd(4,kkk0)=id2
       if(iflag_selected(kkk0).eq.0) then  ! not selected yet
       kkk=kkk+1
       ind_kkk_kkk0(kkk)=kkk0

       sum1=0.d0
       sum2=0.d0
       do i=1,ncase
       tmp=feat_ext2(i,j1,id1)*feat_ext2(i,j2,id2)
       sum1=sum1+dEi_case(i)*tmp
       sum2=sum2+tmp**2
       enddo
       dE_term(kkk)=abs(sum1**2/sum2/ncase)
       endif

2000   continue
!cccccccccccccccccccccccccccccccc
       if(include3.eq.1) then
       do 2500 j1=1,nfeat
       do 2500 j2=1,j1
       do 2500 j3=1,j2
       kkk0=kkk0+1
       idd(0,kkk0)=3
       idd(1,kkk0)=j1
       idd(2,kkk0)=j2
       idd(3,kkk0)=j3
       if(iflag_selected(kkk0).eq.0) then
       kkk=kkk+1
       ind_kkk_kkk0(kkk)=kkk0
       sum1=0.d0
       sum2=0.d0
       do i=1,ncase
       tmp=feat_ext3(i,j1,1)*feat_ext3(i,j2,1)*feat_ext3(i,j3,1)
       sum1=sum1+dEi_case(i)*tmp
       sum2=sum2+tmp**2
       enddo
       dE_term(kkk)=abs(sum1**2/sum2/ncase)
       endif
2500   continue
       endif
   
       nkkk=kkk
       nkkk0=kkk0
!       write(6,*) "nkkk,nkkk0=",nkkk,nkkk0
!! nkkk0 is the original all possible features
!! nkkk is the all possible features except the selected ones (nkkk-mm3*(loop-1))
!! dE_term(kkk) is the kkk feature dot-product with dEi_case(i) (which is the residual one)

       dE_term0=dE_term
!cccccccccccccccccccccccccccccccc

       dE_term=dE_term0

       allocate(index(mm4))
       allocate(index2(mm3))
       allocate(feat_new(mm4,ncase))
       allocate(dE_sum(mm4))

! For the following index:
! kkk is within [1,nkkk]   large set of feature (excluding the selected one) tens of thousands of them
! iii is within [1,mm4]    the preselect subset within kkk (mm4=10*mm3)
! jjj is within [1,mm3]    the final selected subset from mm4, to be added to the selected set
!ccccccccccccccccccccccccccccccc
!cccc  find the mm selected new feature index(iii)
       do iii=1,mm4    ! preselect mm4 features, index index(iii)
       dE_max=0.d0
       do kkk=1,nkkk
       if(dE_term(kkk).gt.dE_max) then
       dE_max=dE_term(kkk)
       kkkm=kkk
       endif
       enddo
       index(iii)=kkkm
       dE_term(kkkm)=0.d0
!       write(6,*) "iii,dE_max",iii,dE_max
       enddo
!cccccccccccccccccccccccccccccccccccc
!  Now, we have selected mm4 feature, index(iii)
! We will further select mm3 features, from this mm4 features
!cccccccccccccccccccccccccccccccc
      

!  generate these mm4 features, and their dot-product
!   also the dE_sum
       do iii=1,mm4
       kkk=index(iii)
       kkk0=ind_kkk_kkk0(kkk)

       if(idd(0,kkk0).eq.1) then
       j1=idd(1,kkk0)
       id1=idd(2,kkk0)
       do i=1,ncase
       feat_new(iii,i)=feat_ext1(i,j1,id1)
       enddo

       elseif(idd(0,kkk0).eq.2) then
       j1= idd(1,kkk0)
       id1=idd(2,kkk0)
       j2=idd(3,kkk0)
       id2=idd(4,kkk0)
       do i=1,ncase
       feat_new(iii,i)=feat_ext2(i,j1,id1)*feat_ext2(i,j2,id2)
       enddo

       elseif(idd(0,kkk0).eq.3) then
       j1=idd(1,kkk0)
       j2=idd(2,kkk0)
       j3=idd(3,kkk0)
       do i=1,ncase
       feat_new(iii,i)=feat_ext3(i,j1,1)*feat_ext3(i,j2,1)* feat_ext3(i,j3,1)
       enddo
       endif

       enddo


! cannot normalize the feature, that is wrong !, keep the feature untouched
       do iii=1,mm4
       sum=0.d0
       do i=1,ncase
       sum=sum+dEi_case(i)*feat_new(iii,i)
       enddo
       dE_sum(iii)=sum
       enddo

       allocate(S(mm4,mm4))

       S=0.d0
       call dgemm('N','T',mm4,mm4,ncase,1.d0, feat_new,mm4,feat_new,mm4,0.d0,S,mm4)

!cccccccccccccccccccccccccccccccccccccccccccccccc
!      Now, we will select mm3 feature out of mm4 candidates  
!      We will use a simplify scheme (one can use a more sophisticated scheme, by diagonalizing
!      the already mm3 selected state, then do the test. But we will use a simplify one. 
!      Add the effect of states, one by one, instead of re-diagonalizing them. 
!cccccccccccccccccccccccccccccccccccccccccccccccc
       do jjj=1,mm3

        dE_max=0.d0
        do iii=1,mm4
        if(dE_sum(iii)**2/S(iii,iii).gt.dE_max) then
        dE_max=dE_sum(iii)**2/S(iii,iii)
        iii_max=iii
        endif
        enddo

        index2(jjj)=iii_max   ! this iii is selected as this jjj
        bc=dE_sum(iii_max)/S(iii_max,iii_max)
        do iii=1,mm4
        dE_sum(iii)=dE_sum(iii)-S(iii,iii_max)*bc    ! the change of dE_sum due to the substract of iii_max
!      This is only an approximation, avoid to choose some very linearly correlated features
        enddo
        do iii=1,jjj   ! the already selected ones
        dE_sum(index2(iii))=0.d0
        enddo
       enddo  ! jjj
!cccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccc
!  Now, the selected index in kkk
     
       do jjj=1,mm3
       iii=index2(jjj)    ! from jjj [1,mm3] to iii [1,mm4]
       kkk=index(iii)     ! from iii [1,mm4] to kkk [1,nkkk]
       kkk0=ind_kkk_kkk0(kkk) ! from kkk [1,nkkk] to kkk0 [1,nkkk0]
       ind_select((loop-1)*mm3+jjj)=kkk0
       iflag_selected(kkk0)=1
       enddo

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Now, the totally selected feature equals to loop*mm3
    
       deallocate(S)
       deallocate(index)
       deallocate(index2)
       deallocate(feat_new)
       deallocate(dE_sum)

       num_sel=loop*mm3
       num_ref=num_sel+nfeat
       allocate(feat_new(num_ref,ncase))
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       do i=1,ncase
       do j=1,nfeat
       feat_new(j,i)=feat_case(j,i)
       enddo
       enddo

       do iii=1,num_sel
       kkk0=ind_select(iii)

       if(idd(0,kkk0).eq.1) then
       j1=idd(1,kkk0)
       id1=idd(2,kkk0)
       do i=1,ncase
       feat_new(iii+nfeat,i)=feat_ext1(i,j1,id1)
       enddo

       elseif(idd(0,kkk0).eq.2) then
       j1= idd(1,kkk0)
       id1=idd(2,kkk0)
       j2=idd(3,kkk0)
       id2=idd(4,kkk0)
       do i=1,ncase
       feat_new(iii+nfeat,i)=feat_ext2(i,j1,id1)*feat_ext2(i,j2,id2)
       enddo

       elseif(idd(0,kkk0).eq.3) then
       j1=idd(1,kkk0)
       j2=idd(2,kkk0)
       j3=idd(3,kkk0)
       do i=1,ncase
       feat_new(iii+nfeat,i)=feat_ext3(i,j1,1)*feat_ext3(i,j2,1)* feat_ext3(i,j3,1)
       enddo
       endif

       enddo

!       write(6,*) "num_ref=",num_ref


       allocate(S(num_ref,num_ref))   ! num_ref=nfeat+num_sel
       allocate(BB(num_ref))
       allocate(ipiv(num_ref))

       S=0.d0
       call dgemm('N','T',num_ref,num_ref,ncase,1.d0, feat_new,num_ref,feat_new,num_ref,0.d0,S,num_ref)

       do j=1,num_ref
       S(j,j)=S(j,j)+0.0001
       enddo

       do j=1,num_ref
       sum=0.d0
       do i=1,ncase
       sum=sum+Ei_case(i)*feat_new(j,i)
       enddo
       BB(j)=sum
       enddo

!cccccccccccc
       call dgesv(num_ref,1,S,num_ref,ipiv,BB,num_ref,info)  
!cccccccccccccccccccccccccccccccccccccccccccccccccc
 
       allocate(E_fit(ncase))

       
       diff1=0.d0
       diff2=0.d0
       diff4=0.d0
       do i=1,ncase
       sum=0.d0
       do j=1,num_ref
       sum=sum+BB(j)*feat_new(j,i)
       enddo
       E_fit(i)=sum
       dEi_case(i)=Ei_case(i)-sum        ! prepare for next loop
       diff1=diff1+abs(E_fit(i)-Ei_case(i))
       diff2=diff2+(E_fit(i)-Ei_case(i))**2
       diff4=diff4+(E_fit(i)-Ei_case(i))**4
       enddo
       diff1a=(diff1/ncase)
       diff2a=dsqrt(diff2/ncase)
       diff4a=dsqrt(dsqrt(diff4/ncase))
       write(6,*) "loop,m_feat=",loop,num_ref
       write(6,"('diff1,2,4=          ',3(E14.7,1x))") diff1a,diff2a,diff4a

       
        open(12,file=fread_dfeat//"OUT.VV_index."//char(48+itype))
        rewind(12)
        write(12,*) num_sel
        do iii=1,num_sel
        kkk0=ind_select(iii)
        write(12,"(5(i6,1x))") idd(0:4,kkk0)
        enddo
        close(12)

       open(10,file=fread_dfeat//"E_fit.VV."//char(48+itype))
       rewind(10) 
       do i=1,ncase
       write(10,"(2(E14.7,1x))") Ei_case(i),E_fit(i)
       enddo
       close(10)

       deallocate(E_fit)

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       deallocate(S)
       deallocate(ipiv)
       deallocate(feat_new)

!       allocate(feat_ext2(ntest,nfeat,ndim))
!       allocate(dE_term((nfeat*ndim)**2))
       allocate(feat_new(num_ref,ntest))
       allocate(E_fit(ntest))

!cccccccccccccccccccccccccccccccc
       do i=1,ntest
       do iii=1,nfeat
       feat_new(iii,i)=feat_test(iii,i)
       enddo
       enddo


       do iii=1,num_sel
       kkk0=ind_select(iii)

       if(idd(0,kkk0).eq.1) then
       j1=idd(1,kkk0)
       id1=idd(2,kkk0)
       do i=1,ntest
       feat_new(iii+nfeat,i)=feat_ext1T(i,j1,id1)
       enddo
       elseif(idd(0,kkk0).eq.2) then
       j1= idd(1,kkk0)
       id1=idd(2,kkk0)
       j2=idd(3,kkk0)
       id2=idd(4,kkk0)
       do i=1,ntest
       feat_new(iii+nfeat,i)=feat_ext2T(i,j1,id1)*feat_ext2T(i,j2,id2)
       enddo
       elseif(idd(0,kkk0).eq.3) then
       j1=idd(1,kkk0)
       j2=idd(2,kkk0)
       j3=idd(3,kkk0)
       do i=1,ntest
       feat_new(iii+nfeat,i)=feat_ext3T(i,j1,1)*feat_ext3T(i,j2,1)* feat_ext3T(i,j3,1)
       enddo
       endif

       enddo
!ccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccc
       diff1=0.d0
       diff2=0.d0
       diff4=0.d0
       do i=1,ntest
       sum=0.d0
       do j=1,num_ref
       sum=sum+BB(j)*feat_new(j,i)
       enddo
       E_fit(i)=sum
       diff1=diff1+abs(E_fit(i)-Ei_test(i))
       diff2=diff2+(E_fit(i)-Ei_test(i))**2
       diff4=diff4+(E_fit(i)-Ei_test(i))**4
       enddo
       diff1t=(diff1/ntest)
       diff2t=dsqrt(diff2/ntest)
       diff4t=dsqrt(dsqrt(diff4/ntest))
       write(6,"('TEST diff1,2,4=     ',3(E14.7,1x))") diff1t,diff2t,diff4t
       write(33,"(2(i6,1x),2(E14.7,1x))") loop,num_ref,diff2a,diff2t 

       open(14,file=fread_dfeat//"E_fit.VV.test."//char(48+itype))
       rewind(14)
       do i=1,ntest
       write(14,"(2(E14.7,1x))") Ei_test(i),E_fit(i)
       enddo
       close(14)
!ccccccccccccccccccccccccccccccccccccccccccccccccccc
       deallocate(feat_new)
       deallocate(E_fit)
       deallocate(BB)

5000   continue
       close(33)

    deallocate(fread_dfeat)
       stop
       end
