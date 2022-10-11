program linear_VV
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
       integer,allocatable,dimension(:) :: index
       integer lwork
       integer iatom_type(10)

       real*8,allocatable,dimension(:,:) :: featCR,featC
       real*8,allocatable,dimension(:,:,:) :: feat_ext2,feat_ext1, feat_ext3
       real*8,allocatable,dimension(:,:,:) :: feat_ext2T,feat_ext1T, feat_ext3T
       real*8,allocatable,dimension(:,:) :: feat_new
       real*8,allocatable,dimension(:) :: dE_term,dE_list,dE_term0
       integer,allocatable,dimension(:,:) :: idd
       integer,allocatable,dimension(:) :: iselect
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
        write(6,*) "iscan_MM, or not, (0: no; 1:scan)" 
        read(5,*) iscan_mm
        if(iscan_mm.eq.0) then
        write(6,*) "input mm (from previous test)"
        read(5,*) mm
        endif


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
!ccccccccccccccccccccccccccccccccccccccccccccc
!ccccccc remove the zero features
        allocate(feat_tmp(nfeat0,num_case))
        allocate(index(nfeat0))
         jj1=0
         do jj=1,nfeat0
         sum=0.d0
         do ii=1,num_case
          sum=sum+abs(feat_case0(jj,ii))  ! bad access approach
          enddo
         if(sum.gt.1.D-8) then
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
!ccccccccccccccccccccccccccccccccccccccccccccc

       allocate(iselect(num_case))

!cccccccccccccccc  two different ways to select the test cases
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
!cccccccccccccccc  two different ways to select the test cases

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


        num_case=ncase   ! training


        num_ref=nfeat

!ccccccccccccccccccccccccccccccccccccccccccccc
       allocate(S(num_ref,num_ref))
       allocate(BB(num_ref))
       allocate(ipiv(num_ref))
       allocate(E_fit(num_case))

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       S=0.d0
       call dgemm('N','T',num_ref,num_ref,num_case,1.d0, feat_case,nfeat,feat_case,nfeat,0.d0,S,num_ref)


       do j=1,num_ref
       S(j,j)=S(j,j)+0.001
       enddo

       do j=1,num_ref
       sum=0.d0
       do i=1,num_case
       sum=sum+Ei_case(i)*feat_case(j,i)
       enddo
       BB(j)=sum
       enddo
!cccccccccccc

       call dgesv(num_ref,1,S,num_ref,ipiv,BB,num_ref,info)  
!cccccccccccccccccccccccccccccccccccccccccccccccccc

       allocate(dEi_case(num_case))


       diff2=0.d0
       do i=1,num_case
       sum=0.d0
       do j=1,num_ref
       sum=sum+BB(j)*feat_case(j,i)
       enddo
       diff2=diff2+(Ei_case(i)-sum)**2
       dEi_case(i)=Ei_case(i)-sum
       enddo
       diff2=sqrt(diff2/num_case)
       write(6,*) "diff2, first linear=",diff2


       diff2=0.d0
       do i=1,ntest
       sum=0.d0
       do j=1,num_ref
       sum=sum+BB(j)*feat_test(j,i)
       enddo
       diff2=diff2+(Ei_test(i)-sum)**2
       enddo
       diff2=sqrt(diff2/ntest)
       write(6,*) "TEST, first linear=",diff2




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

!ccccccccccccccccccccccccccccccccc

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
!ccccccccccccccccccccccccccccccccc
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

!ccccccccccccccccccccccccccccccccc
       kkk=0.d0

       do j1=1,nfeat
       do id1=1,ndim1
       kkk=kkk+1
       idd(0,kkk)=1
       idd(1,kkk)=j1
       idd(2,kkk)=id1
       sum1=0.d0
       sum2=0.d0
       do i=1,ncase
       tmp=feat_ext1(i,j1,id1)
       sum1=sum1+dEi_case(i)*tmp
       sum2=sum2+tmp**2
       enddo
       dE_term(kkk)=abs(sum1**2/sum2/ncase)
       enddo
       enddo
       
!ccccccccccccccccccccccccccccccccc
     
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

       kkk=kkk+1
       idd(0,kkk)=2
       idd(1,kkk)=j1
       idd(2,kkk)=id1
       idd(3,kkk)=j2
       idd(4,kkk)=id2

       sum1=0.d0
       sum2=0.d0
       do i=1,ncase
       tmp=feat_ext2(i,j1,id1)*feat_ext2(i,j2,id2)
       sum1=sum1+dEi_case(i)*tmp
       sum2=sum2+tmp**2
       enddo
       dE_term(kkk)=abs(sum1**2/sum2/ncase)

2000   continue
!ccccccccccccccccccccccccccccccccc
       if(include3.eq.1) then
       do 2500 j1=1,nfeat
       do 2500 j2=1,j1
       do 2500 j3=1,j2
       kkk=kkk+1
       idd(0,kkk)=3
       idd(1,kkk)=j1
       idd(2,kkk)=j2
       idd(3,kkk)=j3
       sum1=0.d0
       sum2=0.d0
       do i=1,ncase
       tmp=feat_ext3(i,j1,1)*feat_ext3(i,j2,1)*feat_ext3(i,j3,1)
       sum1=sum1+dEi_case(i)*tmp
       sum2=sum2+tmp**2
       enddo
       dE_term(kkk)=abs(sum1**2/sum2/ncase)
2500   continue
       endif
   
       nkkk=kkk
       write(6,*) "nkkk=",nkkk

       dE_term0=dE_term
!ccccccccccccccccccccccccccccccccc
       mm1=10
       mm2=8000
       alpha=(mm2*1.d0/mm1)**(1.d0/(30-1))

       nloop=30
       if(iscan_mm.eq.0) nloop=1

       do 5000  loop=1,nloop

       if(iscan_mm.ne.0) then
       mm=mm1*alpha**(loop-1)
       write(6,*) "loop,MM=",loop,mm
       endif
       

       dE_term=dE_term0

       num_ref=mm+nfeat
       allocate(dE_list(mm))
       allocate(index(mm))
       allocate(feat_new(num_ref,ncase))

!ccccccccccccccccccccccccccccccc
!cccc  find the mm selected new feature index(iii)
       do iii=1,mm
       dE_max=0.d0
       do kkk=1,nkkk
       if(dE_term(kkk).gt.dE_max) then
       dE_max=dE_term(kkk)
       kkkm=kkk
       endif
       enddo
       dE_list(iii)=dE_max
       index(iii)=kkkm
       dE_term(kkkm)=0.d0
!       write(6,*) "iii,dE_max",iii,dE_max
       enddo
!ccccccccccccccccccccccccccccccccc
      

       do i=1,ncase
       do iii=1,nfeat
       feat_new(iii,i)=feat_case(iii,i)
       enddo
       enddo

       do iii=1,mm
       kkk=index(iii)

       if(idd(0,kkk).eq.1) then
       j1=idd(1,kkk)
       id1=idd(2,kkk)
       do i=1,ncase
       feat_new(iii+nfeat,i)=feat_ext1(i,j1,id1)
       enddo
       elseif(idd(0,kkk).eq.2) then
       j1= idd(1,kkk)
       id1=idd(2,kkk)
       j2=idd(3,kkk)
       id2=idd(4,kkk)
       do i=1,ncase
       feat_new(iii+nfeat,i)=feat_ext2(i,j1,id1)*feat_ext2(i,j2,id2)
       enddo
       elseif(idd(0,kkk).eq.3) then
       j1=idd(1,kkk)
       j2=idd(2,kkk)
       j3=idd(3,kkk)
       do i=1,ncase
       feat_new(iii+nfeat,i)=feat_ext3(i,j1,1)*feat_ext3(i,j2,1)* feat_ext3(i,j3,1)
       enddo
       endif

       enddo



       allocate(S(num_ref,num_ref))
       allocate(BB(num_ref))
       allocate(ipiv(num_ref))
       allocate(E_fit(num_case))

       S=0.d0
       call dgemm('N','T',num_ref,num_ref,num_case,1.d0, feat_new, num_ref,feat_new,num_ref,0.d0,S,num_ref)

       do j=1,num_ref
       S(j,j)=S(j,j)+0.00001
       enddo

       do j=1,num_ref
       sum=0.d0
       do i=1,num_case
!       sum=sum+dEi_case(i)*feat_new(j,i)
       sum=sum+Ei_case(i)*feat_new(j,i)
       enddo
       BB(j)=sum
       enddo

!cccccccccccc
       call dgesv(num_ref,1,S,num_ref,ipiv,BB,num_ref,info)  
!cccccccccccccccccccccccccccccccccccccccccccccccccc
 
       
       diff1=0.d0
       diff2=0.d0
       diff4=0.d0
       do i=1,num_case
       sum=0.d0
       do j=1,num_ref
       sum=sum+BB(j)*feat_new(j,i)
       enddo
       E_fit(i)=sum
       diff1=diff1+abs(E_fit(i)-Ei_case(i))
       diff2=diff2+(E_fit(i)-Ei_case(i))**2
       diff4=diff4+(E_fit(i)-Ei_case(i))**4
       enddo
       diff1=(diff1/num_case)
       diff2=dsqrt(diff2/num_case)
       diff4=dsqrt(dsqrt(diff4/num_case))
!       write(6,*) "iloop",iloop1,iloop2
       write(6,"('diff1,2,4=',3(E14.7,1x))") diff1,diff2,diff4

3000   continue

       
        if(iscan_mm.eq.0)  then
        open(12,file=fread_dfeat//"OUT.VV_index."//char(48+itype))
        rewind(12)
        write(12,*) mm
        do iii=1,mm
        kkk=index(iii)
        write(12,"(5(i6,1x))") idd(0:4,kkk)
        enddo
        close(12)

       open(10,file=fread_dfeat//"E_fit.VV."//char(48+itype))
       rewind(10) 
       do i=1,num_case
       write(10,"(2(E14.7,1x))") Ei_case(i),E_fit(i)
       enddo
       close(10)




        endif

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

!       deallocate(feat_ext2)
!       deallocate(dE_term)
       deallocate(feat_new)
       deallocate(E_fit)

!       allocate(feat_ext2(ntest,nfeat,ndim))
!       allocate(dE_term((nfeat*ndim)**2))
       allocate(feat_new(num_ref,ntest))
       allocate(E_fit(ntest))

!ccccccccccccccccccccccccccccccccc
       do i=1,ntest
       do iii=1,nfeat
       feat_new(iii,i)=feat_test(iii,i)
       enddo
       enddo


       do iii=1,mm
       kkk=index(iii)

       if(idd(0,kkk).eq.1) then
       j1=idd(1,kkk)
       id1=idd(2,kkk)
       do i=1,ntest
       feat_new(iii+nfeat,i)=feat_ext1T(i,j1,id1)
       enddo
       elseif(idd(0,kkk).eq.2) then
       j1= idd(1,kkk)
       id1=idd(2,kkk)
       j2=idd(3,kkk)
       id2=idd(4,kkk)
       do i=1,ntest
       feat_new(iii+nfeat,i)=feat_ext2T(i,j1,id1)*feat_ext2T(i,j2,id2)
       enddo
       elseif(idd(0,kkk).eq.3) then
       j1=idd(1,kkk)
       j2=idd(2,kkk)
       j3=idd(3,kkk)
       do i=1,ntest
       feat_new(iii+nfeat,i)=feat_ext3T(i,j1,1)*feat_ext3T(i,j2,1)* feat_ext3T(i,j3,1)
       enddo
       endif

       enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
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
       diff1=(diff1/ntest)
       diff2=dsqrt(diff2/ntest)
       diff4=dsqrt(dsqrt(diff4/ntest))
       write(6,"('TEST diff1,2,4=',3(E14.7,1x))") diff1,diff2,diff4

       if(iscan_mm.eq.0) then
       open(14,file=fread_dfeat//"E_fit.VV.test."//char(48+itype))
       rewind(14)
       do i=1,ntest
       write(14,"(2(E14.7,1x))") Ei_test(i),E_fit(i)
       enddo
       close(14)
       endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
       deallocate(feat_new)
       deallocate(E_fit)
       deallocate(index)
       deallocate(dE_list)

!       deallocate(feat_ext2)
!       deallocate(dE_term)

       deallocate(S)
       deallocate(BB)
       deallocate(ipiv)
!       deallocate(idd)
5000   continue

    deallocate(fread_dfeat) 
    stop
end
