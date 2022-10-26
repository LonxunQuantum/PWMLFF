program feat_collect_PCA
       implicit double precision (a-h,o-z)
       integer(4) :: i
       real*8,allocatable,dimension(:,:,:) :: feat_case
       real*8,allocatable,dimension(:,:) :: feat2_case
       real*8,allocatable,dimension(:,:) :: feat2_case_tmp
       real*8,allocatable,dimension(:,:) :: Ei_case
       real*8,allocatable,dimension(:) :: feat_tmp
       real*8,allocatable,dimension(:,:) :: S
       real*8,allocatable,dimension(:,:) :: PV
       real*8,allocatable,dimension(:) :: EW,work,BB
       real*8,allocatable,dimension(:) :: BB2_ave
       real*8,allocatable,dimension(:) :: weight_case
       real*8,allocatable,dimension(:) :: E_fit
       real*8,allocatable,dimension(:) :: feat2_shift,feat2_scale
       integer,allocatable,dimension(:) :: ipiv
       integer iatom_type(10),num_case(10),nfeat_iatype(10)
       integer ifeat_type(100),nfeat(10,100),ncase_tmp(100)
       integer icount_feat(10)
       real*8 Ei_tmp
       integer itmp,lwork
       character(len=200),allocatable,dimension (:) :: trainSetFileDir
       character(len=200) trainSetDir,BadImageDir
       character(len=200) MOVEMENTDir,dfeatDir,infoDir,trainDataDir,MOVEMENTallDir
       integer sys_num,sys
       integer nfeat2_store(100),nfeat1_store(100)
! liuliping for relative path
       integer tmp_i
       character(len=200) fitModelDir
       character(len=:), allocatable :: fread_dfeat

       ! this file should be create by prepare.py
       !write(*,*) "liuliping: test1"
       open(1314, file="input/info_dir")
       rewind(1314)
       read(1314,"(A200)") fitModelDir
       close(1314)
       !write(*,*) "liuliping: test2"
       tmp_i = len(trim(adjustl(fitModelDir)))
       allocate(character(len=tmp_i) :: fread_dfeat)
       fread_dfeat = trim(adjustl(fitModelDir))
       !write(*,*) "liuliping, fread_dfeat: ", fread_dfeat
       ! liuliping, end, all .r .x file should be invoke out of fread_dfeat
       
       open(10,file=fread_dfeat//"feat_collect.in") 
       rewind(10)
       read(10,*) iflag_PCA
       read(10,*) nfeat_type   ! there are nfeat_type of features
       do i=1,nfeat_type
         read(10,*) ifeat_type(i)
       enddo
       read(10,*) ntype      ! there ate ntype of atoms
       do i=1,ntype
          read(10,*)iatom_type(i)
       end do
       close(10)

       open(13,file=fread_dfeat//"location")
       rewind(13)
       read(13,*) sys_num  !,trainSetDir
       read(13,'(a200)') trainSetDir
       close(13)
       ! MOVEMENTallDir=trim(trainSetDir)//"/MOVEMENTall"

       nfeat=0
       ncase_tmp=0

       do kkk=1,nfeat_type  

       trainDataDir=trim(trainSetDir)//"/trainData.txt.Ftype"//char(ifeat_type(kkk)+48)


       open(10,file=trainDataDir)
       rewind(10)
       do i=1,10000000
       read(10,*,IOSTAT=ierr) i2,atom,Ei_tmp,nfeat_tmp 
       if(ierr.ne.0) then
       ncase_tmp(kkk)=i-1
       goto 701
       endif

       itmp=atom+0.0001
       ii=0
       do itype=1,ntype
       if(itmp.eq.iatom_type(itype)) then
       ii=itype
       endif
       enddo
       if(ii.eq.0) then
       write(6,*) "atom type not found in ", trainDataDir
       write(6,*) itmp
       stop
       endif
       if(nfeat(ii,kkk).eq.0) then
       nfeat(ii,kkk)=nfeat_tmp
       else
         if(nfeat(ii,kkk).ne.nfeat_tmp) then
         write(6,*) "nfeat changed for same iat,and feat_type,stop"
         stop
         endif
       endif
       enddo   ! i=1,100000
701    continue
       close(10)
       enddo   ! kkk=1,nfeat_type

       if(nfeat_type.gt.1) then
       do kkk=1,nfeat_type-1
       if(ncase_tmp(kkk).ne.ncase_tmp(kkk+1)) then
       write(6,*)"num of case are different in different trainData.txt.Ftype"
       write(6,*) kkk,ncase_tmp(kkk),ncase_tmp(kkk+1)  
       stop
       endif
       enddo
       endif
       ncase=ncase_tmp(1)

       nfeatM=0
       do ii=1,ntype
       num=0
       do kkk=1,nfeat_type
       num=num+nfeat(ii,kkk)
       enddo
       nfeat_iatype(ii)=num   ! this include all feature types
       if(num.gt.nfeatM) nfeatM=num
       enddo

!   nfeat(ii,kkk) the num of feature for this atom type ii, and feature type kkk
!   nfeat_iatype(iatype) is the total number of feature for this type
!   nfeatM is the maximum num_feature for different type, place holder


       allocate(feat_case(nfeatM,ncase,ntype))
       allocate(Ei_case(ncase,ntype))
       allocate(feat_tmp(nfeatM))



!ccccccccccccccccccccccccccccccccccccccccccccccccccc
          do ii=1,ntype
          icount_feat(ii)=0
          enddo

       do 3300 kkk=1,nfeat_type  
       trainDataDir=trim(trainSetDir)//"/trainData.txt.Ftype"//char(ifeat_type(kkk)+48)

       num_case=0

       open(10,file=trainDataDir)
       rewind(10)
       do iii=1,10000000
       read(10,*,IOSTAT=ierr) i2,atom,Ei_tmp,nfeat_tmp,(feat_tmp(j),j=1,nfeat_tmp) 
       if(ierr.ne.0) then
       goto 700
       endif


       itmp=atom+0.0001
       ii=0
       do itype=1,ntype
       if(itmp.eq.iatom_type(itype)) then
       ii=itype
       endif
       enddo


        num_case(ii)=num_case(ii)+1
        if(kkk.eq.1) then
        Ei_case(num_case(ii),ii)=Ei_tmp
        else
            if(abs(Ei_case(num_case(ii),ii)-Ei_tmp).gt.1.D-9) then
            write(6,*) "Ei not the same in trainData.txt.Ftype,stop"
            write(6,*) i2,Ei_case(num_case(ii),ii), Ei_tmp
            stop
            endif
        endif
      
       do j=1,nfeat_tmp
              feat_case(j+icount_feat(ii),num_case(ii),ii)=feat_tmp(j)
       enddo
       
       enddo  ! do iii=1,10000
700    continue
       write(6,*) "total case=",ncase

       do ii=1,ntype
       icount_feat(ii)=icount_feat(ii)+nfeat(ii,kkk)
       enddo
  
       close(10)
3300   continue

!cccccccccccccccccccccccccccccccccccccccccccccc
       write(6,*) "num_case(itype)", (num_case(i),i=1,ntype)
       write(6,*) "num_feat(itype)", (nfeat_iatype(i),i=1,ntype)


       do 4300 itype=1,ntype

       if(iflag_PCA.eq.1) then
      
!TODO: with PCA:*****************************************
        nfeat1=nfeat_iatype(itype)
        allocate(S(nfeat1,nfeat1))

        S=0.d0
        call dgemm('N','T',nfeat1,nfeat1,num_case(itype),1.d0,feat_case(1,1,itype),&
     &  nfeatM,feat_case(1,1,itype),nfeatM,0.d0,S,nfeat1)


        lwork=10*nfeat1
        allocate(work(lwork))
        allocate(EW(nfeat1))
        call dsyev('V','U',nfeat1,S,nfeat1,EW,work,lwork,info)

        open(10,file=fread_dfeat//"PCA_eigen_feat."//char(itype+48))
        rewind(10)
        do k=1,nfeat1
        write(10,*) i,EW(nfeat1-k+1)
        enddo
        close(10)
        
        num=0
        do k=1,nfeat1
        if(abs(EW(nfeat1-k+1)).gt.1.D-4) num=num+1
        enddo
        nfeat2=num
        nfeat2_store(itype)=nfeat2
        nfeat1_store(itype)=nfeat1

        write(6,*)"PCA,itype,nfeat,nfeat2",itype,nfeat1,nfeat2

        allocate(PV(nfeat1,nfeat2))

        do k=1,nfeat2
        scale=1/dsqrt(abs(EW(nfeat1-k+1)))
        do j=1,nfeat1
        PV(j,k)=S(j,nfeat1-k+1)*scale
        enddo
        enddo
        
        deallocate(S)
        deallocate(work)
        deallocate(EW)

        else    ! no PCA

!*********************************************************
!TODO: without PCA
        nfeat1=nfeat_iatype(itype)  ! the original feature type
        nfeat2=nfeat_iatype(itype)  ! the new PCA feature type
        nfeat2_store(itype)=nfeat2
        nfeat1_store(itype)=nfeat1
       allocate(PV(nfeat1,nfeat2))
       write(6,*)"noPCA,itype,nfeat,nfeat2",itype,nfeat1,nfeat2

       do k=1,nfeat2
       do j=1,nfeat1
       if (j.eq.k) then 
       PV(j,k)=1
       else 
       PV(j,k)=0
       endif
       enddo
       enddo

       endif   ! iflag_PCA
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccc

!  PV(j,k) is the vector for principle component k

       allocate(feat2_case(nfeat2,num_case(itype)))

       call dgemm('T','N',nfeat2,num_case(itype),nfeat1,1.d0,PV, &
     & nfeat1,feat_case(1,1,itype),nfeatM,0.d0,feat2_case,nfeat2)

! change the last feature just equal one (as a constant)
! feat2_case(ifeat,icase) is the new features
! The nfeat2 feature is replaced by 1. 

       open(10,file=fread_dfeat//"feat_new_stored0."//char(itype+48),form="unformatted")
       rewind(10) 
       write(10) num_case(itype),nfeat2
       do ii=1,num_case(itype)
       write(10) ii,Ei_case(ii,itype),feat2_case(:,ii)
! This feat2_case has combined all the feature types:nfeat2
       enddo
       close(10)

       feat2_case(nfeat2,:)=1.d0      ! this practice is a bit strange, always lost one feat. 
       allocate(feat2_scale(nfeat2))
       allocate(feat2_shift(nfeat2))

       do j=1,nfeat2-1
       sum=0.d0
       do i=1,num_case(itype)
       sum=sum+feat2_case(j,i)
       enddo
       sum_ave=sum/num_case(itype)

       feat2_shift(j)=sum_ave

       sum=0.d0
       do i=1,num_case(itype)
       feat2_case(j,i)=feat2_case(j,i)-sum_ave
       sum=sum+feat2_case(j,i)**2
       enddo
       sum=sum/num_case(itype)
       if (abs(sum).lt.1.E-10) then
              sum=1
       endif
       sum=1/dsqrt(sum)

       feat2_scale(j)=sum

       do i=1,num_case(itype)
       feat2_case(j,i)=sum*feat2_case(j,i)
       enddo
       enddo

       feat2_shift(nfeat2)=0.d0
       feat2_scale(nfeat2)=1.d0
!  The deature is then shifted and normalized, 
       open(10,file=fread_dfeat//"feat_PV."//char(itype+48),form="unformatted")
       rewind(10)
       write(10) nfeat1,nfeat2  ! original bug
       write(10) PV
       write(10) feat2_shift
       write(10) feat2_scale
       close(10)
       ! write(6,*) feat2_case
       open(10,file=fread_dfeat//"feat_shift."//char(itype+48))
       rewind(10)
       do i=1,nfeat2
       write(10,*) feat2_shift(i),feat2_scale(i)
       enddo
       close(10)

        write(6,*) "num_case,itype",num_case(itype),itype
       open(10,file=fread_dfeat//"feat_new_stored."//char(itype+48),form="unformatted")
       rewind(10) 
       write(10) num_case(itype),nfeat2
       do ii=1,num_case(itype)
       write(10) ii,Ei_case(ii,itype),feat2_case(:,ii)
       enddo
       close(10)

!  In above, finish the new feature. We could have stopped here. 
!------------------------------------------------------------------
!------------------------------------------------------------------
!------------------------------------------------------------------
!------------------------------------------------------------------

!    In the following, we will do a linear fitting E= \sum_i W(i) feat2(i)
!    We will do several times, so have a average for W(i)^2. 
!    The everage W(i) will be used as a metrix to measure the distrance
!    Between two points. 


!        deallocate(S)
       allocate(S(nfeat2,nfeat2))
       allocate(BB(nfeat2))
       allocate(BB2_ave(nfeat2))
       allocate(ipiv(nfeat2))
       allocate(E_fit(num_case(itype)))
       allocate(weight_case(num_case(itype)))

       allocate(feat2_case_tmp(nfeat2,num_case(itype)))

       iseed=-19287
       BB2_ave=0.d0

!       write(6,*) "input iseed (negative)"
!       read(5,*) iseed


!ccc average over different situations
!ccc try smooth out the zeros in BB
       do 1000 iii=1,100

       do ii=1,num_case(itype)
       ran=ran1(iseed)
       if(ran.gt.0.5) then
       weight_case(ii)=1.d0       ! random selection of the cases
       else
       weight_case(ii)=0.d0
       endif

       do j=1,nfeat2     
       feat2_case_tmp(j,ii)=feat2_case(j,ii)*weight_case(ii)
       enddo
       enddo
      


       S=0.d0
!       call dgemm('N','T',nfeat2,nfeat2,num_case(itype),1.d0,feat2_case,&
!     & nfeat2,feat2_case,nfeat2,0.d0,S,nfeat2)
       call dgemm('N','T',nfeat2,nfeat2,num_case(itype),1.d0,feat2_case_tmp,&
     & nfeat2,feat2_case_tmp,nfeat2,0.d0,S,nfeat2)

       sum=0.d0
       do j1=1,nfeat2
       do j2=1,nfeat2
       sum=sum+abs(S(j1,j2))
       enddo
       enddo
       delta=sum/nfeat2**2
       delta=delta*0.0001
       
       do j=1,nfeat2
       S(j,j)=S(j,j)+delta
       enddo

       do j=1,nfeat2
       sum=0.d0
       do i=1,num_case(itype)
!       sum=sum+Ei_case(i,itype)*feat2_case(j,i)
       sum=sum+Ei_case(i,itype)*feat2_case_tmp(j,i)*weight_case(i)
       enddo
       BB(j)=sum
       enddo


       call dgesv(nfeat2,1,S,nfeat2,ipiv,BB,nfeat2,info)  

!    BB is the linear eight: Ei_case(icase)= \sum_i BB(i)* feat2_case(i,icase)

       do j=1,nfeat2
       BB2_ave(j)=BB2_ave(j)+BB(j)**2
       enddo

1000    continue

       BB2_ave=dsqrt(BB2_ave/100)
       open(10,file=fread_dfeat//"weight_feat."//char(itype+48))
       rewind(10) 

       sum1=0.d0
       do j=1,nfeat2-1
       sum1=sum1+BB2_ave(j)
       enddo
       sum1=sum1/(nfeat2-1)

       do j=1,nfeat2
       write(10,"(i5,1x,1(E15.7,1x))") j,BB2_ave(j)
!cccc do not use this
!       write(10,"(i5,1x,1(E15.7,1x))") j,sum1
       enddo
       close(10)


      deallocate(PV)
      deallocate(feat2_case)
      deallocate(feat2_scale)
      deallocate(feat2_shift)
      
      deallocate(S)
      deallocate(BB)
      deallocate(BB2_ave)
      deallocate(ipiv)
      deallocate(E_fit)
      deallocate(weight_case)

      deallocate(feat2_case_tmp)


4300   continue

       open(11,file=fread_dfeat//"feat.info")
       rewind(11)
       write(11,*) iflag_PCA
       write(11,*) nfeat_type   ! there are nfeat_type of features
       do i=1,nfeat_type
         write(11,*) ifeat_type(i)
       enddo
       write(11,*) ntype      ! there ate ntype of atoms
       do i=1,ntype
          write(11,"(3(i6,2x))") iatom_type(i),nfeat1_store(i),nfeat2_store(i)
       end do
       do ii=1,ntype
       write(11,"(10(i5,1x))") (nfeat(ii,kkk),kkk=1,nfeat_type) 
       enddo
       close(11)



       deallocate(fread_dfeat)
       stop
       end

       
