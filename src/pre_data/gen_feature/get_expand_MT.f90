      subroutine get_expand_MT(ntype,numT,indi,indj,mu,rank,num,jmu_b,itype_b)
! input: ntype,numT,indi,indj,mu,rank,
! output: num,jmub,itype_b
! num: the total number of the expanded contraction
! jmu_b: the mu (radial polynomial index) of this contraction
! itype_b: the type of this contraction. 
! the contfaction itself for the tesnor part follows the index of indi,indj


      implicit double precision (a-h,o-z)

      character*1 txt
      integer ind(10,10)
      integer indi(10,10),indj(10,10)
      integer num,numT,mu(10),rank(10)
      integer iflag_ti(10,10),iflag_tj(10,10)
      integer indpos(10),indpos2(100,10)
      integer jmu_b(10,5000),itype_b(10,5000)
      real*8 check1_st(5000),check2_st(5000),check3_st(5000),check4_st(5000)
      
 
!ccccccccccc The following segment is to figure out the equivalent contraction, 
!ccccccccccc Thus, only take one of them. 
!ccccccccccc We first define an identifier indpos2(kk,i) for each tensor i, and its expansion kk
!ccccccccccc Each tensor will have: mu(i)*ntype expension (for the radial and type part)
       indpos=0
       indpos2=0

       do i=1,numT
       nind=0
       do j=1,rank(i)
       iflag=0
       do j1=1,j-1
       if(indi(j1,i).eq.indi(j,i)) iflag=1
       enddo
       if(iflag.eq.0) nind=nind+1 ! the number of new tensor point to
       enddo
       indpos(i)=indpos(i)+nind   ! the number of tensor this tensor going to contract
       enddo
!  nind is the number of unique tensor this tensor point to for contraction

       do i=1,numT
       do jj=0,mu(i)
       do itype=1,ntype
       kk=itype+jj*ntype
       indpos2(kk,i)=kk*100000+rank(i)*10000+indpos(i)*1000

       do j=1,rank(i)
       indpos2(kk,i)=indpos2(kk,i)+indpos(indi(j,i))+rank(indi(j,i))*100
       enddo
!       write(6,*) "indpos2",i,kk,indpos2(kk,i)
       enddo
       enddo
       enddo
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       if(numT.eq.1) then

       if(rank(1).ne.0) then
       write(6,*) "for numT=0,rank(1) must be zero",numT
       stop
       endif
 
       num=0
       do jj1=0,mu(1)
       do itype1=1,ntype
       kk1=itype1+jj1*ntype
       num=num+1
       if(num.gt.5000) then
        write(6,*) "num.gt.5000,stop"
        stop
       endif
       jmu_b(1,num)=jj1
       itype_b(1,num)=itype1
       enddo
       enddo

       endif

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       if(numT.eq.2) then
 
       num=0
       do jj1=0,mu(1)
       do itype1=1,ntype
       kk1=itype1+jj1*ntype

       do jj2=0,mu(2)
       do itype2=1,ntype
       kk2=itype2+jj2*ntype

       check1=indpos2(kk1,1)+indpos2(kk2,2)
       check2=(indpos2(kk1,1)/1000.d0)**2+(indpos2(kk2,2)/1000.d0)**2
       check3=(indpos2(kk1,1)/1000.d0)**3+(indpos2(kk2,2)/1000.d0)**3
       check4=(indpos2(kk1,1)/1000.d0)**4+(indpos2(kk2,2)/1000.d0)**4

       iflag=0
       do kk=1,num
        check=abs(check1-check1_st(kk))+abs(check2-check2_st(kk))+abs(check3-check3_st(kk))+ &
              abs(check4-check4_st(kk))
        if(check.lt.1.E-5) iflag=1
       enddo
!      if iflag.eq.1, this expanded tensor are the same as before, do not include it
       
       if(iflag.eq.0) then
       num=num+1
       if(num.gt.5000) then
        write(6,*) "num.gt.5000,stop"
        stop
       endif
       check1_st(num)=check1
       check2_st(num)=check2
       check3_st(num)=check3
       check4_st(num)=check4
       jmu_b(1,num)=jj1
       itype_b(1,num)=itype1
       jmu_b(2,num)=jj2
       itype_b(2,num)=itype2
       endif

       enddo
       enddo
       enddo
       enddo

       endif



!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       if(numT.eq.3) then
 
       num=0
       do jj1=0,mu(1)
       do itype1=1,ntype
       kk1=itype1+jj1*ntype

       do jj2=0,mu(2)
       do itype2=1,ntype
       kk2=itype2+jj2*ntype

       do jj3=0,mu(3)
       do itype3=1,ntype
       kk3=itype3+jj3*ntype

       check1=indpos2(kk1,1)+indpos2(kk2,2)+indpos2(kk3,3)
       check2=(indpos2(kk1,1)/1000.d0)**2+(indpos2(kk2,2)/1000.d0)**2+(indpos2(kk3,3)/1000.d0)**2
       check3=(indpos2(kk1,1)/1000.d0)**3+(indpos2(kk2,2)/1000.d0)**3+(indpos2(kk3,3)/1000.d0)**3
       check4=(indpos2(kk1,1)/1000.d0)**4+(indpos2(kk2,2)/1000.d0)**4+(indpos2(kk3,3)/1000.d0)**4

       iflag=0
       do kk=1,num
        check=abs(check1-check1_st(kk))+abs(check2-check2_st(kk))+abs(check3-check3_st(kk))+ &
              abs(check4-check4_st(kk))
        if(check.lt.1.E-5) iflag=1
       enddo
!      if iflag.eq.1, the twp indpos2 are the same as before, do not include it. 
       
       if(iflag.eq.0) then
       num=num+1
       if(num.gt.5000) then
        write(6,*) "num.gt.5000,stop"
        stop
       endif
       check1_st(num)=check1
       check2_st(num)=check2
       check3_st(num)=check3
       check4_st(num)=check4
       jmu_b(1,num)=jj1
       itype_b(1,num)=itype1
       jmu_b(2,num)=jj2
       itype_b(2,num)=itype2
       jmu_b(3,num)=jj3
       itype_b(3,num)=itype3
       endif

       enddo
       enddo
       enddo
       enddo
       enddo
       enddo
       endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       if(numT.eq.4) then
 
       num=0
       do jj1=0,mu(1)
       do itype1=1,ntype
       kk1=itype1+jj1*ntype

       do jj2=0,mu(2)
       do itype2=1,ntype
       kk2=itype2+jj2*ntype

       do jj3=0,mu(3)
       do itype3=1,ntype
       kk3=itype3+jj3*ntype

       do jj4=0,mu(4)
       do itype4=1,ntype
       kk4=itype4+jj4*ntype

       check1=indpos2(kk1,1)+indpos2(kk2,2)+indpos2(kk3,3)+indpos2(kk4,4)
       check2=(indpos2(kk1,1)/1000.d0)**2+(indpos2(kk2,2)/1000.d0)**2+(indpos2(kk3,3)/1000.d0)**2+(indpos2(kk4,4)/1000.d0)**2
       check3=(indpos2(kk1,1)/1000.d0)**3+(indpos2(kk2,2)/1000.d0)**3+(indpos2(kk3,3)/1000.d0)**3+(indpos2(kk4,4)/1000.d0)**3
       check4=(indpos2(kk1,1)/1000.d0)**4+(indpos2(kk2,2)/1000.d0)**4+(indpos2(kk3,3)/1000.d0)**4+(indpos2(kk4,4)/1000.d0)**4

       iflag=0
       do kk=1,num
        check=abs(check1-check1_st(kk))+abs(check2-check2_st(kk))+abs(check3-check3_st(kk))+ &
              abs(check4-check4_st(kk))
        if(check.lt.1.E-5) iflag=1
       enddo
!      if iflag.eq.1, the twp indpos2 are the same as before, do not include it. 
       
       if(iflag.eq.0) then
       num=num+1
       if(num.gt.5000) then
        write(6,*) "num.gt.5000,stop"
        stop
       endif
       check1_st(num)=check1
       check2_st(num)=check2
       check3_st(num)=check3
       check4_st(num)=check4
       jmu_b(1,num)=jj1
       itype_b(1,num)=itype1
       jmu_b(2,num)=jj2
       itype_b(2,num)=itype2
       jmu_b(3,num)=jj3
       itype_b(3,num)=itype3
       jmu_b(4,num)=jj4
       itype_b(4,num)=itype4
       endif

       enddo
       enddo
       enddo
       enddo
       enddo
       enddo
       enddo
       enddo
       endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

!       write(6,*) "num=",num
!       do iii=1,num
!       write(6,"(10(i3,1x,i3,3x))") ((jmu_b(i,iii),itype_b(i,iii)),i=1,numT)
!       enddo

       return
      end subroutine get_expand_MT

