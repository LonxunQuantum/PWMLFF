       subroutine calc_clebsch_gordan(CC,jmm)
       implicit double precision (a-h,o-z)
       real*8 CC(-jmm:jmm,-jmm:jmm,-jmm:jmm,0:jmm,0:jmm,0:jmm)
! m1,m2,j are all doubled index: 2*m1,2*m3,2*j


!  All index has been multiplied by 2

       CC=0.d0
       
       do j=0,jmm   ! double index
       do j1=0,j    ! double index
       do j2=0,j1
       if(abs(j1-j2).le.j.and.j.le.j1+j2.and.mod(j1+j2-j+100,2).eq.0) then

       prod1=(j+1)*factorial((j+j1-j2)/2)*factorial((j-j1+j2)/2)*factorial((j1+j2-j)/2)
       prod1=prod1/factorial((j1+j2+j)/2+1)
       prod1=dsqrt(1.d0*prod1)
       do m=-j,j,2    ! doubled index
       do m1=-j1,j1,2
       do m2=-j2,j2,2
       if(m.ne.m1+m2) goto 200
       prod2=factorial((j+m)/2)*factorial((j-m)/2)
       prod2=prod2*factorial((j1-m1)/2)*factorial((j1+m1)/2)
       prod2=prod2*factorial((j2-m2)/2)*factorial((j2+m2)/2)
       prod2=dsqrt(prod2)
       sum=0.d0
       do k=0,2*jmm
       prod3=(-1)**k/factorial(k)
       jj=(j1+j2-j)/2-k
       if(jj.lt.0) goto 100
       prod3=prod3/factorial(jj)
       jj=(j1-m1)/2-k
       if(jj.lt.0) goto 100
       prod3=prod3/factorial(jj)
       jj=(j2+m2)/2-k
       if(jj.lt.0) goto 100
       prod3=prod3/factorial(jj)
       jj=(j-j2+m1)/2+k
       if(jj.lt.0) goto 100
       prod3=prod3/factorial(jj)
       jj=(j-j1-m2)/2+k
       if(jj.lt.0) goto 100
       prod3=prod3/factorial(jj)
       sum=sum+prod3
100    continue
       enddo
       CC(m,m1,m2,j,j1,j2)=prod1*prod2*sum
200    continue
       enddo
       enddo
       enddo
        endif
       enddo
       enddo
       enddo
   
       return
       end subroutine calc_clebsch_gordan
       


       

