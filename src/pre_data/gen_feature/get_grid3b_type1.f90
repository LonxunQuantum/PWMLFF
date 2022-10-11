    subroutine get_grid3b_type1(grid31,grid32,Rc,Rc2,Rm,n3b1,n3b2)

    implicit double precision (a-h,o-z) 
    real*8 grid31(0:n3b1+1),grid32(0:n3b2+1)

    alpha31=exp(log((Rc+Rm)/Rm)/(n3b1+1))
    grid31(0)=Rm
    do i=1,n3b1+1
    grid31(i)=grid31(i-1)*alpha31
    enddo
    grid31=grid31-Rm
    fact_grid31=1.d0/log(alpha31)

    alpha32=exp(log((Rc2+Rm)/Rm)/(n3b2+1))
    grid32(0)=Rm
    do i=1,n3b2+1
    grid32(i)=grid32(i-1)*alpha32
    enddo
    grid32=grid32-Rm
    fact_grid32=1.d0/log(alpha32)

!cccccccccccccccccccccccccccccccccccccccccccccccccccc
    return
    end subroutine get_grid3b_type1
