    subroutine get_grid2b_type1(grid2,Rc,Rm,n2b)

    implicit double precision (a-h,o-z) 
    real*8 grid2(0:n2b+1)

    alpha2=exp(log((Rc+Rm)/Rm)/(n2b+1))
    grid2(0)=Rm
    do i=1,n2b+1
    grid2(i)=grid2(i-1)*alpha2
    enddo
    grid2=grid2-Rm
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
    return
    end subroutine get_grid2b_type1
