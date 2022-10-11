      subroutine get_ALI(AL,ALI)
******************************************
cc     Written by Lin-Wang Wang, March 30, 2001.  
*************************************************************************
**  copyright (c) 2003, The Regents of the University of California,
**  through Lawrence Berkeley National Laboratory (subject to receipt of any
**  required approvals from the U.S. Dept. of Energy).  All rights reserved.
*************************************************************************

******************************************


      real*8 AL(3,3),ALI(3,3)
      real*8 tmp(3)

      do i=1,3
         do j=1,3
            ALI(j,i)=AL(i,j)
         enddo
         tmp(i)=1
      enddo

      call gaussj(ALI,3,3,tmp,1,1)

*****************************************
*     *  \sum_i AL(i,j1)*ALI(i,j2)= \delta_j1,j2
*     *  2*pi*ALI(i,j) is the jth G vector unit in (i)x,y,z components
*****************************************
      return
      end

