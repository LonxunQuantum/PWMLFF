    subroutine cal_energy_force_lin(feat,dfeat,num_neigh,list_neigh,AL,xatom,natom_tmp,nfeat0_tmp,m_neigh_tmp)
        integer(4)  :: itype,ixyz,i,j,jj
        integer natom_tmp,nfeat0_tmp,m_neigh_tmp
        real(8) :: sum
!        real(8),intent(in) :: feat(nfeat0_tmp,natom_tmp)
!        real*8, intent(in) :: dfeat(nfeat0_tmp,natom_tmp,m_neigh_tmp,3)
        real(8),intent(in) :: feat(nfeat0_tmp,natom_n)
        real*8, intent(in) :: dfeat(nfeat0_tmp,natom_n,m_neigh_tmp,3)
        integer(4),intent(in) :: num_neigh(natom_tmp)
        integer(4),intent(in) :: list_neigh(m_neigh_tmp,natom_tmp)
        real(8), intent(in) :: AL(3,3)
        real(8),dimension(:,:),intent(in) :: xatom
                
        
        real(8),allocatable,dimension(:,:) :: feat2
        real(8),allocatable,dimension(:,:,:) :: feat_type
        real(8),allocatable,dimension(:,:,:) :: feat2_type
        integer(4),allocatable,dimension(:,:) :: ind_type
        real(8),allocatable,dimension(:,:,:) :: dfeat_type
        real(8),allocatable,dimension(:,:,:) :: dfeat2_type
        real(8),allocatable,dimension(:,:,:,:) :: dfeat2
        

        real*8 pi,dE,dFx,dFy,dFz
        real*8 rad1,rad2,rad,dx1,dx2,dx3,dx,dy,dz,dd,yy,w22,dEdd,d,w22_1,w22_2,w22F_1,w22F_2
        integer iat1,iat2,ierr


         pi=4*datan(1.d0)
        
        
        
        allocate(feat2(nfeat2m,natom_n))
        allocate(feat_type(nfeat1m,natom_n,ntype))
        allocate(feat2_type(nfeat2m,natom_n,ntype))
        allocate(ind_type(natom_n,ntype))
        allocate(dfeat_type(nfeat1m,natom_n*m_neigh*3,ntype))
        allocate(dfeat2_type(nfeat2m,natom_n*m_neigh*3,ntype))
        allocate(dfeat2(nfeat2m,natom_n,m_neigh,3))

        istat=0
        error_msg=''

        if (nfeat0_tmp/=nfeat1m .or. natom_tmp/=natom .or. m_neigh_tmp/=m_neigh) then
            write(*,*) "Shape of input arrays don't match the model!"
            stop
        end if

        
        
        
        
        num = 0
        iat1=0
        do i = 1, natom
        if(mod(i-1,nnodes).eq.inode-1) then
        iat1=iat1+1
            itype = iatom_type(i)
            num(itype) = num(itype) + 1
            ind_type(num(itype), itype) = iat1
            feat_type(:, num(itype), itype) = feat(:, iat1)
         endif
         enddo
    
        
        do itype = 1, ntype
            call dgemm('T', 'N', nfeat2(itype), num(itype), nfeat1(itype), 1.d0, pv(1,1,itype), nfeat1m, feat_type(1,1,itype), nfeat1m, 0.d0,feat2_type(1,1,itype), nfeat2m)
        end do
    
        
        do itype = 1, ntype
            do i = 1, num(itype)
                do j = 1, nfeat2(itype) - 1
                    feat2_type(j, i, itype) = (feat2_type(j,i,itype)-feat2_shift(j,itype))*feat2_scale(j, itype)
                end do
                feat2_type(nfeat2(itype), i, itype) = 1.d0
            end do
        end do
          
        
        num = 0
        iat1=0
        do i = 1, natom
        if(mod(i-1,nnodes).eq.inode-1) then
        iat1=iat1+1
            itype = iatom_type(i)
            num(itype) = num(itype) + 1
            feat2(:, iat1) = feat2_type(:, num(itype), itype)
        endif
        enddo
          

        energy_pred_tmp=0.d0
        
        iat1=0
        do i = 1, natom
        if(mod(i-1,nnodes).eq.inode-1) then
        iat1=iat1+1
            itype = iatom_type(i)
            sum = 0.d0
            do j = 1, nfeat2(itype)
              sum = sum + feat2(j, iat1)*bb_type0(j, itype)
            end do
            energy_pred_tmp(i) = sum
        endif
        enddo


        
        
        num = 0
        iat1=0
        do i = 1, natom
        if(mod(i-1,nnodes).eq.inode-1) then
        iat1=iat1+1
                itype = iatom_type(i) 
            do jj = 1, num_neigh(i)
                num(itype) = num(itype) + 1
                dfeat_type(:, num(itype), itype) = dfeat(:, iat1, jj, 1)
                num(itype) = num(itype) + 1
                dfeat_type(:, num(itype), itype) = dfeat(:, iat1, jj, 2)
                num(itype) = num(itype) + 1
                dfeat_type(:, num(itype), itype) = dfeat(:, iat1, jj, 3)
            end do
        endif
        end do
        !cccccccc note: num(itype) is rather large, in the scane of natom*num_neigh
    
        do itype = 1, ntype
            call dgemm('T', 'N', nfeat2(itype), num(itype), nfeat1(itype), 1.d0, pv(1,1,itype), nfeat1m, dfeat_type(1,1,itype), nfeat1m, 0.d0, dfeat2_type(1,1,itype), nfeat2m)
        end do
    
        
        num = 0
        iat1=0
        do i = 1, natom
        if(mod(i-1,nnodes).eq.inode-1) then
        iat1=iat1+1
                itype = iatom_type(i) 
            do jj = 1, num_neigh(i)
                num(itype) = num(itype) + 1
                do j = 1, nfeat2(itype) - 1
                    dfeat2(j, iat1, jj, 1) = dfeat2_type(j, num(itype), itype)*feat2_scale(j, itype)
                end do
                dfeat2(nfeat2(itype), iat1, jj, 1) = 0.d0
                num(itype) = num(itype) + 1
                do j = 1, nfeat2(itype) - 1
                    dfeat2(j, iat1, jj, 2) = dfeat2_type(j, num(itype), itype)*feat2_scale(j, itype)
                end do
                dfeat2(nfeat2(itype), iat1, jj, 2) = 0.d0
                num(itype) = num(itype) + 1
                do j = 1, nfeat2(itype) - 1
                    dfeat2(j, iat1, jj, 3) = dfeat2_type(j, num(itype), itype)*feat2_scale(j, itype)
                end do
                dfeat2(nfeat2(itype), iat1, jj, 3) = 0.d0
            end do
        endif
        end do
        
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !cc  the new dfeat2 is:
        !cc dfeat2(nfeat2,natom,j_neigh,3): dfeat2(j,i,jj,3)= d/dr(jj_neigh)(feat2(j,i))
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !cccc now, we have the new features, we need to calculate the distance to reference state
    

        force_pred_tmp=0.d0
    
        iat1=0
        do i = 1, natom
        if(mod(i-1,nnodes).eq.inode-1) then
        iat1=iat1+1
            itype = iatom_type(i) 
            do jj = 1, num_neigh(i)
              iat2=list_neigh(jj,i)

                do j = 1, nfeat2(itype)    
             force_pred_tmp(1,iat2)=force_pred_tmp(1,iat2)+dfeat2(j,iat1,jj,1)*bb_type0(j,itype)
             force_pred_tmp(2,iat2)=force_pred_tmp(2,iat2)+dfeat2(j,iat1,jj,2)*bb_type0(j,itype)
             force_pred_tmp(3,iat2)=force_pred_tmp(3,iat2)+dfeat2(j,iat1,jj,3)*bb_type0(j,itype)
                enddo
            end do
        endif
        end do
    


       
!ccccccccccccccccccccccccccccccccccccccccccc
       iat1=0
       do i=1,natom
       if(mod(i-1,nnodes).eq.inode-1) then
       iat1=iat1+1


       rad1=rad_atom(iatom_type(i))

       dE=0.d0
       dFx=0.d0
       dFy=0.d0
       dFz=0.d0
       do jj=1,num_neigh(i)
       j=list_neigh(jj,i)
       if(i.ne.j) then
       rad2=rad_atom(iatom_type(j))
       rad=rad1+rad2
       dx1=mod(xatom(1,j)-xatom(1,i)+100.d0,1.d0)
       if(abs(dx1-1).lt.abs(dx1)) dx1=dx1-1
       dx2=mod(xatom(2,j)-xatom(2,i)+100.d0,1.d0)
       if(abs(dx2-1).lt.abs(dx2)) dx2=dx2-1
       dx3=mod(xatom(3,j)-xatom(3,i)+100.d0,1.d0)
       if(abs(dx3-1).lt.abs(dx3)) dx3=dx3-1
       dx=AL(1,1)*dx1+AL(1,2)*dx2+AL(1,3)*dx3
       dy=AL(2,1)*dx1+AL(2,2)*dx2+AL(2,3)*dx3
       dz=AL(3,1)*dx1+AL(3,2)*dx2+AL(3,3)*dx3
       dd=dsqrt(dx**2+dy**2+dz**2)
       if(dd.lt.2*rad) then
!        w22=dsqrt(wp_atom(iatom_type(i))*wp_atom(iatom_type(j)))
!        yy=pi*dd/(4*rad)
! !       dE=dE+0.5*w22*exp((1-dd/rad)*4.0)*cos(yy)**2
! !       dEdd=w22*exp((1-dd/rad)*4.d0)*((-4/rad)*cos(yy)**2
! !     &   -(pi/(2*rad))*cos(yy)*sin(yy))
!        dE=dE+0.5*4*w22*(rad/dd)**12*cos(yy)**2
!        dEdd=4*w22*(-12*(rad/dd)**12/dd*cos(yy)**2  &
!         -(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)
        w22_1=wp_atom(iatom_type(j),iatom_type(i),1)
        w22_2=wp_atom(iatom_type(j),iatom_type(i),2)
        w22F_1=(wp_atom(iatom_type(j),iatom_type(i),1)+wp_atom(iatom_type(i),iatom_type(j),1))/2     ! take the average for force calc.
        w22F_2=(wp_atom(iatom_type(j),iatom_type(i),2)+wp_atom(iatom_type(i),iatom_type(j),2))/2     ! take the average for force calc.

       yy=pi*dd/(4*rad)
! c       dE=dE+0.5*w22*exp((1-dd/rad)*4.0)*cos(yy)**2
! c       dEdd=w22*exp((1-dd/rad)*4.d0)*((-4/rad)*cos(yy)**2
! c     &   -(pi/(2*rad))*cos(yy)*sin(yy))
       dE=dE+0.5*4*(w22_1*(rad/dd)**12*cos(yy)**2+w22_2*(rad/dd)**6*cos(yy)**2)
       dEdd=4*(w22F_1*(-12*(rad/dd)**12/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)   &
      +W22F_2*(-6*(rad/dd)**6/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**6))


       dFx=dFx-dEdd*dx/dd       ! note, -sign, because dx=d(j)-x(i)
       dFy=dFy-dEdd*dy/dd
       dFz=dFz-dEdd*dz/dd
       endif
       endif
       enddo

       energy_pred_tmp(i)=energy_pred_tmp(i)+dE
       force_pred_tmp(1,i)=force_pred_tmp(1,i)+dFx   ! Note, assume force=dE/dx, no minus sign
       force_pred_tmp(2,i)=force_pred_tmp(2,i)+dFy
       force_pred_tmp(3,i)=force_pred_tmp(3,i)+dFz

       endif
       enddo


!ccccccccccccccccccccccccccccccccccccccccccc

       call mpi_allreduce(energy_pred_tmp,energy_pred_lin,natom,MPI_REAL8,MPI_SUM,MPI_COMM_WORLD,ierr)
       call mpi_allreduce(force_pred_tmp,force_pred_lin,3*natom,MPI_REAL8,MPI_SUM,MPI_COMM_WORLD,ierr)



!ccccccccccccccccccccccccccccccccccccccccccc
!    do j=1,add_force_num
        
!            
!        if (axis.eq.0) force_pred(1,add_force_atom(j))= force_pred(1,add_force_atom(j))+(direction(j)-1)*const_f(j)   !give a force on x axis
!        if (axis.eq.1) force_pred(2,add_force_atom(j))= force_pred(2,add_force_atom(j))+(direction(j)-1)*const_f(j)
!        if (axis.eq.2) force_pred(3,add_force_atom(j))= force_pred(3,add_force_atom(j))+(direction(j)-1)*const_f(j)
           
!    enddo
!ccccccccccccccccccccccccccccccccccccccccccc
        
        etot_pred_lin = 0.d0
        do i = 1, natom
            !etot = etot + energy(i)
            etot_pred_lin = etot_pred_lin + energy_pred_lin(i)
        end do

          
        
        deallocate(feat2)
        deallocate(feat_type)
        deallocate(feat2_type)
        deallocate(ind_type)
        deallocate(dfeat_type)
        deallocate(dfeat2_type)
        deallocate(dfeat2)
        ! deallocate(dfeat)
    end subroutine cal_energy_force_lin
