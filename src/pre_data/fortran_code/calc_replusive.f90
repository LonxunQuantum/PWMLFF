module calc_rep
    !implicit double precision (a-h, o-z)
    implicit none
    
    integer(4) :: natom                                    !image的原子个数  
     
    
    real*8,allocatable,dimension(:,:) :: force       !每个原子的受力
    real*8, allocatable,dimension(:) :: energy        !每个原子的能量
    real*8,allocatable,dimension(:) :: rad_atom,E_ave_vdw
    real*8,allocatable,dimension(:,:,:) :: wp_atom
    contains


    subroutine read_wp(fit_dir_input,ntype)
        character(*),intent(in) :: fit_dir_input
        integer(4),intent(in) :: ntype
        character(:),allocatable :: fit_dir,fit_dir_simp
        integer(4) :: len_dir
        integer(4) :: i,itype1,j1
        integer(4) :: ntype_t,nterm,itype_t
        character(80),parameter :: vdw_path0="vdw_fitB.ntype"
        character(200) :: vdw_path=trim(vdw_path0)
        fit_dir_simp=trim(adjustl(fit_dir_input))
        len_dir=len(fit_dir_simp)
        if (len_dir/=0 .and. fit_dir_simp/='.')then
            if (fit_dir_simp(len_dir:len_dir)=='/')then
                fit_dir=fit_dir_simp(:len_dir-1)
            else
                fit_dir=fit_dir_simp
            end if
            vdw_path=fit_dir//'/'//trim(vdw_path0)
        end if
        if (allocated(rad_atom)) then
            deallocate (rad_atom)
            deallocate (wp_atom)
            deallocate (E_ave_vdw)
        end if

        allocate(rad_atom(ntype))
        allocate(E_ave_vdw(ntype))
        allocate(wp_atom(ntype,ntype,2))
        wp_atom=0.0
        open(10,file=trim(vdw_path))
        rewind(10)
        read(10,*) ntype_t,nterm
        if(nterm.gt.2) then
        write(6,*) "nterm.gt.2,stop"
        stop
        endif
        if(ntype_t.ne.ntype) then
        write(6,*) "ntype not same in vwd_fitB.ntype,something wrong"
        stop
        endif
         do itype1=1,ntype
         read(10,*) itype_t,rad_atom(itype1),E_ave_vdw(itype1),((wp_atom(i,itype1,j1),i=1,ntype),j1=1,nterm)
        enddo
        close(10)

    end subroutine read_wp

    subroutine deallo()
        deallocate(energy)
        deallocate(force)
        ! deallocate(feat)
        ! ! deallocate(num_neigh)
        ! deallocate(dfeat)
        ! deallocate(list_neigh)
        ! deallocate(iatom)
    end subroutine deallo
    
   
    subroutine calc_replusive(num_neigh,list_neigh,AL,xatom,iatom_type)
        integer(4)  :: image,nimage,nfeat0_tmp,jj,num_tmp,ii,i_p,i,j

        ! real(8),dimension(:),intent(in) :: rad_atom, wp_atom 
        integer(4),dimension(:), intent(in) :: iatom_type
        real(8) :: dE,dEdd,dFx,dFy,dFz, rad1,rad2,rad, dd,yy, dx1,dx2,dx3,dx,dy,dz, pi,w22_1,w22_2,w22F_1,w22F_2
        ! integer(4),allocatable, dimension (:) :: iatom_type
        integer(4),dimension(:),intent(in) :: num_neigh
        integer(4),dimension(:,:),intent(in) :: list_neigh
        real(8), intent(in) :: AL(3,3)
        ! integer(4),dimension(:),intent(in) :: num_neigh    
        real*8,dimension(:,:),intent(in) :: xatom
     
        natom = size(num_neigh)

         allocate(energy(natom))
         allocate(force(3,natom))
         energy=0.d0
         force=0.d0


            pi=4*datan(1.d0)

            do i=1,natom
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
         !       write(6,"(2(i4,1x),3(f10.5,1x),2x,f13.6)") i,j,dx1,dx2,dx3,dd
        !         w22=dsqrt(wp_atom(iatom_type(i))*wp_atom(iatom_type(j)))
        !         yy=pi*dd/(4*rad)
        !  !       dE=dE+0.5*w22*exp((1-dd/rad)*4.0)*cos(yy)**2
        !  !       dEdd=w22*exp((1-dd/rad)*4.d0)*((-4/rad)*cos(yy)**2
        !  !     &   -(pi/(2*rad))*cos(yy)*sin(yy))
         
        !         dE=dE+0.5*4*w22*(rad/dd)**12*cos(yy)**2
        !         dEdd=4*w22*(-12*(rad/dd)**12/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)
                    w22_1=wp_atom(iatom_type(j),iatom_type(i),1)
                    w22_2=wp_atom(iatom_type(j),iatom_type(i),2)
                    w22F_1=(wp_atom(iatom_type(j),iatom_type(i),1)+wp_atom(iatom_type(i),iatom_type(j),1))/2     ! take the average for force calc.
                    w22F_2=(wp_atom(iatom_type(j),iatom_type(i),2)+wp_atom(iatom_type(i),iatom_type(j),2))/2     ! take the average for force calc.
            
                   yy=pi*dd/(4*rad)

                   dE=dE+0.5*4*(w22_1*(rad/dd)**12*cos(yy)**2+w22_2*(rad/dd)**6*cos(yy)**2)
                   dEdd=4*(w22F_1*(-12*(rad/dd)**12/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)   &
                  +W22F_2*(-6*(rad/dd)**6/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**6))
            
                dFx=dFx-dEdd*dx/dd       ! note, -sign, because dx=d(j)-x(i)
                dFy=dFy-dEdd*dy/dd
                dFz=dFz-dEdd*dz/dd
                endif
                endif
                enddo
         !       write(6,*) "dE,dFx",dE,dFx
                energy(i)=energy(i)+dE
                force(1,i)=force(1,i)+dFx   ! Note, assume force=dE/dx, no minus sign
                force(2,i)=force(2,i)+dFy
                force(3,i)=force(3,i)+dFz
            enddo


    end subroutine calc_replusive
  

end module calc_rep
  
  
