module read_allnn
    !implicit double precision (a-h, o-z)
    implicit none
  
    !!!!!!!!!!!!!          以下为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    integer(4) :: m_neigh                                  !模型所使用的最大近邻数(考虑这个数是否可以不用)
    integer(4) :: nfeat0m                                  !不同种原子的原始feature数目中最大者(目前似无意义)
 
    integer(4) :: dfeat_nnz_total                                !total number of non-zero element in dfeat
    integer(4) :: natom                                    !image的原子个数  
     
    
    real*8,allocatable,dimension(:,:,:) :: force_all       !每个原子的受力
    ! real*8,allocatable,dimension(:,:,:,:) :: dfeat
    real*8,allocatable,dimension(:,:,:) :: feat_all    
    real*8, allocatable,dimension(:,:) :: energy_all        !每个原子的能量
    integer(4),allocatable,dimension(:,:,:) :: list_neigh_all
    integer(4),allocatable,dimension(:,:) :: num_neigh_all
    integer(4),allocatable,dimension(:) :: iatom
    real*8,allocatable,dimension(:) :: rad_atom,E_ave_vdw
    real*8,allocatable,dimension(:,:,:) :: wp_atom

    integer(4),allocatable, dimension (:,:) :: iat_tmp_all,jneigh_tmp_all,ifeat_tmp_all
    real*8, allocatable, dimension (:,:,:) :: dfeat_tmp_all
    integer(4),allocatable,dimension(:)  :: num_tmp_all


    !global array for the sparse storage of feature
    !real*8, allocatable, dimension (:,:) :: dfeat_sparse

    !!!!!!!!!!!!!          以上为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
    contains

    subroutine read_wp(fit_dir_input,ntype)
        character(*),intent(in) :: fit_dir_input
        integer(4),intent(in) :: ntype
        character(:),allocatable :: fit_dir,fit_dir_simp
        
        integer(4) :: len_dir
        integer(4) :: i,itype1,j1
        integer(4) :: ntype_t,nterm,itype_t
        
        character(500),parameter :: vdw_path0="vdw_fitB.ntype"
        character(500) :: vdw_path=trim(vdw_path0)
        
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
        deallocate(energy_all)
        deallocate(force_all)
        deallocate(feat_all)
        deallocate(num_neigh_all)
        deallocate(list_neigh_all)
        deallocate(iatom)
        deallocate(dfeat_tmp_all)
        deallocate(iat_tmp_all)
        deallocate(jneigh_tmp_all)
        deallocate(ifeat_tmp_all)
        deallocate(num_tmp_all)
        !deallocate(dfeat_sparse)

    end subroutine deallo



   
    subroutine read_dfeat(dfeatDir,itype_atom,feat_scale,ipos)
        integer(4)  :: image,nimage,nfeat0_tmp,jj,num_tmp,ii,i_p,i,j,num_tmp_max,ipos
        character(*),intent(in) :: dfeatDir
        real*8,  dimension (:,:),intent(in) ::feat_scale
        !real(8),dimension(:),intent(in) :: rad_atom, wp_atom 
        
        integer(4),dimension(:), intent(in) :: itype_atom
        real(8) :: dE,dEdd,dFx,dFy,dFz, rad1,rad2,rad, dd,yy, dx1,dx2,dx3,dx,dy,dz, pi,w22_1,w22_2,w22F_1,w22F_2
        integer(4)  :: iitype, itype, ntype
        integer(4),allocatable, dimension (:) :: iatom_type
        real*8 AL(3,3)
        integer nfeat1tm(100),nfeat1t(100),ntype_tmp
        
        ! real*8,allocatable,dimension(:,:,:,:) :: dfeat
        ! real*8, allocatable,dimension(:,:) :: feat        
        real*8,allocatable,dimension(:,:) :: force       !每个原子的受力
        real*8,allocatable,dimension(:,:) :: feat    
        real*8, allocatable,dimension(:) :: energy        !每个原子的能量
        integer(4),allocatable,dimension(:,:) :: list_neigh

        real*8, allocatable, dimension (:,:) :: dfeat_tmp
        integer(4),allocatable, dimension (:) :: iat_tmp,jneigh_tmp,ifeat_tmp
        integer(4),allocatable,dimension(:) :: num_neigh    
        real*8,allocatable,dimension(:,:) :: xatom
        
        integer(4) :: dfeat_nnz_idx 

        character(len=500) dfeatDirname

        dfeatDirname=trim(adjustl(dfeatDir))
        num_tmp_max=0

        !dfeat_nnz_total = 0 

        ! read the raw data to get a few parameters
        open(23,file=trim(dfeatDirname),action="read",form="unformatted",access='stream')
        rewind(23)
        read(23) nimage,natom,nfeat0_tmp,m_neigh
        read(23) ntype_tmp,(nfeat1t(ii),ii=1,ntype_tmp)
        allocate(iatom(natom))
        read(23) iatom
        ! write(*,*) nimage,natom,nfeat0_tmp,m_neigh
        ! open(99,file='log.txt')         

        nfeat0m=nfeat0_tmp
        allocate(energy(natom))
        allocate(force(3,natom))
        allocate(feat(nfeat0m,natom))
        allocate(num_neigh(natom))
        allocate(list_neigh(m_neigh,natom))
        allocate(xatom(3,natom))

        do 4333 image=1,nimage
            
            read(23) energy
            
            read(23) force

            read(23) feat

            read(23) num_neigh
            read(23) list_neigh
            !TODO:
            ! read(23) dfeat
            read(23) num_tmp
            
            ! calculate nnz
            !dfeat_nnz_total = num_tmp + dfeat_nnz_total 
            
            ! max nz number in a single image
            if (num_tmp .gt. num_tmp_max) then
                num_tmp_max=num_tmp
            end if

            ! write(*,*) num_tmp
            allocate(dfeat_tmp(3,num_tmp))
            allocate(iat_tmp(num_tmp))
            allocate(jneigh_tmp(num_tmp))
            allocate(ifeat_tmp(num_tmp))
            
            read(23) iat_tmp
            read(23) jneigh_tmp
            read(23) ifeat_tmp
            read(23) dfeat_tmp
            
            read(23) xatom    ! xatom(3,natom)
            read(23) AL       ! AL(3,3)

            !the image number idx is a "compressed" array
            deallocate(dfeat_tmp)
            deallocate(iat_tmp)
            deallocate(jneigh_tmp)
            deallocate(ifeat_tmp)

        4333 continue

        deallocate(energy)
        deallocate(force)
        deallocate(feat)
        deallocate(num_neigh)
        deallocate(list_neigh)
        ! deallocate(iatom)
        deallocate(xatom)

        close(23)

        ! do it again
        open(23,file=trim(dfeatDirname),action="read",form="unformatted",access='stream')
        rewind(23)
        read(23) nimage,natom,nfeat0_tmp,m_neigh
        read(23) ntype_tmp,(nfeat1t(ii),ii=1,ntype_tmp)
        read(23) iatom

        
        allocate(energy_all(natom,nimage))
        allocate(force_all(3,natom,nimage))
        allocate(feat_all(nfeat0m,natom,nimage))
        allocate(num_neigh_all(natom,nimage))
        allocate(list_neigh_all(m_neigh,natom,nimage))
        allocate(num_tmp_all(nimage))
        
        ! num_tmp_max is used here 
        allocate(dfeat_tmp_all(3,num_tmp_max,nimage))
        
        allocate(iat_tmp_all(num_tmp_max,nimage))
        allocate(jneigh_tmp_all(num_tmp_max,nimage))
        allocate(ifeat_tmp_all(num_tmp_max,nimage))


        !dfeat that is going to be saved in the memory
        !allocate(dfeat_sparse(3,dfeat_nnz_total))
        dfeat_nnz_idx = 1 

        !this operation makes dfeat_tmp_all non-sparse
        dfeat_tmp_all(:,:,:)=0.0
        
        iat_tmp_all(:,:)=0
        jneigh_tmp_all(:,:)=0
        ifeat_tmp_all(:,:)=0     
        nfeat0m=nfeat0_tmp


        do 3000 image=1,nimage
            ! write(*,*) image
            allocate(energy(natom))
            allocate(force(3,natom))
            allocate(feat(nfeat0m,natom))
            allocate(num_neigh(natom))
            allocate(list_neigh(m_neigh,natom))
            allocate(xatom(3,natom))
        
            !place holders
            read(23) energy
            
            read(23) force

            read(23) feat

            read(23) num_neigh
            read(23) list_neigh

            num_neigh_all(:,image)=num_neigh(:)
            
            ! read(23) dfeat
            read(23) num_tmp
            num_tmp_all(image)=num_tmp

            ! write(*,*) num_tmp
            allocate(dfeat_tmp(3,num_tmp))
            allocate(iat_tmp(num_tmp))
            allocate(jneigh_tmp(num_tmp))
            allocate(ifeat_tmp(num_tmp))
            
            read(23) iat_tmp

            ! write(*,*) iat_tmp(3)
            read(23) jneigh_tmp
            read(23) ifeat_tmp
            read(23) dfeat_tmp
            read(23) xatom    ! xatom(3,natom)
            read(23) AL       ! AL(3,3)

            ntype=size(itype_atom)
            allocate(iatom_type(natom))

            do i=1,natom
                iitype=0
                do itype=1,ntype
                    if(itype_atom(itype).eq.iatom(i)) then
                        iitype=itype
                    endif
                enddo
                if(iitype.eq.0) then
                    write(6,*) "this type not found", iatom(i)
                endif
                iatom_type(i)=iitype
            enddo

            ! looping over non-zero elements 
            
            !do jj=1,num_tmp
            !    itype=iatom_type(list_neigh(jneigh_tmp(jj),iat_tmp(jj)))
            !    dfeat_tmp(1,jj)=dfeat_tmp(1,jj)*feat_scale(ifeat_tmp(jj)+ipos,itype)
            !    dfeat_tmp(2,jj)=dfeat_tmp(2,jj)*feat_scale(ifeat_tmp(jj)+ipos,itype)
            !    dfeat_tmp(3,jj)=dfeat_tmp(3,jj)*feat_scale(ifeat_tmp(jj)+ipos,itype)
            !enddo
            
            ! dfeat_tmp_all (3, dfeat value index  , image index )
            do i=1,num_tmp
                dfeat_tmp_all(:,i,image)=dfeat_tmp(:,i)
                iat_tmp_all(i,image)=iat_tmp(i)
                jneigh_tmp_all(i,image)=jneigh_tmp(i)
                ifeat_tmp_all(i,image)=ifeat_tmp(i)
                
            enddo   

            ! loading all values in *_sparse 
            !do i=1,num_tmp
                !dfeat_sparse(:,dfeat_nnz_idx) = dfeat_tmp(:,i)
                !update global index
                !dfeat_nnz_idx = dfeat_nnz_idx + 1
            !end do

            deallocate(dfeat_tmp)
            deallocate(iat_tmp)
            deallocate(jneigh_tmp)
            deallocate(ifeat_tmp)

            ! good until here 

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
                energy(i)=energy(i)-dE
                force(1,i)=force(1,i)-dFx   ! Note, assume force=dE/dx, no minus sign
                force(2,i)=force(2,i)-dFy
                force(3,i)=force(3,i)-dFz

            enddo

            do i=1,natom
                energy_all(i,image)=energy(i)
                force_all(:,i,image)=force(:,i)
                feat_all(:,i,image)=feat(:,i)
                ii=num_neigh(i)+1
                list_neigh(ii:,i)=0
                list_neigh_all(:,i,image)=list_neigh(:,i)
            enddo
            
            deallocate(iatom_type)

            deallocate(energy)
            deallocate(force)
            deallocate(feat)
            deallocate(num_neigh)
            deallocate(list_neigh)
            deallocate(xatom)

        3000 continue
    
        ! deallocate(feat)
        ! deallocate(energy)
        ! deallocate(force)
        ! deallocate(feat)
        ! deallocate(num_neigh)
        ! deallocate(list_neigh)
        ! deallocate(iatom)
        ! deallocate(xatom)
        close(23)

    end subroutine read_dfeat

    subroutine read_dfeat_singleimg(dfeatDir,itype_atom,ipos)

        integer(4)  :: image,nimage,nfeat0_tmp,jj,num_tmp,ii,i_p,i,j,num_tmp_max,ipos
        character(*),intent(in) :: dfeatDir
        !real*8,  dimension (:,:),intent(in) ::feat_scale
        !real(8),dimension(:),intent(in) :: rad_atom, wp_atom 
        
        integer(4),dimension(:), intent(in) :: itype_atom
        real(8) :: dE,dEdd,dFx,dFy,dFz, rad1,rad2,rad, dd,yy, dx1,dx2,dx3,dx,dy,dz, pi,w22_1,w22_2,w22F_1,w22F_2
        integer(4)  :: iitype, itype, ntype
        integer(4),allocatable, dimension (:) :: iatom_type
        real*8 AL(3,3)
        integer nfeat1tm(100),nfeat1t(100),ntype_tmp
        
        ! real*8,allocatable,dimension(:,:,:,:) :: dfeat
        ! real*8, allocatable,dimension(:,:) :: feat        
        real*8,allocatable,dimension(:,:) :: force       !每个原子的受力
        real*8,allocatable,dimension(:,:) :: feat    
        real*8, allocatable,dimension(:) :: energy        !每个原子的能量
        integer(4),allocatable,dimension(:,:) :: list_neigh

        real*8, allocatable, dimension (:,:) :: dfeat_tmp
        integer(4),allocatable, dimension (:) :: iat_tmp,jneigh_tmp,ifeat_tmp
        integer(4),allocatable,dimension(:) :: num_neigh    
        real*8,allocatable,dimension(:,:) :: xatom
        
        character(len=500) dfeatDirname

        dfeatDirname=trim(adjustl(dfeatDir))
        num_tmp_max=0

        !dfeat_nnz_total = 0 
        !-----------------------------------------------
        !   read the raw data to get a few parameters
        !-----------------------------------------------

        open(23,file=trim(dfeatDirname),action="read",form="unformatted",access='stream')
        rewind(23)
        read(23) nimage,natom,nfeat0_tmp,m_neigh
        read(23) ntype_tmp,(nfeat1t(ii),ii=1,ntype_tmp)
        allocate(iatom(natom))
        read(23) iatom
        ! write(*,*) nimage,natom,nfeat0_tmp,m_neigh
        ! open(99,file='log.txt')         

        nfeat0m=nfeat0_tmp
        allocate(energy(natom))
        allocate(force(3,natom))
        allocate(feat(nfeat0m,natom))
        allocate(num_neigh(natom))
        allocate(list_neigh(m_neigh,natom))
        allocate(xatom(3,natom))

        !write(*,*) "memory allocation done"

        do 4333 image=1,nimage
            
            read(23) energy
            read(23) force
            read(23) feat
            read(23) num_neigh
            read(23) list_neigh
            read(23) num_tmp
            
            ! max nz number in a single image
            if (num_tmp .gt. num_tmp_max) then
                num_tmp_max=num_tmp
            end if

            ! write(*,*) num_tmp
            allocate(dfeat_tmp(3,num_tmp))
            allocate(iat_tmp(num_tmp))
            allocate(jneigh_tmp(num_tmp))
            allocate(ifeat_tmp(num_tmp))
            
            read(23) iat_tmp
            read(23) jneigh_tmp
            read(23) ifeat_tmp
            read(23) dfeat_tmp

            read(23) xatom    ! xatom(3,natom)
            read(23) AL       ! AL(3,3)
        
            !the image number idx is a "compressed" array
            deallocate(dfeat_tmp)
            deallocate(iat_tmp)
            deallocate(jneigh_tmp)
            deallocate(ifeat_tmp)

        4333 continue

        deallocate(energy)
        deallocate(force)
        deallocate(feat)
        deallocate(num_neigh)
        deallocate(list_neigh)
        deallocate(xatom)

        close(23)

        !-------------------------
        !   form dfeat_tmp_all   
        !-------------------------       

        open(23,file=trim(dfeatDirname),action="read",form="unformatted",access='stream')

        rewind(23)
        read(23) nimage,natom,nfeat0_tmp,m_neigh
        read(23) ntype_tmp,(nfeat1t(ii),ii=1,ntype_tmp)
        read(23) iatom

        ! arrays below are useless 

        !allocate(energy_all(natom,nimage))
        !allocate(force_all(3,natom,nimage))
        !allocate(feat_all(nfeat0m,natom,nimage))

        allocate(num_neigh_all(natom,nimage))
        
        !allocate(list_neigh_all(m_neigh,natom,nimage))
        ! num_tmp_max is used here 
        
        allocate(num_tmp_all(nimage))
        allocate(dfeat_tmp_all(3,num_tmp_max,nimage))
        allocate(iat_tmp_all(num_tmp_max,nimage))
        allocate(jneigh_tmp_all(num_tmp_max,nimage))
        allocate(ifeat_tmp_all(num_tmp_max,nimage))

        !dfeat that is going to be saved in the memory
        !allocate(dfeat_sparse(3,dfeat_nnz_total))

        !this operation makes dfeat_tmp_all non-sparse
        dfeat_tmp_all(:,:,:)=0.0        
        iat_tmp_all(:,:)=0
        jneigh_tmp_all(:,:)=0
        ifeat_tmp_all(:,:)=0   

        nfeat0m=nfeat0_tmp  

        do image=1,nimage
            
            allocate(energy(natom))
            allocate(force(3,natom))
            allocate(feat(nfeat0m,natom))
            allocate(num_neigh(natom))
            allocate(list_neigh(m_neigh,natom))
            allocate(xatom(3,natom))
        
            !place holders
            read(23) energy
            read(23) force
            read(23) feat
            read(23) num_neigh
            read(23) list_neigh

            num_neigh_all(:,image)=num_neigh(:) 

            read(23) num_tmp

            num_tmp_all(image)=num_tmp

            ! write(*,*) num_tmp
            allocate(dfeat_tmp(3,num_tmp))
            allocate(iat_tmp(num_tmp))
            allocate(jneigh_tmp(num_tmp))
            allocate(ifeat_tmp(num_tmp))
            
            read(23) iat_tmp
            ! write(*,*) iat_tmp(3)
            read(23) jneigh_tmp
            read(23) ifeat_tmp
            read(23) dfeat_tmp
            read(23) xatom    ! xatom(3,natom)
            read(23) AL       ! AL(3,3)

            ntype=size(itype_atom)

            allocate(iatom_type(natom))

            do i=1,natom
                iitype=0
                do itype=1,ntype
                    if(itype_atom(itype).eq.iatom(i)) then
                        iitype=itype
                    endif
                enddo
                if(iitype.eq.0) then
                    write(6,*) "this type not found", iatom(i)
                endif
                iatom_type(i)=iitype
            enddo

            ! looping over non-zero elements 
            
            !do jj=1,num_tmp
            !    itype=iatom_type(list_neigh(jneigh_tmp(jj),iat_tmp(jj)))
            !    dfeat_tmp(1,jj)=dfeat_tmp(1,jj)*feat_scale(ifeat_tmp(jj)+ipos,itype)
            !    dfeat_tmp(2,jj)=dfeat_tmp(2,jj)*feat_scale(ifeat_tmp(jj)+ipos,itype)
            !    dfeat_tmp(3,jj)=dfeat_tmp(3,jj)*feat_scale(ifeat_tmp(jj)+ipos,itype)
            !enddo
            
            ! dfeat_tmp_all (3, dfeat value index  , image index )
            do i=1,num_tmp
                dfeat_tmp_all(:,i,image)=dfeat_tmp(:,i)
                iat_tmp_all(i,image)=iat_tmp(i)
                jneigh_tmp_all(i,image)=jneigh_tmp(i)
                ifeat_tmp_all(i,image)=ifeat_tmp(i)
                
            enddo   

            deallocate(energy)
            deallocate(force)
            deallocate(feat)
            deallocate(num_neigh)
            deallocate(list_neigh)
            deallocate(xatom)

            deallocate(iatom_type)

            deallocate(dfeat_tmp)
            deallocate(iat_tmp)
            deallocate(jneigh_tmp)
            deallocate(ifeat_tmp)

        enddo

        deallocate(num_neigh_all)

        close(23)

    end subroutine read_dfeat_singleimg

    subroutine deallo_singleimg()

        !deallocate(energy_all)
        !deallocate(force_all)
        !deallocate(feat_all)
        !deallocate(num_neigh_all)
        !deallocate(list_neigh_all)

        deallocate(iatom)

        !deallocate(dfeat_tmp_all)
        !deallocate(iat_tmp_all)
        !deallocate(jneigh_tmp_all)
        !deallocate(ifeat_tmp_all)
        !deallocate(num_tmp_all)    
        
    end subroutine deallo_singleimg


    subroutine deallocate_dfeat_tmp_all()
        deallocate(dfeat_tmp_all)
    end subroutine deallocate_dfeat_tmp_all


    subroutine deallocate_iat_tmp_all()
        deallocate(iat_tmp_all)
    end subroutine deallocate_iat_tmp_all


    subroutine deallocate_jneigh_tmp_all() 
        deallocate(jneigh_tmp_all)
    end subroutine deallocate_jneigh_tmp_all


    subroutine deallocate_ifeat_tmp_all() 
        deallocate(ifeat_tmp_all) 
    end subroutine deallocate_ifeat_tmp_all


    subroutine deallocate_num_tmp_all()
        deallocate(num_tmp_all)
    end subroutine deallocate_num_tmp_all




end module read_allnn