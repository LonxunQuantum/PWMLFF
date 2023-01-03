module calc_NN

    use mod_mpi
    
    implicit none

    !**********************module varaiables*************************
    
    character(80),parameter :: feat_info_path0="fread_dfeat/feat.info"
    character(80),parameter :: model_Wij_path0="fread_dfeat/Wij.txt"
    character(80),parameter :: model_Scaler_path0="fread_dfeat/data_scaler.txt"
    character(80),parameter :: vdw_path0="fread_dfeat/vdw_fitB.ntype"
    
    character(200) :: feat_info_path=trim(feat_info_path0)
    character(200) :: model_Wij_path=trim(model_Wij_path0)
    character(200) :: model_Scaler_path=trim(model_Scaler_path0)
    character(200) :: vdw_path=trim(vdw_path0)
  
    integer(4) :: ntype                                    !模型所有涉及的原子种类
    integer(4) :: m_neigh                                  !模型所使用的最大近邻数(考虑这个数是否可以不用)
    integer(4) :: nfeat1m                                  !不同种原子的原始feature数目中最大者(目前似无意义)
    integer(4),allocatable,dimension(:) :: nfeat1          !各种原子的原始feature数目
  
    real(8),allocatable,dimension(:) :: bb                 !计算erergy和force时与new feature相乘的系数向量w
    real(8),allocatable,dimension(:,:) :: bb_type          !不明白有何作用,似乎应该是之前用的变量
    real(8),allocatable,dimension(:,:) :: bb_type0         !将bb分别归类到不同种类的原子中，第二维才是代表原子种类
    real(8),allocatable,dimension (:, :) :: w_feat         !不同reference points的权重(对linear无意义)

    integer(4) :: natom                                    !image的原子个数  
    integer(4),allocatable,dimension(:) :: num             !属于每种原子的原子个数，但似乎在calc_linear中无用
    integer(4),allocatable,dimension(:) :: num_atomtype    !属于每种原子的原子个数，似是目前所用的
    integer(4),allocatable,dimension(:) :: itype_atom      !每一种原子的原子属于第几种原子
    integer(4),allocatable,dimension(:) :: iatom           !每种原子的原子序数列表，即atomTypeList
    integer(4),allocatable,dimension(:) :: iatom_type      !每种原子的种类，即序数在种类列表中的序数
    
    real(8),allocatable,dimension(:) :: energy_pred_NN       !每个原子的能量预测值
    real(8),allocatable,dimension(:) :: energy_pred_tmp        !每个原子的能量预测值
    real(8),allocatable,dimension(:,:) :: force_pred_NN       !每个原子的受力预测值
    real(8),allocatable,dimension(:,:) :: force_pred_tmp       !每个原子的受力预测值
    real(8) :: etot_pred_NN
    character(200) :: error_msg
    integer(4) :: istat
    real(8), allocatable, dimension(:) ::  const_fa,const_fb,const_fc,const_fx,const_fy,const_fz
    integer(4),allocatable, dimension(:) :: direction,add_force_atom,const_force_atom
    integer(4) :: add_force_num,power,axis,const_force_num
    real(8) :: alpha, y1, z1
    ! INTEGER*4  access, status
    logical*2::alive

    ! real*8,allocatable,dimension(:) :: rad_atom,wp_atom
    real*8,allocatable,dimension(:) :: rad_atom,E_ave_vdw
    real*8,allocatable,dimension(:,:,:) :: wp_atom
    integer(4) :: nfeat1tm(100),ifeat_type_n(100),nfeat1t(100)
    integer(4) :: nfeat_type_n
    real*8, allocatable,dimension(:,:) :: a_scaler,b_scaler
    integer, allocatable,dimension(:,:) :: nodeNN
    real*8, allocatable,dimension(:,:,:,:) :: Wij_nn
    real*8, allocatable,dimension(:,:,:) :: B_nn
    integer nodeMM,nlayer
    integer ifeat_type(100)
    
    contains
    
    subroutine set_paths_NN(fit_dir_input)
        character(*),intent(in) :: fit_dir_input
        character(:),allocatable :: fit_dir,fit_dir_simp
        integer(4) :: len_dir
        fit_dir_simp=trim(adjustl(fit_dir_input))
        len_dir=len(fit_dir_simp)

        if (len_dir/=0 .and. fit_dir_simp/='.') then
            if (fit_dir_simp(len_dir:len_dir)=='/') then
                fit_dir=fit_dir_simp(:len_dir-1)
            else
                fit_dir=fit_dir_simp
            end if
            
            feat_info_path=fit_dir//'/'//trim(feat_info_path0)
            model_Wij_path=fit_dir//'/'//trim(model_Wij_path0)
            model_Scaler_path=fit_dir//'/'//trim(model_Scaler_path0)
            vdw_path=fit_dir//'/'//trim(vdw_path0)

        end if
    end subroutine set_paths_NN
    
    subroutine load_model_NN()
    
        integer(4) :: nimage,num_refm,num_reftot,nfeat1_tmp,itype,i,k,itmp,itype1,j1
        integer(4) :: iflag_PCA,kkk,ntype_tmp,iatom_tmp,ntype_t,nterm,itype_t
        real(8) :: dist0
        character*20 txt
        integer i1,i2,j,ii,ntmp,nlayer_tmp,j2
        real*8 w_tmp,b_tmp

        ! integer(4),allocatable,dimension(:,:) :: nfeat,ipos_feat


        ! **************** read feat.info ********************
        open(10,file=trim(feat_info_path))
        rewind(10)
        read(10,*) iflag_PCA   ! this can be used to turn off degmm part
        read(10,*) nfeat_type_n
        
        do kkk=1,nfeat_type_n
          read(10,*) ifeat_type_n(kkk)   ! the index (1,2,3) of the feature type
        enddo
        
        read(10,*) ntype,m_neigh

        close(10)
            
        !  m_neight get from gen_feature input file
        !   just to get ntype,m_neight, will read again

        ! **************** read fit_linearMM.input ********************    
        if (allocated(itype_atom)) deallocate(itype_atom)
        if (allocated(nfeat1)) deallocate(nfeat1)
        if (allocated(rad_atom)) deallocate(rad_atom)
        if (allocated(wp_atom)) deallocate(wp_atom)
        if (allocated(E_ave_vdw)) deallocate(E_ave_vdw)
        if (allocated(num)) deallocate(num)                              !image数据,在此处allocate，但在set_image_info中赋值
        if (allocated(num_atomtype)) deallocate(num_atomtype)                     !image数据,在此处allocate，但在set_image_info中赋值
        if (allocated(bb)) deallocate(bb)
        if (allocated(bb_type)) deallocate(bb_type)
        if (allocated(bb_type0)) deallocate(bb_type0)
        if (allocated(add_force_atom)) deallocate(add_force_atom)
        if (allocated(const_fa)) deallocate(const_fa)
        if (allocated(const_fb)) deallocate(const_fb)
        if (allocated(const_fc)) deallocate(const_fc)
        
        allocate (itype_atom(ntype))
        allocate (nfeat1(ntype))
        
        allocate (num(ntype))                              !image数据,在此处allocate，但在set_image_info中赋值
        allocate (num_atomtype(ntype))                     !image数据,在此处allocate，但在set_image_info中赋值
        ! allocate (rad_atom(ntype))
        ! allocate (wp_atom(ntype))
        allocate(rad_atom(ntype))
        allocate(E_ave_vdw(ntype))
        allocate(wp_atom(ntype,ntype,2))
        wp_atom=0.d0

        !ccccccccccccccccccccccccccccccccccccccccc
        open(10,file=trim(feat_info_path))
        rewind(10)
        read(10,*) iflag_PCA   ! this can be used to turn off degmm part
        read(10,*) nfeat_type_n
        do kkk=1,nfeat_type_n
            read(10,*) ifeat_type_n(kkk)   ! the index (1,2,3) of the feature type
        enddo
        read(10,*) ntype,m_neigh
        
        do i=1,ntype
            read(10,*) itype_atom(i),nfeat1(i)   ! these nfeat1,nfeat2 include all ftype
        enddo
        close(10)

        nfeat1m=0   ! the original feature
        do i=1,ntype
           if(nfeat1(i).gt.nfeat1m) nfeat1m=nfeat1(i)
        enddo
        ! **************** read fit_linearMM.input ********************    
        ! ****************** read vdw ************************
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


        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        open (12, file=trim(model_Wij_path))
        rewind (12)
        read(12,*)
        read(12,*)
        read(12,*)
        read(12,*) txt,nlayer_tmp

        nlayer=nlayer_tmp/2/ntype
        if(nlayer_tmp.ne.nlayer*2*ntype) then
            write(6,*) "ERROR! nlayer is wrong",nlayer_tmp
            stop
        endif 
        if (.not. allocated(nodeNN)) then
            allocate(nodeNN(nlayer+1,ntype))
        end if

        do itype=1,ntype
            do ii=1,nlayer
                read(12,*) txt,nodeNN(ii,itype),nodeNN(ii+1,itype)
                do j=1,nodeNN(ii,itype)*nodeNN(ii+1,itype)
                    read(12,*) 
                enddo
            enddo
        enddo
        close(12)   ! just to get nodeNN

        nodeMM=0
        do itype=1,ntype
            do ii=1,nlayer+1
                if(nodeNN(ii,itype).gt.nodeMM) nodeMM=nodeNN(ii,itype)
            enddo
        enddo

        if (.not. allocated(a_scaler)) then
            allocate(a_scaler(nfeat1m,ntype))
            allocate(b_scaler(nfeat1m,ntype))
            allocate(Wij_nn(nodeMM,nodeMM,nlayer,ntype))
            allocate(B_nn(nodeMM,nlayer,ntype))
        end if

        !cccccccccccccccccccccccccccccccccccccccccccccccccccc

        open (12, file=trim(model_Wij_path))
        rewind (12)
        read(12,*)
        read(12,*)


        read(12,*)
        read(12,*) txt,nlayer_tmp


        nlayer=nlayer_tmp/2/ntype

        ! Wij_nn(nodeMM,nodeMM,nlayer,ntype)
        ! 
        do itype=1,ntype
            do ii=1,nlayer
                read(12,*) txt,nodeNN(ii,itype),nodeNN(ii+1,itype)

                if(ii.eq.1.and.nodeNN(1,itype).ne.nfeat1(itype)) then
                    write(6,*) "nodeNN in Wij.txt not correct",nodeNN(1,itype),nfeat1(itype)
                    stop
                endif

                do j=1,nodeNN(ii,itype)*nodeNN(ii+1,itype)
                    read(12,*) i1,i2,w_tmp
                    Wij_nn(i1+1,i2+1,ii,itype)=w_tmp
                enddo

                read(12,*) 
                
                do j=1,nodeNN(ii+1,itype)
                    read(12,*) i1,i2,b_tmp
                    B_nn(j,ii,itype)=b_tmp
                enddo
            enddo
        enddo
        close(12)


        !cccccccccccccccccccccccccccccccccccccccccccccccccccccc

        open (12, file=trim(model_Scaler_path))
        rewind (12)
        read(12,*) 
        read(12,*) 
        read(12,*) 
        read(12,*) txt,ntmp

        if(ntmp.ne.ntype*2) then
            write(6,*) "size not right in data_scale.txt",ii
            stop
        endif

        do itype=1,ntype
            read(12,*) txt,ntmp

            if(ntmp.ne.nfeat1(itype)) then
                write(6,*) "nfeat size not correct in data_scale.txt",ntmp
                stop
            endif

            do j=1,ntmp
                read(12,*) j1,j2, a_scaler(j,itype)
            enddo

            read(12,*) txt,ntmp

            if(ntmp.ne.nfeat1(itype)) then
                write(6,*) "nfeat size not correct in data_scale.txt",ntmp
                stop
            endif

            do j=1,ntmp
                read(12,*) j1,j2,b_scaler(j,itype)
            enddo
        enddo
   
        close(12)

        !********************add_force****************
        inquire(file='add_force',exist=alive)
        !     status = access ("add_force",' ')    ! blank mode
        !   if (status .eq. 0 ) then
        if (alive) then
            open(10,file="add_force")
            rewind(10)
            read(10,*) add_force_num, alpha,y1,z1
            allocate(add_force_atom(add_force_num))
            ! allocate(direction(add_force_num))
            allocate(const_fa(add_force_num))
            allocate(const_fb(add_force_num))
            allocate(const_fc(add_force_num))

            do i=1,add_force_num
                read(10,*) add_force_atom(i), const_fa(i), const_fb(i),const_fc(i)
            enddo

            close(10)
        else
            add_force_num=0
        endif
        !********************
        inquire(file='force_constraint',exist=alive)
        !   status = access ("add_force",' ')    ! blank mode
        !   if (status .eq. 0 ) then
        if (alive) then
            open(10,file="force_constraint")
            rewind(10)
            read(10,*) const_force_num
            allocate(const_force_atom(const_force_num))
            ! allocate(direction(add_force_num))
            allocate(const_fx(const_force_num))
            allocate(const_fy(const_force_num))
            allocate(const_fz(const_force_num))

            do i=1,const_force_num
                read(10,*) const_force_atom(i), const_fx(i), const_fy(i), const_fz(i)
            enddo

            close(10)
        else
            const_force_num=0
        endif

    end subroutine load_model_NN
  
    subroutine set_image_info_NN(iatom_tmp,is_reset,natom_tmp)
        integer(4) :: i,j,itype,iitype
        integer iatom_tmp(natom_tmp)
        logical,intent(in) :: is_reset
        integer(4) :: image_size
        integer :: natom_tmp
        
        image_size=natom_tmp
        if (is_reset .or. (.not. allocated(iatom)) .or. image_size/=natom) then
        
            if (allocated(iatom))then
                if (image_size==natom .and. maxval(abs(iatom_tmp-iatom))==0) then
                    return
                end if
                deallocate(iatom)
                deallocate(iatom_type)
                deallocate(energy_pred_NN)
                deallocate(energy_pred_tmp)
                deallocate(force_pred_NN)
                deallocate(force_pred_tmp)
            end if
              
            natom=image_size
            
            allocate(iatom(natom))
            allocate(iatom_type(natom))
            allocate(energy_pred_NN(natom))
            allocate(energy_pred_tmp(natom))
            allocate(force_pred_NN(3,natom))
            allocate(force_pred_tmp(3,natom))
            !allocate()
            
            iatom=iatom_tmp
              
            do i = 1, natom
                iitype = 0

                do itype = 1, ntype
                    if (itype_atom(itype)==iatom(i)) then
                        iitype = itype
                    end if
                end do
                
                if (iitype==0) then
                    write (6, *) 'this type not found', iatom(i)
                end if

                !write (*,*) "iitype", iitype 
                
                iatom_type(i) = iitype
            end do
      
            num_atomtype = 0
            do i = 1, natom
                itype = iatom_type(i)
                num_atomtype(itype) = num_atomtype(itype) + 1
            end do
        end if
        
    end subroutine set_image_info_NN
    
    subroutine cal_energy_NN(num_neigh,list_neigh,AL,xatom,natom_tmp,nfeat0_tmp,m_neigh_tmp)
        
        !***********************************
        !   Stress tensor calculation 
        !   need to re-calculate feat with input AL
        !   
        !***********************************
        use mod_data, only : iflag_model
        use mod_mpi

        use calc_ftype1, only : feat_M1,dfeat_M1,nfeat0M1,gen_feature_type1,  &
                 nfeat0M1,num_neigh_alltypeM1,list_neigh_alltypeM1,  &
                 natom1,m_neigh1
        use calc_ftype2, only : feat_M2,dfeat_M2,nfeat0M2,gen_feature_type2,  &
                 nfeat0M2,num_neigh_alltypeM2,list_neigh_alltypeM2,natom2,m_neigh2
        use calc_2bgauss_feature, only : feat_M3,dfeat_M3,nfeat0M3,gen_feature_2bgauss,  &
                nfeat0M3,num_neigh_alltypeM3,list_neigh_alltypeM3,natom3,m_neigh3
        use calc_3bcos_feature, only : feat_M4,dfeat_M4,nfeat0M4,gen_3bcos_feature,  &
                nfeat0M4,num_neigh_alltypeM4,list_neigh_alltypeM4,natom4,m_neigh4
        use calc_MTP_feature, only : feat_M5,dfeat_M5,nfeat0M5,gen_MTP_feature,  &
                nfeat0M5,num_neigh_alltypeM5,list_neigh_alltypeM5,natom5,m_neigh5
        use calc_SNAP_feature, only : feat_M6,dfeat_M6,nfeat0M6,gen_SNAP_feature,  &
                nfeat0M6,num_neigh_alltypeM6,list_neigh_alltypeM6,natom6,m_neigh6
        use calc_deepMD1_feature, only : feat_M7,dfeat_M7,nfeat0M7,gen_deepMD1_feature,  &
                nfeat0M7,num_neigh_alltypeM7,list_neigh_alltypeM7,natom7,m_neigh7
        use calc_deepMD2_feature, only : feat_M8,dfeat_M8,nfeat0M8,gen_deepMD2_feature,  &
                nfeat0M8,num_neigh_alltypeM8,list_neigh_alltypeM8,natom8,m_neigh8
        
        integer(4)  :: itype,ixyz,i,j,jj
        integer natom_tmp,nfeat0_tmp,m_neigh_tmp,kk
        integer iat
        real(8) :: sum,direct,mean
        !real(8),intent(in) :: feat(nfeat0_tmp,natom_tmp)
        !real*8, intent(in) :: dfeat(nfeat0_tmp,natom_tmp,m_neigh_tmp,3)
        !real(8),intent(in) :: feat(nfeat0_tmp,natom_n)
        real(8),allocatable, dimension(:,:) :: feat
        !real*8, intent(in) :: dfeat(nfeat0_tmp,natom_n,m_neigh_tmp,3)
        

        integer(4),intent(in) :: num_neigh(natom_tmp)
        integer(4),intent(in) :: list_neigh(m_neigh_tmp,natom_tmp)
        real(8), intent(in) :: AL(3,3)
        real(8),dimension(:,:),intent(in) :: xatom
        
        real(8),allocatable,dimension(:,:,:) :: feat_type
        !real(8),allocatable,dimension(:,:,:) :: dfeat_type
        
        real(8),allocatable,dimension(:,:,:) :: f_in,f_out,f_d,f_back
        real(8),allocatable,dimension(:,:) :: energy_type
        real(8),allocatable,dimension(:,:,:) :: dEdf_type

        real*8 pi,dE,dFx,dFy,dFz
        real*8 rad1,rad2,rad,dx1,dx2,dx3,dx,dy,dz,dd,yy,w22,dEdd,d,w22_1,w22_2,w22F_1,w22F_2
        integer iat1,iat2,ierr
        integer ii
        
        integer nfeat0, count
        
        integer, allocatable, dimension (:,:) :: list_neigh_alltypeM_use
        integer nfeat_type
        real*8 x
        real*8 sum1,sum2,sum3

        integer num_neigh_alltypeM_use(natom_tmp)

        real(8) temp 

        pi=4*datan(1.d0)

        
        !write(*,*) "test natom_n=",natom_n
        
        !featType: feature index, atom index, atom type
        
        allocate(feat_type(nfeat1m,natom_n,ntype))
        allocate(f_in(nodeMM,natom_n,nlayer+1))
        allocate(f_out(nodeMM,natom_n,nlayer+1))
        !allocate(f_d(nodeMM,natom_n,nlayer+1))
        !allocate(f_back(nodeMM,natom_n,nlayer+1))
        allocate(energy_type(natom_n,ntype))
        !allocate(dEdf_type(nfeat1m,natom_n,ntype))

        ! allocate(dfeat_type(nfeat1m,natom_n*m_neigh*3,ntype))
        
        istat=0
        error_msg=''
        
        if (nfeat0_tmp/=nfeat1m .or. natom_tmp/=natom .or. m_neigh_tmp/=m_neigh) then
            write(*,*) "Shape of input arrays don't match the model!"
            write(6,*) nfeat0_tmp,natom_tmp,m_neigh_tmp
            write(6,*) nfeat1m,natom,m_neigh
            stop
        end if


        ! recalculate features 
        if ((iflag_model.eq.3)) then 
        
            !if(iflag_model.eq.3) then
            nfeat_type=nfeat_type_n
            ifeat_type=ifeat_type_n
            !endif
            
            nfeat0=0
            
            !*****************************************
            !         calculate new features 
            !*****************************************

            do kk = 1, nfeat_type
                if (ifeat_type(kk)  .eq. 1) then
                    call gen_feature_type1(AL,xatom)
                    nfeat0=nfeat0+nfeat0M1
                endif
                
                if (ifeat_type(kk)  .eq. 2) then
                    call gen_feature_type2(AL,xatom)
                    nfeat0=nfeat0+nfeat0M2
                endif
                
                if (ifeat_type(kk)  .eq. 3) then
                    call gen_feature_2bgauss(AL,xatom)
                    nfeat0=nfeat0+nfeat0M3
                endif
                
                if (ifeat_type(kk)  .eq. 4) then
                    call gen_3bcos_feature(AL,xatom)
                    nfeat0=nfeat0+nfeat0M4
                endif
                    
                if (ifeat_type(kk)  .eq. 5) then
                    call gen_MTP_feature(AL,xatom)
                    nfeat0=nfeat0+nfeat0M5
                endif
                    
                if (ifeat_type(kk)  .eq. 6) then
                    call gen_SNAP_feature(AL,xatom)
                    nfeat0=nfeat0+nfeat0M6
                endif
                
                if (ifeat_type(kk)  .eq. 7) then
                    call gen_deepMD1_feature(AL,xatom)
                    nfeat0=nfeat0+nfeat0M7
                endif
                
                if (ifeat_type(kk)  .eq. 8) then
                    call gen_deepMD2_feature(AL,xatom)
                    nfeat0=nfeat0+nfeat0M8
                endif

            enddo

            !*****************************************
            !         passing feature params 
            !*****************************************

            if (ifeat_type(1)  .eq. 1) then 
                natom=natom1
                m_neigh=m_neigh1
                num_neigh_alltypeM_use = num_neigh_alltypeM1
                if(allocated(list_neigh_alltypeM_use)) then
                    deallocate(list_neigh_alltypeM_use)
                endif
                allocate(list_neigh_alltypeM_use(m_neigh, natom))
                list_neigh_alltypeM_use = list_neigh_alltypeM1
            endif

            if (ifeat_type(1)  .eq. 2) then 
                natom=natom2   
                m_neigh=m_neigh2
                num_neigh_alltypeM_use = num_neigh_alltypeM2
                if(allocated(list_neigh_alltypeM_use)) then
                    deallocate(list_neigh_alltypeM_use)
                endif
                
                allocate(list_neigh_alltypeM_use(m_neigh, natom))
                list_neigh_alltypeM_use = list_neigh_alltypeM2
            endif    
                          
            if (ifeat_type(1)  .eq. 3) then 
                natom=natom3
                m_neigh=m_neigh3
                num_neigh_alltypeM_use = num_neigh_alltypeM3
                
                if(allocated(list_neigh_alltypeM_use)) then
                    deallocate(list_neigh_alltypeM_use)
                endif   
                
                allocate(list_neigh_alltypeM_use(m_neigh, natom))
                list_neigh_alltypeM_use = list_neigh_alltypeM3
            endif
            
            if (ifeat_type(1)  .eq. 4) then 
                natom=natom4 
                m_neigh=m_neigh4
                num_neigh_alltypeM_use = num_neigh_alltypeM4
                
                if(allocated(list_neigh_alltypeM_use)) then
                    deallocate(list_neigh_alltypeM_use)
                endif
                
                allocate(list_neigh_alltypeM_use(m_neigh, natom))
                list_neigh_alltypeM_use = list_neigh_alltypeM4
            endif

            if (ifeat_type(1)  .eq. 5) then 
                natom=natom5
                m_neigh=m_neigh5
                num_neigh_alltypeM_use = num_neigh_alltypeM5
                if(allocated(list_neigh_alltypeM_use)) then
                    deallocate(list_neigh_alltypeM_use)
                endif
            
                allocate(list_neigh_alltypeM_use(m_neigh, natom))
                list_neigh_alltypeM_use = list_neigh_alltypeM5

            endif
            
            if (ifeat_type(1)  .eq. 6) then 
                natom=natom6   
                m_neigh=m_neigh6
                num_neigh_alltypeM_use = num_neigh_alltypeM6
                
                if(allocated(list_neigh_alltypeM_use)) then
                    deallocate(list_neigh_alltypeM_use)
                endif
                
                allocate(list_neigh_alltypeM_use(m_neigh, natom))
                
                list_neigh_alltypeM_use = list_neigh_alltypeM6
            endif 

            if (ifeat_type(1)  .eq. 7) then 
                natom=natom7
                m_neigh=m_neigh7
                num_neigh_alltypeM_use = num_neigh_alltypeM7
                
                if(allocated(list_neigh_alltypeM_use)) then
                    deallocate(list_neigh_alltypeM_use)
                endif
                
                allocate(list_neigh_alltypeM_use(m_neigh, natom))
                list_neigh_alltypeM_use = list_neigh_alltypeM7
            endif

            if (ifeat_type(1)  .eq. 8) then 
                natom=natom8 
                m_neigh=m_neigh8
                num_neigh_alltypeM_use = num_neigh_alltypeM8
                if(allocated(list_neigh_alltypeM_use)) then
                    deallocate(list_neigh_alltypeM_use)
                endif
                allocate(list_neigh_alltypeM_use(m_neigh, natom))
                
                list_neigh_alltypeM_use = list_neigh_alltypeM8
            endif

            if(natom_tmp.ne.natom) then
                
                write(6,*) "natom.ne.natom_tmp,stop",natom,natom_tmp
                stop
            
            endif

            ! nfeat0=nfeat0M1+nfeat0M2

            !*******************************************
            !    Assemble different feature types
            !*******************************************
            
            allocate(feat(nfeat0,natom_n))
            !allocate(dfeat(nfeat0,natom_n,m_neigh,3))
            
            count =0
            do kk = 1, nfeat_type
                ! features that are passed into the NN
                if (ifeat_type(kk)  .eq. 1) then
                    do iat=1,natom_n
                        do ii=1,nfeat0M1
                            feat(ii+count,iat)=feat_M1(ii,iat)
                        enddo
                    enddo
                    count=count+nfeat0M1
                endif

                if (ifeat_type(kk)  .eq. 2) then
                    do iat=1,natom_n
                        do ii=1,nfeat0M2
                            feat(ii+count,iat)=feat_M2(ii,iat)
                        enddo
                    enddo
                    count=count+nfeat0M2
                endif

                if (ifeat_type(kk)  .eq. 3) then
                    do iat=1,natom_n
                        do ii=1,nfeat0M3
                            feat(ii+count,iat)=feat_M3(ii,iat)
                        enddo
                    enddo
                    count=count+nfeat0M3
                endif

                if (ifeat_type(kk)  .eq. 4) then
                    do iat=1,natom_n
                        do ii=1,nfeat0M4
                            feat(ii+count,iat)=feat_M4(ii,iat)
                        enddo
                    enddo
                    count=count+nfeat0M4
                endif

                if (ifeat_type(kk)  .eq. 5) then
                    do iat=1,natom_n
                        do ii=1,nfeat0M5
                            feat(ii+count,iat)=feat_M5(ii,iat)
                        enddo
                    enddo
                    count=count+nfeat0M5
                endif

                if (ifeat_type(kk)  .eq. 6) then
                    do iat=1,natom_n
                        do ii=1,nfeat0M6
                            feat(ii+count,iat)=feat_M6(ii,iat)
                        enddo
                    enddo
                    count=count+nfeat0M6
                endif

                if (ifeat_type(kk)  .eq. 7) then
                    do iat=1,natom_n
                        do ii=1,nfeat0M7
                            feat(ii+count,iat)=feat_M7(ii,iat)
                        enddo
                    enddo

                    count=count+nfeat0M7
                endif

                if (ifeat_type(kk)  .eq. 8) then
                    do iat=1,natom_n
                        do ii=1,nfeat0M8
                            feat(ii+count,iat)=feat_M8(ii,iat)
                        enddo
                    enddo

                    count=count+nfeat0M8
                endif
            
            enddo

        endif 

        num = 0
        iat1=0
        
        ! MPI workload setting 
        do i = 1, natom
            if(mod(i-1,nnodes).eq.inode-1) then

                iat1=iat1+1
                itype = iatom_type(i)
                num(itype) = num(itype) + 1

                do j=1,nfeat1(itype)
                    feat_type(j,num(itype),itype) = feat(j, iat1)*a_scaler(j,itype)+b_scaler(j,itype)
                enddo

            endif
        enddo

        !***********************************
        !    
        !    forward propagation starts 
        !
        !***********************************
        
        do itype=1,ntype

            do i=1,num(itype)
                do j=1,nodeNN(1,itype)

                    f_in(j,i,1)=feat_type(j,i,itype)
                
                enddo
            enddo

            do ii=1,nlayer
                
                if(ii.ne.1) then
                    do i=1,num(itype)
                        do j=1,nodeNN(ii,itype)
                            x=f_in(j,i,ii)
                            
                            if(x.gt.-150.d0.and.x.lt.150.d0) then
                                f_out(j, i, ii) = (exp(x)-exp(-x)) / (exp(x)+exp(-x))  ! tanh 
                                !f_d(j, i, ii) = 1.0d0 - f_out(j,i,ii)*f_out(j,i,ii)

                            elseif(x.le.-150.d0) then
                                 f_out(j,i,ii)=0.d0
                                 !f_d(j,i,ii)=0.d0
                            elseif(x.ge.150.d0) then 
                                 f_out(j,i,ii)=x
                                 !f_d(j,i,ii)=1.d0
                            endif
                        enddo
                    enddo
                
                elseif(ii.eq.1) then
                    do i=1,num(itype)
                        do j=1,nodeNN(ii,itype)
                            f_out(j,i,ii)=f_in(j,i,ii)
                            !f_d(j,i,ii)=1.d0
                        enddo
                    enddo
                endif
                
                call dgemm('T', 'N', nodeNN(ii+1,itype),  num(itype),  nodeNN(ii,itype), 1.d0, Wij_nn(1,1,ii,itype),nodeMM,f_out(1,1,ii),nodeMM,0.d0,f_in(1,1,ii+1),nodeMM)
                !                     ** # row of C *    *# col of C*  *#row of A*    
                
                !***** test without dgemm *****
                
                do i=1,num(itype)
                    do j=1,nodeNN(ii+1,itype)
                        f_in(j,i,ii+1)=f_in(j,i,ii+1)+B_nn(j,ii,itype)
                    enddo
                enddo                
            enddo

            do i=1,num(itype)
                energy_type(i,itype)=f_in(1,i,nlayer+1)
            enddo

            ! Now, back propagation for the derivative for energy, with respect to the f_in(j,i,1)      
        enddo
        
        energy_pred_tmp = 0.d0
        num = 0
        iat1 = 0
        
        do i = 1, natom
            if(mod(i-1,nnodes).eq.inode-1) then
                iat1=iat1+1
                itype = iatom_type(i)
                num(itype) = num(itype) + 1
                energy_pred_tmp(i) = energy_type(num(itype),itype)
            endif
        enddo

        ! start MPI collect 
        call mpi_allreduce(energy_pred_tmp,energy_pred_NN,natom,MPI_REAL8,MPI_SUM,MPI_COMM_WORLD,ierr)
        
        etot_pred_NN = 0.d0
    
        do i = 1, natom
            !etot = etot + energy(i)
            etot_pred_NN = etot_pred_NN + energy_pred_NN(i)
        enddo 
        

        deallocate(feat)
        deallocate(list_neigh_alltypeM_use)

        deallocate(feat_type)
        deallocate(f_in)
        deallocate(f_out)
        deallocate(energy_type)
        
    end subroutine

    subroutine cal_energy_force_NN(feat,dfeat,num_neigh,list_neigh,AL,xatom,natom_tmp,nfeat0_tmp,m_neigh_tmp)
        integer(4)  :: itype,ixyz,i,j,jj
        integer natom_tmp,nfeat0_tmp,m_neigh_tmp,kk
        real(8) :: sum,direct,mean
        !real(8),intent(in) :: feat(nfeat0_tmp,natom_tmp)
        !real*8, intent(in) :: dfeat(nfeat0_tmp,natom_tmp,m_neigh_tmp,3)
        real(8),intent(in) :: feat(nfeat0_tmp,natom_n)
        real*8, intent(in) :: dfeat(nfeat0_tmp,natom_n,m_neigh_tmp,3)
        integer(4),intent(in) :: num_neigh(natom_tmp)
        integer(4),intent(in) :: list_neigh(m_neigh_tmp,natom_tmp)
        real(8), intent(in) :: AL(3,3)
        real(8),dimension(:,:),intent(in) :: xatom
        
        real(8),allocatable,dimension(:,:,:) :: feat_type
        !real(8),allocatable,dimension(:,:,:) :: dfeat_type
        
        real(8),allocatable,dimension(:,:,:) :: f_in,f_out,f_d,f_back
        real(8),allocatable,dimension(:,:) :: energy_type
        real(8),allocatable,dimension(:,:,:) :: dEdf_type
        
        real*8 pi,dE,dFx,dFy,dFz
        real*8 rad1,rad2,rad,dx1,dx2,dx3,dx,dy,dz,dd,yy,w22,dEdd,d,w22_1,w22_2,w22F_1,w22F_2
        
        integer iat1,iat2,ierr
        integer ii

        real*8 x
        real*8 sum1,sum2,sum3
        real(8) temp 

        pi=4*datan(1.d0)

        !write(*,*) "test natom_n=",natom_n
        
        !featType: feature index, atom index, atom type
        
        allocate(feat_type(nfeat1m,natom_n,ntype))
        allocate(f_in(nodeMM,natom_n,nlayer+1))
        allocate(f_out(nodeMM,natom_n,nlayer+1))
        allocate(f_d(nodeMM,natom_n,nlayer+1))
        allocate(f_back(nodeMM,natom_n,nlayer+1))
        allocate(energy_type(natom_n,ntype))
        allocate(dEdf_type(nfeat1m,natom_n,ntype))

        !allocate(dfeat_type(nfeat1m,natom_n*m_neigh*3,ntype))

        istat=0
        error_msg=''

        if (nfeat0_tmp/=nfeat1m .or. natom_tmp/=natom .or. m_neigh_tmp/=m_neigh) then
            write(*,*) "Shape of input arrays don't match the model!"
            write(6,*) nfeat0_tmp,natom_tmp,m_neigh_tmp
            write(6,*) nfeat1m,natom,m_neigh
            stop
        end if

        num = 0

        iat1=0

        do i = 1, natom
            if(mod(i-1,nnodes).eq.inode-1) then

                iat1=iat1+1
                itype = iatom_type(i)
                num(itype) = num(itype) + 1

                do j=1,nfeat1(itype)
                    feat_type(j,num(itype),itype) = feat(j, iat1)*a_scaler(j,itype)+b_scaler(j,itype)
                enddo

            endif
        enddo

        do itype=1,ntype

            do i=1,num(itype)
                do j=1,nodeNN(1,itype)

                    f_in(j,i,1)=feat_type(j,i,itype)
                
                enddo
            enddo

            do ii=1,nlayer
                
                if(ii.ne.1) then
                    do i=1,num(itype)
                        do j=1,nodeNN(ii,itype)
                            x=f_in(j,i,ii)
                            
                            if(x.gt.-150.d0.and.x.lt.150.d0) then
                                f_out(j, i, ii) = (exp(x)-exp(-x)) / (exp(x)+exp(-x))  ! tanh 
                                f_d(j, i, ii) = 1.0d0 - f_out(j,i,ii)*f_out(j,i,ii)

                            elseif(x.le.-150.d0) then
                                 f_out(j,i,ii)=0.d0
                                 f_d(j,i,ii)=0.d0
                            elseif(x.ge.150.d0) then 
                                 f_out(j,i,ii)=x
                                 f_d(j,i,ii)=1.d0
                            endif
                        enddo
                    enddo
                    
                elseif(ii.eq.1) then
                    do i=1,num(itype)
                        do j=1,nodeNN(ii,itype)
                            f_out(j,i,ii)=f_in(j,i,ii)
                            f_d(j,i,ii)=1.d0
                        enddo
                    enddo
                endif
                
                call dgemm('T', 'N', nodeNN(ii+1,itype),  num(itype),  nodeNN(ii,itype), 1.d0, Wij_nn(1,1,ii,itype),nodeMM,f_out(1,1,ii),nodeMM,0.d0,f_in(1,1,ii+1),nodeMM)
                !                     ** # row of C *    *# col of C*  *#row of A*    

                !***** test without dgemm *****
                
                do i=1,num(itype)
                    do j=1,nodeNN(ii+1,itype)
                        f_in(j,i,ii+1)=f_in(j,i,ii+1)+B_nn(j,ii,itype)
                    enddo
                enddo

            enddo
            
            do i=1,num(itype)
                energy_type(i,itype)=f_in(1,i,nlayer+1)
            enddo

            !  Now, back propagation for the derivative for energy, in respect to the f_in(j,i,1)      
            do i=1,num(itype)
                do j=1,nodeNN(nlayer,itype)
                    f_back(j,i,nlayer)=Wij_nn(j,1,nlayer,itype)*f_d(j,i,nlayer)
                enddo
            enddo

            do ii=nlayer,2,-1
                call dgemm('N', 'N', nodeNN(ii-1,itype),num(itype),nodeNN(ii,itype), 1.d0, Wij_nn(1,1,ii-1,itype),nodeMM,f_back(1,1,ii),nodeMM,0.d0,f_back(1,1,ii-1),nodeMM)

                if(ii-1.ne.1) then
                    do i=1,num(itype)
                        do j=1,nodeNN(ii-1,itype)
                            f_back(j,i,ii-1)=f_back(j,i,ii-1)*f_d(j,i,ii-1)
                        enddo
                    enddo
                endif  
            enddo
            
            do i=1,num(itype)
                do j=1,nfeat1(itype)
                    dEdf_type(j,i,itype)=f_back(j,i,1)
                enddo
            enddo
            
        enddo
        
        energy_pred_tmp=0.d0
        num = 0
        iat1=0
        
        do i = 1, natom
            if(mod(i-1,nnodes).eq.inode-1) then
                iat1=iat1+1
                itype = iatom_type(i)
                num(itype) = num(itype) + 1
                energy_pred_tmp(i)=energy_type(num(itype),itype)
            endif
        enddo

        !***********energy inference ends*************

        !***********force inference starts************

        force_pred_tmp=0.d0
        num = 0
        iat1=0

        do i = 1, natom
            ! take care of the designated segment
            if(mod(i-1,nnodes).eq.inode-1) then
                
                iat1=iat1+1
                itype = iatom_type(i) 

                num(itype) = num(itype) + 1
                
                do jj = 1, num_neigh(i)

                    iat2 = list_neigh(jj,i)
                    
                    sum1=0.d0
                    sum2=0.d0
                    sum3=0.d0
                    
                    do j=1,nfeat1(itype)
                        sum1=sum1+dEdf_type(j,num(itype),itype)*dfeat(j,iat1,jj,1)*a_scaler(j,itype)
                        sum2=sum2+dEdf_type(j,num(itype),itype)*dfeat(j,iat1,jj,2)*a_scaler(j,itype)
                        sum3=sum3+dEdf_type(j,num(itype),itype)*dfeat(j,iat1,jj,3)*a_scaler(j,itype)
                    enddo
                    
                    force_pred_tmp(1,iat2)=force_pred_tmp(1,iat2)+sum1
                    force_pred_tmp(2,iat2)=force_pred_tmp(2,iat2)+sum2    
                    force_pred_tmp(3,iat2)=force_pred_tmp(3,iat2)+sum3
                
                enddo
            endif
        enddo
        ! ***********force inference ends**************


        iat1=0

        ! ***********vdw inference starts**************
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
                
                energy_pred_tmp(i)=energy_pred_tmp(i)+dE
                force_pred_tmp(1,i)=force_pred_tmp(1,i)+dFx   ! Note, assume force=dE/dx, no minus sign
                force_pred_tmp(2,i)=force_pred_tmp(2,i)+dFy
                force_pred_tmp(3,i)=force_pred_tmp(3,i)+dFz
            
            endif
        enddo
        
        ! ***********vdw inference ends****************
        
        ! collecting E and force from all nodes 
        call mpi_allreduce(energy_pred_tmp,energy_pred_NN,natom,MPI_REAL8,MPI_SUM,MPI_COMM_WORLD,ierr)
        call mpi_allreduce(force_pred_tmp,force_pred_NN,3*natom,MPI_REAL8,MPI_SUM,MPI_COMM_WORLD,ierr)
        
        mean=0.0
        do j=1,add_force_num

            ! direct= sign(1.0,xatom(j,3)-0.5)
            
            if ((xatom(3,add_force_atom(j))-0.5).gt.0.0) direct = 1.0
            if ((xatom(3,add_force_atom(j))-0.5).lt.0.0) direct = - 1.0
            if (abs(xatom(3,add_force_atom(j))-0.5).lt.1.0E-5) direct=0.0
        
            const_fa(j)=0.0
            const_fb(j)= - alpha*direct*(xatom(3,add_force_atom(j))-z1)*AL(3,3)
            const_fc(j)=   alpha*direct*(xatom(2,add_force_atom(j))-y1)*AL(2,2)     
            mean=mean+ const_fb(j)
        enddo
        
        ! constant force 
        do j=1,add_force_num
                
            force_pred_NN(1,add_force_atom(j))= force_pred_NN(1,add_force_atom(j))+const_fa(j)   !give a force on x axis
            force_pred_NN(2,add_force_atom(j))= force_pred_NN(2,add_force_atom(j))+const_fb(j)- mean/add_force_num
            force_pred_NN(3,add_force_atom(j))= force_pred_NN(3,add_force_atom(j))+const_fc(j)
                
        enddo

        do j=1,const_force_num
                
            force_pred_NN(1,const_force_atom(j))= const_fx(j)   !give a force on x axis
            force_pred_NN(2,const_force_atom(j))= const_fy(j)
            force_pred_NN(3,const_force_atom(j))= const_fz(j)
            
        enddo
        
        ! calculating final etot
        etot_pred_NN = 0.d0
        do i = 1, natom
            etot_pred_NN = etot_pred_NN + energy_pred_NN(i)
        enddo

        deallocate(feat_type)
        deallocate(energy_type)
        deallocate(dEdf_type)
        deallocate(f_in)
        deallocate(f_out)
        deallocate(f_d)
        deallocate(f_back)
        
        !deallocate(dfeat_type)
        !deallocate(dfeat)
    end subroutine cal_energy_force_NN

   
end module calc_NN
  
  
