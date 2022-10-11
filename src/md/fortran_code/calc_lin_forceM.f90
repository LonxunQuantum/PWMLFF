  !// forquill v1.01 beta www.fcode.cn
module calc_lin
    use mod_mpi
    !implicit double precision (a-h, o-z)
    implicit none

  !!!!!!!!!!!!!          以下为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
    character(80),parameter :: fit_input_path0="fread_dfeat/fit_linearMM.input"
    character(80),parameter :: feat_info_path0="fread_dfeat/feat.info"
    character(80),parameter :: model_coefficients_path0="fread_dfeat/linear_fitB.ntype"
    character(80),parameter :: weight_feat_path_header0="fread_dfeat/weight_feat."
    character(80),parameter :: feat_pv_path_header0="fread_dfeat/feat_PV."
    character(80),parameter :: vdw_path0="fread_dfeat/vdw_fitB.ntype"
    
    character(200) :: fit_input_path=trim(fit_input_path0)
    character(200) :: feat_info_path=trim(feat_info_path0)
    character(200) :: model_coefficients_path=trim(model_coefficients_path0)
    character(200) :: weight_feat_path_header=trim(weight_feat_path_header0)
    character(200) :: feat_pv_path_header=trim(feat_pv_path_header0)
    character(200) :: vdw_path=trim(vdw_path0)
  
    integer(4) :: ntype                                    !模型所有涉及的原子种类
    integer(4) :: m_neigh                                  !模型所使用的最大近邻数(考虑这个数是否可以不用)
    integer(4) :: nfeat1m                                  !不同种原子的原始feature数目中最大者(目前似无意义)
    integer(4) :: nfeat2m                                  !不同种原子的PCA之后feature数目中最大者
    integer(4) :: nfeat2tot                                !PCA之后各种原子的feature数目之和
    integer(4),allocatable,dimension(:) :: nfeat1          !各种原子的原始feature数目
    integer(4),allocatable,dimension(:) :: nfeat2          !各种原子PCA之后的feature数目
    integer(4),allocatable,dimension(:) :: nfeat2i         !用来区分计算时各段各属于哪种原子的分段端点序号
    ! integer(4),allocatable,dimension(:) :: num_ref         !各种原子取的reference points的数目(对linear无意义)
    ! integer(4),allocatable,dimension(:) :: num_refi        !用来区分各种原子的reference points的分段端点序号(对linear无意义)
  
  
    real(8),allocatable,dimension(:) :: bb                 !计算erergy和force时与new feature相乘的系数向量w
    real(8),allocatable,dimension(:,:) :: bb_type          !不明白有何作用,似乎应该是之前用的变量
    real(8),allocatable,dimension(:,:) :: bb_type0         !将bb分别归类到不同种类的原子中，第二维才是代表原子种类
    real(8),allocatable,dimension (:, :) :: w_feat         !不同reference points的权重(对linear无意义)
  
    real(8),allocatable,dimension(:,:,:) :: pv             !PCA所用的转换矩阵
    real(8),allocatable,dimension (:,:) :: feat2_shift     !PCA之后用于标准化feat2的平移矩阵
    real(8),allocatable,dimension (:,:) :: feat2_scale     !PCA之后用于标准化feat2的伸缩系数矩阵
    
    
    integer(4) :: natom                                    !image的原子个数  
    integer(4),allocatable,dimension(:) :: num             !属于每种原子的原子个数，但似乎在calc_linear中无用
    integer(4),allocatable,dimension(:) :: num_atomtype    !属于每种原子的原子个数，似是目前所用的
    integer(4),allocatable,dimension(:) :: itype_atom      !每一种原子的原子属于第几种原子
    integer(4),allocatable,dimension(:) :: iatom           !每种原子的原子序数列表，即atomTypeList
    integer(4),allocatable,dimension(:) :: iatom_type      !每种原子的种类，即序数在种类列表中的序数
    
    real(8),allocatable,dimension(:) :: energy_pred_lin       !每个原子的能量预测值
    real(8),allocatable,dimension(:) :: energy_pred_tmp        !每个原子的能量预测值
    real(8),allocatable,dimension(:,:) :: force_pred_lin       !每个原子的受力预测值
    real(8),allocatable,dimension(:,:) :: force_pred_tmp       !每个原子的受力预测值
    real(8) :: etot_pred_lin
    character(200) :: error_msg
    integer(4) :: istat
    real(8), allocatable, dimension(:) ::  const_f
    integer(4),allocatable, dimension(:) :: direction,add_force_atom
    integer(4) :: add_force_num,power,axis

    ! real*8,allocatable,dimension(:) :: rad_atom,wp_atom
    real*8,allocatable,dimension(:) :: rad_atom,E_ave_vdw
    real*8,allocatable,dimension(:,:,:) :: wp_atom
    integer(4) :: nfeat1tm(100),ifeat_type_l(100),nfeat1t(100)
    integer(4) :: nfeat_type_l
  
  !!!!!!!!!!!!!          以上为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
    contains
    
  
   
    subroutine set_paths_lin(fit_dir_input)
        character(*),intent(in) :: fit_dir_input
        character(:),allocatable :: fit_dir,fit_dir_simp
        integer(4) :: len_dir
        fit_dir_simp=trim(adjustl(fit_dir_input))
        len_dir=len(fit_dir_simp)
        if (len_dir/=0 .and. fit_dir_simp/='.')then
            if (fit_dir_simp(len_dir:len_dir)=='/')then
                fit_dir=fit_dir_simp(:len_dir-1)
            else
                fit_dir=fit_dir_simp
            end if
            fit_input_path=fit_dir//'/'//trim(fit_input_path0)
            feat_info_path=fit_dir//'/'//trim(feat_info_path0)
            model_coefficients_path=fit_dir//'/'//trim(model_coefficients_path0)
            weight_feat_path_header=fit_dir//'/'//trim(weight_feat_path_header0)
            feat_pv_path_header=fit_dir//'/'//trim(feat_pv_path_header0)
            vdw_path=fit_dir//'/'//trim(vdw_path0)
        end if
    end subroutine set_paths_lin
    
    subroutine load_model_lin()
    
        integer(4) :: nimage,num_refm,num_reftot,nfeat1_tmp,nfeat2_tmp,itype,i,k,ntmp,itmp,itype1,j1
        integer(4) :: iflag_PCA,kkk,ntype_tmp,iatom_tmp,ntype_t,nterm,itype_t
        real(8) :: dist0

        ! integer(4),allocatable,dimension(:,:) :: nfeat,ipos_feat

! **************** read fit_linearMM.input ********************    
        open (10, file=trim(fit_input_path))
        rewind(10)
        read(10,*) ntype,m_neigh
        if (allocated(itype_atom)) deallocate(itype_atom)
        if (allocated(nfeat1)) deallocate(nfeat1)
        if (allocated(nfeat2)) deallocate(nfeat2)
        if (allocated(rad_atom)) deallocate(rad_atom)
        if (allocated(wp_atom)) deallocate(wp_atom)
        if (allocated(E_ave_vdw)) deallocate(E_ave_vdw)
        if (allocated(nfeat2i)) deallocate(nfeat2i)
        if (allocated(num)) deallocate(num)                              !image数据,在此处allocate，但在set_image_info中赋值
        if (allocated(num_atomtype)) deallocate(num_atomtype)                     !image数据,在此处allocate，但在set_image_info中赋值
        if (allocated(bb)) deallocate(bb)
        if (allocated(bb_type)) deallocate(bb_type)
        if (allocated(bb_type0)) deallocate(bb_type0)
        if (allocated(pv)) deallocate(pv)
        if (allocated(feat2_shift)) deallocate(feat2_shift)
        if (allocated(feat2_scale)) deallocate(feat2_scale)
        if (allocated(direction)) deallocate(direction)
        if (allocated(const_f)) deallocate(const_f)
        if (allocated(add_force_atom)) deallocate(add_force_atom)
        
        allocate (itype_atom(ntype))
        allocate (nfeat1(ntype))
        allocate (nfeat2(ntype))
        allocate (nfeat2i(ntype))
        
        allocate (num(ntype))                              !image数据,在此处allocate，但在set_image_info中赋值
        allocate (num_atomtype(ntype))                     !image数据,在此处allocate，但在set_image_info中赋值
        ! allocate (rad_atom(ntype))
        ! allocate (wp_atom(ntype))
        allocate(rad_atom(ntype))
        allocate(E_ave_vdw(ntype))
        allocate(wp_atom(ntype,ntype,2))
        wp_atom=0.d0

        do i=1,ntype
            read(10,*) itype_atom(i)!,rad_atom(i),wp_atom(i)
        enddo
        ! read(10,*) weight_E,weight_E0,weight_F
        close(10)
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

! **************** read feat.info ********************
        open(10,file=trim(feat_info_path))
        rewind(10)
        read(10,*) iflag_PCA   ! this can be used to turn off degmm part
        read(10,*) nfeat_type_l
        do kkk=1,nfeat_type_l
          read(10,*) ifeat_type_l(kkk)   ! the index (1,2,3) of the feature type
        enddo
        read(10,*) ntype_tmp
        if(ntype_tmp.ne.ntype) then
            write(*,*)  "ntype of atom not same, fit_linearMM.input, feat.info, stop"
            write(*,*) ntype,ntype_tmp
            stop
        endif
        ! allocate(nfeat(ntype,nfeat_type))
        ! allocate(ipos_feat(ntype,nfeat_type))
        do i=1,ntype
            read(10,*) iatom_tmp,nfeat1(i),nfeat2(i)   ! these nfeat1,nfeat2 include all ftype
            if(iatom_tmp.ne.itype_atom(i)) then
                write(*,*) "iatom not same, fit_linearMM.input, feat.info"
                write(*,*) iatom_tmp,itype_atom(i)
                stop
            endif
        enddo

    ! cccccccc Right now, nfeat1,nfeat2,for different types
    ! cccccccc must be the same. We will change that later, allow them 
    ! cccccccc to be different
           nfeat1m=0   ! the original feature
           nfeat2m=0   ! the new PCA, PV feature
           nfeat2tot=0 ! tht total feature of diff atom type
           nfeat2i=0   ! the starting point
           nfeat2i(1)=0
           do i=1,ntype
           if(nfeat1(i).gt.nfeat1m) nfeat1m=nfeat1(i)
           if(nfeat2(i).gt.nfeat2m) nfeat2m=nfeat2(i)
           nfeat2tot=nfeat2tot+nfeat2(i)
           if(i.gt.1) then
           nfeat2i(i)=nfeat2i(i-1)+nfeat2(i-1)
           endif
    
           enddo


        allocate (bb(nfeat2tot))
        allocate (bb_type(nfeat2m,ntype))
        allocate (bb_type0(nfeat2m,ntype))
        
        open (12, file=trim(model_coefficients_path))
        rewind (12)
        read (12, *) ntmp
        if (ntmp/=nfeat2tot) then
            write (6, *) 'ntmp.not.right,linear_fitb.ntype', ntmp, nfeat2tot
            stop
        end if
        do i = 1, nfeat2tot
            read (12, *) itmp, bb(i)
        end do
        close (12)
        do itype = 1, ntype
            do k = 1, nfeat2(itype)
                bb_type0(k, itype) = bb(k+nfeat2i(itype))
            end do
        end do
        
        allocate (pv(nfeat1m,nfeat2m,ntype))
        allocate (feat2_shift(nfeat2m,ntype))
        allocate (feat2_scale(nfeat2m,ntype))
        
        do itype = 1, ntype
            open (11, file=trim(feat_pv_path_header)//char(itype+48), form='unformatted')
            rewind (11)
            read (11) nfeat1_tmp, nfeat2_tmp
            if (nfeat2_tmp/=nfeat2(itype)) then
                write (6, *) 'nfeat2.not.same,feat2_ref', itype, nfeat2_tmp, nfeat2(itype)
                stop
            end if
            if (nfeat1_tmp/=nfeat1(itype)) then
                write (6, *) 'nfeat1.not.same,feat2_ref', itype, nfeat1_tmp, nfeat1(itype)
                stop
            end if
            read (11) pv(1:nfeat1(itype), 1:nfeat2(itype), itype)
            read (11) feat2_shift(1:nfeat2(itype), itype)
            read (11) feat2_scale(1:nfeat2(itype), itype)
            close (11)
        end do
!********************add_force****************

      add_force_num=0
    
!    open(10,file="add_force")
!    rewind(10)
!    read(10,*) add_force_num,axis
!    allocate(add_force_atom(add_force_num))
!    allocate(direction(add_force_num))
!    allocate(const_f(add_force_num))
!    do i=1,add_force_num
!        read(10,*) add_force_atom(i), direction(i), const_f(i)
!    enddo
!    close(10)
       
    end subroutine load_model_lin
  
    subroutine set_image_info_lin(iatom_tmp,is_reset,natom_tmp)
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
                deallocate(energy_pred_lin)
                deallocate(energy_pred_tmp)
                deallocate(force_pred_lin)
                deallocate(force_pred_tmp)
            end if
              
            natom=image_size
            allocate(iatom(natom))
            allocate(iatom_type(natom))
            allocate(energy_pred_lin(natom))
            allocate(energy_pred_tmp(natom))
            allocate(force_pred_lin(3,natom))
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
                iatom_type(i) = iitype
            end do
      
            num_atomtype = 0
            do i = 1, natom
                itype = iatom_type(i)
                num_atomtype(itype) = num_atomtype(itype) + 1
            end do
        end if
        
    end subroutine set_image_info_lin
  
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

   
end module calc_lin
  
  
