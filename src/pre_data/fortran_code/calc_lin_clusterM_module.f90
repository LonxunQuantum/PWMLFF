  !// forquill v1.01 beta www.fcode.cn
module calc_clst
    !implicit double precision (a-h, o-z)
    implicit none

  !!!!!!!!!!!!!          以下为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  
    character(80),parameter :: fit_input_path0="fit.input"
    character(80),parameter :: model_coefficients_path0="cluster_fitB.ntype"
    character(80),parameter :: weight_feat_path_header0="weight_feat."
    character(80),parameter :: feat_pv_path_header0="feat_PV."
    character(80),parameter :: feat_cent_path_header0="feat_cent."
   
    character(200) :: fit_input_path=trim(fit_input_path0)
    character(200) :: model_coefficients_path=trim(model_coefficients_path0)
    character(200) :: weight_feat_path_header=trim(weight_feat_path_header0)
    character(200) :: feat_pv_path_header=trim(feat_pv_path_header0)
    character(200) :: feat_cent_path_header=trim(feat_cent_path_header0)
  
    integer(4) :: ntype                                    !模型所有涉及的原子种类
    integer(4) :: m_neigh                                  !模型所使用的最大近邻数(考虑这个数是否可以不用)
    integer(4) :: nfeat0m                                  !不同种原子的原始feature数目中最大者(目前似无意义)
    integer(4) :: nfeat2m                                  !不同种原子的PCA之后feature数目中最大者
    integer(4) :: nfeat2tot                                !PCA之后各种原子的feature数目之和
    integer(4),allocatable,dimension(:) :: nfeat0          !各种原子的原始feature数目
    integer(4),allocatable,dimension(:) :: nfeat2          !各种原子PCA之后的feature数目
    integer(4),allocatable,dimension(:) :: nfeat2i         !用来区分计算时各段各属于哪种原子的分段端点序号
    integer(4),allocatable,dimension(:) :: num_ref         !各种原子取的reference points的数目(对linear无意义)
    integer(4),allocatable,dimension(:) :: num_refi        !用来区分各种原子的reference points的分段端点序号(对linear无意义)

    integer(4),allocatable,dimension(:) :: nfeat2N,nfeat2iN  
    integer(4),allocatable,dimension(:) :: clusterNum
    integer kernel_type
    ! real*8,allocatable,dimension(:) :: alpha0,k_dist0
    real(8),allocatable,dimension(:) :: alpha0,k_dist0
    real(8),allocatable,dimension(:,:) :: width
    real(8),allocatable,dimension(:,:,:) ::  feat_cent

    integer(4) :: nfeat2mN
    integer(4) :: nfeat2totN

    real(8),allocatable,dimension(:) :: bb                 !计算erergy和force时与new feature相乘的系数向量w
    real(8),allocatable,dimension(:,:) :: BB_type          !不明白有何作用,似乎应该是之前用的变量
    real(8),allocatable,dimension(:,:) :: BB_type0         !将bb分别归类到不同种类的原子中，第二维才是代表原子种类
    real(8),allocatable,dimension(:,:) :: w_feat         !不同reference points的权重(对linear无意义)
  
    real(8),allocatable,dimension(:,:,:) :: pv             !PCA所用的转换矩阵
    real(8),allocatable,dimension (:,:) :: feat2_shift     !PCA之后用于标准化feat2的平移矩阵
    real(8),allocatable,dimension (:,:) :: feat2_scale     !PCA之后用于标准化feat2的伸缩系数矩阵
    
    
    integer(4) :: natom                                    !image的原子个数  
    integer(4),allocatable,dimension(:) :: num             !属于每种原子的原子个数，但似乎在calc_linear中无用
    integer(4),allocatable,dimension(:) :: numf            
    integer(4),allocatable,dimension(:) :: num_atomtype    !属于每种原子的原子个数，似是目前所用的
    integer(4),allocatable,dimension(:) :: itype_atom      !每一种原子的原子属于第几种原子
    integer(4),allocatable,dimension(:) :: iatom           !每种原子的原子序数列表，即atomTypeList
    integer(4),allocatable,dimension(:) :: iatom_type      !每种原子的种类，即序数在种类列表中的序数
    
    real(8),allocatable,dimension(:) :: energy_pred        !每个原子的能量预测值
    real(8),allocatable,dimension(:,:) :: force_pred       !每个原子的受力预测值
    real(8) :: etot_pred
    character(200) :: error_msg
    integer(4) :: istat
    real(8), allocatable, dimension(:) ::  const_f
    integer(4),allocatable, dimension(:) :: direction,add_force_atom
    integer(4) :: add_force_num,power

    real*8,allocatable,dimension(:) :: rad_atom,wp_atom
    
  
  !!!!!!!!!!!!!          以上为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
    contains
    
  
   
    subroutine set_paths(fit_dir_input)
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
            model_coefficients_path=fit_dir//'/'//trim(model_coefficients_path0)
            weight_feat_path_header=fit_dir//'/'//trim(weight_feat_path_header0)
            feat_pv_path_header=fit_dir//'/'//trim(feat_pv_path_header0)
            feat_cent_path_header=fit_dir//'/'//trim(feat_cent_path_header0)
        end if
    end subroutine set_paths
    
    subroutine load_model()
    
        integer(4) :: nimage,num_refm,num_reftot,nfeat0_tmp,nfeat2_tmp,itype,i,k,ntmp,itmp,j,j1,ii
        real(8) :: dist0
        integer(4) :: clusterNumM
        open (10, file=trim(fit_input_path))
        rewind (10)
        read (10, *) ntype, natom, m_neigh, nimage         !(nimage对md计算的模型无意义，natom是之后set_image_info中应读入的数据，此处读入没有意义)
      !       allocate(itype_atom(ntype))
        if (allocated(itype_atom)) then
            deallocate(itype_atom)
            deallocate(nfeat0)
            deallocate (nfeat2)
            deallocate (num_ref)
            deallocate (num_refi)
            deallocate (rad_atom)
            deallocate (wp_atom)
            deallocate (nfeat2i)
            deallocate (num)                              !image数据,在此处allocate，但在set_image_info中赋值
            deallocate (num_atomtype)                     !image数据,在此处allocate，但在set_image_info中赋值
            deallocate(numf)
            deallocate(nfeat2N)
            deallocate(nfeat2iN)
            deallocate(ClusterNum)
            deallocate(alpha0)
            deallocate(k_dist0)
            deallocate(feat_cent)
            deallocate(width)
            deallocate(w_feat)

            deallocate (bb)
            ! deallocate (bb_type)
            deallocate (bb_type0)
            deallocate (pv)
            deallocate (feat2_shift)
            deallocate (feat2_scale)
            deallocate(add_force_atom)
            deallocate(direction)
            deallocate(const_f)

        end if
        
        allocate (itype_atom(ntype))
        allocate (nfeat0(ntype))
        allocate (nfeat2(ntype))
        allocate (num_ref(ntype))
        allocate (num_refi(ntype))
        allocate (nfeat2i(ntype))
        
        allocate (num(ntype))                              !image数据,在此处allocate，但在set_image_info中赋值
        allocate (num_atomtype(ntype))                     !image数据,在此处allocate，但在set_image_info中赋值
        allocate (rad_atom(ntype))
        allocate (wp_atom(ntype))
        allocate(numf(ntype))
        allocate(nfeat2N(ntype))
        allocate(nfeat2iN(ntype))

        do i = 1, ntype
            read (10, *) itype_atom(i), nfeat0(i), nfeat2(i), num_ref(i), &
    rad_atom(i), wp_atom(i)     !num_ref对linear无意义,nfeat0对每种元素其实都一样
        end do
            !read (10, *) alpha, dist0
            !read (10, *) weight_e, weight_e0, weight_f, delta
        close (10)
        
        ! dist0 = dist0**2
        nfeat0m = 0
        nfeat2m = 0
        num_refm = 0
        num_reftot = 0
        num_refi(1) = 0
        nfeat2tot = 0
        nfeat2i = 0
        nfeat2i(1) = 0
        
        do i = 1, ntype
            if (nfeat0(i)>nfeat0m) nfeat0m = nfeat0(i)
            if (nfeat2(i)>nfeat2m) nfeat2m = nfeat2(i)
            if (num_ref(i)>num_refm) num_refm = num_ref(i)
            num_reftot = num_reftot + num_ref(i)
            nfeat2tot = nfeat2tot + nfeat2(i)
            if (i>1) then
                num_refi(i) = num_refi(i-1) + num_ref(i-1)
                nfeat2i(i) = nfeat2i(i-1) + nfeat2(i-1)
            end if
        end do
        open(99,file='log.txt')
!********************* cluster info *********************
        allocate(ClusterNum(ntype))
        allocate(alpha0(ntype))
        allocate(k_dist0(ntype))
        ClusterNumM=0
        do itype=1,ntype
        open(13,file=trim(feat_cent_path_header)//char(itype+48))
        rewind(13)
        read(13,*) ClusterNum(itype),alpha0(itype)
        if(clusterNum(itype).gt.clusterNumM)  clusterNumM=clusterNum(itype) 
        close(13)
        enddo
 
        allocate(feat_cent(nfeat2m,ClusterNumM,ntype))
        allocate(width(ClusterNumM,ntype))
 
        nfeat2mN=nfeat2m*ClusterNumM

 
        nfeat2totN=0
        do itype=1,ntype
        nfeat2N(itype)=nfeat2(itype)*ClusterNum(itype)
        nfeat2totN=nfeat2totN+nfeat2N(itype)
        enddo
 
        nfeat2iN(1)=0
        do itype=2,ntype
        nfeat2iN(itype)=nfeat2iN(itype-1)+nfeat2N(itype-1)
        enddo
!--------------------------------------------------------
!********************** coefficient BB ******************* 
        allocate(BB(nfeat2totN))
        ! allocate(BB_type(nfeat2mN,ntype))
        allocate(BB_type0(nfeat2mN,ntype))
 

        open (12, file=trim(model_coefficients_path))
        rewind(12)
        read(12,*) ntmp
        if(ntmp.ne.nfeat2totN) then
            write(6,*) "ntmp.not.right,linear_fitB.ntype",ntmp,nfeat2totN
            stop
        endif
        do i=1,nfeat2totN
            read(12,*) itmp, BB(i)
        enddo
        close(12)
        do itype=1,ntype
        do k=1,nfeat2N(itype)
            BB_type0(k,itype)=BB(k+nfeat2iN(itype))
        enddo
        enddo   
!------------------------------------------------------------
!************************** cluster info ********************
        do itype=1,ntype
        open(13,file=trim(feat_cent_path_header)//char(itype+48))
        rewind(13)
        read(13,*) ClusterNum(itype),alpha0(itype),k_dist0(itype),kernel_type
        do ii=1,ClusterNum(itype)
        read(13,*) feat_cent(1:nfeat2(itype),ii,itype)
        enddo
        do ii=1,ClusterNum(itype)
        read(13,*) width(ii,itype)
        enddo
        close(13)
        enddo
!---------------------------------------------------------------
!************************** PV ********************************
        allocate (pv(nfeat0m,nfeat2m,ntype))
        allocate (feat2_shift(nfeat2m,ntype))
        allocate (feat2_scale(nfeat2m,ntype))
        
        do itype = 1, ntype
            open (11, file=trim(feat_pv_path_header)//char(itype+48), form='unformatted')
            rewind (11)
            read (11) nfeat0_tmp, nfeat2_tmp
            if (nfeat2_tmp/=nfeat2(itype)) then
                write (6, *) 'nfeat2.not.same,feat2_ref', itype, nfeat2_tmp, nfeat2(itype)
                stop
            end if
            if (nfeat0_tmp/=nfeat0(itype)) then
                write (6, *) 'nfeat0.not.same,feat2_ref', itype, nfeat0_tmp, nfeat0(itype)
                stop
            end if
            read (11) pv(1:nfeat0(itype), 1:nfeat2(itype), itype)
            read (11) feat2_shift(1:nfeat2(itype), itype)
            read (11) feat2_scale(1:nfeat2(itype), itype)
            close (11)
        end do
!---------------------------------------------------------------
!******************** w_feat ******************
        allocate(w_feat(nfeat2m,ntype))
        ! allocate(feat2_ref(nfeat2m,num_refm,ntype))
        do itype=1,ntype
            open(10,file=trim(weight_feat_path_header)//char(itype+48))
            rewind(10)
            do j=1,nfeat2(itype)
                read(10,*) j1,w_feat(j,itype)
                w_feat(j,itype)=w_feat(j,itype)**2
            enddo
            close(10)
        enddo
!-----------------------------------------------

!********************add_force****************
    open(10,file="add_force")
    rewind(10)
    read(10,*) add_force_num
    allocate(add_force_atom(add_force_num))
    allocate(direction(add_force_num))
    allocate(const_f(add_force_num))
    do i=1,add_force_num
        read(10,*) add_force_atom(i), direction(i), const_f(i)
    enddo
    close(10)
       

    rewind(99)
    write(99,*) "load_model"
    close(99)

    end subroutine load_model
  
    subroutine set_image_info(atom_type_list,is_reset)
        integer(4) :: i,j,itype,iitype
        integer(4),dimension(:),intent(in) :: atom_type_list
        logical,intent(in) :: is_reset
        integer(4) :: image_size
        
        
        image_size=size(atom_type_list)
        if (is_reset .or. (.not. allocated(iatom)) .or. image_size/=natom) then
        
            if (allocated(iatom))then
                if (image_size==natom .and. maxval(abs(atom_type_list-iatom))==0) then
                    return
                end if
                deallocate(iatom)
                !deallocate(num_atomtype) !此似乎应该在load_model中allocate,但赋值似应在此subroutine
                !deallocate(num)          !此似乎应该在load_model中allocate,但赋值似应在此subroutine
                !deallocate(itype_atom)   !此似乎应该在load_model中allocate,但赋值似应在此subroutine
                deallocate(iatom_type)
                deallocate(energy_pred)
                deallocate(force_pred)
            end if
              
            natom=image_size
            allocate(iatom(natom))
            allocate(iatom_type(natom))
            allocate(energy_pred(natom))
            allocate(force_pred(3,natom))
            !allocate()
              
              
              
            iatom=atom_type_list
              
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
        
    end subroutine set_image_info
  
    subroutine cal_energy_force(feat,num_tmp,dfeat_tmp,iat_tmp,jneigh_tmp,ifeat_tmp,num_neigh,list_neigh,AL,xatom)
        integer(4)  :: itype,ixyz,i,j,jj,jjj,iat,jn,jat,jtype
        real(8) :: sum,sum0
        real(8) :: dxx,dexp_dd
        real(8),dimension(:,:),intent(in) :: feat
        real*8, dimension (:,:),intent(in) :: dfeat_tmp
        integer(4), intent(in) :: num_tmp
        integer(4),dimension (:),intent(in) :: iat_tmp,jneigh_tmp,ifeat_tmp
        ! real(8),dimension(:,:,:,:),intent(in) :: dfeat
        integer(4),dimension(:),intent(in) :: num_neigh
        integer(4),dimension(:,:),intent(in) :: list_neigh
        real(8), intent(in) :: AL(3,3)
        real(8),dimension(:,:),intent(in) :: xatom
        !real(8),dimension(:) :: energy_pred
        !real(8),dimension(:,:) :: force_pred
        real*8,allocatable,dimension(:,:,:,:) :: dfeat
        real(8) dsum0(3),d_dd(3)
        real(8) akernel(1000),dkernel(1000,3)        
        real*8,allocatable,dimension(:,:) :: feat22_type,feat2N
        real(8),allocatable,dimension(:,:) :: feat2
        real(8),allocatable,dimension(:,:,:) :: feat_type
        real(8),allocatable,dimension(:,:,:) :: feat2_type,feat2_typeN
        integer(4),allocatable,dimension(:,:) :: ind_type
        real(8),allocatable,dimension(:,:,:) :: dfeat_type
        real(8),allocatable,dimension(:,:,:) :: dfeat2_type
        real(8),allocatable,dimension(:,:,:,:) :: dfeat2
        real(8),allocatable,dimension(:,:,:,:) :: SS
        real(8),allocatable,dimension(:,:,:,:) :: dfeat2N
        integer(4),dimension(2) :: feat_shape,list_neigh_shape
        integer(4),dimension(4) :: dfeat_shape

        real*8 pi,dE,dFx,dFy,dFz
        real*8 rad1,rad2,rad,dx1,dx2,dx3,dx,dy,dz,dd,yy,w22,dEdd,d



         pi=4*datan(1.d0)
        
         open(99,file='log.txt',position='append')
         write(99,*)shape(feat)
         write(99,*) num_tmp
         write(99,*)shape(dfeat_tmp)
         write(99,*) shape(iat_tmp)
         write(99,*) shape(jneigh_tmp)
         write(99,*) shape(ifeat_tmp)
         write(99,*) shape(num_neigh)
         write(99,*) shape(list_neigh)
         write(99,*) shape(xatom)
        ! open(99,file='log.txt')
        
        !allocate(energy_pred(natom))
        !allocate(force_pred(3,natom))
        allocate(dfeat(nfeat0m,natom,m_neigh,3))
        dfeat(:,:,:,:)=0.0
        do jj=1,num_tmp
        dfeat(ifeat_tmp(jj),iat_tmp(jj),jneigh_tmp(jj),:)=dfeat_tmp(:,jj)
        enddo

        allocate(feat2N(nfeat2mN,natom))
        allocate(feat2_typeN(nfeat2mN,natom,ntype))
        allocate(dfeat2N(nfeat2mN,natom,m_neigh,3))

        allocate(feat2(nfeat2m,natom))
        allocate(feat_type(nfeat0m,natom,ntype))
        allocate(feat2_type(nfeat2m,natom,ntype))
        allocate(ind_type(natom,ntype))
        allocate(dfeat_type(nfeat0m,natom*m_neigh*3,ntype))
        allocate(dfeat2_type(nfeat2m,natom*m_neigh*3,ntype))
        allocate(dfeat2(nfeat2m,natom,m_neigh,3))
        allocate(SS(nfeat2mN,natom,3,ntype))
        feat_shape=shape(feat)
        dfeat_shape=shape(dfeat)
        list_neigh_shape=shape(list_neigh)
        istat=0
        error_msg=''
        open(99,file='log.txt',position='append')
        write(*,*)'feat_shape'
        write(*,*)feat_shape
        write(*,*)'dfeat_shape'
        write(*,*)dfeat_shape
        write(*,*)'list_neigh_shape'
        write(*,*)list_neigh_shape
        write(*,*)"nfeat0m,natom,m_neigh"
        write(*,*)nfeat0m,natom,m_neigh
        close(99)
        open(99,file='log.txt',position='append')
        if (feat_shape(1)/=nfeat0m .or. feat_shape(2)/=natom &
             .or. dfeat_shape(1)/=nfeat0m .or. dfeat_shape(2)/=natom .or. dfeat_shape(4)/=3 &
             .or. size(num_neigh)/=natom  .or. list_neigh_shape(2)/=natom) then      
            
            write(99,*) "Shape of input arrays don't match the model!"
            istat=1
            !if (allocated(error_msg)) then
                !deallocate(error_msg)
            !end if
            error_msg="Shape of input arrays don't match the model!"
            return
        end if
        !allocate (feat2(nfeat2m,natom))
        !allocate (feat_type(nfeat0m,natom,ntype))
        !allocate (feat2_type(nfeat2m,natom,ntype))
        !allocate (ind_type(natom,ntype))
        !allocate (dfeat_type(nfeat0m,natom*m_neigh*3,ntype))
        !allocate (dfeat2_type(nfeat2m,natom*m_neigh*3,ntype))
        !allocate (dfeat2(nfeat2m,natom,m_neigh,3))    
        !allocate (ss(nfeat2m,natom,3,ntype))
        
        
        ! write(99,*) "all arrays shape right"
        ! close(99)
        ! open(99,file='log.txt',position='append')
        
        
        numf=0
        do i=1,natom
        itype=iatom_type(i)
        numf(itype)=numf(itype)+1
        ind_type(numf(itype),itype)=i
        feat_type(:,numf(itype),itype)=feat(:,i)
        enddo
    
        ! write(99,*)"feat_type normlly setted"
        ! close(99)
        ! open(99,file='log.txt',position='append')
        
        do itype=1,ntype
        call dgemm('T','N',nfeat2(itype),numf(itype),nfeat0(itype),1.d0,PV(1,1,itype),nfeat0m,feat_type(1,1,itype),nfeat0m,0.d0,feat2_type(1,1,itype),nfeat2m)
        enddo
    
        !write(99,*)"feat2_type normlly setted first time"
        
        do itype=1,ntype
        do i=1,numf(itype)
        do j=1,nfeat2(itype)-1
        feat2_type(j,i,itype)=(feat2_type(j,i,itype)-feat2_shift(j,itype))*feat2_scale(j,itype)
        enddo
        feat2_type(nfeat2,i,itype)=1.d0
        enddo
        enddo

          
        ! write(99,*)"feat2_type normlly setted second time"
        ! close(99)
        ! open(99,file='log.txt',position='append')
        
        numf=0
        do i=1,natom
        itype=iatom_type(i)
        numf(itype)=numf(itype)+1
        feat2(:,i)=feat2_type(:,numf(itype),itype)
        enddo
          
        ! write(99,*)"feat2 normlly setted first time"
        ! close(99)
        ! open(99,file='log.txt',position='append')
        
        do itype=1,ntype
        do i=1,numf(itype)    ! the atom in the system belong to this itype
          iat=ind_type(i,itype)   ! iat, this atom
          sum0=0.d0
          do jj=1,clusterNum(itype)
          sum=0.d0
          do j=1,nfeat2(itype)
          sum=sum+(feat2_type(j,i,itype)-feat_cent(j,jj,itype))**2*w_feat(j,itype)
          enddo
!TODO:
            if (kernel_type.eq.1) then
            akernel(jj)=exp(-(sum/width(jj,itype))**alpha0(itype))
            else if(kernel_type.eq.2) then
            akernel(jj)=1/(sum**alpha0(itype)+k_dist0(itype)**alpha0(itype))
            end if
          sum0=sum0+akernel(jj)
          enddo
            
            if(sum0.lt.1.E-10) then
                sum0=1.0
            end if
          akernel=akernel/sum0
          
          

          do jj=1,clusterNum(itype)
          do j=1,nfeat2(itype)
          jjj=(jj-1)*nfeat2(itype)+j
          feat2_typeN(jjj,i,itype)=feat2_type(j,i,itype)*akernel(jj)
          enddo
          enddo
        enddo
        enddo
!cccccccc feat2_typeN is the new feature, feature number nfeat2N!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        numf=0
        do i=1,natom
        itype=iatom_type(i)
        numf(itype)=numf(itype)+1
        feat2N(:,i)=feat2_typeN(:,numf(itype),itype)
        enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        do i=1,natom
        itype=iatom_type(i)
        sum=0.d0
        do j=1,nfeat2N(itype)
        sum=sum+feat2N(j,i)*BB_type0(j,itype)
        enddo
        energy_pred(i)=sum
        enddo
      
        ! write(99,*)"energy_pred normlly setted"
        ! close(99)
        ! open(99,file='log.txt',position='append')
        
        deallocate(feat_type)
        deallocate(feat2_type)
        deallocate(ind_type)
        deallocate(feat2N)
        deallocate(feat2_typeN)

        num=0
        do i=1,natom
        do jj=1,num_neigh(i)
        itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
        num(itype)=num(itype)+1
        dfeat_type(:,num(itype),itype)=dfeat(:,i,jj,1)
        num(itype)=num(itype)+1
        dfeat_type(:,num(itype),itype)=dfeat(:,i,jj,2)
        num(itype)=num(itype)+1
        dfeat_type(:,num(itype),itype)=dfeat(:,i,jj,3)
        enddo
        enddo

        deallocate(dfeat)
        !cccccccc note: num(itype) is rather large, in the scane of natom*num_neigh
    
        do itype=1,ntype
        call dgemm('T','N',nfeat2(itype),num(itype),nfeat0(itype),1.d0,PV(1,1,itype),nfeat0m,dfeat_type(1,1,itype),nfeat0m,0.d0,dfeat2_type(1,1,itype),nfeat2m)
        enddo
    
        ! write(99,*)"dfeat2_type normlly setted first time"
        ! close(99)
        ! open(99,file='log.txt',position='append')
        
        num=0
        do i=1,natom
        do jj=1,num_neigh(i)
        itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
        num(itype)=num(itype)+1
        do j=1,nfeat2(itype)-1
        dfeat2(j,i,jj,1)=dfeat2_type(j,num(itype),itype)*feat2_scale(j,itype)
        enddo
        dfeat2(nfeat2(itype),i,jj,1)=0.d0
        num(itype)=num(itype)+1
        do j=1,nfeat2(itype)-1
        dfeat2(j,i,jj,2)=dfeat2_type(j,num(itype),itype)*feat2_scale(j,itype)
        enddo
        dfeat2(nfeat2(itype),i,jj,2)=0.d0
        num(itype)=num(itype)+1
        do j=1,nfeat2(itype)-1
        dfeat2(j,i,jj,3)=dfeat2_type(j,num(itype),itype)*feat2_scale(j,itype)
        enddo
        dfeat2(nfeat2(itype),i,jj,3)=0.d0
        enddo
        enddo
        ! write(99,*)"dfeat2 normlly setted"
        ! close(99)
        ! open(99,file='log.txt',position='append')
        
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !cc  now, dfeat2 is:
        !cc dfeat2(nfeat2,natom,j_neigh,3): dfeat2(j,i,jj,3)= d/dr_i(feat2(j,list_neigh(jj,i))
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!  Now, we need to get dfeat2_N(jjj,i,jj,3)

        do iat=1,natom
        do jn=1,num_neigh(iat)
            jat=list_neigh(jn,iat)  ! jat is the neigh of iat
            jtype=iatom_type(jat)  ! the neighbor atom type, the following is doing jat
        
! ccccccccccccc  note: dfeat2(j,iat,jn,3) is the j-th feature of jat, not iat
            sum0=0.d0
            do jj=1,clusterNum(jtype)
            dd=0.d0
            d_dd(:)=0.d0
            do j=1,nfeat2(jtype)   ! right now, this is the sme as nfeat2(itype)
            dxx=feat2(j,jat)-feat_cent(j,jj,jtype)
            dd=dd+dxx**2*w_feat(j,jtype)
            d_dd(:)=d_dd(:) + 2*dxx*dfeat2(j,iat,jn,:)*w_feat(j,jtype)
            enddo
!TODO:
            if (kernel_type.eq.1) then
                akernel(jj)=exp(-(dd/width(jj,jtype))**alpha0(jtype))
                dexp_dd=-akernel(jj)*alpha0(jtype)*dd**(alpha0(jtype)-1)/width(jj,jtype)**alpha0(jtype)
                dkernel(jj,:)=dexp_dd*d_dd(:)
            else if(kernel_type.eq.2) then
                akernel(jj)=1/(dd**alpha0(itype)+k_dist0(itype)**alpha0(itype))
                dexp_dd=-akernel(jj)**2*alpha0(jtype)*dd**(alpha0(jtype)-1) 
                dkernel(jj,:)=dexp_dd*d_dd(:)
            end if
              sum0=sum0+akernel(jj)
            enddo
    
            if (sum0.lt.1.E-10) then
              sum0=1.0
            end if

            dsum0(:)=0.d0
            do jj=1,clusterNum(jtype)
            dsum0(:)=dsum0(:)+dkernel(jj,:)
            enddo

            do jj=1,clusterNum(jtype)
            dkernel(jj,:)=dkernel(jj,:)/sum0-akernel(jj)/sum0**2*dsum0(:)
            akernel(jj)=akernel(jj)/sum0     ! normalize the kernel
            enddo
!cccccccccccccccccccccccccccccccccccccccc
!  The new feat2_N(jjj)=feat2(j)*akernel0(jj)
!  So,  dfeat2)N(jjj)=dfeat2(j)*akernel0(jj)+feat2(j)*dkenerl0(jj)
!  Now, we calculate dkenerl0(jj)
!   dkernel0(jj)= dkernel(jj)/sum0-akernel(jj)/sum0**2*dsum0
!iccccccccccccccccccccccccccccccccccccccccccccccccccc

            do jj=1,clusterNum(jtype) ! neighboring atom cluster
            do j=1,nfeat2(jtype)
            jjj=(jj-1)*nfeat2(jtype)+j
!cccccccccccccccccccccccc
!cccc do we have feat2 here yet? 
            dfeat2N(jjj,iat,jn,1:3)=dfeat2(j,iat,jn,1:3)*akernel(jj)+feat2(j,jat)*dkernel(jj,1:3)   ! this is for jat's feature
    
            enddo
            enddo
        enddo  ! jn
        enddo  ! iat          
      
!*********** Now, we have the new features, we need to calculate the distance to reference state       
        SS=0.d0

        do i=1,natom
        do jj=1,num_neigh(i)
        jtype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
        do j=1,nfeat2N(jtype)
 
        SS(j,i,1,jtype)=SS(j,i,1,jtype)+dfeat2N(j,i,jj,1)
        SS(j,i,2,jtype)=SS(j,i,2,jtype)+dfeat2N(j,i,jj,2)
        SS(j,i,3,jtype)=SS(j,i,3,jtype)+dfeat2N(j,i,jj,3)
        enddo
        enddo
        enddo
    
        ! write(99,*)"ss normlly setted"
        ! close(99)
        ! open(99,file='log.txt',position='append')
         
        do i=1,natom
        do ixyz=1,3
        sum=0.d0
        do itype=1,ntype
        do j=1,nfeat2N(itype)
        sum=sum+SS(j,i,ixyz,itype)*BB_type0(j,itype)
        enddo
        enddo
        force_pred(ixyz,i)=sum
        enddo
        enddo


        deallocate(feat2)
        ! write(*,*) 'feat2 deallocate'
        deallocate(dfeat_type)
        ! write(*,*) 'dfeat_type deallocate'
        deallocate(dfeat2_type)
        ! write(*,*) 'dfeat2_type deallocate'
        deallocate(dfeat2)
        ! write(*,*) 'dfeat2 deallocate'
        deallocate(SS)
        ! write(*,*) 'ss deallocate'
        deallocate(dfeat2N)
        ! write(*,*) 'dfeat2N deallocate'


        do j=1,add_force_num
            do i=1,natom
                if (i .eq. add_force_atom(j) ) then
                    force_pred(1,i)= force_pred(1,i)+(direction(j)-1)*const_f(j)   !give a force on x axis
                end if
            enddo
        enddo
       
!ccccccccccccccccccccccccccccccccccccccccccc
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
       w22=dsqrt(wp_atom(iatom_type(i))*wp_atom(iatom_type(j)))
       yy=pi*dd/(4*rad)
!       dE=dE+0.5*w22*exp((1-dd/rad)*4.0)*cos(yy)**2
!       dEdd=w22*exp((1-dd/rad)*4.d0)*((-4/rad)*cos(yy)**2
!     &   -(pi/(2*rad))*cos(yy)*sin(yy))
       dE=dE+0.5*4*w22*(rad/dd)**12*cos(yy)**2
       dEdd=4*w22*(-12*(rad/dd)**12/dd*cos(yy)**2  &
        -(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)

       dFx=dFx-dEdd*dx/dd       ! note, -sign, because dx=d(j)-x(i)
       dFy=dFy-dEdd*dy/dd
       dFz=dFz-dEdd*dz/dd
       endif
       endif
       enddo
       energy_pred(i)=energy_pred(i)+dE
       force_pred(1,i)=force_pred(1,i)+dFx   ! Note, assume force=dE/dx, no minus sign
       force_pred(2,i)=force_pred(2,i)+dFy
       force_pred(3,i)=force_pred(3,i)+dFz
       enddo
!ccccccccccccccccccccccccccccccccccccccccccc
        ! write(99,*)"force_pred normlly setted"
        ! close(99)
        ! open(99,file='log.txt',position='append')
        
        etot_pred = 0.d0
        do i = 1, natom
            !etot = etot + energy(i)
            etot_pred = etot_pred + energy_pred(i)
        end do
          
        ! write(*,*)"etot_pred normlly setted"
        ! write(*,*)"energy_pred",shape(energy_pred)
        ! ! write(99,*)energy_pred
        ! write(*,*)"force_pred",shape(force_pred)
        !write(99,*)force_pred
        
        

        ! write(*,*)"ended"
        close(99)
    end subroutine cal_energy_force
  
    subroutine cal_only_energy(feat,num_tmp,dfeat_tmp,iat_tmp,jneigh_tmp,ifeat_tmp,num_neigh,list_neigh,AL,xatom)
        integer(4)  :: itype,ixyz,i,j,jj,jjj,iat,jn,jat,jtype
        real(8) :: sum,sum0
        real(8) :: dxx,dexp_dd
        real(8),dimension(:,:),intent(in) :: feat
        real*8, dimension (:,:),intent(in) :: dfeat_tmp
        integer(4), intent(in) :: num_tmp
        integer(4),dimension (:),intent(in) :: iat_tmp,jneigh_tmp,ifeat_tmp
        ! real(8),dimension(:,:,:,:),intent(in) :: dfeat
        integer(4),dimension(:),intent(in) :: num_neigh
        integer(4),dimension(:,:),intent(in) :: list_neigh
        real(8), intent(in) :: AL(3,3)
        real(8),dimension(:,:),intent(in) :: xatom
        !real(8),dimension(:) :: energy_pred
        !real(8),dimension(:,:) :: force_pred
        real*8,allocatable,dimension(:,:,:,:) :: dfeat
        real(8) dsum0(3),d_dd(3)
        real(8) akernel(1000),dkernel(1000,3)        
        real*8,allocatable,dimension(:,:) :: feat22_type,feat2N
        real(8),allocatable,dimension(:,:) :: feat2
        real(8),allocatable,dimension(:,:,:) :: feat_type
        real(8),allocatable,dimension(:,:,:) :: feat2_type,feat2_typeN
        integer(4),allocatable,dimension(:,:) :: ind_type
        real(8),allocatable,dimension(:,:,:) :: dfeat_type
        real(8),allocatable,dimension(:,:,:) :: dfeat2_type
        real(8),allocatable,dimension(:,:,:,:) :: dfeat2
        real(8),allocatable,dimension(:,:,:,:) :: SS
        real(8),allocatable,dimension(:,:,:,:) :: dfeat2N
        integer(4),dimension(2) :: feat_shape,list_neigh_shape
        integer(4),dimension(4) :: dfeat_shape

        real*8 pi,dE,dFx,dFy,dFz
        real*8 rad1,rad2,rad,dx1,dx2,dx3,dx,dy,dz,dd,yy,w22,dEdd,d



         pi=4*datan(1.d0)
        
         open(99,file='log.txt',position='append')
         write(99,*)shape(feat)
         write(99,*) num_tmp
         write(99,*)shape(dfeat_tmp)
         write(99,*) shape(iat_tmp)
         write(99,*) shape(jneigh_tmp)
         write(99,*) shape(ifeat_tmp)
         write(99,*) shape(num_neigh)
         write(99,*) shape(list_neigh)
         write(99,*) shape(xatom)
        ! open(99,file='log.txt')
        
        !allocate(energy_pred(natom))
        !allocate(force_pred(3,natom))
        allocate(dfeat(nfeat0m,natom,m_neigh,3))
        dfeat(:,:,:,:)=0.0
        do jj=1,num_tmp
        dfeat(ifeat_tmp(jj),iat_tmp(jj),jneigh_tmp(jj),:)=dfeat_tmp(:,jj)
        enddo

        allocate(feat2N(nfeat2mN,natom))
        allocate(feat2_typeN(nfeat2mN,natom,ntype))
        allocate(dfeat2N(nfeat2mN,natom,m_neigh,3))

        allocate(feat2(nfeat2m,natom))
        allocate(feat_type(nfeat0m,natom,ntype))
        allocate(feat2_type(nfeat2m,natom,ntype))
        allocate(ind_type(natom,ntype))

        feat_shape=shape(feat)
        dfeat_shape=shape(dfeat)
        list_neigh_shape=shape(list_neigh)
        istat=0
        error_msg=''
        open(99,file='log.txt',position='append')
        write(*,*)'feat_shape'
        write(*,*)feat_shape
        write(*,*)'dfeat_shape'
        write(*,*)dfeat_shape
        write(*,*)'list_neigh_shape'
        write(*,*)list_neigh_shape
        write(*,*)"nfeat0m,natom,m_neigh"
        write(*,*)nfeat0m,natom,m_neigh
        close(99)
        open(99,file='log.txt',position='append')
        if (feat_shape(1)/=nfeat0m .or. feat_shape(2)/=natom &
             .or. dfeat_shape(1)/=nfeat0m .or. dfeat_shape(2)/=natom .or. dfeat_shape(4)/=3 &
             .or. size(num_neigh)/=natom  .or. list_neigh_shape(2)/=natom) then      
            
            write(99,*) "Shape of input arrays don't match the model!"
            istat=1
            !if (allocated(error_msg)) then
                !deallocate(error_msg)
            !end if
            error_msg="Shape of input arrays don't match the model!"
            return
        end if
  
        
        
        ! write(99,*) "all arrays shape right"
        ! close(99)
        ! open(99,file='log.txt',position='append')
        
        
        numf=0
        do i=1,natom
        itype=iatom_type(i)
        numf(itype)=numf(itype)+1
        ind_type(numf(itype),itype)=i
        feat_type(:,numf(itype),itype)=feat(:,i)
        enddo
    
        ! write(99,*)"feat_type normlly setted"
        ! close(99)
        ! open(99,file='log.txt',position='append')
        
        do itype=1,ntype
        call dgemm('T','N',nfeat2(itype),numf(itype),nfeat0(itype),1.d0,PV(1,1,itype),nfeat0m,feat_type(1,1,itype),nfeat0m,0.d0,feat2_type(1,1,itype),nfeat2m)
        enddo
    
        !write(99,*)"feat2_type normlly setted first time"
        
        do itype=1,ntype
        do i=1,numf(itype)
        do j=1,nfeat2(itype)-1
        feat2_type(j,i,itype)=(feat2_type(j,i,itype)-feat2_shift(j,itype))*feat2_scale(j,itype)
        enddo
        feat2_type(nfeat2,i,itype)=1.d0
        enddo
        enddo

          
        ! write(99,*)"feat2_type normlly setted second time"
        ! close(99)
        ! open(99,file='log.txt',position='append')
        
        numf=0
        do i=1,natom
        itype=iatom_type(i)
        numf(itype)=numf(itype)+1
        feat2(:,i)=feat2_type(:,numf(itype),itype)
        enddo
          
        ! write(99,*)"feat2 normlly setted first time"
        ! close(99)
        ! open(99,file='log.txt',position='append')
        
        do itype=1,ntype
        do i=1,numf(itype)    ! the atom in the system belong to this itype
          iat=ind_type(i,itype)   ! iat, this atom
          sum0=0.d0
          do jj=1,clusterNum(itype)
          sum=0.d0
          do j=1,nfeat2(itype)
          sum=sum+(feat2_type(j,i,itype)-feat_cent(j,jj,itype))**2*w_feat(j,itype)
          enddo
!TODO:
            if (kernel_type.eq.1) then
            akernel(jj)=exp(-(sum/width(jj,itype))**alpha0(itype))
            else if(kernel_type.eq.2) then
            akernel(jj)=1/(sum**alpha0(itype)+k_dist0(itype)**alpha0(itype))
            end if
          sum0=sum0+akernel(jj)
          enddo
            
            if(sum0.lt.1.E-10) then
                sum0=1.0
            end if
          akernel=akernel/sum0
          
          

          do jj=1,clusterNum(itype)
          do j=1,nfeat2(itype)
          jjj=(jj-1)*nfeat2(itype)+j
          feat2_typeN(jjj,i,itype)=feat2_type(j,i,itype)*akernel(jj)
          enddo
          enddo
        enddo
        enddo
!cccccccc feat2_typeN is the new feature, feature number nfeat2N!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        numf=0
        do i=1,natom
        itype=iatom_type(i)
        numf(itype)=numf(itype)+1
        feat2N(:,i)=feat2_typeN(:,numf(itype),itype)
        enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        do i=1,natom
        itype=iatom_type(i)
        sum=0.d0
        do j=1,nfeat2N(itype)
        sum=sum+feat2N(j,i)*BB_type0(j,itype)
        enddo
        energy_pred(i)=sum
        enddo
      
        ! write(99,*)"energy_pred normlly setted"
        ! close(99)
        ! open(99,file='log.txt',position='append')
        
        deallocate(feat_type)
        deallocate(feat2_type)
        deallocate(ind_type)
        deallocate(feat2N)
        deallocate(feat2_typeN)


       
!ccccccccccccccccccccccccccccccccccccccccccc
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
       w22=dsqrt(wp_atom(iatom_type(i))*wp_atom(iatom_type(j)))
       yy=pi*dd/(4*rad)
!       dE=dE+0.5*w22*exp((1-dd/rad)*4.0)*cos(yy)**2
!       dEdd=w22*exp((1-dd/rad)*4.d0)*((-4/rad)*cos(yy)**2
!     &   -(pi/(2*rad))*cos(yy)*sin(yy))
       dE=dE+0.5*4*w22*(rad/dd)**12*cos(yy)**2

       endif
       endif
       enddo
       energy_pred(i)=energy_pred(i)+dE
       enddo
!ccccccccccccccccccccccccccccccccccccccccccc
        ! write(99,*)"force_pred normlly setted"
        ! close(99)
        ! open(99,file='log.txt',position='append')
        
        etot_pred = 0.d0
        do i = 1, natom
            !etot = etot + energy(i)
            etot_pred = etot_pred + energy_pred(i)
        end do
          
        ! write(*,*)"etot_pred normlly setted"
        ! write(*,*)"energy_pred",shape(energy_pred)
        ! ! write(99,*)energy_pred
        ! write(*,*)"force_pred",shape(force_pred)
        !write(99,*)force_pred

        close(99)
    end subroutine cal_only_energy
  
   
end module calc_clst
  
  
