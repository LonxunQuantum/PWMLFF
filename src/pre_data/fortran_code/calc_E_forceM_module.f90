module calc_E
    !implicit double precision (a-h, o-z)
    implicit none

  !!!!!!!!!!!!! ****   start module variables   ******  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
    character(80),parameter :: fit_input_path0="fit.input"
    character(80),parameter :: model_coefficients_path0="Gfeat_fit.ntype"
    character(80),parameter :: weight_feat_path_header0="weight_feat."
    character(80),parameter :: feat_pv_path_header0="feat_PV."
    character(80),parameter :: feat_ref_path_header0="feat2_ref."
    character(80),parameter :: featvar_ref_path_header0="feat2_ref."
    
    character(200) :: fit_input_path=trim(fit_input_path0)
    character(200) :: model_coefficients_path=trim(model_coefficients_path0)
    character(200) :: weight_feat_path_header=trim(weight_feat_path_header0)
    character(200) :: feat_pv_path_header=trim(feat_pv_path_header0)
    character(200) :: feat_ref_path_header=trim(feat_ref_path_header0)
    character(200) :: featvar_ref_path_header=trim(featvar_ref_path_header0)
  
    integer(4) :: ntype                                    !模型所有涉及的原子种类
    integer(4) :: m_neigh                                  !模型所使用的最大近邻数(考虑这个数是否可以不用)
    integer(4) :: nfeat0m                                  !不同种原子的原始feature数目中最大者(目前似无意义)
    integer(4) :: nfeat2m                                  !不同种原子的PCA之后feature数目中最大者
    integer(4) :: num_ref2m                                 !max num of num_ref in var
    integer(4) :: num_refm                                 !max num of num_ref
    integer(4) :: nfeat2tot                                !PCA之后各种原子的feature数目之和
    integer(4),allocatable,dimension(:) :: nfeat0          !各种原子的原始feature数目
    integer(4),allocatable,dimension(:) :: nfeat2          !各种原子PCA之后的feature数目
    integer(4),allocatable,dimension(:) :: nfeat2i         !用来区分计算时各段各属于哪种原子的分段端点序号
    integer(4),allocatable,dimension(:) :: num_ref         !各种原子取的reference points的数目(对linear无意义)
    integer(4),allocatable,dimension(:) :: num_refi        !用来区分各种原子的reference points的分段端点序号(对linear无意义)
    integer(4),allocatable,dimension(:) :: num_ref2        !reference num for variance
  
  
    real(8),allocatable,dimension(:) :: bb                 !计算erergy和force时与new feature相乘的系数向量w
    real(8),allocatable,dimension(:,:) :: bb_type          !不明白有何作用,似乎应该是之前用的变量
    real(8),allocatable,dimension(:,:) :: bb_type0         !将bb分别归类到不同种类的原子中，第二维才是代表原子种类
    real(8),allocatable,dimension (:, :) :: w_feat         !不同reference points的权重(对linear无意义)
  
    real(8),allocatable,dimension(:,:,:) :: pv             !PCA所用的转换矩阵
    real(8),allocatable,dimension (:,:) :: feat2_shift     !PCA之后用于标准化feat2的平移矩阵
    real(8),allocatable,dimension (:,:) :: feat2_scale     !PCA之后用于标准化feat2的伸缩系数矩阵

    real(8),allocatable,dimension(:,:,:) :: feat2_ref      !reference feat in GPR
    ! real(8),allocatable,dimension(:,:,:) :: Gfeat_type     !kernel in GPR
    real(8),allocatable,dimension(:,:,:) :: featvar_ref      !reference feat in GPR
    real(8),allocatable,dimension(:,:,:) :: feat2_type      !reference feat in GPR
    real(8),allocatable,dimension(:,:,:) :: Gfeat      !reference feat in GPR
    
    integer(4) :: natom                                    !image的原子个数  
    integer(4),allocatable,dimension(:) :: num             !属于每种原子的原子个数，但似乎在calc_linear中无用
    integer(4),allocatable,dimension(:) :: num_atomtype    !属于每种原子的原子个数，似是目前所用的
    integer(4),allocatable,dimension(:) :: itype_atom      !每一种原子的原子属于第几种原子
    integer(4),allocatable,dimension(:) :: iatom           !每种原子的原子序数列表，即atomTypeList
    integer(4),allocatable,dimension(:) :: iatom_type      !每种原子的种类，即序数在种类列表中的序数
    integer(4),allocatable,dimension(:) :: iatom_in_type   !每个原子在其原子种类中的序数
    
    real(8),allocatable,dimension(:) :: energy_pred        !每个原子的能量预测值
    real(8),allocatable,dimension(:,:) :: force_pred       !每个原子的受力预测值
    real(8) :: etot_pred
    character(200) :: error_msg
    integer(4) :: istat
    real(8) :: alpha, dist0,delta,weight_E,weighr_E2,weight_F,rad3
    
    integer(4),allocatable, dimension(:) :: flag_of_types   !每种原子类型中无法预测的原子个数
    real(8), allocatable, dimension(:) :: var_of_atoms      !每个原子的协方差值
    real(8), allocatable, dimension(:) ::  const_f
    integer(4),allocatable, dimension(:) :: direction,add_force_atom
    integer(4) :: add_force_num,power
    real*8,allocatable,dimension(:) :: rad_atom,wp_atom
    
  
  !!!!!!!!!!!!!  *********       end variables      **************    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
        feat_ref_path_header=fit_dir//'/'//trim(feat_ref_path_header0)
        featvar_ref_path_header=fit_dir//'/'//trim(featvar_ref_path_header0)
    end if
end subroutine set_paths

subroutine load_model()
    
    integer(4) :: nimage,num_reftot,nfeat0_tmp,nfeat2_tmp,itype,i,k,ntmp,itmp,num_ref_tmp,j1,j
    real(8) :: dist

!********************** fit.input  ***********************  
    open (10, file=trim(fit_input_path))
    rewind(10)
    read(10,*) ntype,natom,m_neigh,nimage

    if (allocated(itype_atom)) then
        deallocate(itype_atom)
        deallocate(nfeat0)
        deallocate (nfeat2)
        deallocate (num_ref)
        deallocate (num_refi)
        deallocate (rad_atom)
        deallocate (wp_atom)
        deallocate (nfeat2)
        deallocate (num)                              !image数据,在此处allocate，但在set_image_info中赋值
        deallocate (num_atomtype)                     !image数据,在此处allocate，但在set_image_info中赋值
        deallocate (bb)
        deallocate (bb_type)
        !deallocate (bb_type0)
        deallocate (pv)
        deallocate (feat2_shift)
        deallocate (feat2_scale)
        deallocate(w_feat)
        deallocate(feat2_ref)
        deallocate(Gfeat)
        deallocate(add_force_atom)
        deallocate(direction)
        deallocate(const_f)
    end if
    
    allocate(itype_atom(ntype))
    allocate(nfeat0(ntype))
    allocate(nfeat2(ntype))
    allocate(num_ref(ntype))
    allocate(num_refi(ntype))
    allocate(num_ref2(ntype))
    allocate(rad_atom(ntype))
    allocate(wp_atom(ntype))


    do i=1,ntype
        read(10,*) itype_atom(i),nfeat0(i),nfeat2(i),num_ref(i), &
      rad_atom(i),wp_atom(i)
    enddo
    read(10,*) alpha,dist0
    read(10,*) weight_E,weighr_E2,weight_F,delta,rad3,power
    close(10)

    dist0=dist0**2

    nfeat0m=0
    nfeat2m=0
    num_refm=0
    num_reftot=0
    num_refi(1)=0
    do i=1,ntype
        if(nfeat0(i).gt.nfeat0m) nfeat0m=nfeat0(i)
        if(nfeat2(i).gt.nfeat2m) nfeat2m=nfeat2(i)
        if(num_ref(i).gt.num_refm) num_refm=num_ref(i)
        num_reftot=num_reftot+num_ref(i)
        if(i.gt.1) then
            num_refi(i)=num_refi(i-1)+num_ref(i-1)
        endif
    enddo
    
    num_ref2m=num_refm-10
    num_ref2=num_ref-10
    
    allocate (num(ntype))                              !image数据,在此处allocate，但在set_image_info中赋值
    allocate (num_atomtype(ntype))                     !image数据,在此处allocate，但在set_image_info中赋值

!************************** coefficient BB ********************   
    ! allocate (bb(nfeat2tot))
    ! allocate (bb_type(nfeat2m,ntype))
    ! allocate (bb_type0(nfeat2m,ntype))
    allocate(bb(num_reftot))
    allocate(bb_type(num_refm,ntype))
    open (12, file=trim(model_coefficients_path))
    rewind(12)
    read(12,*) ntmp
    if(ntmp.ne.num_reftot) then
        write(6,*) "ntmp.not.right,Gfeat_fit.ntype",ntmp,num_reftot
        stop
    endif
    do i=1,num_reftot
        read(12,*) itmp, BB(i)
    enddo
    close(12)

    do itype=1,ntype
        do k=1,num_ref(itype)
            BB_type(k,itype)=BB(k+num_refi(itype))
        enddo
    enddo
!**************************************************************

!*************************** PCA matrix  PV ************************    
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
    
!*************************************************************

!*************************** feat_ref ************************
    allocate(w_feat(nfeat2m,ntype))
    allocate(feat2_ref(nfeat2m,num_refm,ntype))
    do itype=1,ntype
        open(10,file=trim(weight_feat_path_header)//char(itype+48))
        rewind(10)
        do j=1,nfeat2(itype)
            read(10,*) j1,w_feat(j,itype)
            w_feat(j,itype)=w_feat(j,itype)**2
        enddo
        close(10)
    enddo
    do itype=1,ntype
        open(10,file=trim(feat_ref_path_header)//char(itype+48),form="unformatted")
        rewind(10)
        read(10) nfeat2_tmp,num_ref_tmp
        if(nfeat2_tmp.ne.nfeat2(itype)) then
            write(6,*) "nfeat2.not.same,feat2_ref",itype,nfeat2_tmp,nfeat2(itype)
            stop
        endif
        if(num_ref_tmp.ne.num_ref(itype)) then
            write(6,*) "num_ref.not.same,feat2_ref",itype,num_ref_tmp,num_ref(itype)
            stop
        endif
        read(10) feat2_ref(1:nfeat2(itype),1:num_ref(itype),itype)
        close(10)
    enddo    
!*************************************************************
    allocate(Gfeat(num_ref2m,num_ref2m,ntype))
    
    do itype=1,ntype
        do i=1,num_ref2(itype)
            do k=1,num_ref2(itype)
                dist=0.d0
                do j=1,nfeat2(itype)-1     
                    dist=dist+(feat2_ref(j,i,itype)-feat2_ref(j,k,itype))**2*w_feat(j,itype)
                enddo
        !ccccc The kernel
                Gfeat(k,i,itype)=exp(-(dist/dist0)**alpha)
            enddo
        enddo
    end do  
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

end subroutine load_model

  
subroutine set_image_info(atom_type_list,is_reset)
    integer(4) :: i,j,itype,iitype
    integer(4),dimension(:),intent(in) :: atom_type_list
    logical,intent(in) :: is_reset
    integer(4) :: image_size
    
    ! write(6,*) "0"
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
            deallocate(iatom_in_type)
            deallocate(energy_pred)
            deallocate(force_pred)
        end if
          
        natom=image_size
        allocate(iatom(natom))
        allocate(iatom_type(natom))
        allocate(iatom_in_type(natom))
        allocate(energy_pred(natom))
        allocate(force_pred(3,natom))
        
        !allocate()
          
        ! write(6,*) "1"
          
        iatom=atom_type_list
        ! write(6,*) "2"
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
        ! write(6,*) "3"
        num_atomtype = 0
        iatom_in_type = 0
        do i = 1, natom
            itype = iatom_type(i)
            ! write(6,*) i,itype,num_atomtype(itype)
            num_atomtype(itype) = num_atomtype(itype) + 1
            iatom_in_type(i) = num_atomtype(itype)
            
        end do
        ! write(6,*) "4"
    end if
    ! write(6,*) "5"
end subroutine set_image_info

subroutine cal_energy_force(feat,dfeat,num_neigh,list_neigh,AL,xatom)
    integer(4)  :: itype,ixyz,i,j,jj,ii,k
    real(8) :: sum,sum1,sum2,sum3
    real(8) :: dist,xx
    real(8),dimension(:,:),intent(in) :: feat
    real(8),dimension(:,:,:,:),intent(in) :: dfeat
    integer(4),dimension(:),intent(in) :: num_neigh
    integer(4),dimension(:,:),intent(in) :: list_neigh
    real(8), intent(in) :: AL(3,3)
    real(8),dimension(:,:),intent(in) :: xatom

    !real(8),dimension(:) :: energy_pred
    !real(8),dimension(:,:) :: force_pred
    real(8),allocatable,dimension(:,:,:) :: Gfeat_type
    real(8),allocatable,dimension(:,:) :: WW,QQ,QQ2
    real(8),allocatable,dimension(:,:) :: Gfeat2,dGfeat2

    real(8),allocatable,dimension(:,:) :: feat2
    real(8),allocatable,dimension(:,:,:) :: feat_type
    real(8),allocatable,dimension(:,:,:) :: feat2_type
    integer(4),allocatable,dimension(:,:) :: ind_type
    real(8),allocatable,dimension(:,:,:) :: dfeat_type
    real(8),allocatable,dimension(:,:,:) :: dfeat2_type
    real(8),allocatable,dimension(:,:,:,:) :: dfeat2
    ! real(8),allocatable,dimension(:,:,:,:) :: ss
    real(8),allocatable,dimension(:) :: V
    integer,allocatable,dimension(:) :: num_inv
    integer,allocatable,dimension(:,:) :: index_inv,index_inv2

    integer(4),dimension(2) :: feat_shape,list_neigh_shape
    integer(4),dimension(4) :: dfeat_shape

    real*8 pi,dE,dFx,dFy,dFz
    real*8 rad1,rad2,rad,dx1,dx2,dx3,dx,dy,dz,dd,yy,w22,dEdd

    pi=4*datan(1.d0)

    
    open(99,file='log.txt',position='append')
    write(99,*) "1111 "
 

    feat_shape=shape(feat)
    dfeat_shape=shape(dfeat)
    list_neigh_shape=shape(list_neigh)
    istat=0
    error_msg=''
    m_neigh=dfeat_shape(3)

    if (feat_shape(1)/=nfeat0m .or. feat_shape(2)/=natom &
         .or. dfeat_shape(1)/=nfeat0m .or. dfeat_shape(2)/=natom .or. dfeat_shape(4)/=3 &
         .or. size(num_neigh)/=natom  .or. list_neigh_shape(2)/=natom) then!此处应该比较m_neigh还是m_neigh+1?        
        
        write(99,*) "Shape of input arrays don't match the model!"
        istat=1
        !if (allocated(error_msg)) then
            !deallocate(error_msg)
        !end if
        error_msg="Shape of input arrays don't match the model!"
        return
    end if  
    !allocate(energy_pred(natom))
    !allocate(force_pred(3,natom))
    allocate(feat2(nfeat2m,natom))
    allocate(feat_type(nfeat0m,natom,ntype))
    allocate(feat2_type(nfeat2m,natom,ntype))
    allocate(ind_type(natom,ntype))
    allocate(dfeat_type(nfeat0m,natom*m_neigh*3,ntype))
    allocate(dfeat2_type(nfeat2m,natom*m_neigh*3,ntype))
    allocate(dfeat2(nfeat2m,natom,m_neigh,3))
    
    allocate(Gfeat2(num_refm,natom))
    allocate(dGfeat2(num_refm,natom))
    allocate(Gfeat_type(natom,num_refm,ntype))

    allocate(num_inv(natom))
    allocate(index_inv(3*m_neigh,natom))
    allocate(index_inv2(3*m_neigh,natom))
    
    allocate(WW(num_refm,natom))
    allocate(V(natom))
    allocate(QQ(nfeat2m,natom))
    allocate(QQ2(nfeat2m,natom))



    write(99,*)'feat_shape'
    write(99,*)feat_shape
    write(99,*)'dfeat_shape'
    write(99,*)dfeat_shape
    write(99,*)'list_neigh_shape'
    write(99,*)list_neigh_shape
    write(99,*)"nfeat0m,natom,m_neigh"
    write(99,*)nfeat0m,natom,m_neigh
    close(99)
    open(99,file='log.txt',position='append')


    num = 0
    do i = 1, natom
        itype = iatom_type(i)
        num(itype) = num(itype) + 1
        ind_type(num(itype), itype) = i
        feat_type(:, num(itype), itype) = feat(:, i)
     end do

    write(99,*)"feat_type normlly setted"
    close(99)
    open(99,file='log.txt',position='append')
    
    do itype = 1, ntype
        call dgemm('T', 'N', nfeat2(itype), num(itype), nfeat0(itype), 1.d0, pv(1,1,itype), nfeat0m, feat_type(1,1,itype), nfeat0m, 0.d0, feat2_type(1,1,itype), nfeat2m)
    end do

    write(99,*)"feat2_type normlly setted first time"
    
    do itype = 1, ntype
        do i = 1, num(itype)
            do j = 1, nfeat2(itype) - 1
                feat2_type(j, i, itype) = (feat2_type(j,i,itype)-feat2_shift(j,itype))*feat2_scale(j, itype)
            end do
            feat2_type(nfeat2(itype), i, itype) = 1.d0
        end do
    end do
      
    
    num = 0
    do i = 1, natom
        itype = iatom_type(i)
        num(itype) = num(itype) + 1
        feat2(:, i) = feat2_type(:, num(itype), itype)
    end do
      
    write(99,*)"feat2 normlly setted first time"
    close(99)
    open(99,file='log.txt',position='append')

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!     We calculate the reference here: Gfeat_type(num(itype),num_ref,itype)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    
    do itype=1,ntype
        write(99,*)"num(i)"
        write(99,*)num(itype),num_ref(itype)
        close(99)
        open(99,file='log.txt',position='append')

        do i=1,num(itype) 
            do k=1,num_ref(itype)-1
                dist=0.d0
                ! write(99,*)i,k,dist
                ! close(99)
                ! open(99,file='log.txt',position='append')
                do j=1,nfeat2(itype)-1     
                    dist=dist+(feat2_type(j,i,itype)-feat2_ref(j,k,itype))**2*w_feat(j,itype)
                    ! write(99,*)i,k,j,dist
                    ! close(99)
                    ! open(99,file='log.txt',position='append')
                enddo
!******** kernel, could  be changed ************************
    !       Gfeat_type(i,k,itype)=1/(dist**alpha+dist0**alpha)
                Gfeat_type(i,k,itype)=exp(-(dist/dist0)**alpha)
!***********************************************************
            enddo
            Gfeat_type(i,num_ref(itype),itype)=1   ! important !
        enddo
        
        write(99,*)"Gfeat(8,30,1)"
        write(99,*)Gfeat_type(8,30,1)
        close(99)
        open(99,file='log.txt',position='append')
    

        do i=1,num(itype)
            sum=0.d0
            do k=1,num_ref(itype)
                sum=sum+Gfeat_type(i,k,itype)*BB_type(k,itype)
            enddo
            energy_pred(ind_type(i,itype))=sum
        enddo
        ! write(99,*)"Gfeat(8,30,1)"
        ! write(99,*)Gfeat_type(8,30,1)
        ! close(99)
        ! open(99,file='log.txt',position='append')
    enddo

!------------------------------------------------------------------
!*********************** energy part end ***************************
!----------------------------------------------------------------

    write(99,*)"energy_pred normlly setted"
    close(99)
    open(99,file='log.txt',position='append')
    
    num = 0
    do i = 1, natom
        do jj = 1, num_neigh(i)
            itype = iatom_type(list_neigh(jj,i)) ! this is this neighbor's typ
            num(itype) = num(itype) + 1
            dfeat_type(:, num(itype), itype) = dfeat(:, i, jj, 1)
            num(itype) = num(itype) + 1
            dfeat_type(:, num(itype), itype) = dfeat(:, i, jj, 2)
            num(itype) = num(itype) + 1
            dfeat_type(:, num(itype), itype) = dfeat(:, i, jj, 3)
        end do
    end do
!cccccccc note: num(itype) is rather large, in the scane of natom*num_neigh

    do itype = 1, ntype
        call dgemm('T', 'N', nfeat2(itype), num(itype), nfeat0(itype), 1.d0, pv(1,1,itype), nfeat0m, dfeat_type(1,1,itype), nfeat0m, 0.d0, dfeat2_type(1,1,itype), nfeat2m)
    end do

    num = 0
    do i = 1, natom
        do jj = 1, num_neigh(i)
            itype = iatom_type(list_neigh(jj,i)) ! this is this neighbor's typ
            num(itype) = num(itype) + 1
            do j = 1, nfeat2(itype) - 1
                dfeat2(j, i, jj, 1) = dfeat2_type(j, num(itype), itype)*feat2_scale(j, itype)
            end do
            dfeat2(nfeat2(itype), i, jj, 1) = 0.d0
            num(itype) = num(itype) + 1
            do j = 1, nfeat2(itype) - 1
                dfeat2(j, i, jj, 2) = dfeat2_type(j, num(itype), itype)*feat2_scale(j, itype)
            end do
            dfeat2(nfeat2(itype), i, jj, 2) = 0.d0
            num(itype) = num(itype) + 1
            do j = 1, nfeat2(itype) - 1
                dfeat2(j, i, jj, 3) = dfeat2_type(j, num(itype), itype)*feat2_scale(j, itype)
            end do
            dfeat2(nfeat2(itype), i, jj, 3) = 0.d0
        end do
    end do
    write(99,*)"dfeat2 normlly setted"
    close(99)
    open(99,file='log.txt',position='append')
    
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cc  now, dfeat2 is:
!cc dfeat2(nfeat2,natom,j_neigh,3): dfeat2(j,i,jj,3)= d/dr_i(feat2(j,list_neigh(jj,i))
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccc now, we have the new features, we need to calculate the distance to reference state
    do i=1,natom
        itype=iatom_type(i)  ! this is this neighbor's type 
        do k=1,num_ref(itype)-1
            dist=0.d0
            do j=1,nfeat2(itype)-1     ! The last feature one is 1. 
                dist=dist+(feat2(j,i)-feat2_ref(j,k,itype))**2*w_feat(j,itype)
            enddo
    !       xx=dist**alpha+dist0**alpha
    !       Gfeat2(k,i)=1/xx
    !       dGfeat2(k,i)=-1/xx**2*alpha*dist**(alpha-1)   ! derivative
            xx=exp(-(dist/dist0)**alpha)
            Gfeat2(k,i)=xx
            dGfeat2(k,i)=-alpha/dist0*(dist/dist0)**(alpha-1)*xx
        enddo
        Gfeat2(num_ref(itype),i)=1   ! important !
        dGfeat2(num_ref(itype),i)=0.d0
    enddo
!************************************************
!***************** Now, the most expensive loop

    num_inv=0
    do i=1,natom
    do j=1,num_neigh(i)
    ii=list_neigh(j,i)
    num_inv(ii)=num_inv(ii)+1
    index_inv(num_inv(ii),ii)=i
    index_inv2(num_inv(ii),ii)=j
    enddo
    enddo

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    QQ2=0.d0
    do itype=1,ntype
        do ii=1,num_atomtype(itype)
            sum=0.d0
            do k=1,num_ref(itype)
                WW(k,ii)=dGfeat2(k,ind_type(ii,itype))*BB_type(k,itype)
                sum=sum+WW(k,ii)
            enddo
            V(ii)=sum
        enddo


        call dgemm('N','N',nfeat2(itype),num_atomtype(itype),num_ref(itype),1.d0,&
    &  feat2_ref(1,1,itype),nfeat2m,WW,num_refm,0.d0,QQ,nfeat2m)


        do ii=1,num_atomtype(itype)
            do j=1,nfeat2(itype)
            QQ2(j,ind_type(ii,itype))=2*(feat2(j,ind_type(ii,itype))*V(ii)-QQ(j,ii))*w_feat(j,itype)
            enddo
        enddo

    enddo ! ityoe



    do i=1,natom

        sum1=0.d0
        sum2=0.d0
        sum3=0.d0
        do jj=1,num_neigh(i)
            ii=list_neigh(jj,i)
            itype=iatom_type(ii) 
            do j=1,nfeat2(itype)
                sum1=sum1+QQ2(j,ii)*dfeat2(j,i,jj,1)
                sum2=sum2+QQ2(j,ii)*dfeat2(j,i,jj,2)
                sum3=sum3+QQ2(j,ii)*dfeat2(j,i,jj,3)
            enddo
        enddo
        force_pred(1,i)=sum1 !give a force on x axis
        force_pred(2,i)=sum2
        force_pred(3,i)=sum3
    enddo

    do j=1,add_force_num
        do i=1,natom
            if (i .eq. add_force_atom(j) ) then
                force_pred(1,i)= force_pred(1,i)+(direction(j)-1)*const_f(j)   !give a force on x axis
            end if
        enddo
    enddo

!*************************    
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
       if (iatom_type(i) .ne. iatom_type(j)) rad=rad3
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
       dE=dE+0.5*4*w22*(rad/dd)**power*cos(yy)**2
       dEdd=4*w22*(-power*(rad/dd)**power/dd*cos(yy)**2   &
        -(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**power)

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

    
    write(99,*)"force_pred normlly setted"
    close(99)
    open(99,file='log.txt',position='append')
    
    etot_pred = 0.d0
    do i = 1, natom
        !etot = etot + energy(i)
        etot_pred = etot_pred + energy_pred(i)
    end do
      
    write(99,*)"etot_pred normlly setted"
    write(99,*)"energy_pred",shape(energy_pred)
    write(99,*)energy_pred
    write(99,*)"force_pred",shape(force_pred)
    write(99,*)force_pred
    close(99)
    
    deallocate(feat2)
    deallocate(feat_type)
    deallocate(feat2_type)
    deallocate(ind_type)
    deallocate(dfeat_type)
    deallocate(dfeat2_type)
    deallocate(dfeat2)

    deallocate(Gfeat2)
    deallocate(dGfeat2)
    deallocate(Gfeat_type)

    deallocate(num_inv)
    deallocate(index_inv)
    deallocate(index_inv2)
    
    deallocate(WW)
    deallocate(V)
    deallocate(QQ)
    deallocate(QQ2)

end subroutine cal_energy_force

subroutine cal_pca(feat,num_neigh,list_neigh)

    integer(4)  :: itype,ixyz,i,j,jj,ii,k
    real(8) :: sum,sum1,sum2,sum3

    real(8),dimension(:,:),intent(in) :: feat
    integer(4),dimension(:),intent(in) :: num_neigh
    integer(4),dimension(:,:),intent(in) :: list_neigh
    !real(8),dimension(:) :: energy_pred
    !real(8),dimension(:,:) :: force_pred

    real(8),allocatable,dimension(:,:) :: feat2
    real(8),allocatable,dimension(:,:,:) :: feat_type
    integer(4),allocatable,dimension(:,:) :: ind_type

    ! real(8),allocatable,dimension(:,:,:,:) :: ss

    integer(4),dimension(2) :: feat_shape,list_neigh_shape
    integer(4),dimension(4) :: dfeat_shape

    open(99,file='log.txt',position='append')
    write(99,*) "1111 "


    feat_shape=shape(feat)
    list_neigh_shape=shape(list_neigh)
    istat=0
    error_msg=''


    if (feat_shape(1)/=nfeat0m .or. feat_shape(2)/=natom &
!        .or. dfeat_shape(1)/=nfeat0m .or. dfeat_shape(2)/=natom .or. dfeat_shape(4)/=3 &
        .or. size(num_neigh)/=natom  .or. list_neigh_shape(2)/=natom) then!此处应该比较m_neigh还是m_neigh+1?        
        
        write(99,*) "Shape of input arrays don't match the model!"
        istat=1
        !if (allocated(error_msg)) then
            !deallocate(error_msg)
        !end if
        error_msg="Shape of input arrays don't match the model!"
        return
    end if  

    allocate(feat2(nfeat2m,natom))
    allocate(feat_type(nfeat0m,natom,ntype))
    allocate(feat2_type(nfeat2m,natom,ntype))
    allocate(ind_type(natom,ntype))



    num = 0
    do i = 1, natom
        itype = iatom_type(i)
        num(itype) = num(itype) + 1
        ind_type(num(itype), itype) = i
        feat_type(:, num(itype), itype) = feat(:, i)
    end do

    write(99,*)"feat_type normlly setted"
    close(99)
    open(99,file='log.txt',position='append')

    do itype = 1, ntype
        call dgemm('T', 'N', nfeat2(itype), num(itype), nfeat0(itype), 1.d0, pv(1,1,itype), nfeat0m, feat_type(1,1,itype), nfeat0m, 0.d0, feat2_type(1,1,itype), nfeat2m)
    end do

    write(99,*)"feat2_type normlly setted first time"

    do itype = 1, ntype
        do i = 1, num(itype)
            do j = 1, nfeat2(itype) - 1
                feat2_type(j, i, itype) = (feat2_type(j,i,itype)-feat2_shift(j,itype))*feat2_scale(j, itype)
            end do
            feat2_type(nfeat2(itype), i, itype) = 1.d0
        end do
    end do

end subroutine cal_pca


subroutine calculate_var(feat,num_neigh,list_neigh)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccc We calculate the reference here: Gfeat_type(num(itype),num_ref,itype)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    ! real*8,allocatable,dimension(:,:,:) :: var
    ! integer,dimension(:) :: num_ref2,num

!**********************************************        



    real(8),dimension(:,:),intent(in) :: feat
    integer(4),dimension(:),intent(in) :: num_neigh
    integer(4),dimension(:,:),intent(in) :: list_neigh
    
    
    integer(4)  :: itype,i,j,k,m,n,info
    integer, dimension(ntype) :: flag
    real(8),allocatable,dimension(:,:,:) :: Gfeat2
    real(8),allocatable,dimension(:,:,:) :: HH,KX,XX,var
    real(8) :: dist
    integer,allocatable,dimension(:) :: ipiv2
    real(8),allocatable,dimension(:,:) :: Gfeat21
    real(8),allocatable,dimension(:,:) :: HH1,KX1,XX1
    ! real(8) :: dist
    integer,allocatable,dimension(:) :: ipiv21

    ! num_ref2=num_ref-1
   
    ! allocate(var(natom,natom,ntype))

    allocate(HH(num_ref2m,num_ref2m,ntype))
    allocate(KX(num_ref2m,natom,ntype))
    allocate(XX(natom,natom,ntype))
    allocate(ipiv2(num_ref2m))      
    allocate(var(natom,natom,ntype))

!******************* calculate K(X*,X) ******************
    allocate(Gfeat2(num_ref2m,natom,ntype))
    
    if (allocated(flag_of_types)) then
        deallocate(flag_of_types)
    end if
    if (allocated(var_of_atoms)) then
        deallocate(var_of_atoms)
    end if
    
    allocate(flag_of_types(ntype))
    allocate(var_of_atoms(natom))
    
    
    call cal_pca(feat,num_neigh,list_neigh)

    write(99,*)"after cal_pca"
    close(99)
    open(99,file='log.txt',position='append')

    do itype=1,ntype
        do i=1,num(itype) 
            do k=1,num_ref2(itype)
                dist=0.d0
                do j=1,nfeat2(itype)-1     
                    dist=dist+(feat2_type(j,i,itype)-feat2_ref(j,k,itype))**2*w_feat(j,itype)
                enddo
        !ccccc The kernel
                Gfeat2(k,i,itype)=exp(-(dist/dist0)**alpha)
            enddo
        enddo
    end do       
!********************************************************
    write(99,*)"Gfeat2 normlly setted"
    close(99)
    open(99,file='log.txt',position='append')
!******************* calculate K(X,X) ******************

!*******************************************************
    write(99,*)"Gfeat normlly setted"
    close(99)
    open(99,file='log.txt',position='append')
    ! do i=1,ntype
    ! write(99,*) num_ref2(i)
    ! enddo
!******************* calculate K(X*,X)K(X,X)^-1K(X,X*) ******************
    HH(:,:,:)=Gfeat(:,:,:)
!    write(99,*)HH(3,3,1)
!    close(99)
!    open(99,file='log.txt',position='append')
    do itype=1,ntype
        do j=1,num_ref2(itype)
            HH(j,j,itype)=HH(j,j,itype)+delta
        enddo
    enddo
    write(99,*)"HH"
    ! write(99,*)HH(3,40,1)
    close(99)
    open(99,file='log.txt',position='append')
    KX(:,:,:)=Gfeat2(:,:,:)
    ! write(99,*)"KX"
    ! write(99,*)KX(3,40,1)
    ! close(99)
    ! open(99,file='log.txt',position='append')

    ! num_ref2=num_ref-1
    do i=1,ntype
        allocate(Gfeat21(num_ref2(i),num(i))) 
        allocate(HH1(num_ref2(i),num_ref2(i)))
        allocate(KX1(num_ref2(i),num(i)))
        allocate(XX1(num(i),num(i)))
        allocate(ipiv21(num_ref2(i)))   
        HH1=HH(:,:,i)
        KX1=KX(:,1:num(i),i)
        Gfeat21=Gfeat2(:,1:num(i),i)


        call dgesv(num_ref2(i),num(i),HH1,num_ref2(i),ipiv21,KX1,num_ref2(i),info)

        call dgemm('T','N',num(i),num(i),num_ref2(i),1.d0,Gfeat21,num_ref2(i),KX1,num_ref2(i),0.d0,XX1,num(i))         
        do m=1,num(i)
            do n=1,num(i)
                var(m,n,i)=1-XX1(m,n)!/KK(m,n)
            enddo
            ! write(99,*)"var"
            ! write(99,*)var(m,m,i)
            ! close(99)
            ! open(99,file='log.txt',position='append')
        enddo
        ! do m=1,num(1)
        !     write(99,*)"XX"
        !     write(99,*)1-XX1(m,m)
        !     close(99)
        !     open(99,file='log.txt',position='append')
        ! enddo
        deallocate(Gfeat21) 
        deallocate(HH1)
        deallocate(KX1)
        deallocate(XX1)
        deallocate(ipiv21) 
    enddo   


    ! do itype=1,ntype
    !   call dgesv(num_ref2(itype),num(itype),HH(1:num_ref2(itype),1:num_ref2(itype),itype),num_ref2(itype),ipiv2(1:num_ref2(itype)),KX(1:num_ref2(itype),1:num(itype),itype),num_ref2(itype),info)
    ! !    call dgesv(num_ref2(itype),num(itype),HH(1,1,itype),num_ref2(itype),ipiv2,KX(1,1,itype),num_ref2(itype),info)
    !     ! write(99,*)"1"
    !     ! write(99,*)KX(3,3,1)
    !     ! close(99)
    !     ! open(99,file='log.txt',position='append')
    !     call dgemm('T','N',num(itype),num(itype),num_ref2(itype),1.d0,Gfeat2(1:num_ref2(itype),1:num(itype),itype),&
    !     & num_ref2(itype),KX(1:num_ref2(itype),1:num(itype),itype),num_ref2(itype),0.d0,XX(1,1,itype),num(itype))      
    !     ! call dgemm('T','N',num(itype),num(itype),num_ref2(itype),1.d0,Gfeat2(1,1,itype),&
    !     ! & num_ref2(itype),KX(1,1,itype),num_ref2(itype),0.d0,XX(1,1,itype),num(itype))      
    ! enddo   


    ! write(99,*)"before cal_var normlly setted"
    ! ! write(99,*)XX(3,3,1)
    ! close(99)
    ! open(99,file='log.txt',position='append')
!******************* calculate var***********************************
    ! do itype=1,ntype
    !     do m=1,num(itype)
    !         do n=1,num(itype)
    !             var(m,n,itype)=1-XX(m,n,itype)!/KK(m,n)

    !         enddo
    !         write(99,*)"var"
    !         write(99,*)var(m,m,itype)
    !         close(99)
    !         open(99,file='log.txt',position='append')
    !     enddo
    ! enddo
!************ output flag ***********     

    do itype=1,ntype
        flag(itype)=0
        do m=1,num(itype)
            if (var(m,m,itype) .gt. 0.2) then
                flag(itype)=flag(itype)+1
            end if 
        enddo
    enddo

    flag_of_types=flag
    var_of_atoms=0.0
    do i=1,natom
        itype=iatom_type(i)        
        var_of_atoms(i)=var(iatom_in_type(i),iatom_in_type(i),itype)
    end do
    
!cccccccccccccccccc output var cccccccccccccccccccccc

    ! open(15,file="var")
    ! ! rewind(15)
    ! do itype=1,ntype
    !     do i=1,num(itype)
    !     write(15,*) var(i,i,itype)!,(Ei_case(i)-E_fit(i))**2
    !     enddo
    ! enddo
    ! close(15)
    deallocate(HH)
    deallocate(KX)
    deallocate(XX)
    deallocate(ipiv2)      
    deallocate(Gfeat2)
    deallocate(feat2_type)

end subroutine
   
end module calc_E

