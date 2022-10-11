  !// forquill v1.01 beta www.fcode.cn
module calc_deepMD
! This version has used the bug in the original DP code, e.g., the ghost
! neighbore in s_neigh, and dxyz_neigh
! It is controlled by iflag_ghost_neigh
! if iflag_ghost_neigh=1, the result depends on m_neigh
    use mod_mpi
    use calc_deepmd_f,only:num_neigh,s_neigh,ds_neigh,dR_neigh,dxyz_neigh,dxyz_dx_neigh,list_neigh,gen_deepMD_feature
    !implicit double precision (a-h, o-z)
    implicit none

  !!!!!!!!!!!!!          以下为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
    character(80),parameter :: feat_info_path0="fread_dfeat/feat.info"
    character(80),parameter :: model_Wij_path01="embeding.net"
    character(80),parameter :: model_Wij_path02="fitting.net"   
    !character(80),parameter :: model_Scaler_path0="fread_dfeat/data_scaler.txt"
    character(80),parameter :: vdw_path0="fread_dfeat/vdw_fitB.ntype"
    
    character(200) :: feat_info_path=trim(feat_info_path0)
    character(200) :: model_Wij_path1=trim(model_Wij_path01)
    character(200) :: model_Wij_path2=trim(model_Wij_path02)
    !character(200) :: model_Scaler_path=trim(model_Scaler_path0)
    character(200) :: vdw_path=trim(vdw_path0)
  
    integer(4) :: ntype                                    !模型所有涉及的原子种类
    integer(4) :: m_neigh                                  !模型所使用的最大近邻数(考虑这个数是否可以不用)
    integer(4) :: nfeat1m                                  !不同种原子的原始feature数目中最大者(目前似无意义)
    integer(4),allocatable,dimension(:) :: nfeat1          !各种原子的原始feature数目
    ! integer(4),allocatable,dimension(:) :: num_ref         !各种原子取的reference points的数目(对linear无意义)
    ! integer(4),allocatable,dimension(:) :: num_refi        !用来区分各种原子的reference points的分段端点序号(对linear无意义)
  
  
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
    real(8) :: etot_pred_deepMD
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
    integer node_em(20),node_NN(20) 
    real*8, allocatable,dimension(:,:,:,:,:) :: Wij_em
    real*8, allocatable,dimension(:,:,:,:) :: B_em
    real*8, allocatable,dimension(:,:,:,:) :: Wij_NN
    real*8, allocatable,dimension(:,:,:) :: B_NN
    real*8, allocatable,dimension(:,:,:) :: W_res_NN
    integer nodeMM_em,nlayer_em,nodeMM_NN,nlayer_NN
    integer nodeMM,nlayer ! to be removed
    integer iflag_resNN(100)
  
    
  
  !!!!!!!!!!!!!          以上为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
    contains
    
  
   
    subroutine set_paths_deepMD(fit_dir_input)
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
            feat_info_path=fit_dir//'/'//trim(feat_info_path0)
            model_Wij_path1=fit_dir//'/'//trim(model_Wij_path01)
            model_Wij_path2=fit_dir//'/'//trim(model_Wij_path02)
            !model_Scaler_path=fit_dir//'/'//trim(model_Scaler_path0)
            vdw_path=fit_dir//'/'//trim(vdw_path0)
        end if
    end subroutine set_paths_deepMD
    
    subroutine load_model_deepMD()
    
        integer(4) :: nimage,num_refm,num_reftot,nfeat1_tmp,itype,i,k,itmp,j1
        integer(4) :: iflag_PCA,kkk,ntype_tmp,iatom_tmp,ntype_t,nterm,itype_t
        real(8) :: dist0
        character*20 txt
        integer i1,i2,j,ii,ntmp,nlayer_tmp,j2
        real*8 w_tmp,b_tmp
        integer itype1,itype2,ntype_pair,ll
        integer node_tmp,node_nn_tmp(100),nlayer_nn_tmp
        real*8 sum

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
        open (12, file=trim(model_Wij_path1))
        rewind (12)
        read(12,*) ntype_pair
        read(12,*) nlayer_em   ! _em: embedding
        read(12,*) (node_em(ll),ll=1,nlayer_em+1)
        if(ntype_pair.ne.ntype**2) then
        write(6,*) "ntype_pair.ne.ntype**2,stop,deepMD",ntype_pair,ntype
        stop
        endif

        nodeMM_em=0
        do itype=1,ntype_pair
        do ii=1,nlayer_em+1
        if(node_em(ii).gt.nodeMM_em) nodeMM_em=node_em(ii)
        enddo
        enddo

        allocate(Wij_em(nodeMM_em,nodeMM_em,nlayer_em,ntype,ntype))
        allocate(B_em(nodeMM_em,nlayer_em,ntype,ntype))


        do itype1=1,ntype
        do itype2=1,ntype
        do ll=1,nlayer_em
         do j1=1,node_em(ll)
          read(12,*) (wij_em(j1,j2,ll,itype2,itype1),j2=1,node_em(ll+1))
! wij_em(j1,j2,ll,itype2,itype1):
! itype1: center atom
! itype2: neigboring atom
! j1: the layer ll node
! j2: the layer ll+1 node

! b_em(j2,ll,itype2,itype1): it is for ll+1 (before the input of ll+1), before
! nonlincear function
         enddo
         read(12,*) (b_em(j2,ll,itype2,itype1),j2=1,node_em(ll+1))
        enddo
        enddo
        enddo
        close(12)



        open (12, file=trim(model_Wij_path2))
        rewind (12)
        read(12,*) ntype
        read(12,*) nlayer_nn   ! _em: embedding
        read(12,*) (node_nn(ll),ll=1,nlayer_nn+1)

        nodeMM_nn=0
        do itype=1,ntype
        do ii=1,nlayer_nn+1
        if(node_nn(ii).gt.nodeMM_nn) nodeMM_nn=node_nn(ii)
        enddo
        enddo

        allocate(Wij_nn(nodeMM_nn,nodeMM_nn,nlayer_nn,ntype))
        allocate(B_nn(nodeMM_nn,nlayer_nn,ntype))

        do itype=1,ntype
        do ll=1,nlayer_nn
         do j1=1,node_nn(ll)
          read(12,*) (Wij_NN(j1,j2,ll,itype),j2=1,node_nn(ll+1))
! Wij_NN(j1,j2,itype):
! itype: center atom
! j1: the layer ll node
! j2: the layer ll+1 node
         enddo
         read(12,*) (B_NN(j2,ll,itype),j2=1,node_nn(ll+1))
        enddo
        enddo
        close(12)


        allocate(W_res_NN(nodeMM_nn,nlayer_nn+1,ntype))


        open(12,file="fittingNet.resnet")
        read(12,*) ntype_tmp
        read(12,*) nlayer_nn_tmp
        read(12,*) (node_nn_tmp(ll),ll=1,nlayer_nn_tmp+1)
        if(ntype_tmp.ne.ntype.or.nlayer_nn_tmp.ne.nlayer_nn) then
        write(6,*) "ntype,nlayer_nn changed,stop",node_nn,nlayer_nn,node_nn_tmp,nlayer_nn_tmp
        stop
        endif
        sum=0.d0
        do ll=1,nlayer_nn
        sum=sum+abs(node_nn(ll)-node_nn_tmp(ll))
        enddo
        if(sum.gt.0.1) then
        write(6,*) "node_nn changed,stop"
        write(6,*) node_nn(1:nlayer_nn)
        write(6,*) node_nn_tmp(1:nlayer_nn)
        stop
        endif
        read(12,*) (iflag_resNN(ll),ll=1,nlayer_nn+1)
        do itype=1,ntype
        do ll=1,nlayer_nn+1
        if(iflag_resNN(ll).eq.1) then
        read(12,*) node_tmp
        if(node_tmp.ne.node_nn(ll)) then
        write(6,*) "node_tmp.ne.node_nn(ll),stop",node_tmp,node_nn(ll)
        stop
        endif
        read(12,*) (W_res_NN(j1,ll,itype),j1=1,node_tmp)
        endif
        enddo
        enddo
        close(12)


        write(6,*) "finished read W_res_NN"


!cccccccccccccccccccccccccccccccccccccccccccccccccccc

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
    !     status = access ("add_force",' ')    ! blank mode
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

    end subroutine load_model_deepMD
  
    subroutine set_image_info_deepMD(iatom_tmp,is_reset,natom_tmp)
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
                iatom_type(i) = iitype
            end do
      
            num_atomtype = 0
            do i = 1, natom
                itype = iatom_type(i)
                num_atomtype(itype) = num_atomtype(itype) + 1
            end do
        end if
        
    end subroutine set_image_info_deepMD
  
    subroutine cal_energy_force_deepMD(AL,xatom,Etot,fatom)

        integer(4)  :: itype,ixyz,i,j,jj
        integer natom_tmp,nfeat0_tmp,m_neigh_tmp,kk
        real(8) :: sum,direct,mean
        real(8), intent(in) :: AL(3,3)
!        real(8),dimension(:,:),intent(in) :: xatom
        real(8) xatom(3,natom),fatom(3,natom)
        real(8) Etot


        integer natom_n_type(50)
        integer,allocatable,dimension(:,:) :: iat_ind
                
        
        real(8),allocatable,dimension(:,:,:) :: feat_type
!        real(8),allocatable,dimension(:,:,:) :: dfeat_type
        
        real(8),allocatable,dimension(:,:,:) :: f_in,f_out,f_back,f_back0
        real(8),allocatable,dimension(:,:,:,:) :: f_d
        real(8),allocatable,dimension(:,:) :: energy_type
        real(8),allocatable,dimension(:,:,:) :: dEdf_type
        real(8),allocatable,dimension(:,:,:) :: ss
        real(8),allocatable,dimension(:,:,:) :: f_in_NN,f_out_NN,f_d_NN
        real(8),allocatable,dimension(:,:,:,:,:,:) :: d_ss
        real(8),allocatable,dimension(:,:,:,:) :: dE_dx
        real(8),allocatable,dimension(:,:,:,:,:) :: d_ss_fout
        real(8),allocatable,dimension(:,:) :: dE_dfout
        real(8),allocatable,dimension(:,:,:) :: f_back0_em,f_back_em
        real(8),allocatable,dimension(:,:) :: force_all,force_all_tmp
        
        real(8),allocatable,dimension(:,:,:) :: s_neigh_tmp

        real*8 pi,dE,dFx,dFy,dFz
        real*8 rad1,rad2,rad,dx1,dx2,dx3,dx,dy,dz,dd,yy,w22,dEdd,d,w22_1,w22_2,w22F_1,w22F_2
        integer iat1,iat2,ierr
        integer ii
        real*8 x
        real*8 sum1,sum2,sum3
        integer natom_m_type,jjm,itype1,itype2,iat,ll,num,nn1,k,m,k1,k2
        real*8 Etot_tmp
        integer j1,j2,kkk
        real*8 y
        real*8 dE0,dE1,energy_type0,dE_df,d_sum,d_sum2
        integer m2
        real*8 energy_st0,dE_dx0,dE2
        real*8 Etot0,test_tmp
        real*8 tt0,tt1,tt2

        integer iflag_ghost_neigh,neigh_add
        real*8 fact1

        ! write(6,*) "nnodes,inode",nnodes,inode

        iflag_ghost_neigh=1   ! 1: use the ghost neigh, 0: not use the ghost neigh

        pi=4*datan(1.d0)
        
        tt0=mpi_wtime()
        call gen_deepMD_feature(AL,xatom)
        tt1=mpi_wtime()

        istat=0
        error_msg=''

        natom_n_type= 0
        iat1=0
        do i = 1, natom
            if(mod(i-1,nnodes).eq.inode-1) then
                iat1=iat1+1
                itype = iatom_type(i)
                natom_n_type(itype) = natom_n_type(itype) + 1
            endif
        enddo

        natom_m_type=0
        do itype=1,ntype
            if(natom_n_type(itype).gt.natom_m_type) natom_m_type=natom_n_type(itype)
        enddo

        write(*,*) "natom_m_type", natom_m_type

        allocate(iat_ind(natom_m_type,ntype))

        natom_n_type= 0
        iat1=0
        do i = 1, natom
            if(mod(i-1,nnodes).eq.inode-1) then
                iat1=iat1+1
                    itype = iatom_type(i)
                    natom_n_type(itype) = natom_n_type(itype) + 1
                    iat_ind(natom_n_type(itype),itype)=i
            endif
        enddo

        !ccccccccccccccccccccc test

        jjm=0

        do itype1=1,ntype
            do itype2=1,ntype
                jj=0
                do i=1,natom_n_type(itype1)
                    iat=iat_ind(i,itype1)

                    neigh_add=0
                    if(iflag_ghost_neigh.eq.1.and.num_neigh(itype2,iat).lt.m_neigh) neigh_add=1
                    ! the neigh_add is the ghost neighbor
                    
                    ! wlj altered
                    do j=1,num_neigh(itype2,iat)+neigh_add
                       jj=jj+1
                    enddo
                    !jj = jj + m_neigh
                    
                enddo
                
                if(jj.gt.jjm) jjm=jj
                
            enddo
        enddo

        nodeMM_em=0
        do ll=1,nlayer_em+1
            if(node_em(ll).gt.nodeMM_em) nodeMM_em=node_em(ll)
        enddo

        nodeMM_NN=0
        do ll=1,nlayer_NN+1
            if(node_NN(ll).gt.nodeMM_NN) nodeMM_NN=node_NN(ll)
        enddo
        

        allocate(f_in(nodeMM_em,jjm,nlayer_em+1))
        allocate(f_out(nodeMM_em,jjm,nlayer_em+1))
        allocate(f_d(nodeMM_em,jjm,nlayer_em+1,ntype))
        allocate(f_back0_em(nodeMM_em,jjm,nlayer_em+1))
        allocate(f_back_em(nodeMM_em,jjm,nlayer_em+1))
        allocate(ss(4,node_em(nlayer_em+1),natom_m_type))
        allocate(f_in_NN(nodeMM_NN,natom_m_type,nlayer_NN+1))
        allocate(f_out_NN(nodeMM_NN,natom_m_type,nlayer_NN+1))
        allocate(f_back(nodeMM_NN,natom_m_type,nlayer_NN+1))
        allocate(f_back0(nodeMM_NN,natom_m_type,nlayer_NN+1))
        allocate(f_d_NN(nodeMM_NN,natom_m_type,nlayer_NN+1))
        allocate(energy_type(natom_m_type,ntype))
        allocate(d_ss(4,3,nodeMM_em,m_neigh,ntype,natom_m_type))
        allocate(dE_dx(3,m_neigh,ntype,natom))
        allocate(d_ss_fout(4,nodeMM_em,m_neigh,ntype,natom_m_type))
        allocate(dE_dfout(nodeMM_em,jjm))

        !do 400 itype1=1,ntype    ! center atom

        write(*,*) "writing num_neigh(itype2,iat)+neigh_add"
        write(*,*) num_neigh(1,1)

        dE_dx=0.d0
        do 400 itype1=1,ntype

            f_in_NN=0.d0
            ss=0.d0
            d_ss=0.d0
            d_ss_fout=0.d0
            ! do 300 itype2=1,ntype    ! neighboring atom

            do itype2=1,ntype

                jj=0 
                
                do i=1,natom_n_type(itype1)
                    iat=iat_ind(i,itype1)
                    neigh_add=0
                    
                    if(iflag_ghost_neigh.eq.1.and.num_neigh(itype2,iat).lt.m_neigh) neigh_add=1
                    
                    if ((itype1.eq.1).and.(itype2.eq.1).and.(i.eq.1)) then
                        write(*,*) "writing s_neigh"    
                        write(*,*) s_neigh(:,1,1) 
                    endif 

                    ! wlj altered
                    do j=1,num_neigh(itype2,iat)+neigh_add
                    !do j=1,m_neigh 

                        jj=jj+1 
                        f_in(1,jj,1)=s_neigh(j,itype2,iat)
                        f_out(1,jj,1)=s_neigh(j,itype2,iat)
                        f_d(1,jj,1,itype2)=1.d0
                        
                    enddo
                enddo

                num=jj     ! the same (itype2,itype1), all the neigh, and all the atomi belong to this CPU

                do ll=1,nlayer_em
            
                    call dgemm('T', 'N', node_em(ll+1),num,node_em(ll), 1.d0,  &
                        Wij_em(1,1,ll,itype2,itype1),nodeMM_em,f_out(1,1,ll),nodeMM_em,0.d0,f_in(1,1,ll+1),nodeMM_em)

                    do i=1,num
                        do j=1,node_em(ll+1)
                            f_in(j,i,ll+1)=f_in(j,i,ll+1)+B_em(j,ll,itype2,itype1)
                        enddo
                    enddo   
                    

                    do i=1,num
                        do j=1,node_em(ll+1)
                            x=f_in(j,i,ll+1)
                            if(x.gt.20.d0) then
                                y=1.d0
                            elseif(x.gt.-20.d0.and.x.le.20.d0) then
                                y=(exp(x)-exp(-x))/(exp(x)+exp(-x))
                            elseif(x.lt.-20.d0) then
                                y=-1.d0
                            endif
                            !  f_out(j, i, ll) = (exp(x)-exp(-x)) / (exp(x)+exp(-x))  ! tanh ! maybe if softplus, sigmoid, tanh
                            f_out(j, i, ll+1) = y
                            f_d(j, i, ll+1,itype2) = 1.0d0 - f_out(j,i,ll+1)*f_out(j,i,ll+1)
                        enddo
                    enddo
                    
                    
                    if (ll.eq.2) then 
                        write(*,*) "before reconnect"
                        if ((itype1.eq.1).and.(itype2.eq.1)) then  
                            !write(*,*) "printing dxyz_neigh"
                            !write(*,*) dxyz_neigh(1,1,1,1:42)
                            write(*,*) "printing f_out"
                            write(*,*) f_out(1,:,3)
                        endif 

                        !write(*,*) "node_em(ll+1), 2*node_em(ll)", node_em(ll+1), 2*node_em(ll)
                    endif 

                    !  This is the reconnect
                    
                    if(node_em(ll+1).eq.node_em(ll)) then
                        !write(*,*) "test", f_out(1,1,ll)
                        do i=1,num
                            do j=1,node_em(ll)
                                f_out(j,i,ll+1)=f_out(j,i,ll+1)+f_out(j,i,ll)
                                !f_out(j+node_em(ll),i,ll+1)=f_out(j+node_em(ll),i,ll+1)+f_out(j,i,ll)
                            enddo
                        enddo
                    endif

                    if(node_em(ll+1).eq.2*node_em(ll)) then
                        !write(*,*) "test", f_out(1,1,ll)
                        do i=1,num
                            do j=1,node_em(ll)
                                f_out(j,i,ll+1)=f_out(j,i,ll+1)+f_out(j,i,ll)
                                f_out(j+node_em(ll),i,ll+1)=f_out(j+node_em(ll),i,ll+1)+f_out(j,i,ll)
                            enddo
                        enddo
                    endif
                    
                    
                    if (ll.eq.2) then 
                        write(*,*) "after reconnect"
                        if ((itype1.eq.1).and.(itype2.eq.1)) then  
                            !write(*,*) "printing dxyz_neigh"
                            !write(*,*) dxyz_neigh(1,1,1,1:42)
                            write(*,*) "printing f_out"
                            write(*,*) f_out(1,:,3)
                        endif 
                    endif 

                enddo
                
                !ccccccccccccc   get the 100 R(s).
                nn1=node_em(nlayer_em+1)

                jj=0 
                
                if ((itype1.eq.1).and.(itype2.eq.1)) then  
                    !write(*,*) "printing dxyz_neigh"
                    !write(*,*) dxyz_neigh(1,1,1,1:42)
                    write(*,*) "printing f_out"
                    write(*,*) f_out(1,:,3)
                endif 
                
                ! index of atom in this type
                do i=1,natom_n_type(itype1)
                    
                    iat=iat_ind(i,itype1)
                    neigh_add=0
                    
                    if(iflag_ghost_neigh.eq.1.and.num_neigh(itype2,iat).lt.m_neigh) neigh_add=1
                        
                    do j=1,num_neigh(itype2,iat)+neigh_add  ! j is sum over

                        jj=jj+1
                        fact1=1
                        
                        if(neigh_add.eq.1.and.j.eq.num_neigh(itype2,iat)+neigh_add) then  ! the ghost neighbor
                            fact1=m_neigh-num_neigh(itype2,iat)
                        endif
                        
                        !if ((itype1.eq.1).and.(itype])2.eq.1)) then
                        !    write(*,*) "fact1",fact1
                        !endif 
                        
                        ! allocate( ss(4,node_em(nlayer_em+1),natom_m_type) )

                        ! write(*,*) "nn1", nn1 
                        ! what is nn1??? 
                        ! dimension of the final layer? 
                        ! print before the ghost 
                        
                        ! cu-cu network, 1st center atom, 
                        if ((j.eq.num_neigh(itype2,iat)+neigh_add).and.(1.eq.1).and.(i.eq.1)) then
                            do k=1,nn1
                                ! cu-cu network
                                if ((itype1.eq.1).and.(itype2.eq.1)) then 
                                    !write(*,*) "k", k, "i", i
                                    write(*,'(F16.12 F16.12 F16.12 F16.12)') ss(:,k,1)
                                endif 
                            enddo
                            write(*,*) "**************************************************************************"
                        endif 
                        
                        ! print before the ghost
                        
                        ! taking care of the ghosts
                        do k=1,nn1
                            do m=1,4 
                                !ss(m,k,i)=ss(m,k,i)+dxyz_neigh(m,j,itype2,iat)*f_out(k,jj,nlayer_em+1)
                                ss(m,k,i)=ss(m,k,i)+dxyz_neigh(m,j,itype2,iat)*f_out(k,jj,nlayer_em+1)*fact1
                                d_ss_fout(m,k,j,itype2,i)=d_ss_fout(m,k,j,itype2,i)+dxyz_neigh(m,j,itype2,iat)*fact1
                                
                                if(j.ne.num_neigh(itype2,iat)+neigh_add) then
                                    
                                    do m2=1,3
                                        
                                        d_ss(m,m2,k,j,itype2,i)=    d_ss(m,m2,k,j,itype2,i) & 
                                                                  + dxyz_dx_neigh(m2,m,j,itype2,iat) * f_out(k,jj,nlayer_em+1)
                                        ! It is possible to do this later, to save memory
                                        ! d_ss is to assume s_neigh, thus f_out is fixed
                                        ! d_ss(m,m2,k,i)=d_SS(m,k,i)/d_x(m2,k,i)
                                        ! d_ss_fout is to take the derivative with respect to f_out (only s_neigh, thus
                                        ! f_out is changing.  
                                        
                                    enddo
                                    
                                endif
                            enddo
                            
                        enddo       
                        
                        !print after the ghost
                        if ((j.eq.num_neigh(itype2,iat)+neigh_add).and.(1.eq.1).and.(i.eq.1)) then

                            do k=1,nn1
                                ! cu-cu network
                                if ((itype1.eq.1).and.(itype2.eq.1)) then 
                                    !write(*,*) "k", k, "i", i
                                    write(*,'(F16.12 F16.12 F16.12 F16.12)') ss(:,k,1)
                                endif 
                            enddo
                            
                        endif 
                        
                    enddo
                enddo  

                ! We need to double check, is it first sum in the ss for different itype2, 
                ! or sum over ss*ss for different itype2
                
                ! allocate(ss(4,node_em(nlayer_em+1),natom_m_type))
                
            enddo ! looping over itype2 end 


            ss=ss/(2*m_neigh)
            d_ss=d_ss/(2*m_neigh)
            d_ss_fout=d_ss_fout/(2*m_neigh)    

            nn1=node_em(nlayer_em+1)
            do i=1,natom_n_type(itype1)
                do k1=1,nn1
                    do k2=1,16   ! fixed, first index
                        kk=(k1-1)*16+k2   ! NN feature index
                        sum=0.d0
                        do m=1,4
                            sum=sum+ss(m,k1,i)*ss(m,k2,i)
                        enddo
                        f_in_NN(kk,i,1)=f_in_NN(kk,i,1)+sum    ! this is sum over itype2

                    enddo
                enddo
            enddo
            
            if(node_NN(1).ne.nn1*16) then
                write(6,*) "node_NN(1).ne.nn1*16,stop",node_NN(1),nn1*16
                stop
            endif
            
            num=natom_n_type(itype1)

            do ll=1,nlayer_NN
        
                if(ll.gt.1) then      
                    do i=1,num
                        do j=1,node_NN(ll)
                            x=f_in_NN(j,i,ll)
                            if(x.gt.20.d0) then
                                y=1.d0
                            elseif(x.gt.-20.d0.and.x.le.20.d0) then
                                y=(exp(x)-exp(-x))/(exp(x)+exp(-x))
                            elseif(x.lt.-20.d0) then
                                y=-1.d0
                            endif
                            !         f_out_NN(j, i, ll) = (exp(x)-exp(-x)) / (exp(x)+exp(-x))  ! tanh ! maybe if softplus, sigmoid, tanh
                            f_out_NN(j, i, ll) = y
                            f_d_NN(j, i, ll) = 1.0d0 - f_out_NN(j,i,ll)*f_out_NN(j,i,ll)
                            !f_out(j, i, ii) = 1.0d0/(exp(-x)+1.0d0)  ! sigmoid ! maybe if softplus, sigmoid, tanh
                            !f_d(j, i, ii) = f_out(j,i,ii) * (1.0d0 - f_out(j,i,ii))
                            ! if(x.gt.-150.d0.and.x.lt.150.d0) then
                            ! f_out(j,i,ii)=log(1.d0+exp(x))  ! softplus
                            ! f_d(j,i,ii)=1.d0/(exp(-x)+1.d0)
                            ! elseif(x.le.-150.d0) then
                            ! f_out(j,i,ii)=0.d0
                            ! f_d(j,i,ii)=0.d0
                            ! elseif(x.ge.150.d0) then 
                            ! f_out(j,i,ii)=x
                            ! f_d(j,i,ii)=1.d0
                            ! endif
                        enddo
                    enddo
                else
                    do i=1,num
                        do j=1,node_NN(ll)
                            f_out_NN(j,i,ll)=f_in_NN(j,i,ll)
                            f_d_NN(j,i,ll)=1.d0
                        enddo
                    enddo
                endif

                ! reconnect NN 
                if(iflag_resNN(ll).eq.1) then
                    do i=1,num
                        do j=1,node_NN(ll)
                            !         f_out_NN(j,i,ll)=f_out_NN(j,i,ll)+W_res_NN(j,ll,itype1)*f_out_NN(j,i,ll-1)
                            f_out_NN(j,i,ll)=f_out_NN(j,i,ll)*W_res_NN(j,ll,itype1)+f_out_NN(j,i,ll-1)
                        enddo
                    enddo
                endif

                !write(*,*) "llp test, f_out(:,1,ii), layer: ", ii
                !write(*,*) f_out(:,1,ii)

                call dgemm('T', 'N', node_NN(ll+1),num,node_NN(ll), 1.d0,  &
                Wij_NN(1,1,ll,itype1),nodeMM_NN,f_out_NN(1,1,ll),nodeMM_NN,0.d0,f_in_NN(1,1,ll+1),nodeMM_NN)
                !write(*,*) "llp test, f_in(:,1,ii+1)=wij*f_out,layer: ", ii+1
                !write(*,*) f_in(:,1,ii+1)

                !if (ii .eq. 1 ) then
                !    write(*,*) "layer0 feature -> layer1. layer1:"
                !    write(*,*) f_in(:,1,ii+1)
                !endif
                do i=1,num
                    do j=1,node_NN(ll+1)
                        f_in_NN(j,i,ll+1)=f_in_NN(j,i,ll+1)+B_NN(j,ll,itype1)
                    enddo
                enddo

                !write(*,*) "llp test, x_ii+1 = wij*x_ii+bj,layer: ", ii+1
                !write(*,*) f_in(:,1,ii+1)

            enddo  

            !ccccccccccccc   get the 100 R(s). 
        
         

            if(node_NN(nlayer_NN+1).ne.1) then
                write(6,*) "node_NN(nlayer_NN+1).ne.1,stop",node_NN(nlayer_NN+1)
                stop
            endif
                !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            do i=1,natom_n_type(itype1)
                energy_type(i,itype1)=f_in_NN(1,i,nlayer_NN+1)
            enddo

            !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            !   Now, we wil do the back propagation
            ! f_back0(j,i,ll)=dE/d_(f_out(j,i,ll))
            ! f_back(j,i,ll)=dE/d_(f_in(j,i,ll))=f_fac0*df(j,i,ll)*W_res
            !  f_out(j,i,ll)=sigma(f_in(j,i,ll))*W_res+f_out(j,i,ll-1)  ! if there are res
            !  f_out(j,i,ll)=sigma(f_in(j,i,ll))                        ! if no rest
            !  f_in(j,i,ll+1)=W(j2,i,ll)*f_out(j2,i,ll)+B(j,i,ii)


            do i=1,num
                do j=1,node_NN(nlayer_NN)
                    f_back0(j,i,nlayer_NN)=Wij_NN(j,1,nlayer_NN,itype1)
                enddo
            enddo

            do ll=nlayer_NN,2,-1

                do i=1,num
                    do j=1,node_NN(ll)
                        f_back(j,i,ll)=f_back0(j,i,ll)*f_d_NN(j,i,ll)
                        !  f_back0=dE/d_(f_out(ll))
                        !  f_back=dE/d_(f_in(ll))
                    enddo
                enddo

                if(iflag_resNN(ll).eq.1) then
                    do i=1,num
                        do j=1,node_NN(ll)
                            f_back(j,i,ll)=f_back(j,i,ll)*W_res_NN(j,ll,itype1)
                        enddo
                    enddo
                endif

                call dgemm('N', 'N', node_NN(ll-1),num,node_NN(ll),1.d0,  &
                Wij_NN(1,1,ll-1,itype1),nodeMM_NN,f_back(1,1,ll),nodeMM_NN,0.d0,f_back0(1,1,ll-1),nodeMM_NN)

                if(iflag_resNN(ll).eq.1) then
                    do i=1,num
                        do j=1,node_NN(ll-1)
                            f_back0(j,i,ll-1)=f_back0(j,i,ll-1)+f_back0(j,i,ll)
                        enddo
                    enddo
                endif
            enddo

            !      f_back0(j,i,1)=dE/d_(f_out(j,i,1))=dE/d_(f_in(j,i,1))=dE/df_NN
            !   j is feature index, i, the itype1 atom index
            ! Now, there are two terms for the force:
            !   (dE/df_NN)*(df_NN/d_x)
            !   (dE/df_NN)*(df_NN/d_fem)*(d_fem/d_s)*(d_s/d_x)
            !  let't do the first term



            !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            ! This part is a bit expensive, it will be nice to accelerate it
            nn1=node_em(nlayer_em+1)
         do itype2=1,ntype
         do i=1,natom_n_type(itype1)
         iat=iat_ind(i,itype1)
         do j=1,num_neigh(itype2,iat)
         do m2=1,3
         d_sum2=0.d0
         do k1=1,nn1
         do k2=1,16   ! fixed, first index
         kk=(k1-1)*16+k2   ! NN feature index
!         d_sum=0.d0
!         do m=1,4
!         d_sum=d_sum+d_ss(m,m2,k1,j,itype2,i)*ss(m,k2,i)+ss(m,k1,i)*d_ss(m,m2,k2,j,itype2,i)
!         enddo
         d_sum=d_ss(1,m2,k1,j,itype2,i)*ss(1,k2,i)+ss(1,k1,i)*d_ss(1,m2,k2,j,itype2,i)+ &
               d_ss(2,m2,k1,j,itype2,i)*ss(2,k2,i)+ss(2,k1,i)*d_ss(2,m2,k2,j,itype2,i)+ &
               d_ss(3,m2,k1,j,itype2,i)*ss(3,k2,i)+ss(3,k1,i)*d_ss(3,m2,k2,j,itype2,i)+ &
               d_ss(4,m2,k1,j,itype2,i)*ss(4,k2,i)+ss(4,k1,i)*d_ss(4,m2,k2,j,itype2,i)
        !cccc d_sum=d_f(kk).dx(m2,j,itype2,iat)
         d_sum2=d_sum2+d_sum*f_back0(kk,i,1)

         enddo
         enddo
        !ccccccccc  This is for assuming s_neigh, is fixed
         dE_dx(m2,j,itype2,iat)=dE_dx(m2,j,itype2,iat)+d_sum2
         enddo
         enddo
         enddo
         enddo
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
 
         nn1=node_em(nlayer_em+1)
         do 130 itype2=1,ntype

         dE_dfout=0.d0

         jj=0
         do i=1,natom_n_type(itype1)
         iat=iat_ind(i,itype1)
         do j=1,num_neigh(itype2,iat)
         jj=jj+1

         d_sum2=0.d0
         do k1=1,nn1
         do k2=1,16   ! fixed, first index
         kk=(k1-1)*16+k2   ! NN feature index
         d_sum=0.d0
         do m=1,4
         d_sum=d_sum+d_ss_fout(m,k1,j,itype2,i)*ss(m,k2,i)
         enddo
         dE_dfout(k1,jj)=dE_dfout(k1,jj)+d_sum*f_back0(kk,i,1)
         d_sum=0.d0
         do m=1,4
         d_sum=d_sum+ss(m,k1,i)*d_ss_fout(m,k2,j,itype2,i)
         enddo
         dE_dfout(k2,jj)=dE_dfout(k2,jj)+d_sum*f_back0(kk,i,1)
         enddo
         enddo
         enddo
         enddo
         
         num=jj

         do i=1,num
         do j=1,node_em(nlayer_em+1)
         f_back0_em(j,i,nlayer_em+1)=dE_dfout(j,i)
         enddo
         enddo

        do 220 ll=nlayer_em+1,2,-1

            do i=1,num
                do j=1,node_em(ll)
                    f_back_em(j,i,ll)=f_back0_em(j,i,ll)*f_d(j,i,ll,itype2)
                enddo
            enddo

            call dgemm('N', 'N', node_em(ll-1),num,node_em(ll),1.d0,  &
            Wij_em(1,1,ll-1,itype2,itype1),nodeMM_em,f_back_em(1,1,ll),nodeMM_em,0.d0,   &
            f_back0_em(1,1,ll-1),nodeMM_em)


            if(node_em(ll).eq.2*node_em(ll-1)) then
                do i=1,num
                    do j=1,node_em(ll-1)
                    f_back0_em(j,i,ll-1)=f_back0_em(j,i,ll-1)+f_back0_em(j,i,ll)+ &
                        f_back0_em(j+node_em(ll-1),i,ll)
                    enddo
                enddo
            endif
            
            ! rec 
            if(node_em(ll).eq.node_em(ll-1)) then
                do i=1,num
                    do j=1,node_em(ll-1)
                        f_back0_em(j,i,ll-1)=f_back0_em(j,i,ll-1)+f_back0_em(j,i,ll)
                    enddo
                enddo
            endif

220     continue

     
         
         jj=0
         do i=1,natom_n_type(itype1)
         iat=iat_ind(i,itype1)
         do j=1,num_neigh(itype2,iat)
         jj=jj+1

         do m2=1,3
         dE_dx(m2,j,itype2,iat)=dE_dx(m2,j,itype2,iat)+f_back0_em(1,jj,1)*ds_neigh(m2,j,itype2,iat)
!1 This is the derivative through s_neigh change, has through the back
!propagation of the embedding net
         enddo
         enddo
         enddo

130      continue

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!  back propagation for the embedding net. 

400     continue

         ! wrapping up final results

         Etot=0.d0
         do itype=1,ntype
         do i=1,natom_n_type(itype)
         Etot=Etot+energy_type(i,itype)
         enddo
         enddo


         allocate(force_all(3,natom))
         allocate(force_all_tmp(3,natom))

         force_all=0.d0
         do itype1=1,ntype
         do i=1,natom_n_type(itype1)
         iat=iat_ind(i,itype1)
         do itype2=1,ntype
         do j=1,num_neigh(itype2,iat)
         iat2=list_neigh(j,itype2,iat)
         force_all(:,iat2)=force_all(:,iat2)+dE_dx(:,j,itype2,iat)
         force_all(:,iat)=force_all(:,iat)-dE_dx(:,j,itype2,iat)   ! the centeratom is a negative derivative
         enddo
         enddo
         enddo
         enddo


        call mpi_allreduce(Etot,Etot_tmp,1,MPI_REAL8,MPI_SUM,MPI_COMM_WORLD,ierr)
        call mpi_allreduce(force_all,force_all_tmp,3*natom,MPI_REAL8,MPI_SUM,MPI_COMM_WORLD,ierr)

        Etot=Etot_tmp
        force_all=force_all_tmp


        fatom(:,1:natom)=force_all(:,1:natom)
        deallocate(force_all)
        deallocate(force_all_tmp)
        deallocate(f_in)
        deallocate(f_out)
        deallocate(f_d)
        deallocate(f_back0_em)
        deallocate(f_back_em)
        deallocate(ss)
        deallocate(f_in_NN)
        deallocate(f_out_NN)
        deallocate(f_back)
        deallocate(f_back0)
        deallocate(f_d_NN)
        deallocate(energy_type)
        deallocate(d_ss)
        deallocate(dE_dx)
        deallocate(d_ss_fout)
        deallocate(dE_dfout)
        deallocate(iat_ind)
        tt2=mpi_wtime()

        !        write(6,"('time find_neigh,calc_force ',2(f10.4,1x))") tt1-tt0,tt2-tt1
        !        write(6,*) "natom_n_type", natom_n_type(1:ntype)

        return
        !  Now, back propagation for the derivative for energy, in respect to the f_in(j,i,1)      


        !ccccccccccccccccccccccccccccccccccccccccccccc
        !ccccccccccccccccccccccccccccccccccccccccccccc
        ! enddo
        !ccccccccccccccccccccccccccccccccccccccccccc
        
    end subroutine cal_energy_force_deepMD

   
end module calc_deepMD
  