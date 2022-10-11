module calc_MTP_feature
    use mod_mpi

    IMPLICIT NONE
    INTEGER :: ierr

    integer,allocatable,dimension (:) :: iatom
    integer num_step, natom, i, j,natom5,m_neigh5
    integer num_step0,num_step1,natom0,max_neigh
    real*8 Etotp_ave,E_tolerance

    integer n2b_t,n3b1_t,n3b2_t,it
    integer n2bm

    real*8 Rc_M
    integer n3b1m,n3b2m,kkk,ii

    real*8,allocatable,dimension (:,:) :: feat_M5
    real*8,allocatable,dimension (:,:,:,:) :: dfeat_M5
    integer,allocatable,dimension (:,:) :: list_neigh_alltypeM5
    integer,allocatable,dimension (:) :: num_neigh_alltypeM5
    integer nfeat0M5

    integer m_neigh,num,itype1,itype2,itype
    integer iat1,max_neigh_M,num_M
    
    integer ntype,n2b,n3b1,n3b2,nfeat0m,nfeat0(100)
    integer ntype1,ntype2,k1,k2,k12,ii_f,iat,ixyz
    integer iat_type(100)
    real*8 Rc, Rc2,Rm
    real*8 alpha31,alpha32

    real*8 Rc_type(100), Rm_type(100)

!cccccccccccccccccccccccccccccccccccccccccccccccc
    character*1 txt
    integer numT_all(20000,10)
    integer rank_all(4,20000,10),mu_all(4,20000,10),jmu_b_all(4,20000,10),itype_b_all(4,20000,10)
    integer indi_all(5,4,20000,10),indj_all(5,4,20000,10)
    integer jmu_b(10,5000),itype_b(10,5000)
    integer numCC_type(10)


    integer indi(10,10),indj(10,10),ind(10,10)
    integer mu(10),rank(10)
    integer iflag_ti(10,10),iflag_tj(10,10)
    integer nMTP_type(100)
    integer numCC,kk,numC,kkc
    integer n2b_type(10)
    integer numc0

    contains
    subroutine load_model_type5()
    

        open(10,file="input/gen_MTP_feature.in",status="old",action="read")
        rewind(10)
        read(10,*) Rc_M,m_neigh
        m_neigh5=m_neigh
        read(10,*) ntype
    
        do 100  itype=1,ntype
        read(10,*) iat_type(itype)
        read(10,*) Rc_type(itype),Rm_type(itype)
         if(Rc_type(itype).gt.Rc_M) then
          write(6,*) "Rc_type must be smaller than Rc_M, gen_3b_feature.in",i,Rc_type(i),Rc_M
          stop
         endif
    
        read(10,*) nMTP_type(itype)   ! number of MTP type (line, each line can be expanded to many MTP)
    
        numCC=0
        do 99 kkk=1,nMTP_type(itype)
    
         read(10,*) num
         backspace(10)
         if(num.gt.4) then
          write(6,*) "we only support contraction upto 4 tensors,stop",num
          stop
          endif
     !  Cannot do double loop with txt
            if(num.eq.1) then
            read(10,*,iostat=ierr) num,(mu(i),i=1,num),(rank(i),i=1,num),&
            txt,(ind(j,1),j=1,rank(1)),txt
            elseif(num.eq.2) then
            read(10,*,iostat=ierr) num,(mu(i),i=1,num),(rank(i),i=1,num),&
            txt,(ind(j,1),j=1,rank(1)),txt,txt,(ind(j,2),j=1,rank(2)),txt
            elseif(num.eq.3) then
            read(10,*,iostat=ierr) num,(mu(i),i=1,num),(rank(i),i=1,num),&
            txt,(ind(j,1),j=1,rank(1)),txt,txt,(ind(j,2),j=1,rank(2)),txt, &
            txt,(ind(j,3),j=1,rank(3)),txt
            elseif(num.eq.4) then
            read(10,*,iostat=ierr) num,(mu(i),i=1,num),(rank(i),i=1,num),&
            txt,(ind(j,1),j=1,rank(1)),txt,txt,(ind(j,2),j=1,rank(2)),txt, &
            txt,(ind(j,3),j=1,rank(3)),txt,txt,(ind(j,4),j=1,rank(4)),txt
            endif
    
          if(ierr.ne.0) then
            write(6,*) "the tensor contraction line is not correct",kkk,itype
            stop
            endif
    
            do i=1,num
            do j=1,rank(i)
            indi(j,i)=ind(j,i)/10
            indj(j,i)=ind(j,i)-indi(j,i)*10
            enddo
            enddo
    
            iflag_ti=0
            iflag_tj=0
            do i=1,num
            do j=1,rank(i)
            if(iflag_ti(indj(j,i),indi(j,i)).ne.0) then
            write(6,*) "contraction error", i,j,indj(j,i),indi(j,i)
            stop
            endif
            iflag_ti(indj(j,i),indi(j,i))=i
            iflag_tj(indj(j,i),indi(j,i))=j
            enddo
            enddo
    
            do i=1,num
            do j=1,rank(i)
            if(iflag_ti(j,i).ne.indi(j,i).or.iflag_tj(j,i).ne.indj(j,i)) then
            write(6,*) "contraction correspondence confluct",i,j
            stop
            endif
            enddo
            enddo
    
            call get_expand_MT(ntype,num,indi,indj,mu,rank,numC,jmu_b,itype_b)
    ! numC is the MTP of this line (after expansion, due mu, and ntype)
    
            numc0=1
            do i=1,num
            numc0=(1+mu(i))*numc0*ntype
            enddo
    
            write(6,"('num_feat,line(kkk,itype,numc,numc(before reduce)',4(i4,1x))") kkk,itype,numc,numc0
            if(numCC+numc.gt.20000) then
             write(6,*) "too many features",numCC+numC,itype
            stop
            endif
    
            do kk=1,numC
            kkc=numCC+kk
            numT_all(kkc,itype)=num       ! num is the number of tensor in this line
              do i=1,num
              rank_all(i,kkc,itype)=rank(i)
              mu_all(i,kkc,itype)=mu(i)
              jmu_b_all(i,kkc,itype)=jmu_b(i,kk)  ! the mu of this kk
              itype_b_all(i,kkc,itype)=itype_b(i,kk)  ! the type of this kk
              enddo
              do i=1,num
              do j=1,rank(i) 
              indi_all(j,i,kkc,itype)=indi(j,i)
              indj_all(j,i,kkc,itype)=indj(j,i)
              enddo
              enddo
           enddo
           numCC=numCC+numc
    99     continue
           numCC_type(itype)=numCC    ! numCC_type is the total MTP term of this itype
    
           write(6,*) "numCC_type",itype,numCC_type(itype)
          
    100   continue
        read(10,*) E_tolerance
        close(10)
    
        write(6,*) "iat_type",iat_type(1:ntype)
    
          nfeat0m=0
          do itype=1,ntype
          if(numCC_type(itype).gt.nfeat0m) nfeat0m=numCC_type(itype)
          nfeat0(itype)=numCC_type(itype)
          enddo
         write(6,*) "itype,nfeat0=",(nfeat0(itype),itype=1,ntype)
    
     
    !cccccccccccccccccccccccccccccccccccccccc
    
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
!  FInish the initial grid treatment

    end subroutine load_model_type5
    
    subroutine set_image_info_type5(atom_type_list,is_reset,natom_tmp)
        integer(4),dimension(:),intent(in) :: atom_type_list(natom_tmp)
        logical,intent(in) :: is_reset
        integer,intent(in) :: natom_tmp
        
            natom=natom_tmp
            natom5=natom

            if(is_reset) then 
              if(allocated(iatom))then
              deallocate(iatom)
              endif
            allocate(iatom(natom))                   
            iatom(1:natom)=atom_type_list(1:natom)
           endif
        
    end subroutine set_image_info_type5

    subroutine gen_MTP_feature(AL,xatom)
        integer(4)  :: itype,i,j
        integer ii,jj,jjm,iat
    
        real(8), intent(in) :: AL(3,3)
        real(8),dimension(:,:),intent(in) :: xatom
    
        integer,allocatable,dimension (:,:,:) :: list_neigh,iat_neigh,iat_neigh_M
        integer,allocatable,dimension (:,:) :: num_neigh
        real*8,allocatable,dimension (:,:,:,:) :: dR_neigh
        integer,allocatable,dimension (:,:) :: map2neigh_alltypeM
        integer,allocatable,dimension (:,:) :: list_tmp
        integer,allocatable,dimension (:) :: nfeat_atom
        integer,allocatable,dimension (:) :: itype_atom
        integer,allocatable,dimension (:,:,:) :: map2neigh_M
        integer,allocatable,dimension (:,:,:) :: list_neigh_M
        integer,allocatable,dimension (:,:) :: num_neigh_M
        integer,allocatable,dimension (:,:) :: list_neigh_alltype
        integer,allocatable,dimension (:) :: num_neigh_alltype
        real*8 :: timestart, timefinish
        real*8,allocatable,dimension (:,:) :: feat
        real*8,allocatable,dimension (:,:,:,:) :: dfeat
        real*8 tt2,tt1,tt0,tt3,tt4,tt5,tt6
    !cccccccccccccccccccccccccccccccccccccccccccccccccccc
        if (allocated(dfeat_M5)) then
            deallocate(feat_M5)
            deallocate(dfeat_M5)
            deallocate(list_neigh_alltypeM5)
            deallocate(num_neigh_alltypeM5)
        endif
    
    ! the dimension of these array, should be changed to natom_n
    ! instead of natom. Each process only needs to know its own natom_n
    
        allocate(list_neigh(m_neigh,ntype,natom))
        allocate(map2neigh_M(m_neigh,ntype,natom)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
        allocate(list_neigh_M(m_neigh,ntype,natom)) ! the neigh list of Rc_M
        allocate(num_neigh_M(ntype,natom))
        allocate(iat_neigh(m_neigh,ntype,natom))
        allocate(dR_neigh(3,m_neigh,ntype,natom))   ! d(neighbore)-d(center) in xyz
        allocate(num_neigh(ntype,natom))
        allocate(list_neigh_alltype(m_neigh,natom))
        allocate(num_neigh_alltype(natom))
    
        allocate(iat_neigh_M(m_neigh,ntype,natom))
        allocate(list_neigh_alltypeM5(m_neigh,natom))
        allocate(num_neigh_alltypeM5(natom))
        allocate(map2neigh_alltypeM(m_neigh,natom)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
        allocate(list_tmp(m_neigh,ntype))
        allocate(itype_atom(natom))
        allocate(nfeat_atom(natom))
    
        allocate(feat(nfeat0m,natom_n))         ! each note, only know its own feat
        allocate(dfeat(nfeat0m,natom_n,m_neigh,3))  ! dfeat is the derivative from the neighboring dR, 
        allocate(feat_M5(nfeat0m,natom_n))
        allocate(dfeat_M5(nfeat0m,natom_n,m_neigh,3))
    
        feat = 0.d0
        dfeat = 0.d0
        feat_M5 = 0.d0
        dfeat_M5 = 0.d0
        
    !ccccccccccccccccccccccccccccccccccccccccccc
        itype_atom=0
        do i=1,natom
            do j=1,ntype
                if(iatom(i).eq.iat_type(j)) then
                    itype_atom(i)=j
                endif
            enddo

            if(itype_atom(i).eq.0) then
                write(6,*) "this atom type didn't found", itype_atom(i)
                stop
            endif
        enddo
    !ccccccccccccccccccccccccccccccccccccccccccc
    !    call cpu_time(timestart)
        call find_neighbore(iatom,natom,xatom,AL,Rc_type,num_neigh,list_neigh, &
        dR_neigh,iat_neigh,ntype,iat_type,m_neigh,Rc_M,map2neigh_M,list_neigh_M, &
        num_neigh_M,iat_neigh_M,inode,nnodes)
    
    !    call cpu_time(timefinish)
    !    write(*,*) 'find_nei time: ', timefinish-timestart
    !ccccccccccccccccccccccccccccccccc
    !ccccccccccccccccccccccccccccccccc
    
        max_neigh=-1
        num_neigh_alltype=0
        max_neigh_M=-1
        num_neigh_alltypeM5=0
        list_neigh_alltypeM5=0
    
        do iat=1,natom
    
            if(mod(iat-1,nnodes).eq.inode-1) then
    
    
                list_neigh_alltype(1,iat)=iat
                list_neigh_alltypeM5(1,iat)=iat
    
    
          num_M=1
          do itype=1,ntype
          do j=1,num_neigh_M(itype,iat)
          num_M=num_M+1
          if(num_M.gt.m_neigh) then
          write(6,*) "total num_neigh.gt.m_neigh,stop",m_neigh
          stop
          endif
          list_neigh_alltypeM5(num_M,iat)=list_neigh_M(j,itype,iat)
          list_tmp(j,itype)=num_M
          enddo
          enddo
    
    
          num=1
          map2neigh_alltypeM(1,iat)=1
          do  itype=1,ntype
          do   j=1,num_neigh(itype,iat)
          num=num+1
          list_neigh_alltype(num,iat)=list_neigh(j,itype,iat)
          map2neigh_alltypeM(num,iat)=list_tmp(map2neigh_M(j,itype,iat),itype)
    ! map2neigh_M(j,itype,iat), maps the jth neigh in list_neigh(Rc) to jth' neigh in list_neigh_M(Rc_M) 
          enddo
          enddo
    
    !ccccccccccccccccccccccccccccccccccccccc
    
    
          num_neigh_alltype(iat)=num
          num_neigh_alltypeM5(iat)=num_M
          if(num.gt.max_neigh) max_neigh=num
          if(num_M.gt.max_neigh_M) max_neigh_M=num_M
    
          endif
    
          enddo  ! iat
    
    !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    ! This num_neigh_alltype(iat) include itself !
    !    dfeat=0.d0
    !    feat=0.d0
    
          call find_feature_MTP(natom,itype_atom,Rc_type,Rm_type,num_neigh,  &
          list_neigh,dR_neigh,iat_neigh, &
          feat,dfeat,nfeat0m,m_neigh,nfeat_atom,  &
          numCC_type,numT_all,mu_all,rank_all,jmu_b_all,itype_b_all,indi_all,indj_all,ntype)
   
    !cccccccccccccccccccccccccccccccccccccccccccccccccccc
    

        do iat1=1,natom_n
        do ii=1,nfeat0m
        feat_M5(ii,iat1)=feat(ii,iat1)   
        enddo
        enddo
    
        iat1=0
        do iat=1,natom
        if(mod(iat-1,nnodes).eq.inode-1) then
        iat1=iat1+1
        do jj=1,num_neigh_alltype(iat)
        jjm=map2neigh_alltypeM(jj,iat)
        do ii=1,nfeat0m
        dfeat_M5(ii,iat1,jjm,1)=dfeat(ii,iat1,jj,1)  ! this is the feature stored in neigh list of Rc_M
        dfeat_M5(ii,iat1,jjm,2)=dfeat(ii,iat1,jj,2)
        dfeat_M5(ii,iat1,jjm,3)=dfeat(ii,iat1,jj,3)
        enddo
        enddo
        endif
        enddo
    
        nfeat0M5=nfeat0m    ! the number of features for feature type 1
    
    
        deallocate(list_neigh)
        deallocate(iat_neigh)
        deallocate(dR_neigh)
        deallocate(num_neigh)
        ! deallocate(feat)
        ! deallocate(dfeat)
        deallocate(list_neigh_alltype)
        deallocate(num_neigh_alltype)
    
    
        deallocate(list_neigh_M)
        deallocate(num_neigh_M)
        deallocate(map2neigh_M)
        deallocate(iat_neigh_M)
        ! deallocate(list_neigh_alltypeM5)
        ! deallocate(num_neigh_alltypeM5)
        deallocate(map2neigh_alltypeM)
        deallocate(list_tmp)
        deallocate(itype_atom)
        deallocate(nfeat_atom)
        deallocate(feat)
        deallocate(dfeat)
    



    end subroutine gen_MTP_feature


end module calc_MTP_feature