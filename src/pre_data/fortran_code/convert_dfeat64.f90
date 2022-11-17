module convert_dfeat
    ! implicit double precision (a-h, o-z)
    ! feature derivative module with double precision
    implicit none

    real*8,allocatable,dimension(:,:,:,:) :: dfeat
    real*8,allocatable,dimension(:,:,:,:) :: dfeat_scaled
  
    contains

    subroutine deallo()

        deallocate(dfeat)

    end subroutine deallo

    subroutine allo(nfeat0m,natom,m_neigh)

        integer(4), intent(in) :: nfeat0m,natom,m_neigh
        allocate(dfeat(nfeat0m,natom,m_neigh,3))
        dfeat(:,:,:,:)=0.0

    end subroutine allo

    subroutine conv_dfeat(image_Num,ipos,natom_p,num_tmp,dfeat_tmp,jneigh_tmp,ifeat_tmp,iat_tmp)

        integer(4)  :: jj,ii,i_p,i,j
        integer(4), intent(in) :: image_Num,num_tmp,ipos,natom_p

        real*8,  dimension (:,:),intent(in) :: dfeat_tmp
        integer(4),dimension (:),intent(in) :: iat_tmp,jneigh_tmp,ifeat_tmp

        do jj=1,num_tmp
            dfeat(ifeat_tmp(jj)+ipos,iat_tmp(jj)+natom_p,jneigh_tmp(jj),1)=dfeat_tmp(1,jj)
            dfeat(ifeat_tmp(jj)+ipos,iat_tmp(jj)+natom_p,jneigh_tmp(jj),2)=dfeat_tmp(2,jj)
            dfeat(ifeat_tmp(jj)+ipos,iat_tmp(jj)+natom_p,jneigh_tmp(jj),3)=dfeat_tmp(3,jj)
        enddo

    end subroutine conv_dfeat

    ! Below are subroutine for single image reading and converting. 
    ! L.Wang 2022.7 

    ! pytorch input format:
    ! atom index within this image, neighbor index, feature index, spatial dimension   

    subroutine allo_singleimg(atom_num,m_neigh,total_feat_num)
        
        ! atom_num = atom number in this image 

        integer(4), intent(in) :: total_feat_num,atom_num,m_neigh

        allocate(dfeat(atom_num, m_neigh,total_feat_num,3))

        dfeat(:,:,:,:)=0.0

    end subroutine allo_singleimg

    subroutine conv_dfeat_singleimg(ipos,natom_p,num_tmp,dfeat_tmp,jneigh_tmp,ifeat_tmp,iat_tmp)
        
        ! this subroutine converts a single image and returns a tensor whose format matches the requirement of pytorch 
        
        integer(4)  :: jj
        integer(4), intent(in) :: num_tmp,ipos,natom_p

        real*8,  dimension (:,:),intent(in) :: dfeat_tmp
        integer(4),dimension (:),intent(in) :: iat_tmp,jneigh_tmp,ifeat_tmp

        
        do jj=1,num_tmp
            
            ! non-continuous memory layout can significantly slow things down  
            
            dfeat(iat_tmp(jj) , jneigh_tmp(jj) , ifeat_tmp(jj)+ipos, 1) = dfeat_tmp(1,jj)
            dfeat(iat_tmp(jj) , jneigh_tmp(jj) , ifeat_tmp(jj)+ipos, 2) = dfeat_tmp(2,jj)
            dfeat(iat_tmp(jj) , jneigh_tmp(jj) , ifeat_tmp(jj)+ipos, 3) = dfeat_tmp(3,jj)

            !dfeat(ifeat_tmp(jj)+ipos,iat_tmp(jj)+natom_p,jneigh_tmp(jj),1)=dfeat_tmp(1,jj)
            !dfeat(ifeat_tmp(jj)+ipos,iat_tmp(jj)+natom_p,jneigh_tmp(jj),2)=dfeat_tmp(2,jj)
            !dfeat(ifeat_tmp(jj)+ipos,iat_tmp(jj)+natom_p,jneigh_tmp(jj),3)=dfeat_tmp(3,jj)
        enddo

    end subroutine conv_dfeat_singleimg




    ! subroutine scale(list_neigh,num_neigh,iatom,feat_scale,itype_atom)
    !     integer(4)  :: nfeat0m,natom,m_neigh,ntype,iitype,itype,i,j,jj
    !     integer(4),dimension (:,:),intent(in) :: list_neigh
    !     real*8,  dimension (:,:),intent(in) ::feat_scale
    !     integer(4),dimension (:),intent(in) :: num_neigh,iatom,itype_atom
    !     integer,allocatable,dimension(:) :: iatom_type
    !     integer(4),dimension (4)::dfeat_shape
    !     real*8 :: t1,t2
    !     dfeat_shape=shape(dfeat)

    !     nfeat0m=dfeat_shape(1)
    !     natom=dfeat_shape(2)
    !     ntype=size(itype_atom)
    !     call cpu_time(t1)
        

    !     allocate(dfeat_scaled(nfeat0m,natom,dfeat_shape(3),3))
    !     allocate(iatom_type(natom))

    !     do i=1,natom
    !         iitype=0
    !         do itype=1,ntype
    !         if(itype_atom(itype).eq.iatom(i)) then
    !         iitype=itype
    !         endif
    !         enddo
    !         if(iitype.eq.0) then
    !         write(6,*) "this type not found", iatom(i)
    !         endif
    !         iatom_type(i)=iitype
    !     enddo

    !     ! num=0
    !     do i=1,natom
    !     do jj=1,num_neigh(i)
    !     itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
        
    !     do j=1,nfeat0m
    !     dfeat_scaled(j,i,jj,:)=dfeat(j,i,jj,:)*feat_scale(j,itype)
    !     enddo

    !     enddo
    !     enddo

    !     deallocate(iatom_type)
    !     call cpu_time(t2)
    !     write(*,*) 'time',t2-t1

    ! end subroutine scale

    ! subroutine link_dfeat(dfeat1,dfeat2)
    !     real*8,dimension(:,:,:,:),intent(in) :: dfeat1
    !     real*8,dimension(:,:,:,:),intent(in) :: dfeat2
    !     real*8,allocatable,dimension(:,:,:,:) :: dfeat
    !     integer(4),dimension(4) :: dfeat1_shape,dfeat2_shape
    !     dfeat1_shape=shape(dfeat1)
    !     dfeat2_shape=shape(dfeat2)
    !     nfeat1=dfeat1_shape(1)
    !     nfeat2=dfeat2_shape(1)
    !     nfeat0m=nfeat1+nfeat2
    !     allocate(dfeat(nfeat0m,natom,m_neigh,3))
    !     dfeat_l(1:nfeat1,:,:,:)=dfeat1(:,:,:,:)


    ! end subroutine link_dfeat

    ! subroutine conv_dfeat(image_Num,num_tmp_all,dfeat_tmp_all,jneigh_tmp_all,ifeat_tmp_all,iat_tmp_all,feat_all,force_all,energy_all,list_neigh_all,iatom_all)

    !     integer(4)  :: jj,num_tmp,ii,i_p,i,j,nfeat0m,natom,m_neigh
    !     integer(4), intent(in) :: image_Num
    !     integer(4), dimension (:,:),intent(in) :: iat_tmp_all,jneigh_tmp_all,ifeat_tmp_all
    !     real*8,  dimension (:,:,:),intent(in) :: dfeat_tmp_all
    !     integer(4),dimension(:),intent(in)  :: num_tmp_all
    !     real*8,dimension(:,:,:),intent(in) :: force_all       !每个原子的受力
    !     real*8,dimension(:,:,:),intent(in) :: feat_all    
    !     real*8, dimension(:,:),intent(in) :: energy_all        !每个原子的能量
    !     integer(4),dimension(:,:,:),intent(in) :: list_neigh_all
    !     integer(4),dimension(:),intent(in) :: iatom_all
    !     real*8, allocatable, dimension (:,:) :: dfeat_tmp
    !     integer(4),allocatable, dimension (:) :: iat_tmp,jneigh_tmp,ifeat_tmp
    !     integer(4),dimension(3) :: feat_shape,list_shape

    !     feat_shape=shape(feat_all)
    !     nfeat0m=feat_shape(1)
    !     natom=feat_shape(2)
    !     list_shape=shape(list_neigh_all)
    !     m_neigh=list_shape(1)

    !      allocate(dfeat(nfeat0m,natom,m_neigh,3))
    !      allocate(energy(natom))
    !      allocate(force(3,natom))
    !      allocate(feat(nfeat0m,natom))
    !      allocate(list_neigh(m_neigh,natom))
    !      allocate(iatom(natom))

    !      do i=1,natom

    !      iatom(i)=iatom_all(i)
    !      energy(i)=energy_all(i,image_Num)
    !      force(:,i)=force_all(:,i,image_Num)
    !      feat(:,i)=feat_all(:nfeat0m,i,image_Num)
    !      list_neigh(:,i)=list_neigh_all(:,i,image_Num)

    !      enddo


    !     num_tmp=num_tmp_all(image_Num)
    !     write(*,*) 'num_tmp', num_tmp
    !     write(*,*) shape(dfeat_tmp_all)
    !     write(*,*) shape(iat_tmp_all)

    !     allocate(dfeat_tmp(3,num_tmp))
    !     allocate(iat_tmp(num_tmp))
    !     allocate(jneigh_tmp(num_tmp))
    !     allocate(ifeat_tmp(num_tmp))
    !     write(*,*) 'allo'
    !     dfeat_tmp(:,:num_tmp)=dfeat_tmp_all(:,1:num_tmp,image_Num)
    !     write(*,*) 'dfeat_tmp'
    !     iat_tmp(:num_tmp)=iat_tmp_all(1:num_tmp,image_Num)
    !     write(*,*) 'iat_tmp'
    !     jneigh_tmp(:num_tmp)=jneigh_tmp_all(1:num_tmp,image_Num)
    !     write(*,*) 'jneigh_tmp'
    !     ifeat_tmp(:num_tmp)=ifeat_tmp_all(1:num_tmp,image_Num)
    !     write(*,*) 'assig'
           
    !         dfeat(:,:,:,:)=0.0
    !         do jj=1,num_tmp
    !         dfeat_tmp(:,jj)=dfeat_tmp_all(:,jj,image_Num)
    !         write(*,*) 'dfeat_tmp'
    !         iat_tmp(jj)=iat_tmp_all(jj,image_Num)
    !         write(*,*) 'iat_tmp'
    !         jneigh_tmp(jj)=jneigh_tmp_all(jj,image_Num)
    !         write(*,*) 'jneigh_tmp'
    !         ifeat_tmp(jj)=ifeat_tmp_all(jj,image_Num)
    !         dfeat(ifeat_tmp(jj),iat_tmp(jj),jneigh_tmp(jj),:)=dfeat_tmp(:,jj)
    !         enddo
    !     deallocate(dfeat_tmp)
    !     deallocate(iat_tmp)
    !     deallocate(jneigh_tmp)
    !     deallocate(ifeat_tmp)

    ! end subroutine conv_dfeat
  
    ! subroutine conv_dfeat(image_Num,nfeat0m,natom,m_neigh,num_tmp,dfeat_tmp,jneigh_tmp,ifeat_tmp,iat_tmp)

    !     integer(4)  :: jj,ii,i_p,i,j
    !     integer(4), intent(in) :: image_Num,nfeat0m,natom,m_neigh,num_tmp

    !     real*8,  dimension (:,:),intent(in) :: dfeat_tmp
    !     integer(4),dimension (:),intent(in) :: iat_tmp,jneigh_tmp,ifeat_tmp
    !     ! integer(4),dimension(3) :: feat_shape,list_shape

    !     ! write(*,*) 'num_tmp', num_tmp

    !     allocate(dfeat(nfeat0m,natom,m_neigh,3))
    !     ! write(*,*) 'allo', shape(dfeat)

           
    !         dfeat(:,:,:,:)=0.0
    !         ! write(*,*) '000'
    !         do jj=1,num_tmp
    !             ! write(*,*) ifeat_tmp(jj)
    !             ! write(*,*) iat_tmp(jj)
    !             ! write(*,*) jneigh_tmp(jj)


    !         dfeat(ifeat_tmp(jj),iat_tmp(jj),jneigh_tmp(jj),:)=dfeat_tmp(:,jj)
    !         enddo


    ! end subroutine conv_dfeat
  
end module convert_dfeat
  
  
