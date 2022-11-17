  !// forquill v1.01 beta www.fcode.cn
module convert_dfeat
    !implicit double precision (a-h, o-z)
    implicit none
  
  !!!!!!!!!!!!!          以下为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ! integer(4) :: m_neigh                                  !模型所使用的最大近邻数(考虑这个数是否可以不用)
    ! integer(4) :: nfeat0m                                  !不同种原子的原始feature数目中最大者(目前似无意义)
 
    
    ! integer(4) :: natom                                    !image的原子个数  
     
    
    ! real*8,allocatable,dimension(:,:) :: force       !每个原子的受力
    real*8,allocatable,dimension(:,:,:,:) :: dfeat
    ! real*8,allocatable,dimension(:,:) :: force       !每个原子的受力
    ! real*8,allocatable,dimension(:,:) :: feat    
    ! real*8, allocatable,dimension(:) :: energy        !每个原子的能量
    ! integer(4),allocatable,dimension(:,:) :: list_neigh
    ! integer(4),allocatable,dimension(:) :: iatom


  !!!!!!!!!!!!!          以上为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
    contains

    subroutine deallo()

        deallocate(dfeat)
        ! deallocate(feat)
        ! deallocate(energy)
        ! deallocate(force)
        ! deallocate(iatom)
        ! deallocate(list_neigh)

    end subroutine deallo
   
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
  
    subroutine conv_dfeat(image_Num,nfeat0m,natom,m_neigh,num_tmp,dfeat_tmp,jneigh_tmp,ifeat_tmp,iat_tmp)

        integer(4)  :: jj,ii,i_p,i,j
        integer(4), intent(in) :: image_Num,nfeat0m,natom,m_neigh,num_tmp

        real*8,  dimension (:,:),intent(in) :: dfeat_tmp
        integer(4),dimension (:),intent(in) :: iat_tmp,jneigh_tmp,ifeat_tmp
        ! integer(4),dimension(3) :: feat_shape,list_shape

        ! write(*,*) 'num_tmp', num_tmp
        allocate(dfeat(nfeat0m,natom,m_neigh,3))
        ! write(*,*) 'allo', shape(dfeat)

           
            dfeat(:,:,:,:)=0.0
            ! write(*,*) '000'
            do jj=1,num_tmp
                ! write(*,*) ifeat_tmp(jj)
                ! write(*,*) iat_tmp(jj)
                ! write(*,*) jneigh_tmp(jj)


              dfeat(ifeat_tmp(jj),iat_tmp(jj),jneigh_tmp(jj),:)=dfeat_tmp(:,jj)
            enddo


    end subroutine conv_dfeat
  
end module convert_dfeat
  
  
