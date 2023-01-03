subroutine ML_FF_EF(Etot,fatom,xatom,AL,natom_tmp,e_stress)
        use IFPORT
        use mod_mpi
        use mod_data, only : iflag_model, e_atom, iatom

        use mod_control, only: MCTRL_iMD     ! get the global variables 

        use data_ewald
        
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

        !  Note: num_neigh)alltypeM1,2; list_neigh_altypeM1,2 should be the same for 1 & 2
        use calc_lin, only : cal_energy_force_lin,Etot_pred_lin,force_pred_lin,nfeat_type_l,ifeat_type_l,energy_pred_lin
        use calc_VV, only : cal_energy_force_VV,Etot_pred_VV,force_pred_VV,nfeat_type_v,ifeat_type_v, energy_pred_vv
        use calc_NN, only : cal_energy_force_NN,cal_energy_NN, Etot_pred_NN,force_pred_NN,nfeat_type_n,ifeat_type_n, energy_pred_nn
        use calc_deepMD, only : cal_energy_force_deepMD 
        implicit none

        integer natom_tmp,natom,m_neigh  ! conflict with the one in calc_lin
        real*8 Etot
        
        real*8 Etot_prime ! used to calculate stress tensor
        real*8 da  

        real*8 fatom(3,natom_tmp)
        real*8 xatom(3,natom_tmp)
        real*8 AL(3,3)
        real*8 e_stress(3,3)

        ! used to accomadate the slightly streched lattice 
        real*8 AL_prime(3,3)
        real*8 stress_ratio 
        real*8 sigma 
        real*8 dAL
        real*8 dS 

        real*8,allocatable,dimension (:,:) :: feat
        real*8,allocatable,dimension (:,:,:,:) :: dfeat
        integer nfeat0, count
        integer ii,jj,iat,kk,i
        real*8 tt1,tt2,tt3,tt4,tt5
        integer nfeat_type
        integer ifeat_type(100)
        integer num_neigh_alltypeM_use(natom_tmp)
        integer, allocatable, dimension (:,:) :: list_neigh_alltypeM_use
        
        !for calling predict.py
        character(200) :: cmd_name, flag, etot_name, ei_name, fi_name
        integer(2):: s

        stress_ratio = 0.0002
        sigma = 0.d0

        ! nothing should be done for dp which is model #4 
        if ((iflag_model.eq.1) .or. (iflag_model.eq.2) .or. (iflag_model.eq.3)) then 
        
            if(iflag_model.eq.1) then
                nfeat_type=nfeat_type_l
                ifeat_type=ifeat_type_l
            endif
            
            if(iflag_model.eq.2) then

                nfeat_type=nfeat_type_v
                ifeat_type=ifeat_type_v
            endif

            if(iflag_model.eq.3) then
                nfeat_type=nfeat_type_n
                ifeat_type=ifeat_type_n
            endif
            
            nfeat0=0
            
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
            
            ! natom_n is the divided number of atom. Defined in mod_mpi.f90
            
            allocate(feat(nfeat0,natom_n))
            allocate(dfeat(nfeat0,natom_n,m_neigh,3))
            
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

            count=0
            do kk = 1, nfeat_type
            
                    if (ifeat_type(kk)  .eq. 1) then
                        do jj=1,m_neigh
                        do iat=1,natom_n
                        do ii=1,nfeat0M1
                        dfeat(ii+count,iat,jj,1)=dfeat_M1(ii,iat,jj,1)
                        dfeat(ii+count,iat,jj,2)=dfeat_M1(ii,iat,jj,2)
                        dfeat(ii+count,iat,jj,3)=dfeat_M1(ii,iat,jj,3)
                        enddo
                        enddo
                        enddo
                        count=count+nfeat0M1
                    endif

                    if (ifeat_type(kk)  .eq. 2) then
                    do jj=1,m_neigh
                    do iat=1,natom_n
                    do ii=1,nfeat0M2
                    dfeat(ii+count,iat,jj,1)=dfeat_M2(ii,iat,jj,1)
                    dfeat(ii+count,iat,jj,2)=dfeat_M2(ii,iat,jj,2)
                    dfeat(ii+count,iat,jj,3)=dfeat_M2(ii,iat,jj,3)
                    enddo
                    enddo
                    enddo
                    count=count+nfeat0M2
                    endif

                    if (ifeat_type(kk)  .eq. 3) then
                    do jj=1,m_neigh
                    do iat=1,natom_n
                    do ii=1,nfeat0M3
                    dfeat(ii+count,iat,jj,1)=dfeat_M3(ii,iat,jj,1)
                    dfeat(ii+count,iat,jj,2)=dfeat_M3(ii,iat,jj,2)
                    dfeat(ii+count,iat,jj,3)=dfeat_M3(ii,iat,jj,3)
                    enddo
                    enddo
                    enddo
                    count=count+nfeat0M3
                    endif

                    if (ifeat_type(kk)  .eq. 4) then
                    do jj=1,m_neigh
                    do iat=1,natom_n
                    do ii=1,nfeat0M4
                    dfeat(ii+count,iat,jj,1)=dfeat_M4(ii,iat,jj,1)
                    dfeat(ii+count,iat,jj,2)=dfeat_M4(ii,iat,jj,2)
                    dfeat(ii+count,iat,jj,3)=dfeat_M4(ii,iat,jj,3)
                    enddo
                    enddo
                    enddo
                    count=count+nfeat0M4
                    endif

                    if (ifeat_type(kk)  .eq. 5) then
                    do jj=1,m_neigh
                    do iat=1,natom_n
                    do ii=1,nfeat0M5
                    dfeat(ii+count,iat,jj,1)=dfeat_M5(ii,iat,jj,1)
                    dfeat(ii+count,iat,jj,2)=dfeat_M5(ii,iat,jj,2)
                    dfeat(ii+count,iat,jj,3)=dfeat_M5(ii,iat,jj,3)
                    enddo
                    enddo
                    enddo
                    count=count+nfeat0M5
                    endif

                    if (ifeat_type(kk)  .eq. 6) then
                    do jj=1,m_neigh
                    do iat=1,natom_n
                    do ii=1,nfeat0M6
                    dfeat(ii+count,iat,jj,1)=dfeat_M6(ii,iat,jj,1)
                    dfeat(ii+count,iat,jj,2)=dfeat_M6(ii,iat,jj,2)
                    dfeat(ii+count,iat,jj,3)=dfeat_M6(ii,iat,jj,3)
                    enddo
                    enddo
                    enddo
                    count=count+nfeat0M6
                    endif

                    if (ifeat_type(kk)  .eq. 7) then
                        do jj=1,m_neigh
                            do iat=1,natom_n
                                do ii=1,nfeat0M7
                                    dfeat(ii+count,iat,jj,1)=dfeat_M7(ii,iat,jj,1)
                                    dfeat(ii+count,iat,jj,2)=dfeat_M7(ii,iat,jj,2)
                                    dfeat(ii+count,iat,jj,3)=dfeat_M7(ii,iat,jj,3)
                                enddo
                            enddo
                        enddo
                        count=count+nfeat0M7
                    endif

                    if (ifeat_type(kk)  .eq. 8) then
                    do jj=1,m_neigh
                    do iat=1,natom_n
                    do ii=1,nfeat0M8
                    dfeat(ii+count,iat,jj,1)=dfeat_M8(ii,iat,jj,1)
                    dfeat(ii+count,iat,jj,2)=dfeat_M8(ii,iat,jj,2)
                    dfeat(ii+count,iat,jj,3)=dfeat_M8(ii,iat,jj,3)
                    enddo
                    enddo
                    enddo
                    count=count+nfeat0M8
                    endif

            enddo    

        endif 
        
        ! branches for imodel 
        ! assume the lattice is symmetric in 3 axis

        if(iflag_model.eq.1) then
            call cal_energy_force_lin(feat,dfeat,num_neigh_alltypeM_use,  &
             list_neigh_alltypeM_use,AL,xatom,natom,nfeat0,m_neigh)
            
            Etot=Etot_pred_lin    ! unit?
            
            e_atom(1:natom_tmp)=energy_pred_lin(1:natom_tmp)
            fatom(:,1:natom_tmp)=force_pred_lin(:,1:natom_tmp)   ! unit, and - sign?
            
            ! update the diagonal element of the stress tensor 
            ! if ((MCTRL_iMD.eq.4).or.(MCTRL_iMD.eq.5).or.(MCTRL_iMD.eq.7)) then
            if (1.eq.0) then
                ! this part is not yet tested 
                ! only for NPT ensembles
                ! plus delta in x direction 
                AL_prime = AL
                AL_prime(1,1) = AL_prime(1,1)*(1.0 + stress_ratio )
                
                ! calculate Etot'
                call cal_energy_force_lin(feat,dfeat,num_neigh_alltypeM_use,  &
                        list_neigh_alltypeM_use,AL_prime,xatom,natom,nfeat0,m_neigh)

                Etot_prime = Etot_pred_lin 

                sigma = sigma + Etot_prime - Etot 
                
                ! minus delta
                AL_prime = AL
                AL_prime(1,1) = AL_prime(1,1)*(1.0 - stress_ratio )
                
                call cal_energy_force_lin(feat,dfeat,num_neigh_alltypeM_use,  &
                        list_neigh_alltypeM_use,AL_prime,xatom,natom,nfeat0,m_neigh)

                Etot_prime = Etot_pred_lin 

                sigma = sigma + Etot_prime - Etot 
                sigma = sigma * 0.5

                e_stress(1,1) = sigma
                e_stress(2,2) = sigma
                e_stress(3,3) = sigma

            endif 

        endif   

        if(iflag_model.eq.2) then
            call cal_energy_force_VV(feat,dfeat,num_neigh_alltypeM_use,  &
             list_neigh_alltypeM_use,AL,xatom,natom,nfeat0,m_neigh)
            Etot=Etot_pred_VV    ! unit?
            e_atom(1:natom_tmp)=energy_pred_vv(1:natom_tmp)
            fatom(:,1:natom_tmp)=force_pred_VV(:,1:natom_tmp)   ! unit, and - sign?
        endif

        if(iflag_model.eq.3) then
            call cal_energy_force_NN(feat,dfeat,num_neigh_alltypeM_use,  &
             list_neigh_alltypeM_use,AL,xatom,natom,nfeat0,m_neigh)
            
            Etot=Etot_pred_NN    ! unit?
            
            !write(*,*) "unperturbed Etot"
            !write(*,*) Etot
            
            e_atom(1:natom_tmp)=energy_pred_nn(1:natom_tmp)
            fatom(:,1:natom_tmp)=force_pred_NN(:,1:natom_tmp)   ! unit, and - sign?
            
            ! test tensor calculation NPT
            
            if ((MCTRL_iMD.eq.4).or.(MCTRL_iMD.eq.5).or.(MCTRL_iMD.eq.7)) then
                
                !write (*,*) "calculating stress tensor"
                ! only for NPT ensembles

                ! x direction
                AL_prime = AL
                dAL = AL_prime(1,1) * stress_ratio 
                AL_prime(1,1) = AL_prime(1,1)*(1.0 + stress_ratio )
                
                ! need to re-calc the feat 
                call cal_energy_NN(num_neigh_alltypeM_use,list_neigh_alltypeM_use,AL_prime,xatom,natom,nfeat0,m_neigh)
                Etot_prime = Etot_pred_NN
                
                sigma = (Etot_prime - Etot)/dAL
                
                ! minus delta
                AL_prime = AL
                AL_prime(1,1) = AL_prime(1,1)*(1.0 - stress_ratio )
                call cal_energy_NN(num_neigh_alltypeM_use,list_neigh_alltypeM_use,AL_prime,xatom,natom,nfeat0,m_neigh)
                Etot_prime = Etot_pred_NN

                sigma = sigma + (Etot_prime - Etot)/dAL

                dS = AL_prime(2,2)*AL_prime(3,3)
                sigma = - sigma * 0.5 / dS
                e_stress(1,1) = sigma
                
                ! y direction
                 
                AL_prime = AL
                dAL = AL_prime(2,2) * stress_ratio 
                AL_prime(2,2) = AL_prime(2,2)*(1.0 + stress_ratio )
                
                call cal_energy_NN(num_neigh_alltypeM_use,list_neigh_alltypeM_use,AL_prime,xatom,natom,nfeat0,m_neigh)
                Etot_prime = Etot_pred_NN
                
                sigma = (Etot_prime - Etot)/dAL
                
                ! minus delta
                AL_prime = AL
                AL_prime(2,2) = AL_prime(2,2)*(1.0 - stress_ratio )
                call cal_energy_NN(num_neigh_alltypeM_use,list_neigh_alltypeM_use,AL_prime,xatom,natom,nfeat0,m_neigh)
                Etot_prime = Etot_pred_NN

                sigma = sigma + (Etot_prime - Etot)/dAL

                dS = AL_prime(1,1)*AL_prime(3,3)
                sigma = - sigma * 0.5 / dS
                e_stress(2,2) = sigma

                ! z direction 

                AL_prime = AL
                dAL = AL_prime(3,3) * stress_ratio 
                AL_prime(3,3) = AL_prime(3,3)*(1.0 + stress_ratio )
                
                !write(*,*) "AL:"
                !write(*,*) AL

                !write(*,*) "AL_prime:"
                !write(*,*) AL_prime

                call cal_energy_NN(num_neigh_alltypeM_use,list_neigh_alltypeM_use,AL_prime,xatom,natom,nfeat0,m_neigh)
                Etot_prime = Etot_pred_NN
                
                !write(*,*) "perturbed Etot z plus"
                !write(*,*) Etot_prime
                
                sigma = (Etot_prime - Etot)/dAL
                
                ! minus delta
                AL_prime = AL
                AL_prime(3,3) = AL_prime(3,3)*(1.0 - stress_ratio )
                call cal_energy_NN(num_neigh_alltypeM_use,list_neigh_alltypeM_use,AL_prime,xatom,natom,nfeat0,m_neigh)
                Etot_prime = Etot_pred_NN

                sigma = sigma + (Etot_prime - Etot)/dAL

                dS = AL_prime(2,2)*AL_prime(1,1) 
                sigma = - sigma * 0.5 / dS
                e_stress(3,3) = sigma

                !write(*,*) "dbg info: e_stress"
                !write(*,*) e_stress

            endif 
        endif
        
        ! fortran inference routine for DP
        if(iflag_model.eq.5) then   
            
            call cal_energy_force_deepMD(AL,xatom,Etot,fatom)
            ! something is missing here 
            ! should copy force and atomic energy into the global variable 
        
            !Etot=Etot_pred_dp
            !e_atom(1:natom_tmp) = energy_pred_dp(1:natom_tmp)
            !fatom(:,1:natom_tmp) = force_pred_dp(:,1:natom_tmp)   ! unit, and - sign?
              
        endif
        
        ! deprecated 
        if(iflag_model.eq.4) then 
            write(*,*) "calling pytorch..."
            s = runqq("predict.py","--dp=True -n DP_cfg_dp 2> err > out")
            write(*,*) "pytorch routine ends"

            ! load from file 
            etot_name = "etot.tmp" 
            ei_name = "ei.tmp"
            fi_name = "force.tmp"

            open(9, file = etot_name, status = "old")
            read(9,*) Etot
            close(9)
            
            open(10, file = fi_name, status = "old")
            do i=1,natom_tmp
                read(10,*) fatom(1:3,i)
            end do
            close(10)

            open(11,file = ei_name, status = "old")
            do i=1,natom_tmp
                read(11,*) e_atom(i)
            end do 
            close(11)

            !do i=1,natom_tmp
            !    write(*,*) e_atom(i)
            !end do
            ! read etot ei force.tmp
        endif        

        if ((iflag_model.eq.1) .or. (iflag_model.eq.2) .or. (iflag_model.eq.3)) then 
        
            deallocate(feat)
            deallocate(dfeat)
            deallocate(list_neigh_alltypeM_use)

        endif
        if(iflag_born_charge_ewald .eq. 1) then
            !write(*,*) "MLFF predict with ewald"
            allocate(ewald_atom(natom))
            allocate(fatom_ewald(3,natom))
            ewald = 0.0d0
            ewald_atom = 0.0d0
            fatom_ewald = 0.0d0
            AL_bohr = AL*Angstrom2Bohr  ! Angstrom2Bohr; to atmoic units
            vol = dabs(AL_bohr(3, 1)*(AL_bohr(1, 2)*AL_bohr(2, 3) - AL_bohr(1, 3)*AL_bohr(2, 2)) &
               + AL_bohr(3, 2)*(AL_bohr(1, 3)*AL_bohr(2, 1) - AL_bohr(1, 1)*AL_bohr(2, 3)) &
               + AL_bohr(3, 3)*(AL_bohr(1, 1)*AL_bohr(2, 2) - AL_bohr(1, 2)*AL_bohr(2, 1)))

            call get_ewald(natom, AL_bohr, iatom, xatom, zatom_ewald, ewald, ewald_atom, fatom_ewald)
            !write(*,*) "MD, before ewald, eatom(1:3): ", e_atom(1:3)
            !write(*,*) "fit-lin-ewald, ewald(1:3) hartree:", ewald_atom(1:3)
            !write(*,*) "Hartree2eV: ", Hartree2eV
            !write(*,*) "ewald(1:3): ", ewald_atom(1:3)*Hartree2eV
            e_atom(1:natom) = e_atom(1:natom) + ewald_atom(1:natom)*Hartree2eV
            fatom(1:3, 1:natom) = fatom(1:3, 1:natom) + fatom_ewald(1:3, 1:natom)*Hartree2eV*Angstrom2Bohr
            deallocate(ewald_atom)
            deallocate(fatom_ewald)
        endif

        return
end subroutine ML_FF_EF
        

