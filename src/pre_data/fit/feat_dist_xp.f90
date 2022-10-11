program feat_dist_xp
    implicit double precision(a - h, o - z)

    integer lwork
    integer, allocatable, dimension(:) :: iatom, iatom_type, itype_atom
    real*8, allocatable, dimension(:) :: Energy, Energyt, energy_pred
    real*8, allocatable, dimension(:, :) :: force_pred
    real*8, allocatable, dimension(:, :) :: feat, feat2
    real*8, allocatable, dimension(:, :, :) :: feat_type, feat2_type
    integer, allocatable, dimension(:) :: num_neigh, num, num_atomtype
    integer, allocatable, dimension(:) :: num_neight
    integer, allocatable, dimension(:, :) :: list_neigh, ind_type

    real*8, allocatable, dimension(:, :, :, :) :: dfeat, dfeat2
    real*8, allocatable, dimension(:, :, :, :) :: dfeat_new
    real*8, allocatable, dimension(:, :, :) :: dfeat_type, dfeat2_type

    real*8, allocatable, dimension(:, :) :: AA
    real*8, allocatable, dimension(:) :: BB

    real*8, allocatable, dimension(:, :, :) :: Gfeat_type
    real*8, allocatable, dimension(:, :) :: Gfeat_tmp

    real*8, allocatable, dimension(:, :, :) :: AA_type
    real*8, allocatable, dimension(:, :) :: BB_type

    real*8, allocatable, dimension(:, :) :: SS_tmp, SS_tmp2

    real*8, allocatable, dimension(:, :, :) :: feat2_ref

    real*8, allocatable, dimension(:, :, :) :: PV
    real*8, allocatable, dimension(:, :) :: feat2_shift, feat2_scale

    integer, allocatable, dimension(:, :, :) :: idd

    real*8, allocatable, dimension(:, :) :: WW, VV, QQ
    real*8, allocatable, dimension(:, :, :, :) :: SS

    real*8, allocatable, dimension(:, :) :: Gfeat2, dGfeat2

    real*8, allocatable, dimension(:, :) :: force

    real*8, allocatable, dimension(:, :) :: xatom
    real*8, allocatable, dimension(:) :: rad_atom, E_ave_vdw
    real*8 AL(3, 3), pi, dE, dFx, dFy, dFz, AL_tmp(3, 3)

    real*8, allocatable, dimension(:, :) :: xatom_tmp

    real*8, allocatable, dimension(:, :) :: feat_new
    real*8, allocatable, dimension(:, :, :) :: feat_new_type
    real*8, allocatable, dimension(:, :, :) :: feat_ext1, feat_ext2, feat_ext3, dfeat_ext1, dfeat_ext2

    integer, allocatable, dimension(:) :: nfeat1, nfeat2
    integer, allocatable, dimension(:, :) :: nfeat, ipos_feat

    real*8, allocatable, dimension(:, :) :: dfeat_tmp
    real*8, allocatable, dimension(:, :) :: feat_ftype
    integer, allocatable, dimension(:) :: iat_tmp, jneigh_tmp, ifeat_tmp
    integer num_tmp, jj
    ! character(len=200) dfeat_n(400)
    character(len=200) trainSetFileDir(400)
    character(len=200) trainSetDir
    character(len=200) MOVEMENTDir, dfeatDir, infoDir, trainDataDir, MOVEMENTallDir
    integer sys_num, sys
    integer nfeat1tm(100), ifeat_type(100), nfeat1t(100)
    integer mm(100), num_ref(100), nkkk0(100), mm0(100)

    integer, allocatable, dimension(:, :) :: iflag_selected, ind_kkk_kkk0

    integer, allocatable, dimension(:, :, :) :: idd0, idd2, idd3

    integer, allocatable, dimension(:, :) :: kkk0_st, kkk0_st2
    real*8, allocatable, dimension(:, :, :) :: S1
    real*8, allocatable, dimension(:, :, :, :) :: S2
    real*8, allocatable, dimension(:, :) :: dE_term, fE_term
    real*8, allocatable, dimension(:, :, :) :: dF_term, fF_term
    real*8, allocatable, dimension(:, :) :: dtot_term, dtot_sum
    real*8, allocatable, dimension(:) :: dE_prev
    real*8, allocatable, dimension(:, :) :: dF_prev
    real*8, allocatable, dimension(:, :) :: S

!       real*8 feat_dist(-100:100,10,20)
    real*8, allocatable, dimension(:, :, :) :: feat_dist
    real*8, allocatable, dimension(:, :, :, :) :: xp, xp1

    real*8 feat_int(-200:200)
! liuliping for relative path
    integer tmp_i
    character(len=200) fitModelDir
    character(len=:), allocatable :: fread_dfeat

    ! this file should be create by prepare.py
    open (1314, file="input/info_dir")
    rewind (1314)
    read (1314, "(A200)") fitModelDir
    close (1314)
    tmp_i = len(trim(adjustl(fitModelDir)))
    allocate (character(len=tmp_i) :: fread_dfeat)
    fread_dfeat = trim(adjustl(fitModelDir))
    write (*, *) "liuliping, fread_dfeat: ", fread_dfeat
! liuliping, end, all .r .x file should be invoke out of fread_dfeat

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    open (10, file=fread_dfeat//"select_VV.input")
    rewind (10)
    read (10, *) nloop
    read (10, *) mm_tot
    read (10, *) nimage_jump
    read (10, *) include3
    read (10, *) ndim1, ndim, width_fact, expd
    close (10)

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    open (10, file=fread_dfeat//"fit_linearMM.input")
    rewind (10)
    read (10, *) ntype, m_neigh
    allocate (itype_atom(ntype))
    allocate (nfeat1(ntype))
    allocate (nfeat2(ntype))
    allocate (rad_atom(ntype))
    allocate (E_ave_vdw(ntype))
    do i = 1, ntype
        read (10, *) itype_atom(i)
    end do
    read (10, *) weight_E, weight_E0, weight_F, delta
    close (10)

    open (10, file=fread_dfeat//"feat.info")
    rewind (10)
    read (10, *) iflag_PCA   ! this can be used to turn off degmm part
    read (10, *) nfeat_type
    do kkk = 1, nfeat_type
        read (10, *) ifeat_type(kkk)   ! the index (1,2,3) of the feature type
    end do
    read (10, *) ntype_tmp
    if (ntype_tmp .ne. ntype) then
        write (6, *) "ntype of atom not same, fit_linearMM.input, feat.info, stop"
        write (6, *) ntype, ntype_tmp
        stop
    end if
    allocate (nfeat(ntype, nfeat_type))
    allocate (ipos_feat(ntype, nfeat_type))
    do i = 1, ntype
        read (10, *) iatom_tmp, nfeat1(i), nfeat2(i)   ! these nfeat1,nfeat2 include all ftype
        if (iatom_tmp .ne. itype_atom(i)) then
            write (6, *) "iatom not same, fit_linearMM.input, feat.info"
            write (6, *) iatom_tmp, itype_atom(i)
            stop
        end if
    end do

    do ii = 1, ntype
        read (10, *) (nfeat(ii, kkk), kkk=1, nfeat_type)
    end do
    close (10)

!   nfeat1(ii) the total (all iftype) num of feature for iatom type ii (sum_kk nfeat(ii,kk))
!   nfeat2(ii) the total num of PCA feature for iatom type ii

    do ii = 1, ntype
        ipos_feat(ii, 1) = 0
        do kkk = 2, nfeat_type
            ipos_feat(ii, kkk) = ipos_feat(ii, kkk - 1) + nfeat(ii, kkk - 1)
        end do
    end do

!ccccccc Right now, nfeat1,nfeat2,for different types
!ccccccc must be the same. We will change that later, allow them
!ccccccc to be different
    nfeat1m = 0   ! the original feature
    nfeat2m = 0   ! the new PCA, PV feature
    do i = 1, ntype
        if (nfeat1(i) .gt. nfeat1m) nfeat1m = nfeat1(i)
        if (nfeat2(i) .gt. nfeat2m) nfeat2m = nfeat2(i)
    end do

    mfeat2 = 0
    do itype = 1, ntype
        if (nfeat2(itype) .gt. mfeat2) mfeat2 = nfeat2(itype)
    end do

!       ndim=4
!       ndim1=20
    allocate (xp(2, ndim, mfeat2, ntype))
    allocate (xp1(2, ndim1, mfeat2, ntype))

    write (6, *) "input: 1 for fixed old xp;2 for new variable xp"
    read (5, *) iflag_tmp

    if (iflag_tmp .eq. 1) then
        if (ndim .ne. 4) then
            write (6, *) "for fixed old xp, ndim must be 4, input_ndim=", ndim
            stop
        end if

        do i = 1, ntype
            do j = 1, mfeat2
                xp(1, 1, j, i) = -3.9
                xp(2, 1, j, i) = 2.6
                xp(1, 2, j, i) = -1.3
                xp(2, 2, j, i) = 2.6
                xp(1, 3, j, i) = 1.3
                xp(2, 3, j, i) = 2.6
                xp(1, 4, j, i) = 3.9
                xp(2, 4, j, i) = 2.6
            end do
        end do

        do i = 1, ntype
            do j = 1, mfeat2
                do id1 = 1, ndim1
                    xp1(1, id1, j, i) = -(id1 - ndim1/2)*3.0/ndim1
                    xp1(2, id1, j, i) = 3.d0/ndim1
                end do
            end do
        end do

        open (12, file=fread_dfeat//"OUT.xp", form="unformatted")
        rewind (12)
        write (12) mfeat2, ntype, ndim, ndim1
        write (12) nfeat2
        write (12) xp
        write (12) xp1
        close (12)

        stop

    end if
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    allocate (PV(nfeat1m, nfeat2m, ntype))
    allocate (feat2_shift(nfeat2m, ntype))
    allocate (feat2_scale(nfeat2m, ntype))
    do itype = 1, ntype
        open (11, file=fread_dfeat//"feat_PV."//char(itype + 48), form="unformatted")
        rewind (11)
        read (11) nfeat1_tmp, nfeat2_tmp
        if (nfeat2_tmp .ne. nfeat2(itype)) then
            write (6, *) "nfeat2.not.same,feat2_ref", itype, nfeat2_tmp, nfeat2(itype)
            stop
        end if
        if (nfeat1_tmp .ne. nfeat1(itype)) then
            write (6, *) "nfeat1.not.same,feat2_ref", itype, nfeat1_tmp, nfeat1(itype)
            stop
        end if
        read (11) PV(1:nfeat1(itype), 1:nfeat2(itype), itype)
!       read(11) ((PV(i1,i2,itype),i1=1,nfeat1(itype)),i2=1,nfeat2(itype))
        read (11) feat2_shift(1:nfeat2(itype), itype)
        read (11) feat2_scale(1:nfeat2(itype), itype)
        close (11)
    end do
!cccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccc  ind_kkk_kkk0(kkk)=kkk0

    allocate (num_atomtype(ntype))
    allocate (num(ntype))   ! only use this as a temp array

    ! sys_num=400
    open (13, file=fread_dfeat//"location")
    rewind (13)
    read (13, *) sys_num  !,trainSetDir
    read (13, '(a200)') trainSetDir
    ! allocate(trainSetFileDir(sys_num))
    do i = 1, sys_num
        read (13, '(a200)') trainSetFileDir(i)
    end do
    close (13)

    ind_kkk_kkk0 = 0

    iflag_selected = 0

    allocate (feat_dist(-200:200, mfeat2, ntype))

    feat_dist = 0.d0

    do 900 sys = 1, sys_num

        do 777 kkk = 1, nfeat_type
            ! MOVEMENTDir=trim(trainSetFileDir(sys))//"/MOVEMENT"
            dfeatDir = trim(trainSetFileDir(sys))//"/dfeat.fbin.Ftype"//char(ifeat_type(kkk) + 48)
            open (1000 + kkk, file=dfeatDir, action="read", access="stream", form="unformatted")
            rewind (1000 + kkk)
            read (1000 + kkk) nimaget, natomt, nfeat1tm(kkk), m_neight

!      nfeat1tm(kkk) is the max(nfeat(ii,kkk)) for all ii(iatype)

            if (kkk .eq. 1) then
                nimage = nimaget
                natom = natomt
                m_neigh = m_neight
            else
                if (nimaget .ne. nimage .or. natomt .ne. natom .or. m_neight .ne. m_neigh) then
                    write (6, *) "param changed in diff ifeat_type"
                    write (6, *) nimage, natom, m_neigh
                    write (6, *) nimaget, natomt, m_neight
                    stop
                end if
            end if

            read (1000 + kkk) ntype_tmp, (nfeat1t(ii), ii=1, ntype_tmp)
!    This is one etra line, perhaps we don't need it

!! for this kkk_ftype, for each atom type, ii, the num of feature is nfeat1t(ii)
!cccccccccccccccccccccccccccccccccccccccccccccccc
            if (ntype_tmp .ne. ntype) then
                write (6, *) "ntype_tmp.ne.ntype,dfeat.fbin,stop"
                write (6, *) ntype_tmp, ntype
                stop
            end if

            do ii = 1, ntype
                if (nfeat1t(ii) .ne. nfeat(ii, kkk)) then   ! the num of feat for ii_th iatype, and kkk_th feat type
                    write (6, *) "nfeat1t not the same, dfeat.fbin,stop"
                    write (6, *) nfeat1t(ii), nfeat(ii, kkk), ii, kkk
                    stop
                end if
            end do

            if (kkk .eq. 1) then
                if (sys .ne. 1 .or. iloop .gt. 1 .or. iloop2 .gt. 1) then
                    deallocate (iatom)
                end if
                allocate (iatom(natom))
            end if
            read (1000 + kkk) iatom      ! The same for different kkk

777         continue
!ccccccccccccccccccccccccccccccccccccccccccccccccc

            if (sys .ne. 1) then

                deallocate (iatom_type)
                deallocate (Energy)
                deallocate (Energyt)
                deallocate (num_neight)
                deallocate (feat)
                deallocate (feat2)
                deallocate (feat_type)
                deallocate (feat2_type)
                deallocate (num_neigh)
                deallocate (list_neigh)
                deallocate (ind_type)
                deallocate (dfeat)
                deallocate (dfeat_type)
                deallocate (dfeat2_type)
                deallocate (dfeat2)
                deallocate (xatom)

                deallocate (force)
                deallocate (SS)

                deallocate (energy_pred)
                deallocate (force_pred)

            end if

!cccccccccccccccccccccccccccccccccccccccccccccccc

            allocate (iatom_type(natom))
            allocate (Energy(natom))
            allocate (Energyt(natom))
            allocate (num_neight(natom))
            allocate (feat(nfeat1m, natom))
! nfeat1m is the max(nfeat1(ii)) for ii(iatype), nfeat1(ii)=sum_kkk nfeat(ii,kkk)
! nfeat1m is the max num of total feature (sum over all feature type)
            allocate (feat2(nfeat2m, natom))
            allocate (feat_type(nfeat1m, natom, ntype))
            allocate (feat2_type(nfeat2m, natom, ntype))
            allocate (num_neigh(natom))
            allocate (list_neigh(m_neigh, natom))
            allocate (ind_type(natom, ntype))
            allocate (dfeat(nfeat1m, natom, m_neigh, 3))
            allocate (dfeat_type(nfeat1m, natom*m_neigh*3, ntype))
            allocate (dfeat2_type(nfeat2m, natom*m_neigh*3, ntype))
            allocate (dfeat2(nfeat2m, natom, m_neigh, 3))
            allocate (xatom(3, natom))
            allocate (energy_pred(natom))
            allocate (force_pred(3, natom))

            dfeat = 0.d0
            dfeat_type = 0.d0

            allocate (force(3, natom))
            allocate (SS(num_refm, natom, 3, ntype))

            pi = 4*datan(1.d0)

            do i = 1, natom
                iitype = 0
                do itype = 1, ntype
                    if (itype_atom(itype) .eq. iatom(i)) then
                        iitype = itype
                    end if
                end do
                if (iitype .eq. 0) then
                    write (6, *) "this type not found", iatom(i)
                end if
                iatom_type(i) = iitype
            end do

            num_atomtype = 0
            do i = 1, natom
                itype = iatom_type(i)
                num_atomtype(itype) = num_atomtype(itype) + 1
            end do

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

            do 3000 image = 1, nimage

!-----------------------------------------------------------------
!-----------------------------------------------------------------
!---- read in the feature from different kkk, and put them together
!-----------------------------------------------------------------
                dfeat(:, :, :, :) = 0.0
                feat(:, :) = 0.0

                do 778 kkk = 1, nfeat_type

                    allocate (feat_ftype(nfeat1tm(kkk), natom))
                    read (1000 + kkk) energy   ! repeated
                    read (1000 + kkk) force    ! repeated
                    read (1000 + kkk) feat_ftype

                    if (kkk .eq. 1) then
                        energyt = energy
                    else
                        diff = 0.d0
                        do ii = 1, natom
                            diff = diff + abs(energyt(ii) - energy(ii))
                        end do
                        if (diff .gt. 1.E-9) then
                            write (6, *) "energy Ei not the same for diff ifeature type, stop"
                            stop
                        end if
                    end if

                    do iat = 1, natom
                        itype = iatom_type(iat)
                        do ii = 1, nfeat(itype, kkk)
                            feat(ii + ipos_feat(itype, kkk), iat) = feat_ftype(ii, iat)   ! put different kkk together
                        end do
                    end do
                    deallocate (feat_ftype)

                    read (1000 + kkk) num_neigh     ! this is actually the num_neighM (of Rc_M)
                    read (1000 + kkk) list_neigh    ! this is actually the list_neighM (of Rc_M)
!    the above should be the same for different kkk.
!    Perhaps we should check it later. Here we proceed without checking
                    if (kkk .eq. 1) then
                        num_neight = num_neigh
                    else
                        diff = 0.d0
                        do ii = 1, natom
                            diff = diff + abs(num_neight(ii) - num_neigh(ii))
                        end do
                        if (diff .gt. 1.E-9) then
                            write (6, *) "num_neigh not the same for diff ifeature type,stop"
                            stop
                        end if
                    end if

!TODO:
                    ! read(10) dfeat
                    read (1000 + kkk) num_tmp
                    allocate (dfeat_tmp(3, num_tmp))
                    allocate (iat_tmp(num_tmp))
                    allocate (jneigh_tmp(num_tmp))
                    allocate (ifeat_tmp(num_tmp))
                    read (1000 + kkk) iat_tmp
                    read (1000 + kkk) jneigh_tmp
                    read (1000 + kkk) ifeat_tmp
                    read (1000 + kkk) dfeat_tmp

                    read (1000 + kkk) xatom    ! xatom(3,natom), repeated for diff kkk
                    read (1000 + kkk) AL       ! AL(3,3), repeated for diff kkk

                    do jj = 1, num_tmp

                        itype2 = iatom_type(list_neigh(jneigh_tmp(jj), iat_tmp(jj))) ! itype2: the type of the neighbor
                        dfeat(ifeat_tmp(jj) + ipos_feat(itype2, kkk), iat_tmp(jj), jneigh_tmp(jj), :) = dfeat_tmp(:, jj)
!  Place dfeat from different iftype into the same dfeat
                    end do
                    deallocate (dfeat_tmp)
                    deallocate (iat_tmp)
                    deallocate (jneigh_tmp)
                    deallocate (ifeat_tmp)

778                 continue
!ccccccccccccccccccccccccccccccccccccccccccccccccc

                    num = 0
                    do i = 1, natom
                        itype = iatom_type(i)
                        num(itype) = num(itype) + 1
                        ind_type(num(itype), itype) = i
                        feat_type(:, num(itype), itype) = feat(:, i)
!  we have to seperate the feature into different iatype, since they have different PV
!  The num of total feature for different iatype is nfeat1(iatype)
                    end do
!cccccccccccccccccccccccccccccccccccccccccc

                    do itype = 1, ntype
                        call dgemm('T', 'N', nfeat2(itype), num_atomtype(itype), nfeat1(itype), 1.d0, PV(1, 1, itype),&
                      & nfeat1m, feat_type(1, 1, itype), nfeat1m, 0.d0, feat2_type(1, 1, itype), nfeat2m)
                    end do

                    do itype = 1, ntype
                        do i = 1, num_atomtype(itype)
!       do j=1,nfeat2(itype)-1
                            do j = 1, nfeat2(itype)
                                feat2_type(j, i, itype) = (feat2_type(j, i, itype) - feat2_shift(j, itype))*feat2_scale(j, itype)

                                itt = feat2_type(j, i, itype)*30
                                if (itt .gt. 200) itt = 200
                                if (itt .lt. -200) itt = -200

                                feat_dist(itt, j, itype) = feat_dist(itt, j, itype) + 1

                            end do
                            feat2_type(nfeat2(itype), i, itype) = 1.d0   ! the special value 1 component
                        end do
                    end do  ! itype

                    num = 0
                    do i = 1, natom
                        itype = iatom_type(i)
                        num(itype) = num(itype) + 1
                        feat2(:, i) = feat2_type(:, num(itype), itype)
!  Here, we collect different iatype back, into feat2(:,i)
!  But actually, different atom (different iatype) will have different number of features
!  But all stored within nfeat2m
                    end do

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cc dfeat(nfeat1,natom,j_neigh,3): dfeat(j,i,jj,3)= d/dR_i(feat(j,list_neigh(jj,i))
!cccccccccccc

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccc We will generate the mm VV feature from these features
!ccccccccccccccccccccccccccccccccccccccccccccccc

3000                continue   ! image

                    do kkk = 1, nfeat_type
                        close (1000 + kkk)     ! dfeat.fbin.Ftype
                    end do

900                 continue   ! system

                    do itype = 1, ntype
                        do j = 1, nfeat2(itype)

                            feat_int(-200) = feat_dist(-200, j, itype)**expd
                            do i = -199, 200
                                feat_int(i) = feat_int(i - 1) + feat_dist(i, j, itype)**expd
                            end do
                            fact = 1.d0/feat_int(200)
                            feat_int = feat_int*fact

                            fact = 1.d0/ndim1
                            do id1 = 1, ndim1
                                fc = (0.5 + id1 - 1)*fact   ! the center point value
                                do i = -200, 200
                                    if (feat_int(i) .ge. fc) exit
                                end do
! i is the fc point.
                                xp1(1, id1, j, itype) = i*1.d0/30
                            end do

!ccccccccccccccccccccccccccccccccccccccccccccccccc
3012                        iflag = 0
                            do id = 1, ndim1 - 1
                                if (abs(xp1(1, id + 1, j, itype) - xp1(1, id, j, itype)) .lt. 1.D-4) then
                                    iflag = iflag + 1
                                    if (id .gt. 1) then
                                        xp1(1, id, j, itype) = (xp1(1, id + 1, j, itype) + xp1(1, id - 1, j, itype))/2
                                    end if
                                    if (id .lt. ndim - 1) then
                                        xp1(1, id + 1, j, itype) = (xp1(1, id + 2, j, itype) + xp1(1, id, j, itype))/2
                                    end if
                                end if
                            end do
                            if (iflag .ne. 0) goto 3012
!ccccccccccccccccccccccccccccccccccccccccccccccccc

                            xp1(2, 1, j, itype) = (xp1(1, 2, j, itype) - xp1(1, 1, j, itype))*width_fact
                            xp1(2, ndim1, j, itype) = (xp1(1, ndim1, j, itype) - xp1(1, ndim1 - 1, j, itype))*width_fact

                            do id1 = 2, ndim1 - 1
                                d1 = xp1(1, id1, j, itype) - xp1(1, id1 - 1, j, itype)
                                d2 = xp1(1, id1 + 1, j, itype) - xp1(1, id1, j, itype)
                                d = d1
                                if (d2 .lt. d) d = d2   ! take the smaller one, another choice, average
                                xp1(2, id1, j, itype) = d*width_fact
                                if (xp1(2, id1, j, itype) .lt. 1.D-4) then
                                    write (6, *) "warning xp1(2).lt.1.D-4", id1, j, itype, xp1(2, id, j, itype)
                                end if
                            end do

!ccccccccccccccccccccccccccccc
                            fact = 1.d0/ndim
                            do id = 1, ndim
                                fc = (0.5 + id - 1)*fact   ! the center point value
                                do i = -200, 200
                                    if (feat_int(i) .ge. fc) exit
                                end do
! i is the fc point.
                                xp(1, id, j, itype) = i*1.d0/30
                            end do

!ccccccccccccccccccccccccccccccccccccccccccccc
!ccc make sure there is no zero interval
3011                        iflag = 0
                            do id = 1, ndim - 1
                                if (abs(xp(1, id + 1, j, itype) - xp(1, id, j, itype)) .lt. 1.D-4) then
                                    iflag = iflag + 1
                                    if (id .gt. 1) then
                                        xp(1, id, j, itype) = (xp(1, id + 1, j, itype) + xp(1, id - 1, j, itype))/2
                                    end if
                                    if (id .lt. ndim - 1) then
                                        xp(1, id + 1, j, itype) = (xp(1, id + 2, j, itype) + xp(1, id, j, itype))/2
                                    end if
                                end if
                            end do
                            if (iflag .ne. 0) goto 3011
!ccccccccccccccccccccccccccccccccccccccccccccc

                            xp(2, 1, j, itype) = (xp(1, 2, j, itype) - xp(1, 1, j, itype))*width_fact
                            xp(2, ndim, j, itype) = (xp(1, ndim, j, itype) - xp(1, ndim - 1, j, itype))*width_fact
                            do id = 2, ndim - 1
                                d1 = xp(1, id, j, itype) - xp(1, id - 1, j, itype)
                                d2 = xp(1, id + 1, j, itype) - xp(1, id, j, itype)
                                d = d1
                                if (d2 .lt. d) d = d2   ! take the smaller one, another choice, average
                                xp(2, id, j, itype) = d*width_fact
                                if (xp(2, id, j, itype) .lt. 1.D-4) then
                                    write (6, *) "warning xp(2).lt.1.D-4", id, j, itype, xp(2, id, j, itype)
                                end if
                            end do

                        end do   !  j
                    end do   ! itype

                    open (12, file=fread_dfeat//"OUT.xp", form="unformatted")
                    rewind (12)
                    write (12) mfeat2, ntype, ndim, ndim1
                    write (12) nfeat2
                    write (12) xp
                    write (12) xp1
                    close (12)

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                    open (13, file=fread_dfeat//"feat.dist")
                    rewind (13)
                    do ii = -200, 200
                        write (13, "(E13.6,2x,10(E13.6,1x))") ii/20.d0, ((feat_dist(ii, j, itype), j=1, 5), itype=1, ntype)
                    end do
                    close (13)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                    write (6, *) "nfeat2=", (nfeat2(i), i=1, ntype)
2000                continue
                    write (6, *) "you can check xp(1)(position),xp(2)(width)"
                    write (6, *) "input jfeat(1:nfeat2),itype(0,0 to end it)"
                    read (5, *) j, itype
                    if (j .eq. 0 .or. itype .eq. 0) stop
                    write (6, *) "------- xp ------"
                    do id = 1, ndim
                        write (6, *) xp(1, id, j, itype), xp(2, id, j, itype)
                    end do
                    write (6, *) "------- xp1 ------"
                    do id1 = 1, ndim1
                        write (6, *) xp1(1, id1, j, itype), xp1(2, id1, j, itype)
                    end do
                    goto 2000

                    deallocate (fread_dfeat)
                    stop
                end

