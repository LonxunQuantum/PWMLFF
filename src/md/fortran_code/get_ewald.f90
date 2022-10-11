!subroutine get_ewald(natom, xatom, fatom_in_ewald, AL, ityatom, ewald_in_ewald, eatom_in_ewald, zatom_in_ewald)
subroutine get_ewald(natom, AL, ityatom, xatom, zatom_in_ewald, ewald_in_ewald, eatom_in_ewald, fatom_in_ewald)
    ! atomic units
    use data_ewald
    implicit none
    !include "mpif.h"
    integer natom
    integer natom_st, natom_fn
    integer jnode, i, j, k, ia, ia1, ia2
    integer ii, jj, kk, i1, j1, k1
    integer ngtot
    real*8 d1, d2, d3, dave, pi, beta
    integer nc1, nc2, nc3
    integer ng_node, ng_used
    real*8 ewald_in_ewald
    real*8 x1, y1, z1, x0, y0, z0, x, y, z, r, derfrdr
    real*8 ch1, ch2, sr, srx, sry, srz
    real*8 sq, yy
    real*8 AL(3, 3)
    real*8 ALI(3, 3)
    real*8 xatom(3, natom), fatom_in_ewald(3, natom)
    real*8 eatom_in_ewald(natom)
    real*8 sr_atom(matom)
    real*8 sq_atom(matom)
    real*8, allocatable :: gk(:, :)
    real*8 gkk, akk, ff1, ff2, s_cos, s_sin, ph, ch, s_tmp
    real*8 zatom_in_ewald(120)  !mtype=10
    integer ityatom(natom)

    pi = 4*datan(1.d0)
    d1 = dsqrt(AL(1, 1)**2 + AL(2, 1)**2 + AL(3, 1)**2)
    d2 = dsqrt(AL(1, 2)**2 + AL(2, 2)**2 + AL(3, 2)**2)
    d3 = dsqrt(AL(1, 3)**2 + AL(2, 3)**2 + AL(3, 3)**2)

    dave = (d1*d2*d3)**0.333333d0
    if (dave .le. 16.d0) then
        beta = 3.0d0/dave
    else
        beta = 5.d0/dave
    end if

    nc1 = 16.d0/beta/d1 + 1
    nc2 = 16.d0/beta/d2 + 1
    nc3 = 16.d0/beta/d3 + 1

    ewald_in_ewald = 0.d0
    eatom_in_ewald = 0.0d0
    sr_atom = 0.0d0
    sq_atom = 0.0d0
    do ia1 = 1, natom
        fatom_in_ewald(1, ia1) = 0.d0
        fatom_in_ewald(2, ia1) = 0.d0
        fatom_in_ewald(3, ia1) = 0.d0
        sr_atom(ia1) = 0.0d0

        do ia2 = 1, natom
            x1 = xatom(1, ia2) - xatom(1, ia1)
            y1 = xatom(2, ia2) - xatom(2, ia1)
            z1 = xatom(3, ia2) - xatom(3, ia1)
            x0 = AL(1, 1)*x1 + AL(1, 2)*y1 + AL(1, 3)*z1
            y0 = AL(2, 1)*x1 + AL(2, 2)*y1 + AL(2, 3)*z1
            z0 = AL(3, 1)*x1 + AL(3, 2)*y1 + AL(3, 3)*z1

            ch1 = zatom_in_ewald(ityatom(ia1))
            ch2 = zatom_in_ewald(ityatom(ia2))

            sr = 0.d0
            srx = 0.d0
            sry = 0.d0
            srz = 0.d0
            do i = -nc1, nc1
                do j = -nc2, nc2
                    do k = -nc3, nc3
                        x = x0 + AL(1, 1)*i + AL(1, 2)*j + AL(1, 3)*k
                        y = y0 + AL(2, 1)*i + AL(2, 2)*j + AL(2, 3)*k
                        z = z0 + AL(3, 1)*i + AL(3, 2)*j + AL(3, 3)*k
                        r = dsqrt(x**2 + y**2 + z**2)

                        if (beta*r .gt. 8.d0) cycle

                        if (ia2 .ne. ia1) then
                            derfrdr = (erfc(beta*(r + 2.d-5))/(r + 2.d-5) - erfc(beta*r)/r)/2.d-5

                            srx = srx - derfrdr*x/r
                            sry = sry - derfrdr*y/r
                            srz = srz - derfrdr*z/r
                        end if

                        if (ia1 .eq. ia2 .and. i .eq. 0 .and. j .eq. 0 .and. k .eq. 0) cycle
                        sr = sr + erfc(beta*r)/r
                        sr_atom(ia1) = sr_atom(ia1) + erfc(beta*r)/r

                    end do
                end do
            end do
!200   continue
            if (ia1 .eq. ia2) sr = sr - 2*beta/dsqrt(pi)

            !ccccc factor 2 for the sum 1, ng2, is only half the sphere

            ewald_in_ewald = ewald_in_ewald + sr*ch1*ch2*0.5d0
            eatom_in_ewald(ia1) = eatom_in_ewald(ia1) + sr*ch1*ch2*0.5d0

            fatom_in_ewald(1, ia1) = fatom_in_ewald(1, ia1) + srx*ch1*ch2
            fatom_in_ewald(2, ia1) = fatom_in_ewald(2, ia1) + sry*ch1*ch2
            fatom_in_ewald(3, ia1) = fatom_in_ewald(3, ia1) + srz*ch1*ch2
        end do
    end do

!1000  continue
    ! liuliping, gen G2 my self in the whole sphere.
    ! column vectors in ALI are BASIS, and vector=matmul(ALI,fraction)

    ! lat in C/C++ for row vector; AL in fortran for column vector
    ! AL = lat.T
    ! reclat = inv(lat)
    ! ALI = inv(AL.T) = reclat
    ! ALI's column vector (2pi) is the BASIS in reciprocal space,
    ! and reclat is only a inv of lat, for inversion of fraction and cartesian
    ! PYTHON: (C, row-vector)
    ! (x,y,z) = (x1,x2,x3) lat; x_cart = np.dot(x_frac, lat)
    ! (x1,x2,x3) = (x,y,z) reclat; x_frac = np.dot(x_cart, reclat)

    ! Fortran:
    ! x      x1
    ! y = AL x2  ; x_cart_fortran = matmul(AL, x_frac_fortran)
    ! z      x3
    ! x1         x
    ! x2 = ALI.T y  ; x_frac_fortran = matmul(transpose(ALI), x_cart_fortran)
    ! x3         z

    call get_ALI(AL, ALI)
    ngtot = n1*n2*n3
    allocate (gk(3, ngtot))
    ng_node = 0
    do kk = 1, n3
        do jj = 1, n2
            do ii = 1, n1
                i1 = ii - 1
                j1 = jj - 1
                k1 = kk - 1
                if (i1 > n1/2) i1 = i1 - n1
                if (j1 > n2/2) j1 = j1 - n2
                if (k1 > n3/2) k1 = j1 - n3
                akk = 0.5d0*(2*pi)**2*( &
                      (ALI(1, 1)*i1 + ALI(1, 2)*j1 + ALI(1, 3)*k1)**2 &
                      + (ALI(2, 1)*i1 + ALI(2, 2)*j1 + ALI(2, 3)*k1)**2 &
                      + (ALI(3, 1)*i1 + ALI(3, 2)*j1 + ALI(3, 3)*k1)**2)
                if (akk > Ecut2) cycle
                if (k1 < 0) cycle
                if (k1 == 0 .and. j1 < 0) cycle
                if (k1 == 0 .and. j1 == 0 .and. i1 < 0) cycle
                ng_node = ng_node + 1
                gk(1, ng_node) = 2*pi*(ALI(1, 1)*i1 + ALI(1, 2)*j1 + ALI(1, 3)*k1)
                gk(2, ng_node) = 2*pi*(ALI(2, 1)*i1 + ALI(2, 2)*j1 + ALI(2, 3)*k1)
                gk(3, ng_node) = 2*pi*(ALI(3, 1)*i1 + ALI(3, 2)*j1 + ALI(3, 3)*k1)
            end do
        end do
    end do
    ! gen g2 end
    !write (*, *) "ng= ", ng_node
    !write (*, *) "gk 1:30"
    !write (*, *) gk(1:3, 1:30)
    !write (*, *) "volume: ", vol
    !write (*, *) "beta: ", beta
    !write (*, *) "AL:"
    !write (*, *) AL
    !write (*, *) "ALI:"
    !write (*, *) ALI
    ng_used = 0
    sq = 0.d0

    do i = 1, ng_node
        gkk = gk(1, i)**2 + gk(2, i)**2 + gk(3, i)**2
        yy = gkk/(4*beta**2)

        if (gkk .le. 1.D-10 .or. yy .gt. 50.d0) cycle

        ng_used = ng_used + 1
        ff1 = dexp(-yy)
        ff2 = ff1*4*pi/vol*2.d0/gkk

        s_cos = 0.d0
        do ia1 = 1, natom
            s_sin = 0.d0
            s_tmp = 0.0d0
            do ia2 = 1, natom
                !if (ia2 == ia1) cycle
                x1 = xatom(1, ia2) - xatom(1, ia1)
                y1 = xatom(2, ia2) - xatom(2, ia1)
                z1 = xatom(3, ia2) - xatom(3, ia1)

                x0 = AL(1, 1)*x1 + AL(1, 2)*y1 + AL(1, 3)*z1
                y0 = AL(2, 1)*x1 + AL(2, 2)*y1 + AL(2, 3)*z1
                z0 = AL(3, 1)*x1 + AL(3, 2)*y1 + AL(3, 3)*z1

                ph = x0*gk(1, i) + y0*gk(2, i) + z0*gk(3, i)

                ch = zatom_in_ewald(ityatom(ia1))*zatom_in_ewald(ityatom(ia2))

                s_cos = s_cos + ch*dcos(ph)
                s_sin = s_sin + ch*dsin(ph)
                s_tmp = s_tmp + ch*dcos(ph)
            end do !ia2

            s_sin = s_sin*ff2
            fatom_in_ewald(1, ia1) = fatom_in_ewald(1, ia1) + s_sin*gk(1, i)
            fatom_in_ewald(2, ia1) = fatom_in_ewald(2, ia1) + s_sin*gk(2, i)
            fatom_in_ewald(3, ia1) = fatom_in_ewald(3, ia1) + s_sin*gk(3, i)
            sq_atom(ia1) = sq_atom(ia1) + ff1*s_tmp/gkk
        end do !ia1

        sq = sq + ff1*s_cos/gkk

    end do ! i; g vectors
    !ccccc factor 2 for the sum 1, ng2, is only half the sphere

    sq = sq*4*pi/vol*2.d0
    sq_atom(1:natom) = sq_atom(1:natom)*4.0*pi/vol*2.d0
    ch = 0.d0
    do ia1 = 1, natom
        do ia2 = 1, natom
            ch = ch + zatom_in_ewald(ityatom(ia1))*zatom_in_ewald(ityatom(ia2))
        end do
    end do

    !write(*,*) "224 ch: ", ch
    sq = sq - ch*pi/beta**2/vol
    ewald_in_ewald = ewald_in_ewald + sq*0.5d0
    do ia1 = 1, natom
        eatom_in_ewald(ia1) = eatom_in_ewald(ia1) + sq_atom(ia1)*0.5d0
    end do 

    !write(*,*) "ewald_in_ewald: ", ewald_in_ewald
    !write(*,*) "sum of eatom_in_ewald: ", sum(eatom_in_ewald(1:natom))
    deallocate (gk)
    return
end

