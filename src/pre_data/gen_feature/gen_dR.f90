PROGRAM gen_dR
    IMPLICIT NONE
    integer :: move_file=1101
    real*8 AL(3,3),Etotp
    real*8,allocatable,dimension (:,:) :: xatom,fatom
    real*8,allocatable,dimension (:) :: Eatom
    integer,allocatable,dimension (:) :: iatom
    logical nextline
    integer num_step, natom, i, j, k
    integer num_step0,num_step1,natom0,max_neigh
    real*8 Etotp_ave,E_tolerance
    character(len=50) char_tmp(20)
    character(len=200) trainSetFileDir(5000)
    character(len=200) trainSetDir
    character(len=200) MOVEMENTDir
    integer sys_num,sys

    integer,allocatable,dimension (:,:,:) :: list_neigh,iat_neigh,iat_neigh_M
    integer,allocatable,dimension (:,:) :: num_neigh
    real*8,allocatable,dimension (:,:,:,:) :: dR_neigh
    real*8 Rc_M

    integer,allocatable,dimension (:) :: itype_atom
    integer,allocatable,dimension (:,:,:) :: map2neigh_M
    integer,allocatable,dimension (:,:,:) :: list_neigh_M
    integer,allocatable,dimension (:,:) :: num_neigh_M

    integer m_neigh
    integer ntype, nfeat0(100)
    integer iat_type(100)

    real*8 Rc_type(100), Rc2_type(100), Rm_type(100),weight_rterm(100)

    integer, parameter :: n = 63  ! Total number of elements

    type dictionary_type
        integer :: order
        real :: atomic_E
    end type dictionary_type

    type(dictionary_type), dimension(n) :: dictionary  
    integer :: o

    !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    INTERFACE
        SUBROUTINE scan_title (io_file, title, title_line, if_find)
            CHARACTER(LEN=200), OPTIONAL :: title_line
            LOGICAL, OPTIONAL :: if_find
            INTEGER :: io_file
            CHARACTER(LEN=*) :: title
        END SUBROUTINE scan_title
    END INTERFACE
    !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

    open(10, file="input/gen_dR_feature.in", status="old", action="read")
    rewind(10)
    read(10,*) Rc_M,m_neigh
    read(10,*) ntype
    do i=1,ntype
        read(10,*) iat_type(i)
        read(10,*) Rc_type(i)
    
        if(Rc_type(i).gt.Rc_M) then
            write(6,*) "Rc_type must be smaller than Rc_M, gen_3b_feature.in",i,Rc_type(i),Rc_M
            stop
        endif
    enddo
    read(10,*) E_tolerance
    close(10)

    open(13,file="input/location")
    rewind(13)
    read(13,*) sys_num  !,trainSetDir
    read(13,'(a200)') trainSetDir
    ! allocate(trainSetFileDir(sys_num))
    do i=1,sys_num
        read(13,'(a200)') trainSetFileDir(i)    
    enddo
    close(13)

    do 2333 sys=1,sys_num
        MOVEMENTDir=trim(trainSetFileDir(sys))//"/MOVEMENT"

!cccccccccccccccccccccccccccccccccccccccccccccccccccc
    OPEN (move_file,file=MOVEMENTDir,status="old",action="read") 
    rewind(move_file)

    num_step0=0
    Etotp_ave=0.d0
1001 continue
    call scan_title (move_file,"ITERATION",if_find=nextline)
    if(.not.nextline) goto 1002
        num_step0=num_step0+1
    backspace(move_file)
    read(move_file,*) natom0
    if(num_step0.gt.1.and.natom.ne.natom0) then
        write(6,*) "The natom cannot change within one MOVEMENT FILE", &
        num_step0,natom0 
    endif
    natom=natom0
    
    CALL scan_title (move_file, "ATOMIC-ENERGY",if_find=nextline)
    if(.not.nextline) then
        write(6,*) "Atomic-energy not found, stop",num_step0
        stop
    endif

    backspace(move_file)
    read(move_file,*) char_tmp(1:4),Etotp
    Etotp_ave=Etotp_ave+Etotp
    goto 1001
1002  continue
    close(move_file)

    Etotp_ave=Etotp_ave/num_step0
    write(6,*) "num_step,natom,Etotp_ave=",num_step0,natom,Etotp_ave
    !cccccccccccccccccccccccccccccccccccccccccccccccccccc
    ALLOCATE (iatom(natom),xatom(3,natom),fatom(3,natom),Eatom(natom))
    !cccccccccccccccccccccccccccccccccccccccccccccccccccc
    OPEN (move_file,file=MOVEMENTDir,status="old",action="read") 
    rewind(move_file)

    num_step1=0
1003 continue
    call scan_title (move_file,"ITERATION",if_find=nextline)
    if(.not.nextline) goto 1004

    CALL scan_title (move_file, "POSITION")
    DO j = 1, natom
        READ(move_file, *) iatom(j),xatom(1,j),xatom(2,j),xatom(3,j)
    ENDDO

    CALL scan_title (move_file, "ATOMIC-ENERGY",if_find=nextline)

    backspace(move_file)
    read(move_file,*) char_tmp(1:4),Etotp

    if(abs(Etotp-Etotp_ave).le.E_tolerance) then
        num_step1=num_step1+1
    endif
    goto 1003
1004  continue
    close(move_file)

    write(6,*) "nstep0,nstep1(used)",num_step0,num_step1

    DEALLOCATE (iatom,xatom,fatom,Eatom)

    !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc!
    OPEN (move_file,file=MOVEMENTDir,status="old",action="read") 
    rewind(move_file)

    max_neigh=-1
    num_step=0
    num_step1=0
1000  continue

    !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    call scan_title (move_file,"ITERATION",if_find=nextline)

    if(.not.nextline) goto 2000
        num_step=num_step+1

        backspace(move_file) 
        read(move_file, *) natom
        open(1317, file='./PWdata/ImageAtomNum.dat', access='append')
        write(1317,"(I6)") natom
        close(1317)

        ALLOCATE (iatom(natom),xatom(3,natom),fatom(3,natom),Eatom(natom))

        CALL scan_title (move_file, "LATTICE")
        DO j = 1, 3
            READ (move_file,*) AL(1:3,j)
        ENDDO

        CALL scan_title (move_file, "POSITION")
        DO j = 1, natom
            READ(move_file, *) iatom(j),xatom(1,j),xatom(2,j),xatom(3,j)
        ENDDO

        CALL scan_title (move_file, "FORCE", if_find=nextline)
        if(.not.nextline) then
            write(6,*) "force not found, stop", num_step
            stop
        endif
        
        open(1315, file='./PWdata/Force.dat', access='append')
        DO j = 1, natom
            READ(move_file, *) iatom(j),fatom(1,j),fatom(2,j),fatom(3,j)
            write(1315, "(3(F24.15, 1x))") fatom(1,j),fatom(2,j),fatom(3,j)
        ENDDO
        close(1315)

        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        ! Initializes the element's Ei dictionary
        data dictionary%order /1, 3, 4, 5, 6, 7, 8, 9, &
        11, 12, 13, 14, 15, 16, 17, &
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, &
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, &
        55, 56, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83/
        data dictionary%atomic_E /-45.140551665, -210.0485218888889, -321.1987119, -146.63024691666666, -399.0110205833333, -502.070125, -879.0771215, -1091.0652775, &
        -1275.295054, -2131.9724644444445, -2412.581311, -787.3439924999999, -1215.4995769047619, -1705.5754946875, -557.9141695, &
        -1544.3553605, -1105.0024515, -1420.574128, -1970.9374273333333, -2274.598644, -2331.976294, -2762.3960913793107, -3298.6401545, -3637.624857, -4140.3502, -5133.970898611111, -5498.13054, -2073.70436625, -2013.83114375, -463.783827, -658.83885375, -495.05260075, &
        -782.22601375, -1136.1897344444444, -1567.6510633333335, -2136.8407, -2568.946113, -2845.9228975, -3149.6645705, -3640.458547, -4080.81555, -4952.347355, -5073.703895555555, -4879.3604305, -2082.8865266666667, -2051.94076125, -2380.010715, -2983.2449, -3478.003375, &
        -1096.984396724138, -969.538106, -2433.925215, -2419.015324, -2872.458516, -4684.01374, -5170.37679, -4678.720765, -5133.04942, -5055.7201, -5791.21431, -1412.194369, -2018.85905225, -2440.8732966666666/

        !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        CALL scan_title (move_file, "ATOMIC-ENERGY",if_find=nextline)
        if(.not.nextline) then
            write(6,*) "Atomic-energy not found, stop",num_step
            stop
        endif
            
        backspace(move_file)
        read(move_file,*) char_tmp(1:4),Etotp

        open(1316, file='./PWdata/Ei.dat', access='append')
        open(1320, file='./PWdata/AtomType.dat', access='append')

        DO j = 1, natom
            READ(move_file, *) iatom(j),Eatom(j)
            do o = 1, n
                if (iatom(j) == dictionary(o)%order) then
                    if (Eatom(j) > 0) then
                        Eatom(j) = dictionary(o)%atomic_E + Eatom(j)
                    else
                        exit
                    end if
                end if
            end do
            write(1316, "(E20.10)") Eatom(j)
            write(1320, "(I6)") iatom(j)
        ENDDO
        
        close(1316)
        close(1320)

        write(6,"('num_step',2(i4,1x),2(E15.7,1x),i5)") num_step,natom,Etotp,Etotp-Etotp_ave,max_neigh

        if(abs(Etotp-Etotp_ave).gt.E_tolerance) then
        write(6,*) "escape this step, dE too large"
        write(333,*) num_step
        deallocate(iatom,xatom,fatom,Eatom)
        goto 1000
        endif

        num_step1=num_step1+1
        
        allocate(list_neigh(m_neigh,ntype,natom))
        allocate(map2neigh_M(m_neigh,ntype,natom)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
        allocate(list_neigh_M(m_neigh,ntype,natom)) ! the neigh list of Rc_M
        allocate(num_neigh_M(ntype,natom))
        allocate(iat_neigh(m_neigh,ntype,natom))
        allocate(dR_neigh(3,m_neigh,ntype,natom))   ! d(neighbore)-d(center) in xyz
        allocate(num_neigh(ntype,natom))

        allocate(iat_neigh_M(m_neigh,ntype,natom))
        allocate(itype_atom(natom))

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

        call find_neighbore(iatom,natom,xatom,AL,Rc_type,num_neigh,list_neigh, &
        dR_neigh,iat_neigh,ntype,iat_type,m_neigh,Rc_M,map2neigh_M,list_neigh_M, &
        num_neigh_M,iat_neigh_M)    
        
        !write(*,*) "dR_neigh in forward" 
        
        !do k=1,m_neigh
        !    write(*,*) dR_neigh(:,k,1,1)
        !enddo 


        open(1314, file='./PWdata/dRneigh.dat', access='append')
        ! m_neigh,ntype,natom
        do k=1, natom
            do j=1, ntype
                do i=1, m_neigh


                    if ((abs(dR_neigh(1, i, j, k)) + abs(dR_neigh(2, i, j, k)) + abs(dR_neigh(3, i, j, k))) > 1.D-8) then
                    !if (abs(dR_neigh(1, i, j, k))>1.D-8) then
                        write(1314, "(3(E17.10, 1x), i4)") dR_neigh(1, i, j, k), dR_neigh(2, i, j, k), dR_neigh(3, i, j, k), list_neigh(i,j,k)
                    else
                        write(1314, "(3(E17.10, 1x), i4)") 0,0,0,0
                    end if
                end do
            end do
        end do
        close(1314)
        
        deallocate(list_neigh)
        deallocate(iat_neigh)
        deallocate(dR_neigh)
        deallocate(num_neigh)
        deallocate(list_neigh_M)
        deallocate(num_neigh_M)
        deallocate(map2neigh_M)
        deallocate(iat_neigh_M)
        deallocate(itype_atom)
        deallocate (iatom,xatom,fatom,Eatom)
        !--------------------------------------------------------
        goto 1000     

2000   continue   

    close(move_file)

2333   continue

    stop
    end
