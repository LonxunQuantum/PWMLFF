!!!!!!!!!!!!!!!!!!!!!!!! THIS IS NUMERIC_CONTAINER !!!!!!!!!!!!!!!!!!!!!
!
! Function uni - generates a random number
!
! Subroutine box_mueller - generates gaussian random numbers of unit
!                          variance (with zero mean and standard
!                          variation of 1)
!
! Subroutine gauss_old - constructs velocity arrays with a gaussian
!                        distribution of unit variance (zero mean) by
!                        an approximation of the Central Limit Theorem
!
! Subroutine gauss - constructs velocity arrays with a gaussian
!                    distribution of unit variance (zero mean) using
!                    the box-mueller method
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Function uni()

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !
        ! dl_poly_4 random number generator based on the universal random number
        ! generator of marsaglia, zaman and tsang
        ! (stats and prob. lett. 8 (1990) 35-39.)
        !
        ! This random number generator originally appeared in "Toward a
        ! Universal Random Number Generator" by George Marsaglia, Arif Zaman and
        ! W.W. Tsang in Florida State University Report: FSU-SCRI-87-50 (1987).
        ! It was later modified by F. James and published in "A Review of
        ! Pseudo-random Number Generators".
        ! THIS IS THE BEST KNOWN RANDOM NUMBER GENERATOR AVAILABLE.
        ! It passes ALL of the tests for random number generators and has a
        ! period of 2^144, is completely portable (gives bit identical results
        ! on all machines with at least 24-bit mantissas in the floating point
        ! representation).
        ! The algorithm is a combination of a Fibonacci sequence (with lags of
        ! 97 and 33, and operation "subtraction plus one, modulo one") and an
        ! "arithmetic sequence" (using subtraction).
        ! Use IJ = 1802 & KL = 9373 (idnode=0) to test the random number
        ! generator. The subroutine RANMAR should be used to generate 20000
        ! random numbers.  Then display the next six random numbers generated
        ! multiplied by 4096*4096.  If the random number generator is working
        ! properly, the random numbers should be:
        !         6533892.0  14220222.0  7275067.0
        !         6172232.0  8354498.0   10633180.0
        !
        ! copyright - daresbury laboratory
        ! author    - w.smith july 1992
        ! amended   - i.t.todorov april 2008
        !
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        use mod_mpi,only : inode
        use mod_control
        Implicit None
        include 'mpif.h'
        integer :: ierr

        Logical,           Save :: newjob = .true.
        Integer,           Save :: ir,jr,idnode
        Integer                 :: i,ii,ij,j,jj,k,kl,l,m
        Real*8, Save :: c,cd,cm,u(1:97)
        Real*8       :: s,t,uni

        ! Random seeding

        Integer, Save :: seed(1:2) = 0
        Logical, Save :: lseed     = .false.
        integer clock

        idnode=1

        ! initialise parameters u,c,cd,cm
        if(newjob) then
            lseed=.true.
            if(MCTRL_SEED_MD.lt.0.d0) then
                if(inode.eq. 1) then
                    call system_clock(count=clock)
                    seed(1)=mod(clock,100000)
                    seed(2)=mod(clock,100000)+37
                endif
            else
                seed(1)=MCTRL_SEED_MD
                seed(2)=MCTRL_SEED_MD+37
            endif
            call mpi_bcast(seed,2,MPI_INT,0,MPI_COMM_WORLD,ierr)
        endif
        If (newjob .or. lseed) Then
            newjob = .false.

            ! If no seeding is specified then default to DL_POLY scheme

            If (lseed) Then

                lseed=.false.

                ! First random number seed must be between 0 and 31328
                ! Second seed must have a value between 0 and 30081

                ij=Mod(Abs(seed(1)+idnode),31328)
                i = Mod(ij/177,177) + 2;
                j = Mod(ij,177)     + 2;

                kl=Mod(Abs(seed(2)+idnode),30081)
                k = Mod(kl/169,178) + 1
                l = Mod(kl,169)

            Else

                ! initial values of i,j,k must be in range 1 to 178 (not all 1)
                ! initial value of l must be in range 0 to 168

                i = Mod(idnode,166) + 12
                j = Mod(idnode,144) + 34
                k = Mod(idnode,122) + 56
                l = Mod(idnode,90)  + 78

            End If

            ir = 97
            jr = 33

            Do ii=1,97

                s = 0.0d0
                t = 0.5d0

                Do jj=1,24

                    m = Mod(Mod(i*j,179)*k,179)
                    i = j
                    j = k
                    k = m
                    l = Mod(53*l+1,169)
                    If (Mod(l*m,64) >= 32) s = s+t
                    t = 0.5d0*t

                End Do

                u(ii)=s

            End Do

            c  =   362436.0d0/16777216.0d0
            cd =  7654321.0d0/16777216.0d0
            cm = 16777213.0d0/16777216.0d0

        End If

        ! calculate random number

        uni=u(ir)-u(jr)
        If (uni < 0.0d0) uni = uni + 1.0d0

        u(ir)=uni

        ir=ir-1
        If (ir == 0) ir = 97

        jr=jr-1
        If (jr == 0) jr = 97

        c = c-cd
        If (c < 0.0d0) c = c+cm

        uni = uni-c
        If (uni < 0.0d0) uni = uni + 1.0d0

End Function uni

Subroutine box_mueller(gauss1,gauss2)

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !
        ! dl_poly_4 routine using the box-mueller method for generating
        ! gaussian random numbers of unit variance (with zero mean and standard
        ! variation of 1).  Otherwise, an approximation of the Central Limit
        ! Theorem must be used: G = (1/A)*[Sum_i=1,N(Ri) - AN/2]*(12/N)^(1/2),
        ! where A is the number of outcomes from the random throw Ri and N is
        ! the number of tries.
        !
        ! dependent on uni
        !
        ! copyright - daresbury laboratory
        ! author    - w.smith may 2008
        !
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        Implicit None

        Real*8, Intent(   Out ) :: gauss1,gauss2

        Logical,save           :: newjob = .true.
        Real*8 :: uni,ran0,ran1,ran2

        ! make sure uni is initialised

        If (newjob) Then
            newjob = .false.
            ran0=uni()
        End If

        ran0=1.0d0

        ! generate uniform random numbers on [-1, 1)

        Do While (ran0 >= 1.0d0 .or. ran0 <=1.d-10)
            ran1=2.0d0*uni()-1.0d0
            ran2=2.0d0*uni()-1.0d0
            ran0=ran1**2+ran2**2
        End Do

        ! calculate gaussian random numbers

        ran0=Sqrt(-2.0d0*Log(ran0)/ran0)
        gauss1=ran0*ran1
        gauss2=ran0*ran2

End Subroutine box_mueller

Subroutine gauss_old(natms,vxx,vyy,vzz)

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !
        ! dl_poly_4 routine for constructing velocity arrays with a gaussian
        ! distribution of unit variance (zero mean), based on the method
        ! described by Allen and Tildesley in "Computer Simulation of Liquids",
        ! Clarendon Press 1987, P347.  It is based on an approximation of the
        ! Central Limit Theorem : G = (1/A)*[Sum_i=1,N(Ri) - AN/2]*(12/N)^(1/2),
        ! where A is the number of outcomes from the random throw Ri and N is
        ! the number of tries.
        !
        ! copyright - daresbury laboratory
        ! author    - w.smith july 1992
        ! amended   - i.t.todorov july 2010
        !
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        Implicit None

        Real*8, Parameter :: a1 = 3.949846138d0
        Real*8, Parameter :: a3 = 0.252408784d0
        Real*8, Parameter :: a5 = 0.076542912d0
        Real*8, Parameter :: a7 = 0.008355968d0
        Real*8, Parameter :: a9 = 0.029899776d0

        Integer,            Intent( In    ) :: natms
        Real*8, Dimension( 1:* ), Intent(   Out ) :: vxx,vyy,vzz

        Integer           :: i,j
        Real*8 :: uni,rrr,rr2

        Do i=1,natms
            rrr=0.0d0
            Do j=1,12
                rrr=rrr+uni()
            End Do
            rrr=(rrr-6.0d0)/4.0d0
            rr2=rrr*rrr
            vxx(i)=rrr*(a1+rr2*(a3+rr2*(a5+rr2*(a7+rr2*a9))))

            rrr=0.0d0
            Do j=1,12
                rrr=rrr+uni()
            End Do
            rrr=(rrr-6.0d0)/4.0d0
            rr2=rrr*rrr
            vyy(i)=rrr*(a1+rr2*(a3+rr2*(a5+rr2*(a7+rr2*a9))))

            rrr=0.0d0
            Do j=1,12
                rrr=rrr+uni()
            End Do
            rrr=(rrr-6.0d0)/4.0d0
            rr2=rrr*rrr
            vzz(i)=rrr*(a1+rr2*(a3+rr2*(a5+rr2*(a7+rr2*a9))))
        End Do

End Subroutine gauss_old

Subroutine gauss(natms,vxx,vyy,vzz)

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !
        ! dl_poly_4 routine for constructing velocity arrays with a gaussian
        ! distribution of unit variance (zero mean), based on the box-mueller
        ! method
        !
        ! copyright - daresbury laboratory
        ! author    - w.smith july 2010
        !
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        Integer,                             Intent( In    ) :: natms
        Real*8, Dimension( 1:* ), Intent(   Out ) :: vxx,vyy,vzz

        Integer           :: i,j
        Real*8 :: gauss1,gauss2

        Do i=1,(natms+1)/2
            j=natms+1-i

            Call box_mueller(gauss1,gauss2)
            vxx(i)=gauss1
            vxx(j)=gauss2

            Call box_mueller(gauss1,gauss2)
            vyy(i)=gauss1
            vyy(j)=gauss2

            Call box_mueller(gauss1,gauss2)
            vzz(i)=gauss1
            vzz(j)=gauss2
        End Do

End Subroutine gauss

