**==ranf.spg  processed by SPAG 4.52O  at 18:54 on 27 Mar 1996
      FUNCTION RANFX(Idum)
c
c     random number generator
c
c     Idum (input): can be used as seed (not used in present
c                   random number generator.
 
      IMPLICIT NONE
      INTEGER Idum
      DOUBLE PRECISION RANFX, RCARRY
      RANFX = RCARRY()
      RETURN
C ----------------------------------------------------C
      END
**==randx.spg  processed by SPAG 4.52O  at 18:54 on 27 Mar 1996
 
      FUNCTION RANDX(Iseed)
C----------------------------------------------------------------------C
C  Random number generator, fast and rough, machine independent.
C  Returns an uniformly distributed deviate in the 0 to 1 interval.
C  This random number generator is portable, machine-independent and
C  reproducible, for any machine with at least 32 bits / real number.
C  REF: Press, Flannery, Teukolsky, Vetterling, Numerical Recipes (1986)
C----------------------------------------------------------------------C
      IMPLICIT NONE
      INTEGER IA, IC, Iseed, M1
      DOUBLE PRECISION RANDX, RM
      PARAMETER (M1=714025, IA=1366, IC=150889, RM=1.D+0/M1)
c
      Iseed = MOD(IA*Iseed+IC, M1)
      RANDX = Iseed*RM
      IF (RANDX.LT.1.D-10) THEN
         Iseed = 12345
         Iseed = MOD(IA*Iseed+IC, M1)
         RANDX = Iseed*RM
         write(*,*) "random number is negative"
         !STOP '*** Random number is negative ***'
      END IF
c
      RETURN
      END
**==ranset.spg  processed by SPAG 4.52O  at 18:54 on 27 Mar 1996
 
      SUBROUTINE RANSET(Iseed)
c     --- initializes random number generator
      IMPLICIT NONE
      INTEGER Iseed
 
      CALL RSTART(Iseed)
      RETURN
      END
**==rstart.spg  processed by SPAG 4.52O  at 18:54 on 27 Mar 1996
 
      SUBROUTINE RSTART(Iseeda)
C----------------------------------------------------------------------C
C       Initialize Marsaglia list of 24 random numbers.
C----------------------------------------------------------------------C
      use common_module_99
      IMPLICIT NONE

      DOUBLE PRECISION ran, RANDX
      INTEGER i, Iseeda
 
      I24_99 = 24
      J24_99 = 10
      CARRY_99 = 0.D+0
      ISEED_99 = Iseeda
c
c       get rid of initial correlations in rand by throwing
c       away the first 100 random numbers generated.
c
      DO i = 1, 100
         ran = RANDX(ISEED_99)
      END DO
c
c       initialize the 24 elements of seed
c
 
      DO i = 1, 24
         SEED_99(i) = RANDX(ISEED_99)
      END DO
 
      RETURN
      END
**==rcarry.spg  processed by SPAG 4.52O  at 18:54 on 27 Mar 1996
 
 
      FUNCTION RCARRY()
C----------------------------------------------------------------------C
C       Random number generator from Marsaglia.
C----------------------------------------------------------------------C
      use common_module_99

      IMPLICIT NONE
      DOUBLE PRECISION RCARRY, TWOm24, TWOp24, uni
      PARAMETER (TWOp24=16777216.D+0, TWOm24=1.D+0/TWOp24)
c
c       F. James Comp. Phys. Comm. 60, 329  (1990)
c       algorithm by G. Marsaglia and A. Zaman
c       base b = 2**24  lags r=24 and s=10
c
      uni = SEED_99(I24_99) - SEED_99(J24_99) - CARRY_99
      IF (uni.LT.0.D+0) THEN
         uni = uni + 1.D+0
         CARRY_99 = TWOm24
      ELSE
         CARRY_99 = 0.D+0
      END IF
      SEED_99(I24_99) = uni
      I24_99 = I24_99 - 1
      IF (I24_99.EQ.0) I24_99 = 24
      J24_99 = J24_99 - 1
      IF (J24_99.EQ.0) J24_99 = 24
      RCARRY = uni
 
      RETURN
      END
