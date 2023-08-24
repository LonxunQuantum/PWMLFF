MODULE LeastSquaresSolver
   USE, INTRINSIC :: iso_c_binding
   IMPLICIT NONE
   ! LAPACK 的接口
   ! INTERFACE
   !    SUBROUTINE dgels(TRANS, M, N, NRHS, A, LDA, B, LDB, WORK, LWORK, INFO) &
   !       BIND(C, NAME="dgels_")
   !       CHARACTER, INTENT(IN) :: TRANS
   !       INTEGER, INTENT(IN) :: M, N, NRHS, LDA, LDB, LWORK
   !       INTEGER, INTENT(OUT) :: INFO
   !       DOUBLE PRECISION, INTENT(INOUT) :: A(LDA, *)
   !       DOUBLE PRECISION, INTENT(INOUT) :: B(LDB, *)
   !       DOUBLE PRECISION, INTENT(INOUT) :: WORK(LWORK)
   !    END SUBROUTINE dgels
   ! END INTERFACE

CONTAINS

   SUBROUTINE SolveLeastSquares(A, B, X, M, N, NRHS, INFO)
      INTEGER, INTENT(IN) :: M, N, NRHS
      DOUBLE PRECISION, INTENT(IN) :: A(M, N)
      DOUBLE PRECISION, INTENT(IN) :: B(MAX(M, N))
      !   DOUBLE PRECISION, INTENT(INOUT) :: B(M)
      DOUBLE PRECISION, INTENT(OUT) :: X(N)
      INTEGER, INTENT(OUT) :: INFO
      DOUBLE PRECISION :: WORK(9)   ! 工作数组，大小是 3 * n，n 是 A 的列数
      INTEGER :: LWORK

      LWORK = MAX(1, MIN(M, N) + MAX(M, N, NRHS))   ! 工作数组的大小
      !> Solves the linear least squares problem using the QR or LQ factorization of A.
      !!>
      !!> @param[in] 'N' specifies that the matrix A is not transposed.
      !!> @param[in] M is the number of rows of the matrix A.
      !!> @param[in] N is the number of columns of the matrix A.
      !!> @param[in] NRHS is the number of columns of the matrix B.
      !!> @param[in] A is the matrix of size (M,N) containing the coefficients of the linear system.
      !!> @param[in] LDA is the leading dimension of A.
      !!> @param[in] B is the matrix of size (MAX(M,N),NRHS) containing the right-hand side of the linear system.
      !!> @param[in] LDB is the leading dimension of B.
      !!> @param[in] WORK is a workspace array of size LWORK.
      !!> @param[in] LWORK is the length of the array WORK.
      !!> @param[out] INFO is an integer output variable. If INFO = 0, the execution is successful.
      !!>
      !!> @note This subroutine uses LAPACK routine dgels.
      !!> @note The matrix A is overwritten by its QR or LQ factorization.
      !!> @note The solution X is returned in B.
      !!> @note If M >= N, the QR factorization is used. If M < N, the LQ factorization is used.
      !!> @note If INFO > 0, the i-th argument had an illegal value.
      !!> @note If INFO = -i, the i-th diagonal element of the triangular factor of A is zero, so that A does not have full rank.
      !!> @note If INFO = i, the i-th row of A and/or the i-th element of b is invalid.
      !!>
      !   CALL dgels('N', M, N, NRHS, A, M, B, N, WORK, LWORK, INFO)
      CALL dgels('N', M, N, NRHS, A, M, B, MAX(M,N), WORK, LWORK, INFO)
      X(1:N) = B(1:N)
   END SUBROUTINE SolveLeastSquares

END MODULE LeastSquaresSolver


