MODULE CountsATOMS;
   IMPLICIT NONE

CONTAINS

   SUBROUTINE num_elem_type(natom,ntypes,iatom,num_step,iatom_type,iatom_type_num)
      !!> Counts the type and number of elements
      !!>
      !!> @param natom (input) number of atoms
      !!> @param ntypes (input) number of atom types
      !!> @param iatom (input) atom index
      !!> @param num_step (input) number of steps
      !!> @param iatom_type (input) atom type
      !!> @param iatom_type_num (input) atom numbers of atom types

      INTEGER, INTENT(IN) :: natom
      INTEGER, INTENT(IN) :: ntypes
      INTEGER, INTENT(IN) :: iatom(natom)
      INTEGER, INTENT(IN) :: num_step
      INTEGER, INTENT(IN) :: iatom_type(ntypes)
      INTEGER, INTENT(INOUT) :: iatom_type_num(ntypes)

      INTEGER :: i,j

      ! Initialize the counters
      iatom_type_num = 0

      ! Loop over all atoms and count the number of each atom type
      DO i = 1, natom
         DO j =1, ntypes
            IF (iatom(i) == iatom_type(j)) THEN
               iatom_type_num(j) = iatom_type_num(j) + 1
               EXIT
            END IF
         END DO
      END DO

   END SUBROUTINE num_elem_type

end module CountsATOMS;



