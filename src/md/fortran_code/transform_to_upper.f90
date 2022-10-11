    SUBROUTINE transform_to_upper ( in_words, upper_words )
    !
    IMPLICIT NONE
    !
    INTEGER :: i
    CHARACTER(LEN=*), INTENT(IN) :: in_words
    CHARACTER(LEN=*), INTENT(OUT) :: upper_words
    !
    upper_words = in_words
    DO i = 1, LEN_TRIM ( upper_words )   !  transform "in_words" to CAPITAL "IN_WORDS"
        IF( 97 <= IACHAR ( upper_words(i:i) ) .AND.  &
            IACHAR ( upper_words(i:i) ) <= 122 ) THEN
            upper_words(i:i) = ACHAR( IACHAR ( upper_words(i:i) ) - 32 )
        ENDIF
    ENDDO
    !
    RETURN
    !
    END SUBROUTINE transform_to_upper

