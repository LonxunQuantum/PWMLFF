!
!   This subroutine will read the opened file(fileio), and Scan the file
! about keywords, which is no case sensitive. The expression is like this:
! keywords = right. When right is nothing, but only keywords left, it equals
! not setting keywords.
!
SUBROUTINE read_key_words ( fileio, keywords, length, right, readit)
    !
    IMPLICIT NONE
    !
    INTEGER, INTENT(IN) :: fileio
    INTEGER, INTENT(IN) :: length ! the length of the keywords
    CHARACTER(LEN=*) :: keywords  
    CHARACTER(LEN=length) :: keywords_upper ! transform the keywords to the UPPER
    CHARACTER(LEN=200), INTENT(OUT) :: right
    INTEGER :: ierror
    logical, intent(out) :: readit
    !
    INTEGER :: i, equal_index, keywords_index, sharp_index
    CHARACTER(LEN=200) :: line, left
    CHARACTER(LEN=200) :: upperline
    !
    readit = .true.
    CALL transform_to_upper (keywords, keywords_upper)
    REWIND (fileio)
    DO
        !
        READ ( fileio, "(A200)", IOSTAT = ierror ) line
        if (ierror < 0) then
            readit = .false.
            exit
        endif    
        CALL normal_char(line)
        ! the following are the commenting symbols.
        !sharp_index = INDEX(line, "#")
        !if(sharp_index .gt. 0) line(sharp_index:) = '' 
        !sharp_index = INDEX(line, "$")
        !if(sharp_index .gt. 0) line(sharp_index:) = '' 
        !sharp_index = INDEX(line, "!")
        !if(sharp_index .gt. 0) line(sharp_index:) = '' 
        !sharp_index = INDEX(line, "@")
        !if(sharp_index .gt. 0) line(sharp_index:) = '' 
        !
        CALL transform_to_upper ( line, upperline )
        keywords_index = INDEX (upperline, keywords_upper )
        !
        IF ( keywords_index <= 0 .AND. ierror == 0 ) CYCLE
        !
        IF ( keywords_index >= 1 .AND. ierror == 0 ) THEN
            !upperline = upperline(keywords_index:)
            equal_index = INDEX ( upperline, "=" )
            left = trim(upperline(:equal_index-1))
            left = ADJUSTL( left )
            if ( TRIM(left) /= TRIM(keywords_upper)) CYCLE
            right = line(equal_index+1:)
            right = ADJUSTL( right )
            !
            IF (LEN_TRIM(right) == 0) THEN
                ierror = 1
                readit = .true.
                EXIT
            ENDIF
            !
            EXIT
        ENDIF
        IF ( ierror /= 0 ) then
            readit = .false.
            EXIT
        endif    
    ENDDO
    !
    RETURN
    !
END SUBROUTINE read_key_words
!
!
!
!
!
!
SUBROUTINE normal_char(in_char)
    IMPLICIT NONE
    CHARACTER(LEN=*) in_char
    INTEGER :: i
    DO i = 1, LEN_TRIM(in_char)
        IF (in_char(i:i) == ACHAR(9)) in_char(i:i) = ACHAR(32)
    ENDDO
    RETURN
END SUBROUTINE normal_char
