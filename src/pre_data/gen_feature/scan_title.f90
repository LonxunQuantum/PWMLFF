    SUBROUTINE scan_title (io_file, title, title_line, if_find)
    !
    IMPLICIT NONE
    !
    INTEGER :: io_file
    CHARACTER(LEN=200) :: oneline, tmp_char
    CHARACTER(LEN=200), OPTIONAL :: title_line
    CHARACTER(LEN=*) :: title
    INTEGER :: stat
    LOGICAL, OPTIONAL :: if_find 
    !
    IF (PRESENT(if_find)) if_find = .FALSE.
    !REWIND (io_file)
    DO
        READ (io_file, "(A200)", IOSTAT = stat) tmp_char
        CALL transform_to_upper (tmp_char, oneline)
        IF (PRESENT(title_line)) title_line = oneline
        IF (INDEX(oneline,title) > 0) THEN
            IF (PRESENT(if_find)) if_find = .TRUE.
            EXIT
        ENDIF
        IF (stat /= 0) THEN
            IF (PRESENT(if_find)) if_find = .FALSE.
            EXIT
        ENDIF
    ENDDO
    !
    END SUBROUTINE scan_title

