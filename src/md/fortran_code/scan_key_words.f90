    subroutine scan_key_words (fileio, keywords, length, scanit)
    !
    implicit none
    !
    integer :: fileio, length, ierror, sharp_index, keywords_index
    !
    logical :: scanit
    !
    character(len=*) :: keywords
    character(len=length) :: upper_keywords
    character(len=200) :: line, upper_line
    !
    call transform_to_upper (keywords, upper_keywords)
    rewind (fileio)
    do
        read (fileio, "(A200)", iostat = ierror) line
        if (ierror < 0) then
	    scanit = .false.
	    exit
	endif    
        call normal_char (line)
	!
        sharp_index = INDEX(line, "#")
        if(sharp_index .gt. 0) line(sharp_index:) = '' 
        sharp_index = INDEX(line, "$")
        if(sharp_index .gt. 0) line(sharp_index:) = '' 
        sharp_index = INDEX(line, "!")
        if(sharp_index .gt. 0) line(sharp_index:) = '' 
        sharp_index = INDEX(line, "@")
        if(sharp_index .gt. 0) line(sharp_index:) = '' 
        !
        CALL transform_to_upper ( line, upper_line )
        keywords_index = INDEX (upper_line, upper_keywords )
        !
        IF ( keywords_index <= 0 .AND. ierror == 0 ) CYCLE
        !
        IF ( keywords_index >= 1 .AND. ierror == 0 ) scanit = .true.
	exit
	!
    enddo
    end subroutine scan_key_words
