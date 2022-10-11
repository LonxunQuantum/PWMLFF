subroutine get_zatom(natom)
    use data_ewald
    implicit none
    integer ierr, tmp_atom
    integer ia, ib, ic, natom
    
    open(1314, file='input/ewald.input', action='read', iostat=ierr)
    if (ierr .ne. 0) then
        iflag_born_charge_ewald = 0
    else 
        read(1314, *) iflag_born_charge_ewald, ewald_epsilon
        read(1314, *) n_born_type
        close(1314)
        open(1315, file='input/zatom.input', action='read', iostat=ierr)
        do ia = 1, natom
            read(1315, *) tmp_zatom
            zatom_ewald(ia) = tmp_zatom
        end do
        close(1315)
    end if
end subroutine
