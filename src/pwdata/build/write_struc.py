import numpy as np
import os
from collections import Counter
from build.cell import cell_to_cellpar

def write_config(filepath,
                 filename,
                 atoms,
                 direct=True,
                 sort=None,
                 symbol_count=None,
                 long_format=True,
                 ignore_constraints=False,
                 wrap=False):
    """Method to write PWmat position (atom.config) files.

    Writes label, scalefactor, unitcell, # of various kinds of atoms,
    positions in cartesian or scaled coordinates (Direct), and constraints
    to file. fractional coordinates is default.
    """

    if isinstance(atoms, (list, tuple)):
        if len(atoms) > 1:
            raise RuntimeError('Don\'t know how to save more than ' +
                               'one image to PWmat input')
        else:
            atoms = atoms[0]

    # Check lattice vectors are finite
    """Get unit cell parameters. Sequence of 6 numbers.

    First three are unit cell vector lengths and second three
    are angles between them::

        [len(a), len(b), len(c), angle(b,c), angle(a,c), angle(a,b)]

    in degrees.

    radians: bool, Convert angles to radians.

    See also :func:`cell.cell_to_cellpar`."""
    cellpar = cell_to_cellpar(atoms.lattice, radians = False)
    if np.any(cellpar == 0.):
        raise RuntimeError(
            'Lattice vectors must be finite and not coincident. '
            'At least one lattice length or angle is zero.')

    # Write atom positions in scaled or cartesian coordinates
    if direct:
        coord = atoms.get_scaled_positions(wrap=wrap)
    else:
        coord = atoms.get_positions(wrap=wrap)

    '''constraints = atoms.constraints and not ignore_constraints

    if constraints:
        sflags = np.zeros((len(atoms), 3), dtype=bool)
        for constr in atoms.constraints:
            if isinstance(constr, FixScaled):
                sflags[constr.a] = constr.mask
            elif isinstance(constr, FixAtoms):
                sflags[constr.index] = [True, True, True]
            elif isinstance(constr, FixedPlane):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5,
                              axis=1)
                if sum(mask) != 1:
                    raise RuntimeError(
                        'VASP requires that the direction of FixedPlane '
                        'constraints is parallel with one of the cell axis')
                sflags[constr.a] = mask
            elif isinstance(constr, FixedLine):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5,
                              axis=1)
                if sum(mask) != 1:
                    raise RuntimeError(
                        'VASP requires that the direction of FixedLine '
                        'constraints is parallel with one of the cell axis')
                sflags[constr.a] = ~mask'''

    if sort:
        ind = np.argsort(atoms.get_atomic_numbers())
        symbols = np.array(atoms.get_atomic_numbers())[ind]
        coord = coord[ind]
        # if constraints:
        #     sflags = sflags[ind]
    else:
        symbols = atoms.atom_types_image

    # Create a list sc of (symbol, count) pairs
    if symbol_count:
        # sc = symbol_count
        pass
    else:
        sc = Counter(symbols)
        atom_type = list(sc.keys())
        atom_type_num = list(sc.values())

    # Write to file
    output_file = open(os.path.join(filepath, filename), 'w')
    output_file.write('%d\n' % len(atoms))
    output_file.write('Lattice vector\n')
    for i in range(3):
        output_file.write('%19.16f %19.16f %19.16f\n' % tuple(atoms.lattice[i]))
    output_file.write('Position, move_x, mov_y, move_z\n')
    for i in range(len(atoms)):
        output_file.write('%4d %19.16f %19.16f %19.16f %2d %2d %2d\n' %
                          (symbols[i], coord[i][0], coord[i][1], coord[i][2],
                           1, 1, 1))
    output_file.close()
    
def write_vasp(filepath,
               filename,
               atoms,
               direct=False,
               sort=None,
               symbol_count=None,
               long_format=True,
               vasp5=True,
               ignore_constraints=False,
               wrap=False):
    """Method to write VASP position (POSCAR/CONTCAR) files.

    Writes label, scalefactor, unitcell, # of various kinds of atoms,
    positions in cartesian or scaled coordinates (Direct), and constraints
    to file. Cartesian coordinates is default and default label is the
    atomic species, e.g. 'C N H Cu'.
    """

    if isinstance(atoms, (list, tuple)):
        if len(atoms) > 1:
            raise RuntimeError('Don\'t know how to save more than ' +
                               'one image to VASP input')
        else:
            atoms = atoms[0]

    # Check lattice vectors are finite
    """Get unit cell parameters. Sequence of 6 numbers.

    First three are unit cell vector lengths and second three
    are angles between them::

        [len(a), len(b), len(c), angle(b,c), angle(a,c), angle(a,b)]

    in degrees.

    radians: bool, Convert angles to radians.

    See also :func:`cell.cell_to_cellpar`."""
    cellpar = cell_to_cellpar(atoms.lattice, radians = False)
    if np.any(cellpar == 0.):
        raise RuntimeError(
            'Lattice vectors must be finite and not coincident. '
            'At least one lattice length or angle is zero.')

    # Write atom positions in scaled or cartesian coordinates
    if direct:
        coord = atoms.get_scaled_positions(wrap=wrap)
    else:
        coord = atoms.get_positions(wrap=wrap)

    '''constraints = atoms.constraints and not ignore_constraints

    if constraints:
        sflags = np.zeros((len(atoms), 3), dtype=bool)
        for constr in atoms.constraints:
            if isinstance(constr, FixScaled):
                sflags[constr.a] = constr.mask
            elif isinstance(constr, FixAtoms):
                sflags[constr.index] = [True, True, True]
            elif isinstance(constr, FixedPlane):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5,
                              axis=1)
                if sum(mask) != 1:
                    raise RuntimeError(
                        'VASP requires that the direction of FixedPlane '
                        'constraints is parallel with one of the cell axis')
                sflags[constr.a] = mask
            elif isinstance(constr, FixedLine):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5,
                              axis=1)
                if sum(mask) != 1:
                    raise RuntimeError(
                        'VASP requires that the direction of FixedLine '
                        'constraints is parallel with one of the cell axis')
                sflags[constr.a] = ~mask'''
    
    if sort:
        ind = np.argsort(atoms.get_atomic_numbers())
        symbols = np.array(atoms.get_atomic_numbers())[ind]
        coord = coord[ind]
        # if constraints:
        #     sflags = sflags[ind]
    else:
        symbols = atoms.atom_types_image

    # Create a list sc of (symbol, count) pairs
    if symbol_count:
        # sc = symbol_count
        pass
    else:
        sc = Counter(symbols)
        atom_type = list(sc.keys())
        atom_type_num = list(sc.values())
    
    from const import elements
    atom_type = [elements[_] for _ in atom_type]
    # Write to file
    output_file = open(os.path.join(filepath, filename), 'w')
    output_file.write('Created from %s\n' % filename)
    output_file.write('1.0\n')
    for i in range(3):
        output_file.write('%19.16f %19.16f %19.16f\n' % tuple(atoms.lattice[i]))
    output_file.write(' '.join(atom_type) + '\n')
    output_file.write(' '.join([str(_) for _ in atom_type_num]) + '\n')

    # if constraints:
    #     output_file.write('Selective dynamics\n')

    if direct:
        output_file.write('Direct\n')
    else:
        output_file.write('Cartesian\n')

    for i in range(len(atoms)):
        output_file.write('%19.16f %19.16f %19.16f' % tuple(coord[i]))
        # if constraints:
        #     output_file.write(' %s %s %s' % tuple(sflags[i]))
        output_file.write('\n')
    output_file.close()