import numpy as np
import copy
import os 
from build.write_struc import write_config, write_vasp, write_lammps
from calculators.const import elements
from build.geometry import wrap_positions
from build.cell import scaled_positions

# 1. initial the image class
class Image(object):
    def __init__(self, 
                 atom_type = None, atom_type_num = None, atom_nums = None, atom_types_image = None, 
                 iteration = None, Etot = None, Ep = None, Ek = None, scf = None, lattice = None, 
                 stress = None, position = None, force = None, atomic_energy = None,
                 content = None, image_nums = None, pbc = None, cartesian = None):
        """
        Represents an image in a AIMD trajectory.

        Args:
            atom_type (str): The type of atom.
            atom_type_num (int): The number of atom types.
            atom_nums (list): The number of atoms.
            atom_types_image (list): The types of atoms in the image.
            iteration (int): The iteration number.
            Etot (float): The total energy.
            Ep (float): The potential energy.
            Ek (float): The kinetic energy.
            scf (float): The index of the self-consistent field.
            lattice (list): The lattice vectors.
            stress (list): The stress tensor.
            position (list): The atomic positions.
            force (list): The atomic forces.
            atomic_energy (list): The atomic energies.
            content (str): The content of the image.
            image_nums (int): The number of images.
            pbc (list): three bool, Periodic boundary conditions flags.  Examples: [True, True, False] or [1, 1, 0]. True (1) means periodic, False (0) means non-periodic. Default: [False, False, False].
        """
        self.atom_nums = atom_nums
        self.iteration = iteration
        self.atom_type = atom_type
        self.atom_type_num = atom_type_num
        self.atom_types_image = atom_types_image if atom_types_image is not None else []
        self.Etot = Etot
        self.Ep = Ep
        self.Ek = Ek
        self.scf = scf
        self.image_nums = image_nums
        self.lattice = lattice if lattice is not None else []
        self.stress = stress if stress is not None else []
        self.position = position if position is not None else []    # this position can be fractional coordinates or cartesian coordinates
        self.force = force if force is not None else []
        self.atomic_energy = atomic_energy if atomic_energy is not None else []
        self.content = content if content is not None else []
        self.cartesian = cartesian if cartesian is not None else False
        self.pbc = pbc if pbc is not None else np.zeros(3, bool)
        self.arrays = self.prim_dict() # here, position will be convert to cartesian coordinates
    
    def copy(self):
        """Return a copy."""
        atoms = self.__class__(lattice=self.lattice, position=self.position, pbc=self.pbc, cartesian=self.cartesian)
        if self.cartesian:
            pass
        else:
            self._set_cartesian()
        atoms.arrays = {}
        # atoms.cartesian = self.cartesian
        prim_dict = self.prim_dict()
        for name, a in prim_dict.items():
            atoms.arrays[name] = a.copy()
        return atoms
    
    def to(self, file_path, file_name, file_format, direct, sort, wrap = False):
        """Write atoms object to a new file."""
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if file_format.lower() == 'config' or file_format.lower() == 'pwmat':
            write_config(file_path, file_name, self, sort=sort, wrap=wrap)
        elif file_format.lower() == 'poscar' or file_format.lower() == 'vasp':
            write_vasp(file_path, file_name, self, direct=direct, sort=sort, wrap=wrap)
        elif file_format.lower() == "lammps":
            write_lammps(file_path, file_name, self, sort=sort, wrap=wrap)
        else:
            raise RuntimeError('Unknown file format')
    
    def prim_dict(self):
        """Return a dictionary of the primitive image data."""
        return {'atom_types_image': np.array(self.atom_types_image, dtype=np.int64), 'position': np.array(self.position).reshape(-1, 3)}
    
    def extend(self, other):
        """Extend atoms object by appending atoms from *other*."""
        n1 = len(self)
        n2 = len(other)

        for name, a1 in self.arrays.items():
            a = np.zeros((n1 + n2,) + a1.shape[1:], a1.dtype)
            a[:n1] = a1
            if name == 'masses':
                pass
            else:
                a2 = other.arrays.get(name)
            if a2 is not None:
                a[n1:] = a2
            self.arrays[name] = a

        for name, a2 in other.arrays.items():
            if name in self.arrays:
                continue
            a = np.empty((n1 + n2,) + a2.shape[1:], a2.dtype)
            a[n1:] = a2
            if name == 'masses':
                pass
            else:
                a[:n1] = 0

            self.set_array(name, a)

    def wrap(self, **wrap_kw):
        """Wrap positions to unit cell.

        Parameters:

        wrap_kw: (keyword=value) pairs
            optional keywords `pbc`, `center`, `pretty_translation`, `eps`,
            see :func:`ase.geometry.wrap_positions`
        """

        if 'pbc' not in wrap_kw:
            wrap_kw['pbc'] = self.pbc

        self.position[:] = self.get_positions(wrap=True, **wrap_kw)

    def get_positions(self, wrap=False, **wrap_kw):
        """Get array of positions.

        Parameters:

        wrap: bool
            wrap atoms back to the cell before returning positions
        wrap_kw: (keyword=value) pairs
            optional keywords `pbc`, `center`, `pretty_translation`, `eps`,
            see :func:`ase.geometry.wrap_positions`
        """
        if wrap:
            if 'pbc' not in wrap_kw:
                wrap_kw['pbc'] = self.pbc
            position = self._get_positions()
            return wrap_positions(position, self.lattice, **wrap_kw)
        else:
            return self.arrays['position'].copy()
        
    def get_scaled_positions(self, wrap=True):
        """Get positions relative to unit cell.

        If wrap is True, atoms outside the unit cell will be wrapped into
        the cell in those directions with periodic boundary conditions
        so that the scaled coordinates are between zero and one.

        If any cell vectors are zero, the corresponding coordinates
        are evaluated as if the cell were completed using
        ``cell.complete_cell()``.  This means coordinates will be Cartesian
        as long as the non-zero cell vectors span a Cartesian axis or
        plane."""

        fractional = scaled_positions(self.lattice, self.position)

        if wrap:
            for i, periodic in enumerate(self.pbc):
                if periodic:
                    # Yes, we need to do it twice.
                    # See the scaled_positions.py test.
                    fractional[:, i] %= 1.0
                    fractional[:, i] %= 1.0

        return fractional

    def get_atomic_numbers(self):
        """Get integer array of atomic numbers."""
        return self.arrays['atom_types_image'].copy()
    
    def _get_positions(self):
        """Return reference to positions-array for in-place manipulations."""
        return self.arrays['position']
    
    def _set_cartesian(self):
        """Set positions in Cartesian coordinates."""
        self.position = frac2cart(self.position, self.lattice)
        self.cartesian = True
        return self
    
    def _set_fractional(self):
        """Set positions in fractional coordinates.
            no use, see get_scaled_positions(wrap=wrap) instead"""
        self.position = cart2frac(self.position, self.lattice)
        self.cartesian = False
        return self
    
    def __len__(self):
        return len(self.arrays['position'])

'''follow functions shoule be merged into the Image class later!!!'''
def elements_to_order(atom_names, atom_types_image, atom_nums):
    """
    Replaces the atom types's order (from 1) to the order of the elements in the atom_names list.

    Args:
        atom_names (list): List of atom names.
        atom_types_image (list): List of atom types.
        atom_nums (int): Number of atoms.

    Example:
        >>> atom_names = ['C', 'N']
        >>> atom_types_image = [1, 1, 1, 1, 1, ... , 2, 2, 2, 2, 2, ... , 2]
        >>> atom_nums = 56
        >>> elements_to_order(atom_names, atom_types_image, atom_nums)
        [6, 6, 6, 6, 6, ... , 7, 7, 7, 7, 7, ... , 7]
        
    Returns:
        list: Updated list of atom types per atom.
    """
    for idx, name in enumerate(atom_names):
        for ii in range(atom_nums):
            if name in elements and atom_types_image[ii] == idx+1:
                atom_types_image[ii] = elements.index(name)
    return atom_types_image

def frac2cart(position, lattice):
    """
    Convert fractional coordinates to Cartesian coordinates.

    Args:
        position (list): List of fractional coordinates.
        lattice (list): List of lattice vectors.

    Example:
        >>> position = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        >>> lattice = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        >>> frac2cart(position, lattice)
        [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]

    Returns:
        list: List of Cartesian coordinates.
    """
    position = np.array(position).reshape(-1, 3)
    lattice = np.array(lattice).reshape(3, 3)
    return np.dot(position, lattice)

def cart2frac(position, lattice):
    """
    Convert Cartesian coordinates to fractional coordinates.

    Args:
        position (list): List of Cartesian coordinates.
        lattice (list): List of lattice vectors.

    Example:
        >>> position = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
        >>> lattice = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        >>> cart2frac(position, lattice)
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]

    Returns:
        list: List of fractional coordinates.
    """
    position = np.array(position).reshape(-1, 3)
    lattice = np.array(lattice).reshape(3, 3)
    return np.dot(position, np.linalg.inv(lattice))