"""Adapted from ase: http://wiki.fysik.dtu.dk/ase """
import numpy as np
from image import Image

class SupercellError(Exception):
    """Use if construction of supercell fails"""

def make_supercell(image_data: Image, supercell_matrix: list, pbc: list = None, wrap=True, tol=1e-5):
    """Construct supercell from image_data and supercell_matrix

    Args:
        image_data (list): image_data list including prim atom types, positions, lattice, etc.
        supercell_matrix (list): supercell matrix (3x3)
        pbc (list): Periodic boundary conditions flags.
    """
    prim = image_data
    supercell_matrix = np.array(supercell_matrix)
    supercell = clean_matrix(supercell_matrix @ prim.lattice)
    # cartesian lattice points
    lattice_points_frac = lattice_points_in_supercell(supercell_matrix)
    lattice_points = np.dot(lattice_points_frac, supercell)

    superatoms = Image(lattice=supercell, pbc=pbc)
    for lp in lattice_points:
        shifted_atoms = prim.copy()
        shifted_atoms.arrays['position'] += lp
        superatoms.extend(shifted_atoms)
    superatoms.cartesian = True

    # check number of atoms is correct
    n_target = int(np.round(np.linalg.det(supercell_matrix) * len(prim.position)))
    if n_target != len(superatoms):
        msg = "Number of atoms in supercell: {}, expected: {}".format(
            n_target, len(superatoms)
        )
        raise SupercellError(msg)

    if wrap:
        superatoms.wrap(eps=tol)
    
    return superatoms

    
def lattice_points_in_supercell(supercell_matrix):
    """Find all lattice points contained in a supercell.
    1) define the diagonal array including all vertexes of a cube.
    2) The corresponding vertices in the superlattice are obtained by multiplying the diagonals array with the supercell_matrix matrix.
    3) The minimum and maximum values of the vertices in the superlattice are obtained, and the range of the vertices in the superlattice is obtained.
    4) The vertices in the superlattice are obtained by traversing the range of the vertices in the superlattice. (all_points)
    5) The vertices in the superlattice are converted into fractional coordinates. (frac_points)
    6) The vertices in the superlattice are filtered according to the fractional coordinates, and only retain those points inside the superlattice. (tvects)
    7) The number of vertices in the superlattice is equal to the volume of the superlattice.

    Args:
        supercell_matrix (list): supercell matrix (3x3)

    Returns:
        list: list of lattice points in supercell"""

    diagonals = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    d_points = np.dot(diagonals, supercell_matrix)

    mins = np.min(d_points, axis=0)
    maxes = np.max(d_points, axis=0) + 1

    ar = np.arange(mins[0], maxes[0])[:, None] * np.array([1, 0, 0])[None, :]
    br = np.arange(mins[1], maxes[1])[:, None] * np.array([0, 1, 0])[None, :]
    cr = np.arange(mins[2], maxes[2])[:, None] * np.array([0, 0, 1])[None, :]

    all_points = ar[:, None, None] + br[None, :, None] + cr[None, None, :]
    all_points = all_points.reshape((-1, 3))

    frac_points = np.dot(all_points, np.linalg.inv(supercell_matrix))

    tvects = frac_points[
        np.all(frac_points < 1 - 1e-10, axis=1)
        & np.all(frac_points >= -1e-10, axis=1)
    ]
    assert len(tvects) == round(abs(np.linalg.det(supercell_matrix)))
    return tvects

def clean_matrix(matrix, eps=1e-12):
    """ clean from small values"""
    matrix = np.array(matrix)
    for ij in np.ndindex(matrix.shape):
        if abs(matrix[ij]) < eps:
            matrix[ij] = 0
    return matrix