"""Adapted from ase: http://wiki.fysik.dtu.dk/ase """

"""Utility tools for atoms/geometry manipulations.
   - convenient creation of slabs and interfaces of
different orientations.
   - detection of duplicate atoms / atoms within cutoff radius
"""
import numpy as np
from build.cell import complete_cell

def translate_pretty(fractional, pbc):
    """Translates atoms such that fractional positions are minimized."""

    for i in range(3):
        if not pbc[i]:
            continue

        indices = np.argsort(fractional[:, i])
        sp = fractional[indices, i]

        widths = (np.roll(sp, 1) - sp) % 1.0
        fractional[:, i] -= sp[np.argmin(widths)]
        fractional[:, i] %= 1.0
    return fractional

def pbc2pbc(pbc):
    newpbc = np.empty(3, bool)
    newpbc[:] = pbc
    return newpbc

def wrap_positions(positions, cell, pbc=True, center=(0.5, 0.5, 0.5),
                   pretty_translation=False, eps=1e-7):
    """Wrap positions to unit cell.

    Returns positions changed by a multiple of the unit cell vectors to
    fit inside the space spanned by these vectors.  See also the
    :meth:`ase.Atoms.wrap` method.

    Parameters:

    positions: float ndarray of shape (n, 3)
        Positions of the atoms
    cell: float ndarray of shape (3, 3)
        Unit cell vectors.
    pbc: one or 3 bool
        For each axis in the unit cell decides whether the positions
        will be moved along this axis.
    center: three float
        The positons in fractional coordinates that the new positions
        will be nearest possible to.
    pretty_translation: bool
        Translates atoms such that fractional coordinates are minimized.
    eps: float
        Small number to prevent slightly negative coordinates from being
        wrapped.

    Example:

    >>> wrap_positions([[-0.1, 1.01, -0.5]],
    ...                [[1, 0, 0], [0, 1, 0], [0, 0, 4]],
    ...                pbc=[1, 1, 0])
    array([[ 0.9 ,  0.01, -0.5 ]])
    """

    if not hasattr(center, '__len__'):
        center = (center,) * 3

    pbc = pbc2pbc(pbc)
    shift = np.asarray(center) - 0.5 - eps

    # Don't change coordinates when pbc is False
    shift[np.logical_not(pbc)] = 0.0

    assert np.asarray(cell)[np.asarray(pbc)].any(axis=1).all(), (cell, pbc)

    cell = complete_cell(cell)
    fractional = np.linalg.solve(cell.T,
                                 np.asarray(positions).T).T - shift

    if pretty_translation:
        fractional = translate_pretty(fractional, pbc)
        shift = np.asarray(center) - 0.5
        shift[np.logical_not(pbc)] = 0.0
        fractional += shift
    else:
        for i, periodic in enumerate(pbc):
            if periodic:
                fractional[:, i] %= 1.0
                fractional[:, i] += shift[i]

    return np.dot(fractional, cell)