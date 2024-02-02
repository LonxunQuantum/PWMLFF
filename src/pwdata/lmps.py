import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
      return [0, 0, 0]
    return v/norm

def Box2l(lattice):
    """
        converting box to lammps style upper-triangle
    """
  
    A = lattice[0]
    B = lattice[1]
    C = lattice[2]
  
  
    nA = normalize(A)
    Al = np.linalg.norm(A)
    Bl = np.linalg.norm(B)
    Cl = np.linalg.norm(C)
  
    ax = np.linalg.norm(A)
    bx = np.dot(B,nA)
    by = np.sqrt(Bl*Bl-bx*bx)
    cx = np.dot(C,nA)
    cy = (np.dot(B,C)-bx*cx)/by
    cz = np.sqrt(Cl*Cl - cx*cx - cy*cy)
  
    xx = ax
    yy = by
    zz = cz
    xy = bx
    xz = cx
    yz = cy 
  
    return [xx,xy,yy,xz,yz,zz]
    
def l2Box(diagdisp, offdiag):
    """
    Convert lammps style upper-triangle to box.
  
    Parameters:
    diagdisp: cell dimension convoluted with the displacement vector
    offdiag: off-diagonal cell elements
  
    Returns:
    lattice: lattice vectors
    lattice_disp: lattice displacement vector
    """
    xy = offdiag[0]
    xz = offdiag[1]
    yz = offdiag[2]
    xlo = diagdisp[0][0]
    xhi = diagdisp[0][1]
    ylo = diagdisp[1][0]
    yhi = diagdisp[1][1]
    zlo = diagdisp[2][0]
    zhi = diagdisp[2][1]
    
    xhilo = (xhi - xlo) - abs(xy) - abs(xz)
    yhilo = (yhi - ylo) - abs(yz)
    zhilo = (zhi - zlo)
    
    lattice_displacementx = xlo - min(0, xy) - min(0, xz)
    lattice_displacementy = ylo - min(0, yz)
    lattice_displacementz = zlo
  
    lattice = np.array([[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]])
    lattice_disp = np.array([lattice_displacementx, lattice_displacementy, lattice_displacementz])
  
    return lattice, lattice_disp

  