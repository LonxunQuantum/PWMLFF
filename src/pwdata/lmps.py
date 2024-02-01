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
    