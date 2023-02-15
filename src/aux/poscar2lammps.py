"""
    Data pre-processing for LAMMPS
    L. Wang
    2023.1
"""

import numpy as np

idx_table = {'H': 1,
 'He': 2,
 'Li': 3,
 'Be': 4,
 'B': 5,
 'C': 6,
 'N': 7,
 'O': 8,
 'F': 9,
 'Ne': 10,
 'Na': 11,
 'Mg': 12,
 'Al': 13,
 'Si': 14,
 'P': 15,
 'S': 16,
 'Cl': 17,
 'Ar': 18,
 'K': 19,
 'Ca': 20,
 'Sc': 21,
 'Ti': 22,
 'V': 23,
 'Cr': 24,
 'Mn': 25,
 'Fe': 26,
 'Co': 27,
 'Ni': 28,
 'Cu': 29,
 'Zn': 30,
 'Ga': 31,
 'Ge': 32,
 'As': 33,
 'Se': 34,
 'Br': 35,
 'Kr': 36,
 'Rb': 37,
 'Sr': 38,
 'Y': 39,
 'Zr': 40,
 'Nb': 41,
 'Mo': 42,
 'Tc': 43,
 'Ru': 44,
 'Rh': 45,
 'Pd': 46,
 'Ag': 47,
 'Cd': 48,
 'In': 49,
 'Sn': 50,
 'Sb': 51,
 'Te': 52,
 'I': 53,
 'Xe': 54,
 'Cs': 55,
 'Ba': 56,
 'La': 57,
 'Ce': 58,
 'Pr': 59,
 'Nd': 60,
 'Pm': 61,
 'Sm': 62,
 'Eu': 63,
 'Gd': 64,
 'Tb': 65,
 'Dy': 66,
 'Ho': 67,
 'Er': 68,
 'Tm': 69,
 'Yb': 70,
 'Lu': 71,
 'Hf': 72,
 'Ta': 73,
 'W': 74,
 'Re': 75,
 'Os': 76,
 'Ir': 77,
 'Pt': 78,
 'Au': 79,
 'Hg': 80,
 'Tl': 81,
 'Pb': 82,
 'Bi': 83,
 'Po': 84,
 'At': 85,
 'Rn': 86,
 'Fr': 87,
 'Ra': 88,
 'Ac': 89,
 'Th': 90,
 'Pa': 91,
 'U': 92,
 'Np': 93,
 'Pu': 94,
 'Am': 95,
 'Cm': 96,
 'Bk': 97,
 'Cf': 98,
 'Es': 99,
 'Fm': 100,
 'Md': 101,
 'No': 102,
 'Lr': 103,
 'Rf': 104,
 'Db': 105,
 'Sg': 106,
 'Bh': 107,
 'Hs': 108,
 'Mt': 109,
 'Ds': 110,
 'Rg': 111,
 'Uub': 112}

mass_table = {  1:1.007,2:4.002,3:6.941,4:9.012,5:10.811,6:12.011,
                        7:14.007,8:15.999,9:18.998,10:20.18,11:22.99,12:24.305,
                        13:26.982,14:28.086,15:30.974,16:32.065,17:35.453,
                        18:39.948,19:39.098,20:40.078,21:44.956,22:47.867,
                        23:50.942,24:51.996,25:54.938,26:55.845,27:58.933,
                        28:58.693,29:63.546,30:65.38,31:69.723,32:72.64,33:74.922,
                        34:78.96,35:79.904,36:83.798,37:85.468,38:87.62,39:88.906,
                        40:91.224,41:92.906,42:95.96,43:98,44:101.07,45:102.906,46:106.42,
                        47:107.868,48:112.411,49:114.818,50:118.71,51:121.76,52:127.6,
                        53:126.904,54:131.293,55:132.905,56:137.327,57:138.905,58:140.116,
                        59:140.908,60:144.242,61:145,62:150.36,63:151.964,64:157.25,65:158.925,
                        66:162.5,67:164.93,68:167.259,69:168.934,70:173.054,71:174.967,72:178.49,
                        73:180.948,74:183.84,75:186.207,76:190.23,77:192.217,78:195.084,
                        79:196.967,80:200.59,81:204.383,82:207.2,83:208.98,84:210,85:210,86:222,
                        87:223,88:226,89:227,90:232.038,91:231.036,92:238.029,93:237,94:244,
                        95:243,96:247,97:247,98:251,99:252,100:257,101:258,102:259,103:262,104:261,105:262,106:266}

def pBox2l(A):
    
    from numpy.linalg import norm
    from numpy import dot
    from numpy import cross
    """
        converting PWMAT box to lammps style upper-triangle
    """
    xx = 0
    xy = 0
    yy = 0
    zz = 0
    yz = 0
    xz = 0    
    
    x = A[0]
    y = A[1]
    z = A[2]
    
    cross_xy = cross(x,y)

    # align with lammps rule 
    if (dot(cross_xy,z) < 0 ):
        z = -z 
    abs_x = norm(x)
    abs_y = norm(y)
    abs_z = norm(z)
    
    # run         
    xx = abs_x
    
    xy = dot(x,y)/xx
    yy = (abs_y**2 - xy**2)**0.5
    
    zz = dot(cross_xy/norm(cross_xy), z)
    yz = dot(z-zz*cross_xy/norm(cross_xy), y / abs_y)
    xz = dot(z-zz*cross_xy/norm(cross_xy), x / abs_x)
    
    return [xx,xy,yy,xz,yz,zz]
    
def p2l(filename = "POSCAR"):
    """
        poscar to lammps.data
        
        NOTE: in PWMAT, each ROW represnets a edge vector
    """
    natoms = 0
    atype = []
    
    A = np.zeros([3,3],dtype=float)
    
    infile = open(filename, 'r')
    
    raw = infile.readlines()
    raw = raw[2:]
    
    infile.close()
    
    for idx, line in enumerate(raw):
        raw[idx] = line.split() 
    
    # pwmat box
    for i in range(3):
        #print (raw[i])
        A[i,0] = float(raw[i][0])
        A[i,1] = float(raw[i][1])
        A[i,2] = float(raw[i][2])
        
    lammps_box = pBox2l(A)

    # number of type
    typeNum = len(raw[4])
    
    p = 1 
    # atom num & type list 
    for item in raw[4]:
        natoms += int(item)
        atype += [p for i in range(int(item))]
        p +=1
    
    # x array 
    x = np.zeros([natoms,3],dtype=float)
    
    for idx,line in enumerate(raw[6:]):
        
        x[idx,0] = float(line[0])
        x[idx,1] = float(line[1])
        x[idx,2] = float(line[2])
    
        
    # output preparation 
    # return [xx,xy,yy,xz,yz,zz]
    xlo = 0.0
    xhi = lammps_box[0]
    ylo = 0.0
    yhi = lammps_box[2]
    zlo = 0.0
    zhi = lammps_box[5]
    xy = lammps_box[1]
    xz = lammps_box[3]
    yz = lammps_box[4]

    LX = np.zeros([natoms,3],dtype=float)
    A = np.zeros([3,3],dtype=float) 
    
    A[0,0] = lammps_box[0]
    A[0,1] = lammps_box[1]
    A[1,1] = lammps_box[2]
    A[0,2] = lammps_box[3]
    A[1,2] = lammps_box[4]
    A[2,2] = lammps_box[5]
    
    print("converted LAMMPS upper trangualr box:")
    
    print(A)
    
    print("Ref:https://docs.lammps.org/Howto_triclinic.html")
    # convert lamda (fraction) coords x to box coords LX
    # A.T x = LX
    # LX = A*x in LAMMPS. see https://docs.lammps.org/Howto_triclinic.html
    """
    for i in range(natoms):
        LX[i,0] = A[0,0]*x[i,0] + A[1,0]*x[i,1] + A[2,0]*x[i,2]
        LX[i,1] = A[0,1]*x[i,0] + A[1,1]*x[i,1] + A[2,1]*x[i,2]
        LX[i,2] = A[0,2]*x[i,0] + A[1,2]*x[i,1] + A[2,2]*x[i,2]
    """
    
    for i in range(natoms):
        LX[i,0] = A[0,0]*x[i,0] + A[0,1]*x[i,1] + A[0,2]*x[i,2]
        LX[i,1] = A[1,0]*x[i,0] + A[1,1]*x[i,1] + A[1,2]*x[i,2]
        LX[i,2] = A[2,0]*x[i,0] + A[2,1]*x[i,1] + A[2,2]*x[i,2]

    #print(A)
    #AI = np.linalg.inv(A)
    #print(AI)

    # output LAMMPS data
    ofile = open('lammps.data', 'w')

    ofile.write("#converted from POSCAR\n\n")

    ofile.write("%-12d atoms\n" % (natoms))
    ofile.write("%-12d atom types\n\n" % (typeNum))

    ofile.write("%16.12f %16.12f xlo xhi\n" %  (xlo, xhi))
    ofile.write("%16.12f %16.12f ylo yhi\n" %  (ylo, yhi))
    ofile.write("%16.12f %16.12f zlo zhi\n" %  (zlo, zhi))  
    ofile.write("%16.12f %16.12f %16.12f xy xz yz\n\n" %  (xy, xz, yz))

    ofile.write("Masses\n\n")
    
    for idx,sym in enumerate(raw[3]):
        out_line = str(idx+1)+" "
        out_line += str(mass_table[idx_table[sym]])+"\n"
        #print (out_line)
        ofile.write(out_line)
        
    #ofile.write("1 6.94000000      #Li\n")
    #ofile.write("2 180.94788000    #Ta\n")
    #.write("3 15.99900000     #O\n")
    #ofile.write("4 1.00800000      #H\n\n")

    ofile.write("\nAtoms # atomic\n\n")

    for i in range(natoms):
        ofile.write("%12d %5d %21.15f %21.15f %21.15f\n" % (i+1, atype[i], LX[i,0], LX[i,1], LX[i,2]) )

    ofile.close()

if __name__ =="__main__":
    
    p2l() 
