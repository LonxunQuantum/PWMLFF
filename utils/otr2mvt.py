#!/usr/bin/env python3
import numpy as np
import numpy.linalg as LA
import os
import argparse

element = ["0", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg"]

def outcar2raw():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='specify input movement filename', type=str, default='OUTCAR')
    parser.add_argument('-n', '--number', help='specify number of samples per set', type=int, default=10000)
    parser.add_argument('-d', '--directory', help='specify stored directory of raw data', type=str, default='.')
    args = parser.parse_args()

    Labelsystem(args.input, fmt='outcar', num=args.number, directory=args.directory)

class Labelsystem():
    def __init__(self, filename, fmt='outcar', num=10000, directory='.'):
        self.filename = filename
        self.fmt = fmt
        self.num = num
        self.directory = directory
        self.blocks = []  # List to store data for each "Ionic step" block
        self.current_block = {}

        self.readfile(self.filename, self.fmt, self.num)
        self.writefile(self.directory)

    def readfile(self, filename, fmt, num):
        if fmt == 'outcar':
            self.read_outcar(filename, num)
        else:
            print('Error: unsupported file format')
            exit(1)

    def writefile(self, directory):
        # self.write_raw()
        # self.load_raw()
        # self.transform_raw()
        self.transform2movement(directory)  # tmp use
        
    def write_raw(self):
        # Write raw data
        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=True)
        np.savetxt(os.path.join(self.directory, 'type_map.raw'), self.atom_types, fmt='%d')
        np.savetxt(os.path.join(self.directory, 'atom_name.raw'), self.atom_names, fmt='%s')
        np.savetxt(os.path.join(self.directory, 'box.raw'), np.reshape(self.cells, [self.ionic_step
, 9]))
        np.savetxt(os.path.join(self.directory, 'coord.raw'), np.reshape(self.coords, [self.ionic_step
,self.atom_numbs*3]))
        np.savetxt(os.path.join(self.directory, 'energy.raw'), np.reshape(self.energys, [self.ionic_step
, 1]))
        np.savetxt(os.path.join(self.directory, 'force.raw'), np.reshape(self.forces, [self.ionic_step
,self.atom_numbs*3]))
        if self.virial is not None:
            np.savetxt(os.path.join(self.directory, 'virial.raw'), np.reshape(self.virial, [self.ionic_step
,9]))
   
    def read_outcar(self, filename, num):
        f = open(filename, 'r')
        lines = f.readlines()
        f.close()

        # Temporarily retain
        self.atom_names, self.atom_numbs, self.atom_types, self.cells, self.coords, self.energys, self.forces, self.virial, self.ionic_step = self.get_frames(lines, begin=0, step=1)
      

    def get_frames(self, lines, begin=0, step=1):
        atom_names = []
        atom_numbs = None
        nelm = None
        nsw = None

        # Initialize variables to store data
        all_coords = []
        all_cells = []
        all_energies = []
        all_forces = []
        all_virials = []

        for ii in lines: 
            if "TITEL  =" in ii:
                # get atom names from POTCAR info, tested only for PAW_PBE ...
                _ii=ii.split()[3]
                if '_' in _ii:
                    atom_names.append(_ii.split('_')[0])
                else:
                    atom_names.append(_ii)
            elif 'NELM   =' in ii :
                nelm = int(ii.split()[2][:-1])
                break;
            elif 'NSW =' in ii :
                nsw = int(ii.split()[2])
            elif 'ions per type' in ii :
                atom_numbs_ = [int(s) for s in ii.split()[4:]]
                if atom_numbs is None :                
                    atom_numbs = atom_numbs_
                else :
                    assert (atom_numbs == atom_numbs_), "inconsistent number of atoms in OUTCAR"
        assert(nelm is not None), "cannot find maximum steps for each SC iteration"
        assert(atom_numbs is not None), "cannot find ion type info in OUTCAR"
        atom_names = atom_names[:len(atom_numbs)]
        atom_types = []
        for idx,ii in enumerate(atom_numbs) :
            for jj in range(ii) :
                atom_types.append(idx+1)

        num_atoms = sum(atom_numbs)

        atom_names, atom_types = self.type_map(atom_names, atom_numbs, atom_types, num_atoms)

        all_images = self.analyze_frames(lines, num_atoms, nelm, nsw)
        # 每个image中的数据是一个字典，包含ionic_step, coord, cell, energy, force, virial, is_converge，is_converge暂时没用
        
        for image in all_images:
            if len(image) > 0:
                all_coords.append(image["coord"])
                all_cells.append(image["cell"])
                all_energies.append(image["energy"])
                all_forces.append(image["force"])
                all_virials.append(image["virial"])
        ionic_step = all_images[-1]["ionic_step"]
        
        if len(all_virials) == 0 :
            all_virials = None
        else :
            all_virials = np.array(all_virials)
        
        return atom_names, num_atoms, atom_types, np.array(all_cells), np.array(all_coords), np.array(all_energies), np.array(all_forces), all_virials, ionic_step
        
    def analyze_frames(self, lines, num_atoms, nelm, nsw):
        energy = []
        virial = []
        is_converge = True  # Whether the calculation is converged, need to be modified
        all_coords = []
        all_cells = []
        all_energies = []
        all_forces = []
        all_virials = []
        prev_insw = None    # 跟踪上一个循环中的离子步

        for idx, ii in enumerate(lines):
            if 'Iteration' in ii:
                # insw = int(ii.split()[3])
                insw = int(ii.split()[2][:-1])
                if insw != prev_insw:
                    iflag = 1
                else:
                    iflag = 0

                prev_insw = insw

                if iflag == 1:   
                    if self.current_block is not None:
                        self.blocks.append(self.current_block)
                    self.current_block = {
                        "ionic_step": insw,
                        "electronic_step": [],
                        "coord": [],
                        "cell": [],
                        "energy": [],
                        "force": [],
                        "virial": [],
                        "is_converge": True
                    }

            if 'Iteration' in ii:
                sc_index = int(ii.split()[3][:-1])
                self.current_block["electronic_step"].append(sc_index)
                if sc_index >= nelm:
                    is_converge = False
                    self.current_block["is_converge"] = is_converge
            elif 'free  energy   TOTEN' in ii:
                energy = float(ii.split()[4])
                # all_energies.append(energy)
                self.current_block["energy"] = [energy]
            elif 'VOLUME and BASIS' in ii:
                cell = []
                for dd in range(3):
                    tmp_l = lines[idx+5+dd]
                    cell.append([float(ss) for ss in tmp_l.replace('-', ' -').split()[0:3]])
                self.current_block["cell"] = cell
                # all_cells.append(cell)
            elif 'in kB' in ii:
                prev_line = lines[idx-1]    # eV line
                tmp_v = [float(ss) for ss in prev_line.split()[1:]]
                virial = np.zeros([3, 3])
                virial[0][0] = tmp_v[0]     # xx
                virial[1][1] = tmp_v[1]     # yy    
                virial[2][2] = tmp_v[2]     # zz
                virial[0][1] = tmp_v[3]     # xy
                virial[1][0] = tmp_v[3]     # xy
                virial[1][2] = tmp_v[4]     # yz    
                virial[2][1] = tmp_v[4]     # yz
                virial[0][2] = tmp_v[5]     # zx
                virial[2][0] = tmp_v[5]     # zx
                # all_virials.append(virial)
                self.current_block["virial"] = virial
            elif 'TOTAL-FORCE' in ii:
                coord = []
                force = []
                for jj in range(idx+2, idx+2+num_atoms):
                    tmp_l = lines[jj]
                    info = [float(ss) for ss in tmp_l.split()]
                    coord.append(info[:3])
                    force.append(info[3:])
                self.current_block["coord"] = coord
                self.current_block["force"] = force
                # all_coords.append(coord)    
                # all_forces.append(force)

        return self.blocks

    def type_map(self, atom_names, atom_numbs, atom_types, num_atoms):
        # element mappping to atom_type and atom_name
        expand_atom_names = atom_names
        for idx, name in enumerate(atom_names):
            for ii in range(num_atoms):
                if name in element and atom_types[ii] == idx+1:
                    atom_types[ii] = element.index(name)

        return atom_names, atom_types

    def transform2movement(self, directory):
        # Transform raw data to movement data
        num_image = self.ionic_step
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        mvmt = open(os.path.join(directory, 'MOVEMENT'), 'w')
        # mvmt = open('MOVEMENT', 'w')
        for i in range(1,num_image+1):
            write_image(mvmt, self.blocks[i], self.atom_numbs, self.atom_types)
        mvmt.close()

def write_image(mvmt, image, atom_numbs, atom_types):

    image['coord'] = np.dot(image['coord'], LA.inv(image['cell']))
    
    mvmt.write(" %d atoms, Iteration (fs) = %16.10E, Etot,Ep,Ek (eV) = %16.10E  %16.10E   %16.10E, SCF = %d\n"\
                % (atom_numbs, 0.0, image['energy'][0], image['energy'][0], 0.0, image['electronic_step'][-1]))
    mvmt.write(" MD_INFO: METHOD(1-VV,2-NH,3-LV,4-LVPR,5-NHRP) TIME(fs) TEMP(K) DESIRED_TEMP(K) AVE_TEMP(K) TIME_INTERVAL(fs) TOT_TEMP(K) \n")
    mvmt.write("          1    0.5000000000E+00   0.59978E+03   0.30000E+03   0.59978E+03   0.50000E+02   0.59978E+03\n")
    mvmt.write(" MD_VV_INFO: Basic Velocity Verlet Dynamics (NVE), Initialized total energy(Hartree)\n")
    mvmt.write("          -0.1971547257E+05\n")
    mvmt.write("Lattice vector (Angstrom)\n")
    for i in range(3):
        if len(image['virial']) == 0:
            mvmt.write("  %16.10E    %16.10E    %16.10E\n" % (image['cell'][i][0], image['cell'][i][1], image['cell'][i][2]))
        else:
            mvmt.write("  %16.10E    %16.10E    %16.10E     stress (eV): %16.10E    %16.10E    %16.10E\n" % (image['cell'][i][0], image['cell'][i][1], image['cell'][i][2], image['virial'][i][0], image['virial'][i][1], image['virial'][i][2]))
    mvmt.write("  Position (normalized), move_x, move_y, move_z\n")
    for i in range(atom_numbs):
        mvmt.write(" %4d    %20.15F    %20.15F    %20.15F    1 1 1\n"\
                % (atom_types[i], image['coord'][i][0], image['coord'][i][1], image['coord'][i][2]))
    mvmt.write("  Force (-force, eV/Angstrom)\n")
    for i in range(atom_numbs):
        mvmt.write(" %4d    %20.15F    %20.15F    %20.15F\n"\
                % (atom_types[i], -image['force'][i][0], -image['force'][i][1], -image['force'][i][2]))  # minus sign here
    mvmt.write(' -------------------------------------\n')

if __name__ == '__main__':
    outcar2raw()
