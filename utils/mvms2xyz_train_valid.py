"""
Script for converting multi movement files to xyz format:

You only need to write in the movement path, which supports multiple movement conversions
 without requiring consistent configurations between movements.

Ratio control:
    if random_valid is set to True, the test set and will be randomly extracted from each movement, 
    while False will take the last ratio of each movement
"""

import os, shutil, sys
import numpy as np
import re
import math

import pymatgen as pm
from glob import glob
from dpdata.vasp.outcar import get_frames

from collections import Counter
ANGSTROM2BOHR=1.8897161646320724
EV2HA=0.0367493

element_table = ['', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                 'Ar',
                 'K', 'Ca',
                 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                 'Rb', 'Sr', 'Y',
                 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
                 'Ba', 'La',
                 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                 'W', 'Re',
                 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                 'U', 'Np',
                 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

class Image(object):
    def __init__(self, atom_type=None, atom_num = None, iteration=None, Etot = None, Ep = None, Ek = None, scf = None) -> None:
        #76 atoms,Iteration (fs) =   -0.1000000000E+01, Etot,Ep,Ek (eV) =   -0.1335474369E+05  -0.1335474369E+05   0.0000000000E+00, SCF =    14
        self.atom_num = atom_num
        self.iteration = iteration
        self.atom_type = atom_type
        self.Etot = Etot
        self.Ep = Ep
        self.Ek = Ek
        self.scf = scf
        self.lattice = []
        self.stress = []
        self.position = []
        self.force = []
        self.atomic_energy = []
        self.content = []

    def set_md_info(self, method = None ,time = None ,temp = None ,desired_temp = None ,avg_temp = None ,time_interval = None ,tot_temp = None):
        #  MD_INFO: METHOD(1-VV,2-NH,3-LV,4-LVPR,5-NHRP) TIME(fs) TEMP(K) DESIRED_TEMP(K) AVE_TEMP(K) TIME_INTERVAL(fs) TOT_TEMP(K)
        #   2    0.1000000000E+01   0.19941E+04   0.10000E+04   0.19941E+04   0.10000E+03   0.19941E+04\
        self.method = method
        self.time = time
        self.temp = temp
        self.desired_temp = desired_temp
        self.avg_temp = avg_temp
        self.time_interval = time_interval
        self.tot_temp = tot_temp

    def set_energy_info(self, energy_content):
        # "76 atoms,Iteration (fs) =   -0.1000000000E+01, Etot,Ep,Ek (eV) =   -0.1335474369E+05  -0.1335474369E+05   0.0000000000E+00, SCF =    14"
        # after re extract, numbers = ['76', '-0.1000000000E+01', '-0.1335474369E+05', '-0.1335474369E+05', '0.0000000000E+00', '14']
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", energy_content)
        self.atom_num = int(numbers[0])
        self.iteration = format(float(numbers[1]), '.2f')
        self.Etot, self.Ep, self.Ek = float(numbers[2]), float(numbers[3]), float(numbers[4])
        if len(numbers) >= 5:
            self.scf = int(numbers[5])
        # self.content.append(energy_content)

    def set_lattice_stress(self, lattice_content):
        lattic1 = [float(_) for _ in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", lattice_content[0])]
        lattic2 = [float(_) for _ in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", lattice_content[1])]
        lattic3 = [float(_) for _ in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", lattice_content[2])]
        if "stress" in lattice_content[0]:
            self.stress = [lattic1[3:], lattic2[3:], lattic3[3:]]
        self.lattice = [lattic1[:3], lattic2[:3], lattic3[:3]]
        # self.content.append(lattice_content)
    
    def set_position(self, position_content):
        atom_type = []
        for i in range(0, len(position_content)):
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", position_content[i])
            atom_type.append(int(numbers[0]))
            self.position.append([float(_) for _ in numbers[1:4]])
        counter = Counter(atom_type)
        self.atom_type = list(counter.keys())
        self.atom_type_num = list(counter.values())
        assert self.atom_num == sum(self.atom_type_num)
        # self.content.append(position_content)

    def set_force(self, force_content):
        for i in range(0, len(force_content)):
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", force_content[i])
            self.force.append([float(_) for _ in numbers[1:4]])
        assert self.atom_num == len(self.force)

    def set_atomic_energy(self, atomic_energy):
        for i in range(0, len(atomic_energy)):
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", atomic_energy[i])
            self.atomic_energy.append(float(numbers[1]))
        assert self.atom_num == len(self.atomic_energy)

    def set_content(self, content):
        self.content = content

class MOVEMENT(object):
    def __init__(self, movement_file) -> None:
        self.image_list:list[Image] = []
        self.image_nums = 0
        self.load_movement_file(movement_file)

    '''
    description: 
        load movement file, seperate the file to each image contents and get the image nums
    param {*} self
    param {*} movement_file
    return {*}
    author: wuxingxing
    ''' 
    def load_movement_file(self, movement_file):
        # seperate content to image contents
        with open(movement_file, 'r') as rf:
            mvm_contents = rf.readlines()
        i = 0
        while i < len(mvm_contents):
            if "Iteration" in mvm_contents[i]:
                image_start = i
                # set energy info
                image = Image()
                self.image_list.append(image)
                image.set_energy_info(mvm_contents[i])
                i += 1
            elif "Lattice" in mvm_contents[i]:
                # three line for lattic info
                image.set_lattice_stress(mvm_contents[i+1:i+4])
                i += 4
            elif " Position" in mvm_contents[i]:
                # atom_nums line for postion
                image.set_position(mvm_contents[i+1:i+image.atom_num+1])
                i = i + 1 + image.atom_num
            elif "Force" in mvm_contents[i]:
                image.set_force(mvm_contents[i+1: i+ image.atom_num+1])
                i = i + 1 + image.atom_num
            elif "Atomic-Energy" in mvm_contents[i]:
                image.set_atomic_energy(mvm_contents[i+1: i+ image.atom_num+1])
                i = i + 1 + image.atom_num
            else:
                i = i + 1   #to next line
            # image content end at the line "------------END"
            if "-------------" in mvm_contents[i]:
                i = i + 1
                image_end = i
                image.set_content(mvm_contents[image_start:image_end])

        self.image_nums = len(self.image_list)

    '''
    description: 
        Sample the movement file according to the interval value and save the sampled result to file
    param {*} self
    param {*} save_file
    param {*} interval
    return {*}
    author: wuxingxing
    '''
    def save_image_interval(self, save_file, interval=1):
         with open(save_file, 'w') as wf:
             for i in range(0, self.image_nums):
                 if i % interval == 0:
                    for line in self.image_list[i].content:
                        wf.write(line)

    def save_image_range(self, save_file, start=0, end=100):
         with open(save_file, 'w') as wf:
             for i in range(start, end):
                for line in self.image_list[i].content:
                    wf.write(line)

    def save_to_2parts(self, mid):
        part_1 = "part_1_movement_{}".format(mid)
        part_2 = "part_2_movement_{}".format(self.image_nums-mid)
        with open(part_1, 'w') as wf:
            for i in range(0, mid):
                for line in self.image_list[i].content:
                    wf.write(line)

        with open(part_2, 'w') as wf:
            for i in range(mid, self.image_nums):
                for line in self.image_list[i].content:
                    wf.write(line)

class Structure:
    '''
    Now, the code can only to deal with one type of MOVEMENT.
    MOVEMENT: Energy(eV) Force(eV/Angstrom) lattice(Angstrom)
    input.data: Energy(Ha) Force(Ha/bohr) lattice(bohr)
    '''


    def __init__(self, path, path_2='./POSCAR',type='MOVEMENT',is_charge=False):
        self.element_table = element_table

        self.NELM=200
        self.atom_num=0
        self.lattice = []
        self.eles_list = []
        self.eles_num_list=[]
        self.atom_position = []
        self.cartesian_position = []
        self.etot = []
        self.dE = []
        # etot_dE=[]
        self.atom_force = []
        self.atom_energy=[]
        self.mag=[]
        self.charge = []
        self.charge_tot = []
        self.vol=[]
        self.length_cell=[]
        nimages = 0
        self.is_atom_config = False
        if type == 'MOVEMENT':
            if "config".upper() in os.path.basename(path).upper(): #convert atom.config file
                self.is_atom_config = True
            self.MOVEMENTloader(path)
        if type == 'RuNNer':
            self.RuNNerdataloader(path)
        if type == 'vasprun':
            self.VASPRUNloader(path)
            if is_charge==True:
                self.is_charge=True
                self.ACFDATAloader(path_2)
            #self.OSZICARloader(path_OSZICAR)
        if type == 'XDATCAR':
            self.POSCARloader(path_2)
            self.XDATCARloader(path)
        if type == 'OUTCAR':
            # self.OSZICAR_dir=path_2
            # self.OUTCARloader(path)
            self.OUTCARloader_dp(path)
        if type=='POSCAR':
            self.POSCARloader(path)


    #def unit_conversion(self):
        #self.lattice=self.lattice*ANGSTROM2BOHR
        #self.etot=self.etot*EV2HA
    def OUTCARloader_dp(self,path):
        '''
        A code from https://docs.deepmodeling.com/projects/dpdata/en/master/_modules/dpdata/vasp/outcar.html#get_frames
        '''
        self.eles_type,\
        self.eles_num_list,\
        self.eles_list,\
        self.lattice,\
        self.cartesian_position,\
        self.etot,\
        self.atom_force,\
        _=get_frames(path)
        self.nimages=len(self.etot)
        for i in self.eles_num_list:
            self.atom_num+=int(i)
        return

    def ACFDATAloader(self,path):
        '''
        Because bader method can't deal with MD traj, we split traj into every step to get ACF.dat.
        So the path should be the total directory of all ACF.dat
        '''
        tmp=0
        total_charge=[0.0 for i in range(self.nimages)]
        z_list=[3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0]
        charge_list=[[] for i in range(self.nimages)]

        for nimage in range(self.nimages):
            tmp = 0
            with open(path+'/'+str(nimage+1)+'/ACF.dat','r') as f:
                lines=f.readlines()
                for i in range(len(lines)):
                    if 1<i<(len(lines)-4):
                        charge_list[nimage].append(z_list[tmp]-float(lines[i].split()[4]))
                        total_charge[nimage]+=z_list[tmp]-float(lines[i].split()[4])
                        tmp+=1

        self.charge=charge_list
        self.charge_tot=total_charge
        return

    def POSCARloader(self,path):
        # structure = pm.Structure.from_file(path, False)
        # self.lattice=structure.lattice.matrix
        # self.atom_num=len(structure.atomic_numbers)
        self.lattice = [[] for i in range(3)]

        num=0
        with open(path,'r') as f:
            all_data=f.readlines()
            for i in range(2,5):
                for j in range(3):
                    self.lattice[num].append(float(all_data[i].split()[j]))
                num+=1
            self.eles_type=len(all_data[5].split())
            for i in range(self.eles_type):
                self.eles_list.append(all_data[5].split()[i])
            for i in range(self.eles_type):
                self.eles_num_list.append(int(all_data[6].split()[i]))
            for i in range(self.eles_type):
                self.atom_num+=self.eles_num_list[i]
            self.eles_position=[[[] for ii in range(self.eles_num_list[i])] for i in range(self.eles_type)]
            num=0
            num1=0
            for i in range(8,8+self.atom_num):
                for j in range(3):
                    self.eles_position[num][num1].append(float(all_data[i].split()[j]))
                num1+=1
                if num1>=self.eles_num_list[num]:
                    num+=1
                    num1=0
        return

    def XDATCARloader(self,path):
        self.fcoord = np.loadtxt(path, comments='D', skiprows=7).reshape((-1, self.atom_num, 3))
        self.nimages=len(self.fcoord)
        with open(path,'r') as f:
            tmp=f.readlines()
            self.heads=tmp[:7]
        return

    def RuNNerdataloader(self,path):
        '''
        now it can only deal with one type of atom_num
        '''
        tmp_nimage=0
        tmp_lattice=0
        tmp_natom=0
        self.nimages=int(os.popen('grep begin {}|wc -l'.format(path)).read())
        if int(os.popen('grep atom {}|wc -l'.format(path)).read())%self.nimages!=0:
            print('Warning!!! every images in input.data have different atom number!')
            return
        self.atom_num=int(int(os.popen('grep atom {}|wc -l'.format(path)).read())/self.nimages)
        self.lattice = [[] for i in range(self.nimages)]
        self.atom_position = [[[] for i in range(self.atom_num)] for i in range(self.nimages)]
        self.cartesian_position = [[[] for i in range(self.atom_num)] for i in range(self.nimages)]
        self.atom_force = [[[] for i in range(self.atom_num)] for i in range(self.nimages)]
        with open(path,'r') as f:
            file_tmp=f.readlines()
            for i in range(len(file_tmp)):
                if file_tmp[i].split()[0].count('lattice')==1:
                    self.lattice[tmp_nimage].append([])
                    for ii in range(1,4):
                        self.lattice[tmp_nimage][tmp_lattice].append(float(file_tmp[i].split()[ii])/ANGSTROM2BOHR)
                    tmp_lattice+=1

                if file_tmp[i].split()[0].count('atom')==1:
                    for ii in range(1,4):
                        self.cartesian_position[tmp_nimage][tmp_natom].append(float(file_tmp[i].split()[ii])/ANGSTROM2BOHR)
                    self.eles_list.append(self.element_table.index(file_tmp[i].split()[4]))
                    for ii in range(7,10):
                        self.atom_force[tmp_nimage][tmp_natom].append(float(file_tmp[i].split()[ii])/EV2HA*ANGSTROM2BOHR)
                    tmp_natom += 1

                if file_tmp[i].split()[0].count('energy')==1:
                    self.etot.append(float(file_tmp[i].split()[1])/EV2HA)

                if file_tmp[i].split()[0].count('end')==1:
                    tmp_nimage+=1
                    tmp_lattice=0
                    tmp_natom=0

        return

    def OSZICARloader(self,path):
        with open(path,'r') as f:
            file_tmp=f.readlines()
            for i in range(len(file_tmp)):
                if re.search('T=',file_tmp[i]):
                    self.etot.append(float(file_tmp[i].split()[10]))

        return

    def VASPRUNloader(self,path):
        tmp=0
        tmp_natom=0
        tmp_nlattice=0
        tmp_nposition=0
        tmp_nbasis=0
        tmp_nforce=0
        structure=pm.Structure.from_file(path)
        self.atom_num=len(structure.frac_coords)
        self.eles_list=list(structure.atomic_numbers)
        with open(path,'r') as f:
            file_tmp=f.readlines()
            for i in range(len(file_tmp)):
                if re.search('<calculation>',file_tmp[i]):
                    self.nimages+=1

            self.lattice = [[] for i in range(self.nimages)]
            self.atom_position = [[[] for i in range(self.atom_num)] for i in range(self.nimages)]
            self.cartesian_position = [[[] for i in range(self.atom_num)] for i in range(self.nimages)]
            self.atom_force = [[[] for i in range(self.atom_num)] for i in range(self.nimages)]

            for i in range(len(file_tmp)):
                if re.search('<varray name="basis" >', file_tmp[i]):
                    tmp_nbasis=tmp_nbasis+1
                    if 2<tmp_nbasis<self.nimages+3:
                        for ii in range(i+1,i+4):
                            self.lattice[tmp_nbasis-3].append([])
                            for j in range(1,4):
                                self.lattice[tmp_nbasis-3][tmp_nlattice].append(float(file_tmp[ii].split()[j]))
                            tmp_nlattice+=1
                        tmp_nlattice=0

                if re.search('<varray name="positions" >', file_tmp[i]):
                    tmp_nposition=tmp_nposition+1
                    if 2<tmp_nposition<self.nimages+3:
                        for ii in range(i+1,i+self.atom_num+1):
                            for j in range(1,4):
                                self.atom_position[tmp_nposition-3][tmp_natom].append(float(file_tmp[ii].split()[j]))
                            tmp_natom+=1
                        tmp_natom=0

                if re.search('<varray name="forces" >', file_tmp[i]):
                    for ii in range(i+1,i+self.atom_num+1):
                        for j in range(1,4):
                            self.atom_force[tmp_nforce][tmp_natom].append(float(file_tmp[ii].split()[j]))
                        tmp_natom+=1
                    tmp_nforce+=1
                    tmp_natom=0

                if re.match('   <i name="e_fr_energy">',file_tmp[i]):
                    self.etot.append(float(file_tmp[i].split()[2]))

        return

    def MOVEMENTloader(self,path):
        '''
        now it can only deal with one type of atom_num
        Attension!!! PWmat Force unit is (-force, eV/Angstrom), so we should take negative value
        '''
        tmp=0
        iter_loop=[]
        lattice_loop=[]
        position_loop=[]
        force_loop=[]
        dE_loop=[]
        if self.is_atom_config:
            self.nimages = 1
        else:
            self.nimages=int(os.popen('grep Iter {}|wc -l'.format(path)).read())
        with open(path,'r') as f:
            file_tmp = f.readlines()
            self.atom_num = int(file_tmp[0].split()[0])
            self.lattice = [[[],[],[]]for i in range(self.nimages)]
            self.atom_position = [[[] for i in range(self.atom_num)] for i in range(self.nimages)]
            self.cartesian_position=[[[] for i in range(self.atom_num)] for i in range(self.nimages)]
            self.atom_force=[[[] for i in range(self.atom_num)] for i in range(self.nimages)]
            self.atom_energy=[[[] for i in range(self.atom_num)] for i in range(self.nimages)]
            for i in range(len(file_tmp)):
                # if re.match(file_tmp[i].split()[0], '{}'.format(self.atom_num)) != None:
                if 'Iteration' in file_tmp[i]:
                    iter_loop.append(i)
                if file_tmp[i].split()[0].count('Lattice')==1:
                    lattice_loop.append(i)
                # if file_tmp[i].split()[0].count('Position')==1:
                # if re.match(file_tmp[i].split()[0],)!=None:
                if ' Position' in file_tmp[i]:
                    position_loop.append(i)
                if file_tmp[i].split()[0].count('Force')==1:
                    force_loop.append(i)
                if file_tmp[i].split(',')[0].count('Atomic-Energy')==1:
                    dE_loop.append(i)

            #self.nimages=len(iter_loop)
            if self.is_atom_config:
                if len(lattice_loop)!=self.nimages or len(position_loop)!=self.nimages:
                    print("Warning!, loop not equal to nimages!!!")
                    return
            else:
                if len(lattice_loop)!=self.nimages or len(position_loop)!=self.nimages or len(force_loop)!=self.nimages:
                    print("Warning!, loop not equal to nimages!!!")
                    return

            '''be careful! the loop only consider about NVT with MD type 6'''
            for j in range(self.nimages):
                '''NOTICE！ the MOVEMENT Etot should not be Ep! we use Ei_sum as Etot, it has a shift between Ep! '''
                tmp=0
                self.etot.append(float(file_tmp[iter_loop[j]].split()[9])) #Ep
                for i in range(lattice_loop[j]+1,lattice_loop[j]+4):
                    for ii in range(0, 3):
                        self.lattice[j][tmp].append(float(file_tmp[i].split()[ii]))
                    tmp = tmp + 1
                tmp = 0
                for i in range(position_loop[j]+1,position_loop[j]+self.atom_num+1):
                    if j==0 :
                        self.eles_list.append(int(file_tmp[i].split()[0]))
                    for ii in range(1, 4):
                        self.atom_position[j][tmp].append(float(file_tmp[i].split()[ii]))
                    tmp = tmp + 1
                tmp=0
                for i in range(force_loop[j]+1,force_loop[j]+self.atom_num+1):
                    for ii in range(1,4):
                        self.atom_force[j][tmp].append(-float(file_tmp[i].split()[ii]))
                    tmp=tmp+1
                tmp=0
                if len(dE_loop) > 0:
                    self.dE.append(float(file_tmp[dE_loop[j]].split()[-1]))
                    for i in range(dE_loop[j]+1,dE_loop[j]+self.atom_num+1):
                        self.atom_energy[j][tmp].append(float(file_tmp[i].split()[1]))
                        tmp=tmp+1

        return

    def OUTCARloader(self,path):
        tmp=0
        mag_loop=[]
        lattice_loop=[]
        position_loop=[]
        force_loop=[]
        Etot_loop=[]
        vol_loop=[]
        length_loop=[]
        self.nimages=int(os.popen('grep POSITION {}|wc -l'.format(path)).read())
        with open(path,'r') as f:
            file_tmp = f.readlines()

            for i in range(len(file_tmp)):
                try:
                    file_tmp[i].split()[0]
                except:
                    continue

                if file_tmp[i].split()[0].count('POSITION')==1:
                    position_loop.append(i)
                if file_tmp[i].split()[0].count('direct')==1:
                    lattice_loop.append(i)
                if file_tmp[i].split()[0].count('POSITION')==1:
                    force_loop.append(i)
                if file_tmp[i].split()[0].count('magnetization')==1:
                    mag_loop.append(i)
                if file_tmp[i].split('=')[0].count('energy  without entropy')==1:
                    Etot_loop.append(i)
                if file_tmp[i].split(':')[0].count('volume of cell')==1:
                    vol_loop.append(i)
                if file_tmp[i].split()[0].count('length')==1:
                    length_loop.append(i)

            num=0
            for i in range(position_loop[0],position_loop[0]+100000):
                num+=1
                if file_tmp[i].split()[0].count('total')==1:
                    self.atom_num=num-4
                    break


            #self.nimages=len(iter_loop)

            '''a really strange thing is, at the end of OUTCAR, it will find a add magnetization and total-charge?'''
            lattice_loop.pop(0)
            mag_loop.pop()
            if len(lattice_loop)!=self.nimages or len(position_loop)!=self.nimages or len(force_loop)!=self.nimages:
                print("Warning!, loop not equal to nimages!!!")
                return

            unconv_list=self.image_conv_judgment(self.OSZICAR_dir)

            if unconv_list.count(False)!=0:
                nimage_tmp=self.nimages-unconv_list.count(False)
            else:
                nimage_tmp=self.nimages
            self.lattice = [[[],[],[]]for i in range(nimage_tmp)]
            self.atom_position = [[[] for i in range(self.atom_num)] for i in range(nimage_tmp)]
            self.cartesian_position=[[[] for i in range(self.atom_num)] for i in range(nimage_tmp)]
            self.atom_force=[[[] for i in range(self.atom_num)] for i in range(nimage_tmp)]
            self.mag=[[] for i in range(nimage_tmp)]
            self.vol=[]
            self.length_cell=[[]for i in range(nimage_tmp)]

            n_iter=0
            for j in range(self.nimages):
                if  unconv_list[j]==False:
                    continue
                '''NOTICE！ the MOVEMENT Etot should not be Ep! we use Ei_sum as Etot, it has a shift between Ep! '''
                self.etot.append(float(file_tmp[Etot_loop[j]].split('=')[-1]))
                for i in range(lattice_loop[j]+1,lattice_loop[j]+4):
                    for ii in range(0, 3):
                        self.lattice[n_iter][tmp].append(float(file_tmp[i].split()[ii]))
                    tmp = tmp + 1
                tmp = 0
                for i in range(position_loop[j]+2,position_loop[j]+self.atom_num+2):
                    # if j==0 :
                    #     self.eles_list.append(int(file_tmp[i].split()[0]))
                    for ii in range(3):
                        self.cartesian_position[n_iter][tmp].append(float(file_tmp[i].split()[ii]))
                    tmp = tmp + 1
                tmp=0
                for i in range(force_loop[j]+2,force_loop[j]+self.atom_num+2):
                    for ii in range(3,6):
                        self.atom_force[n_iter][tmp].append(float(file_tmp[i].split()[ii]))
                    tmp=tmp+1
                tmp=0
                for i in range(mag_loop[j]+4,mag_loop[j]+self.atom_num+4):
                    self.mag[n_iter].append(float(file_tmp[i].split()[-1]))
                # self.dE.append(float(file_tmp[dE_loop[j]].split()[-1]))
                self.vol.append(float(file_tmp[vol_loop[j]].split(':')[-1]))
                for i in range(3):
                    self.length_cell[n_iter].append(float(file_tmp[length_loop[j+1]+1].split()[i]))
                n_iter+=1

            self.nimages=nimage_tmp

        return

    def spin_judgement(self):
        for i in range(self.nimages):
            for j in range(self.atom_num):
                if self.mag[i][j]>0.5 :
                    self.eles_list[i][j]+=1
                if self.mag[i][j]<-0.5:
                    self.eles_list[i][j]-=1
        return


    def image_conv_judgment(self,path):
        iter_loop=[]
        unconv_list=[]
        with open(path,'r') as f:
            file_tmp=f.readlines()
            for i in range(len(file_tmp)):
                if file_tmp[i].split()[1].count('T=') == 1:
                    iter_loop.append(i)

        for i in range(len(iter_loop)):
            if i ==0:
                iter=iter_loop[i]-1
            else:
                iter=iter_loop[i]-iter_loop[i-1]
            if iter >= self.NELM:
                unconv_list.append(False)
            else:
                unconv_list.append(True)

        return unconv_list

    def coordinate2cartesian(self):  # 需要转成矩阵乘法以提速
        for j in range(self.nimages):
            for i in range(self.atom_num):
                for ii in range(3):
                    self.cartesian_position[j][i].append(
                        float(self.atom_position[j][i][0]) * float(self.lattice[j][0][ii])
                        + float(self.atom_position[j][i][1]) * float(self.lattice[j][1][ii])
                        + float(self.atom_position[j][i][2]) * float(self.lattice[j][2][ii]))
        return

    def cartesian2coordinate(self):
        '''
        consider cartesian position as array(A) (atom_num*3)
        coordinate position as array(a) (atom_num*3)
        lattice as array(L) (3*3)
        A=aL
        a=AL^-1
        '''
        tmp=0

        for j in range(self.nimages):
            A=np.array(self.cartesian_position[j])
            L=np.array(self.lattice[j])
            a=np.dot(A,np.linalg.inv(L))
            self.atom_position[j]=a.tolist()

        return

    def out_extxyz(self,path,select_index):
        '''
        out as a extxyz, for ase input
        The total energy and force will include
        the default format is element/position_x/position_y/position_z/force_x/force_y/force_z
        the unit is Ångström
        '''
        
        file_op_type = "w" if self.is_atom_config else "a+"
        with open(path,file_op_type) as f:
            for j in range(self.nimages):
                # if select_index.count(j):
                    f.write('{}\n'.format(self.atom_num))
                    # f.write('{}\n'.format(self.atom_num[j]))
                    f.write('pbc="T T T" Lattice="')
                    Lattice_value = ""
                    for i in range(3):
                        for ii in range(3): #" ".join(str(item) for item in value)
                            Lattice_value += '{} '.format(self.lattice[j][i][ii])
                    f.write(Lattice_value.strip())
                    #f.write('" Properties=species:S:1:pos:R:3:forces:R:3:energies:R:1 energy={} pbc="T T T"\n'.format(self.etot[j]-self.dE[j]))
                    if self.is_atom_config:
                        f.write('" Properties=species:S:1:pos:R:3\n')
                    else:
                        f.write('" Properties=species:S:1:pos:R:3:forces:R:3 energy={}\n'.format(self.etot[j]))
                    # for i in range(self.atom_num):
                    for i in range(int(self.atom_num)):
                        # f.write('{}      '.format(self.eles_type[self.eles_list[i]]))
                        f.write('{}      '.format(self.element_table[self.eles_list[i]]))
                        # f.write('{}      '.format(self.element_table[self.eles_list[j][i]]))
                        for ii in range(3):
                            f.write('{}      '.format(self.cartesian_position[j][i][ii]))

                        if not self.is_atom_config:
                            for ii in range(3):
                                f.write('{}      '.format(self.atom_force[j][i][ii]))
                        #f.write('{}'.format(self.atom_energy[j][i][0]))
                        f.write('\n')
        return


    def out_MOVEMENT(self,path):
        '''
        out as MD_DETAIL=6(NVT)
        '''
        with open(path,'w') as f:
            for j in range(self.nimages):
                f.write(' {}  atoms,Iteration (fs) = 0.01, Etot,Ep,Ek (eV) = {} {} 0.0\n'.format(self.atom_num,self.etot[j],self.etot[j]))
                f.write(' MD_INFO: METHOD(1-VV,2-NH,3-LV,4-LVPR,5-NHRP) TIME(fs) TEMP(K) DESIRED_TEMP(K) AVE_TEMP(K) TIME_INTERVAL(fs) TOT_TEMP(K)\n')
                f.write('             6    0.2000000000E+01   0.44759E+03   0.25000E+03   0.44759E+03   0.20000E+03   0.44759E+03\n')
                f.write(' Lattice vector (Angstrom)\n')
                for i in range(3):
                    f.write('   {}    {}    {}\n'.format(self.lattice[j][i][0],self.lattice[j][i][1],self.lattice[j][i][2]))
                f.write(' Position (normalized), move_x, move_y, move_z\n')
                for i in range(self.atom_num):
                    f.write('   {}    {}     {}     {}    1   1   1\n'.format(self.eles_list[i],self.atom_position[j][i][0],self.atom_position[j][i][1],self.atom_position[j][i][2]))
                f.write(' Force (-force, eV/Angstrom)\n')
                for i in range(self.atom_num):
                    f.write('   {}    {}     {}     {}\n'.format(self.eles_list[i],-self.atom_force[j][i][0],-self.atom_force[j][i][1],-self.atom_force[j][i][2]))
                f.write('Atomic-Energy, Etot(eV),E_nonloc(eV),Q_atom:dE(eV)=  0.0\n')
                for i in range(self.atom_num):
                    f.write('   {}    {}\n'.format(self.eles_list[i],self.etot[j]/self.atom_num))
                f.write(' -------------------------------------------------\n')
        return

    def out_npz(self,path):
        eles_list=np.array(self.eles_list)
        eles_list=eles_list.reshape(1,-1)
        eles_list=eles_list.repeat(1000,axis=0)
        etot=np.array(self.etot)
        dE=np.array(self.dE)
        energy=np.array(etot-dE)
        cartesian_position=np.array(self.cartesian_position)
        lattice=np.array(self.lattice)
        force=np.array(self.atom_force)
        nxyz_data=np.dstack((eles_list,cartesian_position))

        np.savez(path,nxyz=nxyz_data,energy=energy,lattice=lattice,force=force)
        return


    def out_data_dE(self,path):
        '''
        for MOVEMENT2runner
        :param path:
        :return:
        '''
        with open(path,'w') as f:
            for j in range(self.nimages):
                f.write('begin\n')
                f.write('command trans from MOVEMENT\n')
                for i in range(3):
                    f.write('lattice   {}   {}   {}\n'.format(self.lattice[j][i][0]*ANGSTROM2BOHR,self.lattice[j][i][1]*ANGSTROM2BOHR,self.lattice[j][i][2]*ANGSTROM2BOHR))
                for i in range(self.atom_num):
                    f.write('atom   {}   {}   {}   {}   0.00000   0.00000   {}   {}   {}\n'.format(self.cartesian_position[j][i][0]*ANGSTROM2BOHR,self.cartesian_position[j][i][1]*ANGSTROM2BOHR,
                                                                                                   self.cartesian_position[j][i][2]*ANGSTROM2BOHR,self.element_table[self.eles_list[i]],
                                                                                                   self.atom_force[j][i][0]*EV2HA/ANGSTROM2BOHR,self.atom_force[j][i][1]*EV2HA/ANGSTROM2BOHR,
                                                                                                   self.atom_force[j][i][2]*EV2HA/ANGSTROM2BOHR))

                f.write('energy   {}\n'.format((self.etot[j]-self.dE[j])*EV2HA))
                f.write('charge   0.00000\n')
                f.write('end\n')
        return

    def out_data_Etot(self,path):
        '''
        for vasp2runner
        :param path:
        :return:
        '''
        with open(path,'w') as f:
            for j in range(self.nimages):
                f.write('begin\n')
                f.write('command trans from MOVEMENT\n')
                for i in range(3):
                    f.write('lattice   {}   {}   {}\n'.format(self.lattice[j][i][0]*ANGSTROM2BOHR,self.lattice[j][i][1]*ANGSTROM2BOHR,self.lattice[j][i][2]*ANGSTROM2BOHR))
                for i in range(self.atom_num):
                    if self.is_charge==False:
                        f.write('atom   {}   {}   {}   {}   0.00000   0.00000   {}   {}   {}\n'.format(self.cartesian_position[j][i][0]*ANGSTROM2BOHR,self.cartesian_position[j][i][1]*ANGSTROM2BOHR,
                                                                                                   self.cartesian_position[j][i][2]*ANGSTROM2BOHR,self.element_table[self.eles_list[i]],
                                                                                                   self.atom_force[j][i][0]*EV2HA/ANGSTROM2BOHR,self.atom_force[j][i][1]*EV2HA/ANGSTROM2BOHR,
                                                                                                   self.atom_force[j][i][2]*EV2HA/ANGSTROM2BOHR))
                    else :
                        f.write('atom   {}   {}   {}   {}   {}   0.00000   {}   {}   {}\n'.format(self.cartesian_position[j][i][0]*ANGSTROM2BOHR,self.cartesian_position[j][i][1]*ANGSTROM2BOHR,
                                                                                                   self.cartesian_position[j][i][2]*ANGSTROM2BOHR,self.element_table[self.eles_list[i]],self.charge[j][i],
                                                                                                   self.atom_force[j][i][0]*EV2HA/ANGSTROM2BOHR,self.atom_force[j][i][1]*EV2HA/ANGSTROM2BOHR,
                                                                                                   self.atom_force[j][i][2]*EV2HA/ANGSTROM2BOHR))
                f.write('energy   {}\n'.format(self.etot[j]*EV2HA))
                f.write('charge   {}\n'.format(self.charge_tot[j]))
                f.write('end\n')
        return

    def out_data_test(self,path):
        with open(path,'w') as f:
            for j in range(self.nimages):
                f.write('begin\n')
                f.write('command trans from MOVEMENT\n')
                for i in range(3):
                    f.write('lattice   {}   {}   {}\n'.format(self.lattice[j][i][0],self.lattice[j][i][1],self.lattice[j][i][2]))
                for i in range(self.atom_num):
                    f.write('atom   {}   {}   {}   {}   0.00000   0.00000   {}   {}   {}\n'.format(self.cartesian_position[j][i][0],self.cartesian_position[j][i][1],
                                                                                                   self.cartesian_position[j][i][2],self.element_table[self.eles_list[i]],
                                                                                                   self.atom_force[j][i][0],self.atom_force[j][i][1],
                                                                                                   self.atom_force[j][i][2]))
                f.write('energy   {}\n'.format(self.etot[j]-self.dE[j]))
                f.write('charge   0.00000\n')
                f.write('end\n')
        return

    def select_tmp(self,path):
        select_list=[1,18,30,59,61,62,71,92,107,123,126,146,160,164,166,183,188,192,193]
        with open(path,'w') as f:
            for j in range(self.nimages):
                if select_list.count(j+1)==1:
                    f.write('begin\n')
                    f.write('command trans from MOVEMENT\n')
                    for i in range(3):
                        f.write('lattice   {}   {}   {}\n'.format(self.lattice[j][i][0]*ANGSTROM2BOHR,self.lattice[j][i][1]*ANGSTROM2BOHR,self.lattice[j][i][2]*ANGSTROM2BOHR))
                    for i in range(self.atom_num):
                        f.write('atom   {}   {}   {}   {}   0.00000   0.00000   {}   {}   {}\n'.format(self.cartesian_position[j][i][0]*ANGSTROM2BOHR,self.cartesian_position[j][i][1]*ANGSTROM2BOHR,
                                                                                                       self.cartesian_position[j][i][2]*ANGSTROM2BOHR,self.element_table[self.eles_list[i]],
                                                                                                       self.atom_force[j][i][0]*EV2HA/ANGSTROM2BOHR,self.atom_force[j][i][1]*EV2HA/ANGSTROM2BOHR,
                                                                                                       self.atom_force[j][i][2]*EV2HA/ANGSTROM2BOHR))
                    f.write('energy   {}\n'.format((self.etot[j]-self.dE[j])*EV2HA))
                    f.write('charge   0.00000\n')
                    f.write('end\n')
        return

    def split2POSCAR(self,path):
        for i in range(self.nimages):
            with open(path+'_{}'.format(i+1),'w') as f:
                for tmp in self.heads:
                    f.write(str(tmp))
                f.write('Direct\n')
                for ii in range(self.atom_num):
                    f.write('  {}  {}  {}\n'.format(self.fcoord[i][ii][0],self.fcoord[i][ii][1],self.fcoord[i][ii][2]))
        return

def get_eles_list(eles_list,eles_num_list):
    '''
    for get the eles_list from the POSCAR format 
    '''
    tmp=[]
    for i in range(len(eles_list)):
        for ii in range(eles_num_list[i]):
            tmp.append(element_table.index(eles_list[i]))

    return tmp

'''
description: 
    configuration to xyz format
param {str} input_file
param {str} file_type  "POSCAR", "OUTCAR"
return {*}
author: wuxingxing
'''
def POSCAR_OUTCAR2xyz(input_file:str, out_file:str, file_type:str):
    reader=Structure(input_file,type=file_type)
    reader.out_extxyz(out_file,None)

def OUTCAR2xyz(path):
    '''
    A flow for deal with multi files
    '''
    file_list=os.listdir(path)
    input_list=['OUTCAR','OSZICAR','POSCAR']
    for i in file_list:
        sub_list=os.listdir(path+i)
        for ii in sub_list:
            path_list = []
            for j in input_list:
                path_list.append(path+i+'/'+ii+'/'+j)
            a=Structure(path_list[0],path_list[1],type='OUTCAR')
            b=Structure(path_list[2],type='POSCAR')
            tmp = get_eles_list(b.eles_list, b.eles_num_list)
            a.eles_list = np.array(tmp).reshape(1, -1).repeat(a.nimages, axis=0).tolist()
            a.cartesian2coordinate()
            a.out_extxyz(path+i+'/'+ii+'/'+i+'.xyz')
            print(path+i+'/'+ii+' finish!')
    return

def OUTCAR2xyz_dp(path,train_ratio):
    a = glob('{}/*/*/OUTCAR'.format(path))
    b = glob('{}/*/*/*/OUTCAR'.format(path))
    c = glob('{}/*/*/*/*/OUTCAR'.format(path))
    total = a + b + c

    reader=[]
    for i in range(len(total)):
        reader=Structure(total[i],type='OUTCAR')
        length=reader.nimages
        train_index=list(np.linspace(0,int(length*train_ratio)-1,num=int(length*train_ratio)))
        valid_index=list(np.linspace(int(length*train_ratio),length,num=int(length)-int(length*train_ratio)+1))
        reader.out_extxyz('./train_data.xyz',train_index)
        reader.out_extxyz('./valid_data.xyz',valid_index)

def atomconfig2xyz(config_file, save_file):
    reader = Structure(config_file, type="MOVEMENT")
    reader.coordinate2cartesian()
    reader.out_extxyz(save_file, None)

def convert_mvmfiles_to_xyz(mvm_file_list:list, train_save_path:str, valid_save_path:str, valid_shuffle:bool=False, ratio:float=0.2):
    mvm_classed_list = classify_mvm(mvm_file_list)
    # saperated movements to training and valid by random or last 20%
    write_train_valid_movement(train_save_path, valid_save_path, mvm_classed_list, ratio, valid_shuffle)
        
def classify_mvm(mvm_file_list: list[str]):
    mvm_sorted = {}
    mvm_dict = {}
    mvm_obj = []
    for i, mvm_file in enumerate(mvm_file_list):
        mvm = MOVEMENT(mvm_file)
        mvm_obj.append(mvm)
        atom_type = mvm.image_list[0].atom_type
        atom_type_num_list = mvm.image_list[0].atom_type_num
        key1 = "_".join(str(item) for item in atom_type_num_list)
        key2 = '_'.join(str(item) for item in atom_type)
        mvm_dict[i] = "{}_{}".format(key1, key2)
    tmp = sorted(mvm_dict.items(), key = lambda x: len(x[1]), reverse=True)
    for t in tmp:
        if t[1] not in mvm_sorted.keys():
            mvm_sorted[t[1]] = [{"file": mvm_file_list[t[0]], "obj":mvm_obj[t[0]]}]
        else:
            mvm_sorted[t[1]].append({"file": mvm_file_list[t[0]], "obj":mvm_obj[t[0]]})
    return mvm_sorted

def write_train_valid_movement(train_save_path, valid_save_path, mvm_sorted:dict, ratio:float, valid_shuffle:bool):
    # separate mvm files to train_movement and valid_movement
    train_file_list = []
    valid_file_list = []
    for i, mvm_type_key in enumerate(mvm_sorted.keys()):
        mvm_list = mvm_sorted[mvm_type_key]
        tmp_train = "train_mvm_{}_{}".format(mvm_type_key, i)
        tmp_valid = "valid_mvm_{}_{}".format(mvm_type_key, i)

        for mvm in mvm_list:
            train_indexs, valid_indexs = random_index(mvm["obj"].image_nums, ratio, valid_shuffle)
    
            with open(tmp_train, 'a') as af:
                for j in train_indexs:
                    for line in mvm["obj"].image_list[j].content:
                        af.write(line)

            with open(tmp_valid, 'a') as af:
                for j in valid_indexs:
                    for line in mvm["obj"].image_list[j].content:
                        af.write(line)
            print("Tmp file {} separted to train and valid momvement done (valid shuffle {})!".format(mvm['file'], valid_shuffle))
        
        train_file_list.append(tmp_train)
        valid_file_list.append(tmp_valid)
        
    # convert movement to xyz 
    mvm2xyz(train_file_list, train_save_path)
    mvm2xyz(valid_file_list, valid_save_path)
    # delete tmp files
    for mvm in train_file_list:
        os.remove(mvm)
    for mvm in valid_file_list:
        os.remove(mvm)

def mvm2xyz(mvm_list:list[str], file_name:str):
    # if exist file_name, delete them
    if os.path.exists(file_name):
        os.remove(file_name)
        
    for mvm in mvm_list:
        a=Structure(path=mvm, type="MOVEMENT")
        a.coordinate2cartesian()
        a.out_extxyz(file_name,None) # Write to the file in append mode
        print("{} convert to xyz format done!".format(mvm))

'''
description: 
    seperate a int list [0, end) with a ratio
param {int} image_nums
param {float} ratio
param {bool} is_random, random seperate
return {*}
author: wuxingxing
'''
def random_index(image_nums:int, ratio:float, is_random:bool=False):
    arr = np.arange(image_nums)
    if is_random is True:
        np.random.shuffle(arr)
    split_idx = math.ceil(image_nums*ratio)
    train_data = arr[:split_idx]
    test_data = arr[split_idx:]
    return sorted(train_data), sorted(test_data)

def conert_mvm_files_to_xyz(movements, ratio, random_valid):
    # random_valid = False # if True, the valid set will be random selected from MOVMENTs
    # ratio = 0.8 # the training data ratio
    # movements = ["./lisi/MOVEMENT", "./si/500k/MOVEMENT", "./si/800k/MOVEMENT"]
    convert_mvmfiles_to_xyz(movements, "train.xyz", "test.xyz", random_valid, ratio)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    print("Note that this script only supports conversions of the same atomic type and quantity!")


    parser.add_argument('-t', '--work_type', help="specify input work type:\n \
                        \t\t1 for mvms to xyz, with train.xyz and test.xyz output files\n\
                        \t\t2 for config (POSCAR, OUTCAR, atom.config) to xyz with default 'model.xyz' output file (or specified name by the -o param)\n", 
                        type=int, default=1)
    parser.add_argument('-w', '--work_dir', help='specify work dir, default is current dir', type=str, default=os.getcwd())
    parser.add_argument('-m', '--mvms', help='specify input movement file paths', nargs='+', type=str, default=None)
    parser.add_argument('-r', '--ratio', help='Specify the ratio between the training set and valid set', type=float, default=0.8)
    parser.add_argument('-s', '--shuffle', help='the valid set will randomly selected in each movements with the ratio if true', type=bool, default=False)
    parser.add_argument('-c', '--config', help='specify input config file path', type=str, default=None)
    parser.add_argument('-o', '--out_file_name', help='specify stored file name', type=str, default="model.xyz")
    args = parser.parse_args()
    cwd = os.getcwd()
    os.chdir(args.work_dir)
    if args.work_type == 1:
        conert_mvm_files_to_xyz(args.mvms, args.ratio, args.shuffle)
        print('movements to train.xyz and valid.xyz done!')

    if args.work_type == 2:
        if "config".upper() in args.config.upper():
            # atom.config to xyz format
            atomconfig2xyz(args.config, args.out_file_name)
        elif "outcar".upper() in args.config.upper():
            POSCAR_OUTCAR2xyz(args.config, args.out_file_name, "OUTCAR")
        elif "poscar".upper() in args.config.upper():
            POSCAR_OUTCAR2xyz(args.config, args.out_file_name, "POSCAR")
        print('{} to {} done!'.format(args.config.upper(), args.out_file_name))
    
    os.chdir(cwd)