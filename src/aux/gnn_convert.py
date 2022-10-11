
import os
import re
import numpy as np
import pymatgen as pym


# import nff.data as d # for data transform to the graph nff model
#from pymatgen.core import Structure

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

class Structure:
    '''
    Now, the code can only to deal with one type of MOVEMENT.
    MOVEMENT: Energy(eV) Force(eV/Angstrom) lattice(Angstrom)
    input.data: Energy(Ha) Force(Ha/bohr) lattice(bohr)
    '''

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

    def __init__(self, path,path_2='./POSCAR',type='MOVEMENT',is_charge=False):
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
        self.mag=[]
        self.charge = []
        self.charge_tot = []
        nimages = 0

        if type == 'MOVEMENT':
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
            self.OSZICAR_dir=path_2
            self.OUTCARloader(path)
        if type=='POSCAR':
            self.POSCARloader(path)


    #def unit_conversion(self):
        #self.lattice=self.lattice*ANGSTROM2BOHR
        #self.etot=self.etot*EV2HA

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
        structure=pym.Structure.from_file(path)
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
        self.nimages=int(os.popen('grep Iter {}|wc -l'.format(path)).read())
        with open(path,'r') as f:
            file_tmp = f.readlines()
            self.atom_num = int(file_tmp[0].split()[0])
            self.lattice = [[[],[],[]]for i in range(self.nimages)]
            self.atom_position = [[[] for i in range(self.atom_num)] for i in range(self.nimages)]
            self.cartesian_position=[[[] for i in range(self.atom_num)] for i in range(self.nimages)]
            self.atom_force=[[[] for i in range(self.atom_num)] for i in range(self.nimages)]
            for i in range(len(file_tmp)):
                #if re.match(file_tmp[i].split()[0], '{}'.format(self.atom_num)) != None:
                if file_tmp[i].split()[0].count('{}'.format(self.atom_num))==1:
                    iter_loop.append(i)
                if file_tmp[i].split()[0].count('Lattice')==1:
                    lattice_loop.append(i)
                # if file_tmp[i].split()[0].count('Position')==1:
                if re.match(file_tmp[i].split()[0],'Position')!=None:
                    position_loop.append(i)
                if file_tmp[i].split()[0].count('Force')==1:
                    force_loop.append(i)
                if file_tmp[i].split(',')[0].count('Atomic-Energy')==1:
                    dE_loop.append(i)

            #self.nimages=len(iter_loop)

            if len(lattice_loop)!=self.nimages or len(position_loop)!=self.nimages or len(force_loop)!=self.nimages:
                print("Warning!, loop not equal to nimages!!!")
                return

            '''be careful! the loop only consider about NVT with MD type 6'''
            for j in range(self.nimages):
                '''NOTICE！ the MOVEMENT Etot should not be Ep! we use Ei_sum as Etot, it has a shift between Ep! '''
                #print (file_tmp[iter_loop[j]].split())
                self.etot.append(float(file_tmp[iter_loop[j]].split()[9]))
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
                self.dE.append(float(file_tmp[dE_loop[j]].split()[-1]))

        return

    def OUTCARloader(self,path):
        tmp=0
        mag_loop=[]
        lattice_loop=[]
        position_loop=[]
        force_loop=[]
        Etot_loop=[]
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

    def out_extxyz(self,path):
        '''
        out as a extxyz, for ase input
        The total energy and force will include
        the default format is element/position_x/position_y/position_z/force_x/force_y/force_z
        the unit is Ångström
        ''' 
        
        #print ("number of totol image:",self.nimage)
        with open(path,'w') as f:
            for j in range(self.nimages):

                if j%100 == 0:
                    print ("writing image ", j)
                
                f.write('{}\n'.format(self.atom_num))
                f.write('Lattice="')
                for i in range(3):
                    for ii in range(3):
                        f.write('{} '.format(self.lattice[j][i][ii]))
                f.write('" Properties=species:S:1:pos:R:3:forces:R:3 energy={} pbc="T T T"\n'.format(self.etot[j]-self.dE[j]))
                # f.write('" Properties=species:S:1:pos:R:3:forces:R:3 energy={} pbc="T T T"\n'.format(self.etot[j]))
                for i in range(self.atom_num):
                    f.write('{}      '.format(self.element_table[self.eles_list[i]]))
                    # f.write('{}      '.format(self.element_table[self.eles_list[j][i]]))
                    for ii in range(3):
                        f.write('{}      '.format(self.cartesian_position[j][i][ii]))
                    for ii in range(3):
                        f.write('{}      '.format(self.atom_force[j][i][ii]))
                    f.write('\n')
           
        print ("total number of images:",self.nimages) 
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

    def test(self):
        print(ANGSTROM2BOHR)

def get_eles_list(eles_list,eles_num_list):
    '''
    for get the eles_list from the POSCAR format 
    '''
    tmp=[]
    for i in range(len(eles_list)):
        for ii in range(eles_num_list[i]):
            tmp.append(element_table.index(eles_list[i]))
    
    return tmp

if __name__ == '__main__':
    '''
    path = './vasprun.xml'
    path_charge='./total_charge'
    a = Structure(path,path_charge,type='vasprun',is_charge=True)
    a.coordinate2cartesian()
    a.out_data_Etot('input.data')
    '''

    # path='./Li0.5CoO_test/10/OUTCAR'
    # path2='./Li0.5CoO_test/10/OSZICAR'
    # path3='./Li0.5CoO_test/10/POSCAR'
    # a= Structure(path,path2,type='OUTCAR')
    # b= Structure(path3,type='POSCAR')
    # tmp=get_eles_list(b.eles_list,b.eles_num_list)
    # a.eles_list=np.array(tmp).reshape(1,-1).repeat(a.nimages,axis=0).tolist()
    # a.cartesian2coordinate()
    # a.spin_judgement()
    # a.out_extxyz('./Li0.5CoO_test/10/Li0.5CoO.xyz')
    # a.coordinate2cartesian()
    # a.out_npz('./Cu1000/Cu1000.npz')
    #    a.out_data_test('input.data')
    #     a.out_extxyz('./CuO/CuO.xyz')

    # for MOVEMENT2xyz
    path='./PWdata/MOVEMENT'
    path_out='./PWdata/Cu.xyz'
    a=Structure(path,type='MOVEMENT')
    a.coordinate2cartesian()
    a.out_extxyz(path_out)  

