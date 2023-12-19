"""
This is a movement utility class that encapsulates movements into an image list and adds some methods 
for manipulating the movement, such as interval-based image extraction.
"""
import numpy as np
import os
import sys
import shutil
import re
from collections import Counter

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

    def save_to_2parts(self, mid, write_patten="w", part_1=None, part_2=None):
        if part_1 is None:
            part_1 = "part_1_movement_{}".format(mid)
        if part_2 is None:
            part_2 = "part_2_movement_{}".format(self.image_nums-mid)
        with open(part_1, write_patten) as wf:
            for i in range(0, mid):
                for line in self.image_list[i].content:
                    wf.write(line)

        with open(part_2, write_patten) as wf:
            for i in range(mid, self.image_nums):
                for line in self.image_list[i].content:
                    wf.write(line)

    def separate_mvms():
        os.chdir("/data/home/wuxingxing/codespace/PWMLFF/pwmat_mlff_workdir/alloy")
        train_movement_file = [
        "./mvm_files/movement_0",   
        "./mvm_files/movement_11",  
        "./mvm_files/movement_14",  
        "./mvm_files/movement_17",  
        "./mvm_files/movement_2",   
        "./mvm_files/movement_22",  
        "./mvm_files/movement_25",  
        "./mvm_files/movement_28",  
        "./mvm_files/movement_30",  
        "./mvm_files/movement_33",  
        "./mvm_files/movement_36",  
        "./mvm_files/movement_5",  
        "./mvm_files/movement_8",
        "./mvm_files/movement_1",   
        "./mvm_files/movement_12",  
        "./mvm_files/movement_15",  
        "./mvm_files/movement_18",  
        "./mvm_files/movement_20",  
        "./mvm_files/movement_23",  
        "./mvm_files/movement_26",  
        "./mvm_files/movement_29",  
        "./mvm_files/movement_31",  
        "./mvm_files/movement_34",  
        "./mvm_files/movement_37",  
        "./mvm_files/movement_6",  
        "./mvm_files/movement_9",
        "./mvm_files/movement_10",  
        "./mvm_files/movement_13",  
        "./mvm_files/movement_16",  
        "./mvm_files/movement_19",  
        "./mvm_files/movement_21",  
        "./mvm_files/movement_24",  
        "./mvm_files/movement_27",  
        "./mvm_files/movement_3",   
        "./mvm_files/movement_32",  
        "./mvm_files/movement_35",  
        "./mvm_files/movement_4",   
        "./mvm_files/movement_7",
        "./mvm_init/movement_0",
        "./mvm_init/movement_10",
        "./mvm_init/movement_12",
        "./mvm_init/movement_14",
        "./mvm_init/movement_16",
        "./mvm_init/movement_18",
        "./mvm_init/movement_2",
        "./mvm_init/movement_21",
        "./mvm_init/movement_23",
        "./mvm_init/movement_25",
        "./mvm_init/movement_4",
        "./mvm_init/movement_6",
        "./mvm_init/movement_8",
        "./mvm_init/movement_1",
        "./mvm_init/movement_11",
        "./mvm_init/movement_13",
        "./mvm_init/movement_15",
        "./mvm_init/movement_17",
        "./mvm_init/movement_19",
        "./mvm_init/movement_20",
        "./mvm_init/movement_22",
        "./mvm_init/movement_24",
        "./mvm_init/movement_3",
        "./mvm_init/movement_5",
        "./mvm_init/movement_7",
        "./mvm_init/movement_9"
    ]
        train_file = "train_mvm"
        valid_file = "valid_mvm"
        sperate_log = "split_norand.log"
        is_shaffule = False
        ratio = 0
        with open(sperate_log, "w") as wf:
            wf.readline("is shaffule: {}\t ratio is: {}".format(is_shaffule, ratio))
        for mvm in train_movement_file:
            mvm = MOVEMENT(source_movement_file)
            mid = int(mvm.image_nums * ratio)
            mvm.save_to_2parts(mid, write_patten="a", part_1=train_file, part_2=valid_file)
            with open(sperate_log, "a") as wf:
                wf.readline("mvm: {}\t atomlist: {}\t train: {}\t valid:{}\n"\
                            .format(mvm, mvm.image_list[0].atom_type, mid, mvm.image_nums-mid))    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='specify input movement filename', type=str, default='MOVEMENT')
    parser.add_argument('-s', '--savepath', help='specify stored directory', type=str, default='movement_save')
    parser.add_argument('-wt', '--work_type', help='specify work type: 0 for interval cut, 1 for range cut, 2 for split 2 parts of movement}', type=int, default=0)
    
    parser.add_argument('-t', '--interval', help='specify interval', type=int, default=10)
    parser.add_argument('-f', '--first_range', help='specify start of range', type=int, default=0)
    parser.add_argument('-e', '--end_range', help='specify end of range', type=int, default=1)
    parser.add_argument('-m', '--mid', help='specify split Position of 2 parts', type=int, default=100)
    args = parser.parse_args()

    source_movement_file = args.input
    mvm = MOVEMENT(source_movement_file)
    if args.work_type == 0:
        mvm.save_image_interval(args.savepath, interval=args.interval)
    elif args.work_type == 1:
        mvm.save_image_range(args.savepath, start= args.first_range, end=args.end_range)
    elif args.work_type == 2:
        mvm.save_to_2parts(args.mid)