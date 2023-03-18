"""
    Adaptive trainer for DPKF
    2023.3, L. Wang  
"""

import os
import pathlib
import sys
import subprocess 
import multiprocessing as mp 
import numpy as np 

codepath = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(codepath)

sys.path.append(codepath+'/../PWmatMLFF')
sys.path.append(codepath+'/../aux')

#import dp_network

from poscar2lammps import p2l 
from slice import slice

class adaptive_trainer():
    
    def __init__(
                    self, 
                    model_num = 4, 
                    process_num = 1,     # num of process for trajectory generation
                    iter_num = 1,        # num of meta iteration 
                    temp = [], 
                    pressure = [],
                    traj_step = 300, 
                    ensemble = "nvt",  # npt, nvt 
                    kspacing = 0.16, 
                    silent_mode = True, 
                    num_select_per_group = 200,
                    psp_dir = "/share/psp/NCPP-SG15-PBE", 
                    node_num = 1, 
                    
                ):
        """
            DIR: train
                All the training stuff
            
            DIR: seed 
                seed configs for exploration
                there will be groups with different ratio, box, etc.
                num of atom is the same with in a group 
                
            DIR: explore 
                You need 3 directories
                * models: 1.ff 2.ff ... n.ff. Force fields 
                * results: candidates that will be used for train. 
                * subsys: all configurations to be explored 
                *** which contian sys1, sys2, sys3, ...         
                *** in each of them, generate trajectory, and 
            
        """
        
        self.num_cand_per_cfg = 10                 #  number of candidate per config
        self.silent_mode = silent_mode
        self.process_num = process_num

        self.num_select_per_group = num_select_per_group

        self.iter_num = iter_num
        # lammps related
        self.temperature = temp
        self.pressure = pressure
        self.ensemble = ensemble
        self.traj_step = traj_step
        # dir for sub systems explore
        # ./subsys/***
        
        # for traj generation
        self.ksapcing  = kspacing
        self.model_num = model_num

        self.working_dir = os.getcwd() 
        self.psp_dir = psp_dir

        self.train_path = self.working_dir+"/train"
        self.explore_path = self.working_dir+"/explore"
        self.seed_path = self.working_dir+"/seed"

        self.explore_model_dir  = self.explore_path + "/model"
        self.explore_result_dir = self.explore_path + "/result"
        self.explore_subsys_dir = self.explore_path + "/subsys"

        self.write_lmp_config = {   "init_config":"lmp.init",
                                    "ff_name":self.explore_model_dir+"/1.ff",
                                    "init_temp":300,
                                    "end_temp":300,
                                    "ensemble":self.ensemble,
                                    "timestep":0.001,
                                    "output_name":"traj",
                                    "is_traj":True,
                                    "traj_step":self.traj_step}
        
        # stats in each iteration 
        self.stats_list = []
        self.node_num = node_num

    def initialize(self):
        # housekeeping. Set up dirs etc. 

        if not os.path.exists(self.train_path):
            print("creating dir \'train\'")
            os.mkdir(self.train_path)
        
        if not os.path.exists(self.explore_path):
            print("creating dir \'explore\'")
            os.mkdir(self.explore_path)
        
        if not os.path.exists(self.seed_path):
            print("creating dir \'seed\'")
            os.mkdir(self.seed_path)

        if not os.path.exists(self.explore_model_dir):
            print("creating dir \'explore/model\'")
            os.mkdir(self.explore_model_dir)

        if not os.path.exists(self.explore_result_dir):
            print("creating dir \'explore/result\'")
            os.mkdir(self.explore_result_dir)

        if not os.path.exists(self.explore_subsys_dir):
            print("creating dir \'explore/subsys\'")
            os.mkdir(self.explore_subsys_dir)
        
        #print (os.path.exists(current_dir+"/explore"))
    
    """
       ****************************************** 
               selecting configurations 
       ****************************************** 
    """

    def select_init_config(self):
        
        """
            move all data from /seed to /subsys    
            add new data to /seed step by step 
            Note: will create directoroes in both /subsys and /result 
        """
        #remove all the subsys 
        #subprocess.run(["rm "+self.explore_subsys_dir+" * -r"],shell=True)
        
        # collect all dirs in /seed
        for a, b, c in os.walk(self.seed_path):
            dirs = b
            break
        
        for subdir in dirs:
            # create dirs in /subsys
            if not os.path.exists(self.explore_subsys_dir+"/"+subdir):
                os.mkdir(self.explore_subsys_dir+"/"+subdir)

            # create dirs in /result
            if not os.path.exists(self.explore_result_dir+"/"+subdir):
                os.mkdir(self.explore_result_dir+"/"+subdir)
            
            for a,b, c in os.walk(self.seed_path+"/"+subdir):
                cfg_names = c
                #print (cfg_names)
                break
            
            cfg_root = self.seed_path+"/"+subdir

            for cfg in cfg_names: 
                # all tempratures 
                for temp in self.temperature:
                    for pres in self.pressure:
                        
                        tmp_path_subsys = self.explore_subsys_dir+"/"+subdir+"/"+cfg + "_T="+str(temp) + "_P="+str(pres)
                        tmp_path_result = self.explore_result_dir+"/"+subdir+"/"+cfg + "_T="+str(temp) + "_P="+str(pres)
                        
                        if not os.path.exists(tmp_path_subsys):
                            os.mkdir(tmp_path_subsys)

                        #if not os.path.exists(tmp_path_result):
                        #    os.mkdir(tmp_path_result)
                        
                        # copy configuration
                        cmd = "cp "+cfg_root+"/"+cfg + " " + tmp_path_subsys+"/atom.config"
                        subprocess.run([cmd],shell=True)

                        # generate lmp.cfg
                        os.chdir(tmp_path_subsys)
                        subprocess.run(["config2poscar.x atom.config > /dev/null"], shell = True)
                        p2l(output_name = "lmp.init")
                        subprocess.run(["rm","atom.config","POSCAR"])

                        print("lmp input config generated in "+tmp_path_subsys)

                        write_cfg = self.write_lmp_config  
                        write_cfg["init_temp"] = temp
                        write_cfg["end_temp"] = temp 
                        write_cfg["init_pres"] = pres
                        write_cfg["end_pres"] = pres
                            
                        self.write_lmp_in(write_cfg)
            

    def write_lmp_in(self,write_config):

        # write lammps.in 
        out_name = "lammps.in"

        head  =  ["units           metal\n",
                   "boundary        p p p\n",
                   "atom_style      atomic\n",
                   "processors    * * *\n",
                   "neighbor        2.0 bin\n",
                   "neigh_modify    every 10 delay 0 check no\n"]
        
        init_config = write_config["init_config"]

        vline1 = "velocity         all create " 
        vline2 = " 500000 dist gaussian\n"

        dumpline1 = "dump             1 all custom 1 "
        dumpline2 = " id type x y z  vx vy vz fx fy fz\n"

        f = open(out_name,"w")
 
        for line in head:
            f.writelines(line)
        
        f.writelines("read_data   "+init_config+"\n")

        f.writelines("pair_style  pwmatmlff\n")
        #f.writelines("pair_coeff  * * 5 5 3 14 "+write_config["ff_name"]+"\n")
        f.writelines("pair_coeff  * * 5 5 "+str(self.model_num)+" 3 14 ")
        
        for i in range(1,self.model_num+1):
            write_config["ff_name"] = self.explore_model_dir + "/" + str(i) +  ".ff "
            f.writelines(write_config["ff_name"])

        f.writelines("\n")

        # initial velocity gen
        f.writelines(vline1+str(write_config["init_temp"])+vline2)
        
        f.writelines("timestep         "+str(write_config["timestep"]) + "\n") 

        if write_config["ensemble"] == "nvt":
            line = "fix              1 all nvt temp "
            line = line + str(write_config["init_temp"]) + " "
            line = line + str(write_config["end_temp"]) + " "
            line  = line + str(write_config["timestep"]*100) + "\n"
            f.writelines(line)
        elif write_config["ensemble"] == "npt":
            line = "fix              1 all npt temp "
            line = line + str(write_config["init_temp"]) + " "
            line = line + str(write_config["end_temp"]) + " "
            line  = line + str(write_config["timestep"]*100) + " "
            line = line + "tri " +  str(write_config["init_pres"]) + " " + str(write_config["end_pres"])+ " "+ str(write_config["timestep"]*100) + "\n"
            f.writelines(line)

        f.writelines("thermo_style     custom step pe ke etotal temp\n")
        f.writelines("thermo           1\n")

        f.writelines(dumpline1+write_config["output_name"]+dumpline2)
        
        # how many lammps steps 
        f.writelines("run              ")
        
        if write_config["is_traj"] is True:
            f.writelines(str(write_config["traj_step"])+"\n")
        else:
            f.writelines("0\n")

        f.close()

    """
       ****************************************** 
          running lmp to generate trajectories 
       ****************************************** 
    """
    
    def run_lmp_mp(self,tgt_dirs):
        for dir in tgt_dirs:
            print ("generating trajectory for " + dir)
            os.chdir(dir)

            if self.silent_mode is True:
                subprocess.run(["lmp_mpi -in lammps.in > /dev/null"], shell=True)   
            else:
                subprocess.run(["lmp_mpi -in lammps.in"], shell=True)   
            
            print("trajectory generated for "+dir+"\n")
            
            
    def run_lmp(self):
        """
            using LAMMPS to generate trajectories 
        """
        lmp_dirs = [] 
        lmp_dir_mp = [[] for i in range(self.process_num)] 

        # look for sub dirs
        for a, b, c in os.walk(self.explore_subsys_dir):
            # exclude those not at the lowest level 
            if len(b) == 0:
                lmp_dirs.append(a)

        num_dirs = len(lmp_dirs)

        print ("number of target dirs:", num_dirs)

        for i in range(num_dirs):
            lmp_dir_mp[i%self.process_num].append(lmp_dirs[i])
        
        # distribute across processes
        pool = mp.Pool(self.process_num)
        pool.map(self.run_lmp_mp,lmp_dir_mp) 

        num_success = 0 
        num_fail = 0
        num_cand = 0
        
        # collect stats  
        print ("*****************SUMMARY OF EXPLORATION*****************")

        for dir in lmp_dirs: 
            
            if not os.path.exists(os.path.join(dir,"explr.stat")):
                print("exploration stat not foudn in", dir)
                continue 
            
            f = open(os.path.join(dir,"explr.stat"),"r")
            raw = f.readlines()[0].split()  
            f.close()
            
            tmp_success = int(raw[0])
            tmp_cand = int(raw[1])
            tmp_fail = int(raw[2])
            tmp_total = tmp_cand+tmp_success+tmp_fail

            print("in",dir)
            print ("ratio of success:", float(tmp_success)/tmp_total)
            print ("ratio of candidate:", float(tmp_cand)/tmp_total)
            print ("ratio of failure", float(tmp_fail)/tmp_total)
            print("\n")

            num_success += int(raw[0])
            num_cand += int(raw[1])
            num_fail += int(raw[2])
        
        num_total = num_success+num_cand+num_fail 

        print ("********************************************************")

        print ("num of all img:", num_success+num_cand+num_fail)
        print ("ratio of success:", float(num_success)/num_total)
        print ("ratio of candidate:", float(num_cand)/num_total)
        print ("ratio of failure", float(num_fail)/num_total)
        
        print ("********************************************************\n")

    """
       ****************************************** 
                        run scf 
       ****************************************** 
    """

    def run_scf(self):
        """
            use selected structures for SCF. 
        """
        self.collect_cfgs() 
        
        self.write_scf_in_and_run() 
        
        self.collect_mvmt()

    def write_etot_input(self, dir, num_atom, norm_b_list):
        """
            write etot.input
        """
        from math import floor 

        psp_path = self.psp_dir

        mp_line = "mp_n123 = "

        round_up = lambda x: int(floor(x)+1) if x-floor(x)>0.5 else int(floor(x))

        if num_atom > 200:
            mp_line = "mp_n123 = 1 1 1 0 0 0"
        else:
            mp_line = mp_line + str(round_up(norm_b_list[0]/self.ksapcing)) + " "
            mp_line = mp_line + str(round_up(norm_b_list[1]/self.ksapcing)) + " "
            mp_line = mp_line + str(round_up(norm_b_list[2]/self.ksapcing)) + " "
            mp_line = mp_line + "0 0 0"
        
        psp_lines = ["in.psp1 = "+psp_path+"/Li.SG15.PBE.UPF\n","in.psp2 = "+psp_path+"/Si.SG15.PBE.UPF\n"] 

        # start
        """
            !!! Si Li pseudopotential on PWMAT only, at this moment 
        """ 
        with open(dir+"/etot.input","w") as file:
            file.writelines("4    1\n")
            file.writelines("job = scf\n")
            file.writelines("in.atom = atom.config\n")

            for line in psp_lines:
                file.writelines(line)
            
            #file.writelines("in.psp1 = Li.SG15.PBE.UPF\n")
            #file.writelines("in.psp2 = Si.SG15.PBE.UPF\n")
            
            file.writelines("accuracy = high\n")
            file.writelines("ecut = 50.0\n")
            file.writelines("wg_error = 0.0\n")
            file.writelines("e_error = 0.0001\n")
            file.writelines("rho_error = 0.0\n")
            file.writelines("out.wg = F\n")
            file.writelines("out.rho = F\n")
            file.writelines("out.vr = F\n")
            file.writelines("out.force = T\n")
            file.writelines("out.stress = T\n")
            file.writelines("out.mlmd = T\n")
            file.writelines("scf_iter0_1 = 6 4 3 0.0 0.1 2\n")
            file.writelines("scf_iter0_2 = 94 4 3 1.0 0.1 2\n")
            file.writelines("energy_decomp = T\n")
            file.writelines("energy_decomp_special2 = 2 0 0 1 1\n")
            file.writelines(mp_line)

    def calc_recip_latt(self,a1,a2,a3):
        """
            return norms of reciprocal lattice vector
        """
        from numpy.linalg import norm
        from numpy import dot
        from numpy import cross

        pi = 3.14159268

        denominator = 1.0/dot(a1,cross(a2,a3))
        
        b1 = norm(2*pi*denominator*cross(a2,a3))
        b2 = norm(2*pi*denominator*cross(a3,a1)) 
        b3 = norm(2*pi*denominator*cross(a1,a2))

        return b1,b2,b3

    def write_scf_in_and_run(self):
        """
            For etot.input: 
            1. check atom number. Use single K-point when > 200
            2. use kspacing para to determine K mesh
        """    
        import time 

        to_array = lambda x: np.array([float(i) for i in x.split()])

        scf_dirs = [] 

        for a, b, c in os.walk(self.explore_result_dir):
            # exclude those not at the lowest level 
            if len(b) == 0:
                scf_dirs.append(a)   

        for dir in scf_dirs:
            print ("writting etot.input in",dir)

            with open(dir+"/atom.config","r") as file:
                raw = file.readlines()[:5]

            num_atom = int(raw[0])
            
            a1 = to_array(raw[2])
            a2 = to_array(raw[3])
            a3 = to_array(raw[4])
            
            norm_b1,norm_b2,norm_b3 = self.calc_recip_latt(a1,a2,a3)
            
            #print (norm_b1,norm_b2,norm_b3)
            self.write_etot_input(dir, num_atom, [norm_b1,norm_b2,norm_b3] )

            print ("SCF starts in", dir)
            
            os.chdir(dir)
            begin = time.time()
            # silent mode 
            pwmat_cmd = "mpirun -np $SLURM_NPROCS -iface ib0 PWmat -host 10.0.0.2 50002 > /dev/null"
            #pwmat_cmd = "mpirun -np $SLURM_NPROCS -iface ib0 PWmat -host 10.0.0.2 50002 | tee output"
            subprocess.run(pwmat_cmd,shell=True) 
            
            end = time.time() 
            
            print ("SCF ends successfully in", end-begin, "s")
    
            
    def collect_cfgs(self):
        """
            select configs in /subsys to /result
        """
        from random import shuffle
        lmp_dirs = [] 

        # for all groups pick out all config.x in lmp dir 
        for root, dirs, files in os.walk(self.explore_subsys_dir, topdown=False):
            group_dirs = dirs
        
        for group in group_dirs:
            # walk in this group 
            cfg_dot_dirs = [] 
            for root, dirs, files in os.walk(os.path.join(self.explore_subsys_dir,group), topdown=False):
                """
                    some lmp run will just die out. 
                    No config.(x) is generated
                """
                for name in files:
                    tmp = os.path.join(root, name)
                    if "config." in tmp:
                        cfg_dot_dirs.append(tmp)

            shuffle(cfg_dot_dirs)

            # result dir 
            tgt = (self.explore_result_dir + "/"+group).replace("subsys","result")

            #print (os.path.exists(tgt))
            for idx, cfg in enumerate(cfg_dot_dirs[:self.num_select_per_group]):
                
                tgt_sub = tgt+ "/" + str(idx)
                print("creating:",tgt_sub)

                if not os.path.exists(tgt_sub):
                    os.mkdir(tgt_sub)
                
                cmd = "cp "+cfg+" "+tgt_sub+"/atom.config"
                subprocess.run([cmd], shell=True) 

    def collect_mvmt(self):
        """
            collect and concatenate movements from all scf dirs
            all OUT.MLMD -> to group_(x)/MOVEMENT  
        """
        scf_dirs = [] 
        tgt_name = "OUT.MLMD" 

        for root, dirs, files in os.walk(self.explore_result_dir):
            group_dirs = dirs 
            break
        
        for group in group_dirs:
            cat_cmd = "cat "
            # walk in this group 
            for root, dirs, files in os.walk(os.path.join(self.explore_result_dir,group)):
                cand_dirs = dirs 
                break 
                
            for cand in cand_dirs:
                # check if OUT.MLMD exist 
                if os.path.exists(os.path.join(self.explore_result_dir,group,cand,tgt_name)):
                    cat_cmd = cat_cmd + os.path.join(self.explore_result_dir,group,cand,tgt_name) + " " 
                #print (type(os.path.join(self.explore_result_dir,group,cand,tgt_name))
            if cat_cmd != "cat ":
                cat_cmd = cat_cmd + " > " + os.path.join(self.explore_result_dir, group, "MOVEMENT")
                #print(cat_cmd) 
                subprocess.run([cat_cmd],shell=True)
                print ("MOVEMENT generated in", group)
        
    """
       ****************************************** 
                        run scf 
       ****************************************** 
    """
    def explore(self):
        
        # prepare directories
        #self.select_init_config() 

        # generate trajectories and select candidates
        #self.run_lmp()      

        # scf for selected configs
        self.run_scf()

    def train(self):
        """

        """
        
        pass 
        
    
if __name__ == "__main__":
    
    temp_range = [200*i for i in range(1,9)]
    pressure_range= [1.0,10,100,1000,5000,10000]   

    #temp_range = [500,1000]
    #pressure_range = [100,500]
    
    adpt_trainer = adaptive_trainer(
                                temp = temp_range,
                                pressure= pressure_range, 
                                process_num = 64,           # for lmp traj gen 
                                model_num = 4,              
                                kspacing = 0.16, 
                                ensemble= "npt",
                                traj_step = 300, 
                                num_select_per_group = 50,
                              ) 
    #trainer.initialize() 
    adpt_trainer.explore() 
    
