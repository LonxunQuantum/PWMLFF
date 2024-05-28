from utils.json_operation import get_parameter, get_required_parameter
from src.user.nep_param import NepParam

class OptimizerParam(object):
    def __init__(self) -> None:
        pass

    def set_optimizer(self, json_source:dict, nep_param:NepParam=None):
        optimizer_dict = get_parameter("optimizer", json_source, {})
        self.opt_name = get_parameter("optimizer", optimizer_dict, "LKF")
        self.batch_size = get_parameter("batch_size", optimizer_dict, 1)
        self.epochs = get_parameter("epochs", optimizer_dict, 30)
        self.print_freq = get_parameter("print_freq", optimizer_dict, 10)
        # the start epoch could be reset at the resume model code block
        self.reset_epoch = get_parameter("reset_epoch", optimizer_dict, False)
        self.start_epoch = get_parameter("start_epoch", optimizer_dict, 1)
        # self.optimizer_param = OptimizerParam(optimizer_type, start_epoch=start_epoch, epochs=epochs, batch_size=batch_size, \
                                        #  print_freq=print_freq)
        if "KF" in self.opt_name.upper():  #set Kalman Filter Optimizer params
            self.kalman_lambda = get_parameter("kalman_lambda", optimizer_dict, 0.98)
            self.kalman_nue = get_parameter("kalman_nue", optimizer_dict, 0.9987)
            self.block_size = get_parameter("block_size", optimizer_dict, 5120)
            self.nselect = get_parameter("nselect", optimizer_dict, 24)
            self.groupsize = get_parameter("groupsize", optimizer_dict, 6)
        elif "ADAM" in self.opt_name.upper():   # set ADAM Optimizer params
            self.learning_rate = get_parameter("learning_rate", optimizer_dict, 0.001)
            self.weight_decay = get_parameter("weight_decay", optimizer_dict, 1e-4)
            self.momentum = get_parameter("momentum", optimizer_dict, 0.9)
            self.gamma = get_parameter("gamma", optimizer_dict, 0.99) # used in nn optimizer
            self.step = get_parameter("step", optimizer_dict, 100) # used in nn optimizer
            self.scheduler = get_parameter("scheduler", optimizer_dict, None) # used in nn optimizer

            self.stop_step = get_parameter("stop_step", optimizer_dict, 1000000)
            self.decay_step = get_parameter("decay_step", optimizer_dict, 5000)
            self.stop_lr = get_parameter("stop_lr",optimizer_dict, 3.51e-8)
            # self.set_adam_sgd_params(learning_rate, weight_decay, momentum,\
            #                                 gamma, step, scheduler, stop_step, decay_step, stop_lr)
        else:
            pass

        self.train_energy = get_parameter("train_energy", optimizer_dict, True) 
        self.train_force = get_parameter("train_force", optimizer_dict, True) 
        self.train_ei = get_parameter("train_ei", optimizer_dict, False) 
        self.train_virial = get_parameter("train_virial", optimizer_dict, False) 
        self.train_egroup = get_parameter("train_egroup", optimizer_dict, False) 

        self.lambda_1 = None
        self.lambda_2 = None
        self.force_delta = None
        self.population = None
        self.generation = None
        self.pre_fac_force = 2.0
        self.pre_fac_etot = 1.0
        self.pre_fac_ei = 1.0
        self.pre_fac_virial = 1.0
        self.pre_fac_egroup = 0.1

        if "KF" in self.opt_name.upper():
            self.pre_fac_force = get_parameter("pre_fac_force", optimizer_dict, 2.0) 
            self.pre_fac_etot = get_parameter("pre_fac_etot", optimizer_dict, 1.0) 
            self.pre_fac_ei = get_parameter("pre_fac_ei", optimizer_dict, 1.0) 
            self.pre_fac_virial = get_parameter("pre_fac_virial", optimizer_dict, 1.0) 
            self.pre_fac_egroup = get_parameter("pre_fac_egroup", optimizer_dict, 0.1) 
            
        elif "ADAM" in self.opt_name.upper():
            self.start_pre_fac_force = get_parameter("start_pre_fac_force", optimizer_dict, 1000) 
            self.start_pre_fac_etot = get_parameter("start_pre_fac_etot", optimizer_dict, 0.02) 
            self.start_pre_fac_ei = get_parameter("start_pre_fac_ei", optimizer_dict, 0.1) 
            self.start_pre_fac_virial = get_parameter("start_pre_fac_virial", optimizer_dict, 50.0) 
            self.start_pre_fac_egroup = get_parameter("start_pre_fac_egroup", optimizer_dict, 0.02) 

            self.end_pre_fac_force = get_parameter("end_pre_fac_force", optimizer_dict, 1.0) 
            self.end_pre_fac_etot = get_parameter("end_pre_fac_etot", optimizer_dict, 1.0) 
            self.end_pre_fac_ei = get_parameter("end_pre_fac_ei", optimizer_dict, 2.0) 
            self.end_pre_fac_virial = get_parameter("end_pre_fac_virial", optimizer_dict, 1.0) 
            self.end_pre_fac_egroup = get_parameter("end_pre_fac_egroup", optimizer_dict, 1.0) 

        elif "SNES" in self.opt_name.upper():# natural evolution strategies
            # if get_parameter("nep_in_file", json_source, None) is not None \
            #     or get_parameter("nep_txt_file", json_source, None) is not None:
            try: # read from nep.in file
                self.pre_fac_etot = nep_param.lambda_e
                self.pre_fac_force = nep_param.lambda_f
                self.pre_fac_virial = nep_param.lambda_v
                self.pre_fac_egroup = nep_param.lambda_eg
                self.pre_fac_ei = nep_param.lambda_ei
                self.lambda_1 = nep_param.lambda_1
                self.lambda_2 = nep_param.lambda_2
                self.force_delta = nep_param.force_delta
                self.population = nep_param.population
                self.generation = nep_param.generation
                self.batch_size = nep_param.batch
                self.eta_m = None
                self.eta_s = None
                return

            except Exception:
                print('Read snes optimizer param from json file')
            # from 'optimizer' dict
            self.lambda_1 = get_parameter("lambda_1", optimizer_dict, -1) # weight of regularization term
            if self.lambda_1 != -1 and self.lambda_1 < 0:
                raise Exception("ERROR! the lambda_1 should >= 0 or lambda_1 = -1 for automatically determined in training!")

            self.lambda_2 = get_parameter("lambda_2", optimizer_dict, -1) # weight of norm regularization term
            if self.lambda_2 != -1 and self.lambda_2 < 0:
                raise Exception("ERROR! the lambda_2 should >= 0 or lambda_2 = -1 for automatically determined in training!")

            self.pre_fac_ei = get_parameter("lambda_ei", optimizer_dict, 1.0) # weight of energy loss term
            self.pre_fac_egroup = get_parameter("lambda_eg", optimizer_dict, 0.1) # weight of energy loss term
            self.pre_fac_etot = get_parameter("lambda_e", optimizer_dict, 1.0) # weight of energy loss term
            self.pre_fac_force = get_parameter("lambda_f", optimizer_dict, 1.0) # weight of force loss term
            self.pre_fac_virial = get_parameter("lambda_v", optimizer_dict, 0.1) # weight of virial loss term
            self.force_delta = get_parameter("force_delta", optimizer_dict, None) # bias term that can be used to make smaller forces more accurate
            self.batch_size = get_parameter("batch_size", optimizer_dict, 1000) # batch size for training
            self.population = get_parameter("population", optimizer_dict, 50) # population size used in the SNES algorithm [Schaul2011]
            self.generation = get_parameter("generation", optimizer_dict, 100000) # number of generations used by the SNES algorithm [Schaul2011]
            self.eta_m = get_parameter("eta_m", optimizer_dict, None) # population size used in the SNES algorithm [Schaul2011]
            self.eta_s = get_parameter("eta_s", optimizer_dict, None) # number of generations used by the SNES algorithm [Schaul2011]
        
    def to_linear_dict(self):
        opt_dict = {}
        opt_dict["train_energy"] = self.train_energy
        opt_dict["train_force"] = self.train_force
        opt_dict["train_ei"] = self.train_ei
        opt_dict["pre_fac_force"] = self.pre_fac_force
        opt_dict["pre_fac_etot"] = self.pre_fac_etot
        opt_dict["pre_fac_ei"] = self.pre_fac_ei
        return opt_dict
    
    def to_dict(self):
        opt_dict = {}
        opt_dict["optimizer"]=self.opt_name
        # opt_dict["start_epoch"] = self.start_epoch
        opt_dict["epochs"] = self.epochs
        opt_dict["batch_size"] = self.batch_size
        opt_dict["print_freq"] = self.print_freq
        if "KF" in self.opt_name:
            if "LKF" in self.opt_name:
                opt_dict["block_size"] = self.block_size 
            opt_dict["kalman_lambda"] = self.kalman_lambda
            opt_dict["kalman_nue"] = self.kalman_nue

            opt_dict["train_energy"] = self.train_energy
            opt_dict["train_force"] = self.train_force
            opt_dict["train_ei"] = self.train_ei
            opt_dict["train_virial"] = self.train_virial
            opt_dict["train_egroup"] = self.train_egroup
    
            opt_dict["pre_fac_force"] = self.pre_fac_force
            opt_dict["pre_fac_etot"] = self.pre_fac_etot
            opt_dict["pre_fac_ei"] = self.pre_fac_ei
            opt_dict["pre_fac_virial"] = self.pre_fac_virial
            opt_dict["pre_fac_egroup"] = self.pre_fac_egroup
        elif "SGD" in self.opt_name or "ADAM" in self.opt_name:
            if "SGD" in self.opt_name:
                opt_dict["weight_decay"]= self.weight_decay
                opt_dict["momentum"]= self.momentum

            opt_dict["learning_rate"]= self.learning_rate
            opt_dict["stop_lr"] = self.stop_lr
            opt_dict["stop_step"] = self.stop_step
            opt_dict["decay_step"] = self.decay_step

            opt_dict["train_energy"] = self.train_energy
            opt_dict["train_force"] = self.train_force
            opt_dict["train_ei"] = self.train_ei
            opt_dict["train_virial"] = self.train_virial
            opt_dict["train_egroup"] = self.train_egroup

            opt_dict["start_pre_fac_force"] = self.start_pre_fac_force
            opt_dict["start_pre_fac_etot"] = self.start_pre_fac_etot
            opt_dict["start_pre_fac_ei"] = self.start_pre_fac_ei
            opt_dict["start_pre_fac_virial"] = self.start_pre_fac_virial
            opt_dict["start_pre_fac_egroup"] = self.start_pre_fac_egroup

            opt_dict["end_pre_fac_force"] = self.end_pre_fac_force
            opt_dict["end_pre_fac_etot"] = self.end_pre_fac_etot
            opt_dict["end_pre_fac_ei"] = self.end_pre_fac_ei
            opt_dict["end_pre_fac_virial"] = self.end_pre_fac_virial
            opt_dict["end_pre_fac_egroup"] = self.end_pre_fac_egroup
        elif "SNES" in self.opt_name:
            opt_dict["train_energy"] = self.train_energy
            opt_dict["train_force"] = self.train_force
            opt_dict["train_ei"] = self.train_ei
            opt_dict["train_virial"] = self.train_virial
            opt_dict["train_egroup"] = self.train_egroup
    
            opt_dict["pre_fac_force"] = self.pre_fac_force
            opt_dict["pre_fac_etot"] = self.pre_fac_etot
            opt_dict["pre_fac_ei"] = self.pre_fac_ei
            opt_dict["pre_fac_virial"] = self.pre_fac_virial
            opt_dict["pre_fac_egroup"] = self.pre_fac_egroup

            opt_dict["lambda_1"] =  self.lambda_1
            opt_dict["lambda_2"] =  self.lambda_2
            opt_dict["force_delta"] =  self.force_delta
            opt_dict["population"] =  self.population
            opt_dict["generation"] =  self.generation
        return opt_dict

    def snes_to_nep_txt(self):
        content = ""
        content += "lambda_e    {}\n".format(self.pre_fac_etot)
        content += "lambda_f    {}\n".format(self.pre_fac_force)
        content += "lambda_v    {}\n".format(self.pre_fac_virial)
        content += "batch       {}\n".format(self.batch_size)        
        # content += "lambda_eg   {}\n".format(self.pre_fac_egroup)
        # content += "lambda_ei   {}\n".format(self.pre_fac_ei)
        if self.lambda_1 is not None:
            content += "lambda_1    {}\n".format(self.lambda_1)
        else:
            content += "lambda_1    {}\n".format(-1)
        if self.lambda_2 is not None:
            content += "lambda_2    {}\n".format(self.lambda_2)
        else:
            content += "lambda_2    {}\n".format(-1)
        if self.force_delta is not None:
            content += "force_delta {}\n".format(self.force_delta)
        else:
            content += "force_delta {}\n".format(0)
        if self.population is not None:
            content += "population  {}\n".format(self.population)
        else:
            content += "population  {}\n".format(100)
        if self.generation is not None:
            content += "generation  {}\n".format(self.generation)
        else:
            content += "generation  {}\n".format(10000)
        return content