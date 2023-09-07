from utils.json_operation import get_parameter, get_required_parameter
class OptimizerParam(object):
    def __init__(self) -> None:
        pass

    def set_optimizer(self, json_input:dict):
        self.opt_name = get_parameter("optimizer", json_input, "LKF")
        self.batch_size = get_parameter("batch_size", json_input, 1)
        self.epochs = get_parameter("epochs", json_input, 30)
        self.print_freq = get_parameter("print_freq", json_input, 10)
        # the start epoch could be reset at the resume model code block
        self.start_epoch = get_parameter("start_epoch", json_input, 1)
        # self.optimizer_param = OptimizerParam(optimizer_type, start_epoch=start_epoch, epochs=epochs, batch_size=batch_size, \
                                        #  print_freq=print_freq)
        if "KF" in self.opt_name.upper():  #set Kalman Filter Optimizer params
            self.kalman_lambda = get_parameter("kalman_lambda", json_input, 0.98)
            self.kalman_nue = get_parameter("kalman_nue", json_input, 0.9987)
            self.block_size = get_parameter("block_size", json_input, 5120)
            self.nselect = get_parameter("nselect", json_input, 24)
            self.groupsize = get_parameter("groupsize", json_input, 6)
        else:   # set ADAM Optimizer params
            self.learning_rate = get_parameter("learning_rate", json_input, 0.001)
            self.weight_decay = get_parameter("weight_decay", json_input, 1e-4)
            self.momentum = get_parameter("momentum", json_input, 0.9)
            self.gamma = get_parameter("gamma", json_input, 0.99) # used in nn optimizer
            self.step = get_parameter("step", json_input, 100) # used in nn optimizer
            self.scheduler = get_parameter("scheduler", json_input, None) # used in nn optimizer

            self.stop_step = get_parameter("stop_step", json_input, 1000000)
            self.decay_step = get_parameter("decay_step", json_input, 5000)
            self.stop_lr = get_parameter("stop_lr",json_input, 3.51e-8)
            # self.set_adam_sgd_params(learning_rate, weight_decay, momentum,\
            #                                 gamma, step, scheduler, stop_step, decay_step, stop_lr)

        self.train_energy = get_parameter("train_energy", json_input, True) 
        self.train_force = get_parameter("train_force", json_input, True) 
        self.train_ei = get_parameter("train_ei", json_input, False) 
        self.train_virial = get_parameter("train_virial", json_input, False) 
        self.train_egroup = get_parameter("train_egroup", json_input, False) 

        if "KF" in self.opt_name:
            self.pre_fac_force = get_parameter("pre_fac_force", json_input, 2.0) 
            self.pre_fac_etot = get_parameter("pre_fac_etot", json_input, 1.0) 
            self.pre_fac_ei = get_parameter("pre_fac_ei", json_input, 1.0) 
            self.pre_fac_virial = get_parameter("pre_fac_virial", json_input, 1.0) 
            self.pre_fac_egroup = get_parameter("pre_fac_egroup", json_input, 0.1) 
            
            # self.set_train_pref(train_energy = train_energy, train_force = train_force, 
            #     train_ei = train_ei, train_virial = train_virial, train_egroup = train_egroup, 
            #         pre_fac_force = pre_fac_force, pre_fac_etot = pre_fac_etot, 
            #             pre_fac_ei = pre_fac_ei, pre_fac_virial = pre_fac_virial, pre_fac_egroup = pre_fac_egroup
            #         )
        else:
            self.start_pre_fac_force = get_parameter("start_pre_fac_force", json_input, 1000) 
            self.start_pre_fac_etot = get_parameter("start_pre_fac_etot", json_input, 0.02) 
            self.start_pre_fac_ei = get_parameter("start_pre_fac_ei", json_input, 0.1) 
            self.start_pre_fac_virial = get_parameter("start_pre_fac_virial", json_input, 50.0) 
            self.start_pre_fac_egroup = get_parameter("start_pre_fac_egroup", json_input, 0.02) 

            self.end_pre_fac_force = get_parameter("end_pre_fac_force", json_input, 1.0) 
            self.end_pre_fac_etot = get_parameter("end_pre_fac_etot", json_input, 1.0) 
            self.end_pre_fac_ei = get_parameter("end_pre_fac_ei", json_input, 2.0) 
            self.end_pre_fac_virial = get_parameter("end_pre_fac_virial", json_input, 1.0) 
            self.end_pre_fac_egroup = get_parameter("end_pre_fac_egroup", json_input, 1.0) 

            # self.set_adam_sgd_train_pref(
            #         train_energy = train_energy, train_force = train_force, 
            #             train_ei = train_ei, train_virial = train_virial, train_egroup = train_egroup, 
            #         start_pre_fac_force = start_pre_fac_force, start_pre_fac_etot = start_pre_fac_etot, 
            #             start_pre_fac_ei = start_pre_fac_ei, start_pre_fac_virial = start_pre_fac_virial, 
            #             start_pre_fac_egroup = start_pre_fac_egroup,
            #         end_pre_fac_force = end_pre_fac_force, end_pre_fac_etot = end_pre_fac_etot, 
            #             end_pre_fac_ei = end_pre_fac_ei, end_pre_fac_virial = end_pre_fac_virial, 
            #             end_pre_fac_egroup = end_pre_fac_egroup,
            #         ) 
    
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
        else:
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
            
        return opt_dict