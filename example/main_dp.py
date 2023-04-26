from PWmatMLFF.dp_network import dp_network

if __name__ == "__main__":

    atom_type = [3,14]
     
    dp_trainer = dp_network(
                            atom_type = atom_type,
                            optimizer = "LKF",
                            gpu_id = 1, 
                            session_dir = "record",
                            n_epoch = 10,
                            batch_size = 10,
                            Rmax = 5.0,
                            Rmin = 0.5,
                            M2 = 16,
                            block_size= 5120, 
                            pre_fac_force = 1,
                            pre_fac_etot = 0.5
                           )
     
    # pre-process trianing data. ONLY NEED TO BE DONE ONCE
    dp_trainer.generate_data()
    
    # load data and train 
    dp_trainer.load_and_train()
      
    dp_trainer.extract_force_field()
     
