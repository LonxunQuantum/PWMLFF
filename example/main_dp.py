from PWMLFF.dp_network import dp_network                                    
                                                                               
if __name__ == "__main__":                                                     
                                                                               
    atom_type = [14]                                                         
                                                           
    """                                                                        
        command line option:                                                   
        python main.py --gpu 0 -b 20 --opt LKF --epochs 1 -s record            
    """                                                                                                               
                                                                                                           
    dp_trainer = dp_network(                                               
                            atom_type = atom_type,                             
                            optimizer = "LKF",                                 
                            gpu_id = 0,                                        
                            session_dir = "record",                          
                            n_epoch = 20,                                      
                            batch_size = 5,                                    
                            Rmax = 6.0,                                        
                            Rmin = 0.5,                                        
                            M2 = 16,                                           
                            dataset_size = 1000,                               
                            block_size= 10240,               
                            is_virial = False,   # default is false            
                            is_egroup = False,   # default is false
                            #is_resume = True,
                            #is_evaluate= True,
                            #model_name = 'checkpoint.pth.tar',
                            pre_fac_force = 1.,                               
                            pre_fac_etot = .5,                                
                            # pre_fac_virial = 1.,                              
                            # pre_fac_egroup = .1
                           )                                                                                                                
    # pre-process trianing data. ONLY NEED TO BE DONE ONCE                     
    dp_trainer.generate_data()     # is_real_Ep = True means use Etot = Ep, otherwise Etot = sum Ei                
                                                                               
    # load data and train                                                      
    dp_trainer.load_and_train()                                            
                                                             
    dp_trainer.extract_force_field()                                       
                                                                 
