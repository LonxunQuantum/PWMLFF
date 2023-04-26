from PWmatMLFF.nn_network import nn_network 

if __name__ == '__main__':
 
    # initialization
    nn_layer_dim = [15,15,1]
    atom_type = [3,14]
    feature_type = [1,2]
    
    Rc_M = 5.0
    Rc_min = 0.5 
    # custom feature parameters 
    f1 = {           
        'numOf2bfeat':[24 for tmp in atom_type],     
        'Rc':[Rc_M for tmp in atom_type],
        'Rm':[Rc_min for tmp in atom_type],
        'iflag_grid':[3 for tmp in atom_type],                      
        'fact_base':[0.2 for tmp in atom_type],
        'dR1':[0.5 for tmp in atom_type],
        'iflag_ftype':3       
        }
    
    # custom feature parameters 

    f2 ={       
        'numOf3bfeat1':[3 for tmp in atom_type],
        'numOf3bfeat2':[3 for tmp in atom_type],
        'Rc':[3 for tmp in atom_type],
        'Rc2':[3 for tmp in atom_type],
        'Rm':[3 for tmp in atom_type],

        'iflag_grid':[3 for tmp in atom_type],            
        'fact_base':[3 for tmp in atom_type],

        'dR1':[3 for tmp in atom_type],
        'dR2':[3 for tmp in atom_type],
        'iflag_ftype':3   
    }


    # create an instance
    kfnn_trainer = nn_network(  nn_layer_config = nn_layer_dim, 
                                atom_type = atom_type,   
                                feature_type = feature_type, 
                                kalman_type = "LKF",
                                device = "cpu",
                                session_dir = "record",
                                block_size = 2560, 
                                batch_size = 1,
                                custom_feat_1= f1,
                                custom_feat_2= f2, 
                                kf_prefac_etot = 1.0,
                                kf_prefac_force = 2.5,
                                n_epoch = 10,
                            )
    
    # generate training data and load 
    kfnn_trainer.generate_data()
    
    kfnn_trainer.load_and_train()  
    
    # extract 
    kfnn_trainer.extract_force_field() 
