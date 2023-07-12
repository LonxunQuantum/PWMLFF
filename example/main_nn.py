#import the regressor module
from PWMLFF.nn_network import nn_network 

if __name__ == '__main__':

    # atom type to be used. MUST BE SPECIFIED 
    atom_type = [22,8]

    # feature to be used. MUST BE SPECIFIED 
    feature_type = [7]

    # create an instance. MUST BE DONE. 
    kfnn_trainer = nn_network(
                                atom_type = atom_type,   
                                feature_type = feature_type, 
                                n_epoch = 20,              # number of epochs
                                Rmax = 6.0,
                                Rmin = 0.5,
                                # is_trainEi = True,           # train atomic energy
                                kalman_type = "LKF",      # using global Kalman filter
                                device = "cuda",              # run training on gpu
                                recover = False,             # recover previous training
                                session_dir = "record"       # directory that contains 
                                )
    
    # generate data from MOVEMENT files
    # ONLY NEED TO BE DONE ONCE
    kfnn_trainer.generate_data()

    kfnn_trainer.load_and_train()

    # extract network parameters for inference module. MUST-HAVE, ONLY ONCE
    kfnn_trainer.extract_model_para()

    # run evaluation
    kfnn_trainer.evaluate() 

    # plot the evaluation result
    kfnn_trainer.plot_evaluation(plot_elem = False, save_data = False)

    # md_detail array
    md_detail = [1,1000,1,500,500]

    # run MD  
    kfnn_trainer.run_md(md_details = md_detail, follow = False)

    # extract force field
    kfnn_trainer.extract_force_field()
