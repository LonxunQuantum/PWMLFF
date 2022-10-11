"""
    Example of NN force field workflow 
"""
#import the regressor module
from PWmatMLFF.nn_network import nn_network 

if __name__ == '__main__':

    # atom type to be used 
    atom_type = [29,8]
    # feature to be used
    feature_type = [1,2]
    
    # create an instance 
    kfnn_trainer = nn_network(   
                                atom_type = atom_type,   
                                feature_type = feature_type, 
                                kalman_type = "global",     # using global Kalman filter
                                device = "cpu",             #
                                recover = False             # recover previous training
                             )
    
    # besides passing in args when creating the instance, 
    # can also call set_xxx()
    kfnn_trainer.set_working_dir("record")
    

    # generate data from MOVEMENT files
    # ONLY NEED TO BE DONE ONCE
    kfnn_trainer.generate_data()
    # transform data
    kfnn_trainer.load_data()
    # initialize the network   
    kfnn_trainer.set_model() 
    # set optimizers and related scheduler
    kfnn_trainer.set_optimizer()
    # set epoch number for training
    kfnn_trainer.set_epoch_num(20)
    # training 
    kfnn_trainer.train() 
    
    # the md_detail array as in PWmat
    

    # prepare force field for MD
    kfnn_trainer.extract_model_para()

    # run evaluation
    kfnn_trainer.evaluate()
    
    # run MD 
    md_detail = [1,1000,1,300,300] 
    kfnn_trainer.run_md(md_details = md_detail, num_thread = 12, follow = False)
    
    