from PWmatMLFF.gnn_network import gnn_network


if __name__ == "__main__":
    
    atom_type = ["Cu"]

    gnn_trainer = gnn_network(  device = "cuda", # choose the device for training 
                                chemical_symbols = atom_type
                                )
    
    gnn_trainer.set_epoch_num(20)

    # set number of image in training and validation
    # Notice that nequip picks up training and validation set randomly. 

    # set number of images for trianing and validation
    gnn_trainer.set_num_train_img(400)
    gnn_trainer.set_num_valid_img(400)
    
    gnn_trainer.set_working_dir("record")
    gnn_trainer.set_task_name("20220902-test")
    
    # generate data.
    # ONLY NEED TO BE DONE ONCE! 
    gnn_trainer.generate_data() 
    
    # lanuch training 
    gnn_trainer.train() 
    
    # evaluating the model. Must specify train_dir 
    gnn_trainer.evaluate(device = "cuda",train_dir = "record/20220902-test") 
    
    # save the model
    gnn_trainer.deploy(train_dir = "record/20220902-test", out_file = "./20220902-test.pth") 
    
