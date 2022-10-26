from PWmatMLFF.dp_network import dp_network

if __name__ == "__main__":

    atom_type = [29,8]

    # create class instance
    dp_trainer = dp_network(device = "cuda", atom_type = atom_type)
    
    dp_trainer.set_session_dir("kfdp_record")
    
    # generating trianing data. ONLY NEED TO BE DONE ONCE
    dp_trainer.generate_data() 
    
    # load data into memeory 
    dp_trainer.load_data()  
        
    # initialize network 
    dp_trainer.set_model()  
    
    # set optimzer 
    dp_trainer.set_optimizer()
    
    # set epoch num
    dp_trainer.set_epoch_num(10)
    #dp_trainer.test_dbg()  
    
    #start training 
    dp_trainer.train()  
    
    """
        Lines Below Are For MD 
    """
    # extract network parameters. MUST HAVE. 
    dp_trainer.extract_model_para()
    
    # the md_detail array as in PWmat
    md_detail = [1,1000,1,300,300] 
    
    # run MD
    dp_trainer.run_md(init_config = "atom.config", md_details = md_detail, num_thread = 4, follow = False)