"""
    example of linear fitting workflow
"""
from PWMLFF.linear_regressor import linear_regressor 

if __name__ == "__main__":

    # training 
    # atom type to be used 
    atom_type = [8,6,1]
    # feature to be used
    feature_type = [7]
    # create an instance
    linReg = linear_regressor(atom_type = atom_type,
                              feature_type = feature_type, 
                              etot_weight = 0.5, 
                              force_weight = 0.5,
                              ei_weight = 1.0)
    # generate data 
    # ONLY NEED TO BE DONE ONCE
    linReg.generate_data() 
    # training 
    linReg.train() 

    """
        perform evaulation and plot 
    """ 
    linReg.evaluate()
    linReg.plot_evaluation(plot_elem = False, save_data = False)
    # run MD 
    # PWmat-style md_detail array
    md_detail = [1,1000,1,500,500]
    linReg.run_md(md_details = md_detail, follow = False)
    
