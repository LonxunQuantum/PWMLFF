"""
    example of linear fitting workflow
"""
from PWmatMLFF.linear_regressor import linear_regressor 

if __name__ == "__main__":

    # training 
    # atom type to be used 
    atom_type = [29,8]
    # feature to be used
    feature_type = [1,2]
    # create an instance
    linReg = linear_regressor(atom_type = atom_type, feature_type = feature_type)
    # generate data 
    # ONLY NEED TO BE DONE ONCE
    linReg.generate_data() 
    # training 
    linReg.train() 
    
    # run MD 
    # PWmat-style md_detail array
    md_detail = [1,1000,1,500,500]
    linReg.run_md(md_details = md_detail, follow = False)
    
