import os
import numpy as np
import use_para as pm

# empty use_para, import all attributes from calculation_dir/parameters.py, import other needed attributes from default_para.py
def parse_input():
    print('parsing parameters ... ')
    import parameters
    import default_para
    default_attrs = default_para.__dir__()
    diy_parameters = parameters.__dir__()
    not_inputs = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__file__', '__cached__', '__builtins__', 'np', 'os']
    for tmp in not_inputs: # these attributes in pm won't be overrided 
        if tmp in diy_parameters:
            diy_parameters.remove(tmp)
        if tmp in default_attrs:
            default_attrs.remove(tmp)
    for tmp in diy_parameters:
        setattr(pm, tmp, getattr(parameters, tmp))
        if tmp in default_attrs:
            default_attrs.remove(tmp)
        else:
            print('the input parameter "' + tmp +'" is not in default parameters, make sure you set right parameters')
    for tmp in default_attrs:
        setattr(pm, tmp, getattr(default_para, tmp))
   
    # some patches
    #pm.test_val = 0 
   	
  	#formating nNodes 

    numAtomType = len((pm.atomType))

    nodeDim = pm.nodeDim

    nNodesTemp = []     

    for dim in nodeDim:
        nNodesTemp.append([dim for i in range(numAtomType)])


    pm.nNodes = np.array(nNodesTemp) 

    #batches 
    pm.nfeat_type=len(pm.use_Ftype)
    pm.ntypes=len(pm.atomType)
    pm.atomTypeNum=len(pm.atomType)
    if hasattr(pm, 'nFeatures') and not hasattr(pm, 'MLFF_dmirror_cfg'):
        pm.MLFF_dmirror_cfg = [
                            ('linear', pm.nFeatures, 1, True),
                       ]

    if hasattr(pm, 'nFeatures') and  not hasattr(pm, 'MLFF_dmirror_cfg1'):
        pm.MLFF_dmirror_cfg1 = [
                            ('linear', pm.nFeatures, 30, True),
                            ('activation',),
                            ('linear', 30, 60, True),
                            ('activation',),
                            ('linear', 60, 1, True)
                       ]
    if hasattr(pm, 'nFeatures') and  not hasattr(pm, 'MLFF_dmirror_cfg2'):
        pm.MLFF_dmirror_cfg2 = [
                            ('linear', pm.nFeatures, 1, True),
                            ('activation',),
                            ('linear', 1, 1, True)
                       ]
    if hasattr(pm, 'nFeatures') and  not hasattr(pm, 'MLFF_dmirror_cfg3'):
        pm.MLFF_dmirror_cfg3 = [
                            ('linear', pm.nFeatures, 10, True),
                            ('activation',),
                            ('linear', 10, 3, True),
                            ('activation',),
                            ('linear', 3, 1, True)
                       ]
    print('done')
    print('writing parameters to output/out.parameters.py... ', end='')
    all_para = pm.__dir__()
    for tmp in not_inputs:
        all_para.remove(tmp)
    os.system('mkdir input')
    os.system('mkdir output')
    fout = open('output/out.parameters.py', 'w')
    for tmp in all_para:
        fout.write(tmp+' = ')
        tmp_type = type(getattr(pm,tmp))
        if tmp_type == str:
            fout.write('\'' + getattr(pm,tmp) + '\'')
        elif tmp_type == np.ndarray:
            fout.write('np.array(' + str((getattr(pm,tmp).tolist())) + ')')
        else:
            fout.write(str(getattr(pm,tmp)))
        fout.write('\n')
    fout.close()
    print('done')
