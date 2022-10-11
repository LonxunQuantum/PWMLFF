"""
    calculate feature for dp only. 
    BAD VERSION CONTROL KILLS. 
"""

def calc_feat():

    import default_para as pm 
    import os
    import sys
    import prepare as pp

    if os.path.exists('./input/'):
        pass
    else:
        os.mkdir('./input/')
        os.mkdir('./output')

    """
        VdW related
    """

    print('auto creating vdw_fitB.ntype, donot use your own vdw_fitB.ntype file')
    print('please modify parameters.py to specify your vdw parameters')
    
    if not os.path.exists(pm.fitModelDir):
        os.makedirs(pm.fitModelDir)
    
    strength_rad = 0.0
    
    if pm.isFitVdw == True:
        strength_rad = 1.0
    vdw_input = {
        'ntypes': pm.ntypes,
        'nterms': 1,
        'atom_type': pm.atomType,
        'rad': [strength_rad for i in range(pm.ntypes)],
        'e_ave': [0.0 for i in range(pm.ntypes)],
        'wp': [ [0.0 for i in range(pm.ntypes*1)] for i in range(pm.ntypes)]
    }
    if hasattr(pm, 'vdwInput'):
        vdw_input = pm.vdwInput
    pp.writeVdwInput(pm.fitModelDir, vdw_input)

    """
        feature calculation starts
    """

    if os.path.exists(os.path.join(os.path.abspath(pm.trainSetDir), 'trainData.txt.Ftype1')):
        os.system(
            'rm '+os.path.join(os.path.abspath(pm.trainSetDir), 'trainData.txt.*'))
        os.system(
            'rm '+os.path.join(os.path.abspath(pm.trainSetDir), 'inquirepos*'))
        os.system('rm ' + pm.dRneigh_path)
    else:
        pass

    if os.path.exists(os.path.join(pm.trainSetDir, 'lppData.txt')):
        os.system('rm '+os.path.join(pm.trainSetDir, 'lppData.txt'))
    else:
        pass

    pp.collectAllSourceFiles()
    pp.savePath()
    pp.combineMovement()
    pp.writeGenFeatInput()
    os.system('cp '+os.path.abspath(pm.fbinListPath)+' ./input/')

    print (pm.atomType)

    calFeatGrid = False
    for i in range(len(pm.atomType)):
        if pm.Ftype1_para['iflag_grid'][i] == 3 or pm.Ftype2_para['iflag_grid'][i] == 3:
    
           pp.calFeatGrid()

    for i in pm.use_Ftype:
        # import ipdb;ipdb.set_trace()
        command = pm.Ftype_name[i]+".x" 
        print(command)
        os.system(command)

    return 

    