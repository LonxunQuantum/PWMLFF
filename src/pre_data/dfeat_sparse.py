"""
    dfeat_sparse module 
    location: src/pre_data
    L.Wang 2022.7
"""


import numpy as np 
import torch 
import pandas as pd
import os
import sys

import default_para as pm 
import prepare  

#codepath=os.path.abspath(sys.path[0])

#sys.path.append(codepath+'/../lib')            # for debug 

#sys.path.append(codepath+'/lib')                # location of read_all.o as seen by train.py 

from read_all import read_allnn 

"""
    determine the module to be loaded 
"""
if (pm.feature_dtype == 'float64'):
    from convert_dfeat64 import convert_dfeat 
elif (pm.feature_dtype == 'float32'):
    from convert_dfeat import convert_dfeat
else:
    raise RuntimeError( "unsupported feature_dtype:", pm.feature_dtype)

class dfeat_raw:
    
    """
        dfeat_tmp_all is the sparse array from fortran
        
    """
    
    """
        input_dfeat_record_path = pm.f_train_dfeat or pm.f_test_dfeat, in fread_dfeat  
        input_feat_path = pm.f_train_feat or pm.f_test_feat 
        input_natoms_path = f_train_natoms or f_test_natoms
    """

    def __init__(self,
                 # for training set transform  
                 input_dfeat_record_path_train, 
                 input_feat_path_train, 
                 input_natoms_path_train,
                 # for valid set transform 
                 input_dfeat_record_path_valid, 
                 input_feat_path_valid, 
                 input_natoms_path_valid,

                 scaler):

        """
            dictionary: {image index: raw sparse dfeat data}
            Only raw sparse dfeat data is saved in memory.
            Call transfrom() when to be used. 
        """
            
        """
            raw data 
            dfeat_tmp_all: non-zero elements of dfeat
            others: axuliary ones
        """

        self.dfeat_tmp_all = {} 
        self.num_tmp_all = {}
        self.iat_tmp_all = {}
        self.jneigh_tmp_all = {}
        self.ifeat_tmp_all = {}
        
        # feature data type (64bit or 32bit)
        self.feature_dtype = pm.feature_dtype

        # feature used
        self.use_Ftype = pm.use_Ftype 
        self.atomType = pm.atomType

        self.ntypes = pm.ntypes

        self.indImg_train = None 
        self.indImg_valid = None

        self.total_img_num_train = 0
        self.total_img_num_valid = 0
        
        """
            format:
            path to the binary file, image index included in this file, ? 

            EXAMPLE: 
            ....
            /data/home/wlj_pwmat/working_dir/CuO_dbg/PWdata/dfeat.fbin.Ftype1, 1, 285 
            /data/home/wlj_pwmat/working_dir/CuO_dbg/PWdata/dfeat.fbin.Ftype1, 2, 535625

            ....
        """

        self.nfeat = {} 
        self.total_feature_num = 0 

        self.maxNeighborNum = pm.maxNeighborNum 

        self.dfeat_names_global_train = {}
        self.image_nums_global_train = {} 

        self.dfeat_names_global_valid = {}
        self.image_nums_global_valid = {} 
        
        # for traning set 
        self.dfeat_record_path_train = input_dfeat_record_path_train
        self.feat_path_train = input_feat_path_train
        self.natoms_path_train = input_natoms_path_train

        #for valid set 
        self.dfeat_record_path_valid = input_dfeat_record_path_valid
        self.feat_path_valid = input_feat_path_valid
        self.natoms_path_valid = input_natoms_path_valid

        self.batch_size = pm.batch_size 

        # sacler
        self.scaler = scaler 

    def load(self):
        
        """ 
            
            The sparse array for the whole MOVEMENT file is loaded.
            some variables from the previous code

        """
        #itypes, feat, engy = prepare.r_feat_csv(self.feat_path_train) 

        #nfeat0m = feat.shape[1] 

        itype_atom = np.asfortranarray(np.array(self.atomType).transpose()) 

        dfeatdirs = {} 

        """
            feature index base
        """
        self.nfeat = {}
        self.nfeat[0] = 0

        feature_flag = 0

        # ----------------------------------------------------------------

        """ 
            load indImg, FOR BOTH VALID AND TRAINING
        """ 

        # training 
        natoms = np.loadtxt(self.natoms_path_train, dtype=int)
        natoms = np.atleast_2d(natoms)
        
        nImg = natoms.shape[0]
        
        indImg = np.zeros((nImg+1,), dtype=int)
        indImg[0] = 0
        
        for i in range(nImg):
            indImg[i+1] = indImg[i] + natoms[i, 0]

        self.indImg_train = indImg.copy() 
        self.total_img_num_train = nImg

        # ----------------------------------------------------------------

        # valid 

        natoms = np.loadtxt(self.natoms_path_valid, dtype=int)
        natoms = np.atleast_2d(natoms)
        
        nImg = natoms.shape[0]
        
        indImg = np.zeros((nImg+1,), dtype=int)
        indImg[0] = 0
        
        for i in range(nImg):
            indImg[i+1] = indImg[i] + natoms[i, 0]

        self.indImg_valid = indImg.copy() 
        self.total_img_num_valid = nImg

        # ----------------------------------------------------------------

        """
            Load data arrays. Will be used for both training and validation
            using "self.dfeat_record_path_train" here has no impact  
        """   

        for feature in self.use_Ftype:

            # obtain binary file path. Both training and valid are ok 
            dfeatdirs[feature] = np.unique(pd.read_csv(self.dfeat_record_path_train+str(feature), header=None, encoding= 'unicode_escape').values[:, 0])
            
            for dfeatBinIdx in dfeatdirs[feature]:

                #print ("calling fortran")
                read_allnn.read_dfeat_singleimg(dfeatBinIdx, itype_atom, self.nfeat[feature_flag])
                #print ("calling fortran ends")

                """
                    below are the sparse data arrays 
                """     
                # max feature number of the current feature type
                self.nfeat[feature_flag+1] = int(read_allnn.nfeat0m)             

                #self.nfeat[feature_flag+1] = np.array(read_allnn.feat_all).shape[0]

                # This step involves 2 copies of dfeat  
                self.dfeat_tmp_all[dfeatBinIdx] = np.array(read_allnn.dfeat_tmp_all).astype(self.feature_dtype)
                read_allnn.deallocate_dfeat_tmp_all() 

                #print("dfeat_tmp_all loaded")
                #print ("nnz:", len(np.where(self.dfeat_tmp_all[dfeatBinIdx]!=0)[0]) )
                #print ("size of tensor:", self.dfeat_tmp_all[dfeatBinIdx].shape)

                self.num_tmp_all[dfeatBinIdx] = np.array(read_allnn.num_tmp_all).astype(np.uint32) 
                read_allnn.deallocate_num_tmp_all()
                
                self.iat_tmp_all[dfeatBinIdx] = np.array(read_allnn.iat_tmp_all).astype(np.uint32)
                read_allnn.deallocate_iat_tmp_all() 

                self.jneigh_tmp_all[dfeatBinIdx] = np.array(read_allnn.jneigh_tmp_all).astype(np.uint32)
                read_allnn.deallocate_jneigh_tmp_all() 

                self.ifeat_tmp_all[dfeatBinIdx] = np.array(read_allnn.ifeat_tmp_all).astype(np.uint32)
                read_allnn.deallocate_ifeat_tmp_all()

                read_allnn.deallo_singleimg()     

                """ 
                    dfeat_tmp_all: (spatial dimenstion , dfeat nz element index , image index )
                    others: (dfeat nz element index , image index)
                """ 

            feature_flag += 1
            
            """
                fortran convert_dfeat routine: 
                    
                conv_dfeat(image_Num,ipos,natom_p,num_tmp,dfeat_tmp,jneigh_tmp,ifeat_tmp,iat_tmp)

                do jj=1,num_tmp
                    dfeat(ifeat_tmp(jj)+ipos,iat_tmp(jj)+natom_p,jneigh_tmp(jj),1)=dfeat_tmp(1,jj)
                    dfeat(ifeat_tmp(jj)+ipos,iat_tmp(jj)+natom_p,jneigh_tmp(jj),2)=dfeat_tmp(2,jj)
                    dfeat(ifeat_tmp(jj)+ipos,iat_tmp(jj)+natom_p,jneigh_tmp(jj),3)=dfeat_tmp(3,jj)
                enddo

                natom_p: indImg[imageIdx] starting absolute atom index of this image
                
                ipos: nfeat[featureIdx], starting index for this feature 
            """
        # total feature number  
        
        self.total_feature_num = sum(self.nfeat.values()) 

        """
            two global arrays for transformaiton 
        """

        # setting path 
        for featureIdx in self.use_Ftype:
            # train
            values = pd.read_csv(self.dfeat_record_path_train+str(featureIdx), header=None, encoding= 'unicode_escape').values
     
            self.dfeat_names_global_train[featureIdx] = values[:, 0]
            self.image_nums_global_train[featureIdx] = values[:, 1].astype(int)
            
            # valid
            # filter out dfeat files that didn't show up in train
            values = pd.read_csv(self.dfeat_record_path_valid+str(featureIdx), header=None, encoding= 'unicode_escape').values
            
            self.dfeat_names_global_valid[featureIdx] = values[:, 0]
            self.image_nums_global_valid[featureIdx] = values[:, 1].astype(int)
            """
            values_filtered = [] 
            
            for item in values[:,0]:
                if item in self.dfeat_names_global_train[featureIdx]:
                    values_filtered.append(item)
                
            #self.dfeat_names_global_valid[featureIdx] = values[:, 0]
            self.dfeat_names_global_valid[featureIdx] = np.asarray(values_filtered)
            self.image_nums_global_valid[featureIdx] = values[:, 1].astype(int)
            """
            #print ("dbg starts")
            #print (type(self.dfeat_names_global_valid[featureIdx]))
            #print ("dbg ends")
        

    def transform(self, batchIdx, target = "train" ):
        """
            returns dfeat tensor of a specific batch
            
            index order of dfeat_scaled: 
            absolute atom index, neighbor index, feature index, spatial dimension

            around 0.01 sec per transformation, acceptable
        """     

        if target == "train":
            self.total_img_num = self.total_img_num_train
            self.indImg = self.indImg_train
            
            self.dfeat_names_global = self.dfeat_names_global_train
            self.image_nums_global = self.image_nums_global_train

        elif target == "valid":
            self.total_img_num = self.total_img_num_valid
            self.indImg = self.indImg_valid

            self.dfeat_names_global = self.dfeat_names_global_valid
            self.image_nums_global = self.image_nums_global_valid

        else:
            raise Exception("target type must be either train or valid")
        
        imageIdx_start = self.batch_size * batchIdx 
        imageIdx_end = min(imageIdx_start + self.batch_size, self.total_img_num) # watch out the boundary
        
        #print ("imageIdx_start",imageIdx_start)
        #print ("imageIdx_end",imageIdx_end)
        #print ("self.total_img_num", self.total_img_num)

        if target == "valid" and self.dfeat_names_global[self.use_Ftype[0]][imageIdx_start] not in self.dfeat_tmp_all:
            return "aborted"

        dfeat = [] 
                
        for imageIndex in range(imageIdx_start, imageIdx_end):
        
            dfeat_name = {} 
            image_num = {} 

            # atom number in this image 
            atom_num = self.indImg[imageIndex+1] - self.indImg[imageIndex]
            convert_dfeat.allo_singleimg(atom_num,self.maxNeighborNum,self.total_feature_num)  

            for feature in self.use_Ftype:
                dfeat_name[feature] = self.dfeat_names_global[feature][imageIndex]
                image_num[feature] = self.image_nums_global[feature][imageIndex]

            featureIdx = 0 
            
            for feature in self.use_Ftype:

                # feature value array 
                """
                    index order of dfeat_tmp_all:
                    spatial dimension, non-zero element index, image index 
                """     
                # image index within this movement file 

                img_idx_within_mvmt = image_num[feature]-1 

                dfeat_tmp=np.asfortranarray(self.dfeat_tmp_all[dfeat_name[feature]][:,:,img_idx_within_mvmt])
                
                #neighbor index array
                jneigh_tmp=np.asfortranarray(self.jneigh_tmp_all[dfeat_name[feature]][:,img_idx_within_mvmt])
                
                # feature index array
                ifeat_tmp=np.asfortranarray(self.ifeat_tmp_all[dfeat_name[feature]][:,img_idx_within_mvmt])
                
                # atom index array 
                iat_tmp=np.asfortranarray(self.iat_tmp_all[dfeat_name[feature]][:,img_idx_within_mvmt])

                convert_dfeat.conv_dfeat_singleimg(self.nfeat[featureIdx],    # ipos
                                                   self.indImg[imageIndex],
                                                   self.num_tmp_all[dfeat_name[feature]][img_idx_within_mvmt],
                                                   dfeat_tmp,
                                                   jneigh_tmp,
                                                   ifeat_tmp,
                                                   iat_tmp)

                featureIdx += 1 
                         
            """
                return type? 5 dimension tensor
            """

            dfeat.append(np.array(convert_dfeat.dfeat).astype(self.feature_dtype)) 

            #dfeat.append(torch.tensor(np.array(convert_dfeat.dfeat).astype(self.feature_dtype)))    
        
            """
                perform scaling
            """
            if pm.is_scale: 
                # using the scaler obtained from 
                if self.scaler == None:
                    raise Exception("scaler is not found")

                trans = lambda x : x.transpose(0,1,3,2) 
                dfeat[-1] =  trans(trans(dfeat[-1]) * self.scaler.scale_ )
    
            dfeat[-1] = torch.tensor(dfeat[-1])

            convert_dfeat.deallo() 

        if (self.batch_size == 1):
            return dfeat[0][None,:]

        else:
            # return a "stack" with an extra dimension to match the format of torch data 
            return torch.stack(tuple(dfeat))

    def update_scaler(self,input_scaler):

        self.scaler = input_scaler

    def debug(self):

        return 
"""
    usage example 
"""

#initialize 


"""
    debuging 
"""


"""

from data_loader_2type import MovementDataset, get_torch_data
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
import time

batch_size = 1

train_data_path = pm.train_data_path
torch_train_data = get_torch_data(train_data_path)

valid_data_path=pm.test_data_path
torch_valid_data = get_torch_data(valid_data_path)

#
pm.is_scale = True

print(pm.is_scale)
scaler = None


if pm.is_scale:

    # generate new 
    print("using new scaler")
    scaler=MinMaxScaler()

    torch_train_data.feat = scaler.fit_transform(torch_train_data.feat)
    torch_valid_data.feat = scaler.transform(torch_valid_data.feat)
    #batch index, atom index within this image, neighbor index, feature index, spatial dimension   
    
    trans = lambda x: x.transpose(0, 1, 3, 2) 

    torch_train_data.dfeat =   trans(trans(torch_train_data.dfeat) * scaler.scale_ )
    torch_valid_data.dfeat =   trans(trans(torch_valid_data.dfeat) * scaler.scale_ )

loader_train = Data.DataLoader(torch_train_data, batch_size=batch_size, shuffle=False)   
loader_valid = Data.DataLoader(torch_valid_data, batch_size=batch_size, shuffle=False) 

dfeat = dfeat_raw(input_dfeat_record_path_train = pm.f_train_dfeat, 
                  input_feat_path_train = pm.f_train_feat,
                  input_natoms_path_train = pm.f_train_natoms,

                  input_dfeat_record_path_valid = pm.f_test_dfeat, 
                  input_feat_path_valid = pm.f_test_feat,
                  input_natoms_path_valid = pm.f_test_natoms,

                  scaler = scaler)

dfeat.load() 

print("raw data loaded")


for batchIdx, batchData in enumerate(loader_train):

    print(batchIdx)
    
    t1 = time.time()

    dfeat_img = dfeat.transform(batchIdx, target = "train")
    t2 = time.time() 

    print(t2-t1, "seconds")
    #dfeat_img = dfeat_img[None,:]

    q = batchData["input_dfeat"] 

    num = len(np.where(q!=dfeat_img)[0])

    print("how many values are different?",num)

"""