import numpy as np
import prepare
import use_para as pm


class MinMaxScaler:
    ''' a*x +b = x_scaled like sklearn's MinMaxScaler
        note cp.atleast_2d and self.a[xmax==xmin] = 0
    '''

    def __init__(self, feature_range=(0, 1)):
        self.fr = feature_range
        self.a = 0
        self.b = 0

    def fit_transform(self, x):

        if len(x) == 0:
            self.a = 0
            self.b = 0
            return x
        x = np.atleast_2d(x)
        xmax = x.max(axis=0)
        xmin = x.min(axis=0)
        self.a = (self.fr[1] - self.fr[0]) / (xmax-xmin)
        self.a[xmax-xmin <= 0.1] = 10

        self.a[xmax < 0.01] = 1
        self.a[xmax == xmin] = 0  # important !!!
        self.b = self.fr[0] - self.a*xmin
        return self.transform(x)

    def transform(self, x):
        x = np.atleast_2d(x)
        return self.a.astype("float64")*x + self.b.astype("float64")

    def inverse_transform(self, y):
        y = np.atleast_2d(y)
        return (y - self.b) / self.a


class DataScaler:
    def __init__(self):
        self.feat_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feat_a = None
        self.engy_scaler = MinMaxScaler(feature_range=(0, 1))
        self.engy_a = None

    #===========================================================================

    def get_scaler(self, f_feat, f_ds, b_save=True):
        itypes,feat,engy = prepare.r_feat_csv(f_feat)
        print('=DS.get_scaler ', f_feat, 'feat.shape, feat.dtype', feat.shape, feat.dtype)
        
        _ = self.feat_scaler.fit_transform(feat)
        _ = self.engy_scaler.fit_transform(engy)
        
        feat_b      = self.feat_scaler.transform(np.zeros((1, feat.shape[1])))    
        self.feat_a = self.feat_scaler.transform(np.ones((1, feat.shape[1]))) - feat_b
        engy_b      = self.engy_scaler.transform(0)
        self.engy_a = self.engy_scaler.transform(1) - engy_b

        #return self.feat_scaler

    #===========================================================================
    
    def pre_feat(self, feat):
        return self.feat_scaler.transform(feat)


class DataScalers:
    '''
    The wrapper for multiple elements. Generally a dictionary with data scalers for each element.
    Notice the 's'. It is important.
    '''

    def __init__(self, f_ds, f_feat, load=False):

        self.scalers = {}
        self.feat_as = {}
        self.engy_as = {}
        
        for i in range(pm.ntypes):
            self.scalers[pm.atomType[i]] = DataScaler()

        from os import path

        if load and path.isfile(f_ds):
            self.loadDSs_np(f_ds)
        elif path.isfile(f_feat):
            self.get_scalers(f_feat, f_ds, b_save=True)
        else:
            exit(["===Error in DataScaler, don't find ", f_ds, f_feat, '==='])

        return

    def get_scalers(self, f_feat, f_ds, b_save=True):
        itypes,feat,engy = prepare.r_feat_csv(f_feat)
        print('=DS.get_scaler ', f_feat, 'feat.shape, feat.dtype', feat.shape, feat.dtype)
        print('=DS.get_scaler ', f_feat, 'engy.shape, feat.dtype', engy.shape, engy.dtype)
        print('=DS.get_scaler ', f_feat, 'itypes.shape, feat.dtype', itypes.shape, itypes.dtype)

        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            subfeat = feat[itypes == itype]
            subengy = engy[itypes == itype]
            _ = self.scalers[itype].feat_scaler.fit_transform(subfeat) # 确定scale 参数   a 和 b
            _ = self.scalers[itype].engy_scaler.fit_transform(subengy)
            feat_b = self.scalers[itype].feat_scaler.transform(np.zeros((1, subfeat.shape[1]))) # b 的值
            engy_b = self.scalers[itype].engy_scaler.transform(np.zeros((1, subengy.shape[1])))
            self.feat_as[itype] = self.scalers[itype].\
                                 feat_scaler.transform(np.ones((1, subfeat.shape[1]))) - feat_b # a 的值
            self.engy_as[itype] = self.scalers[itype].\
                                 engy_scaler.transform(np.ones((1, subengy.shape[1]))) - engy_b

        if b_save:
            self.save2np(f_ds)
    
        #return self.feat_scalers
    def pre_engy(self, engy, itypes):
            # engy_scaled = cp.zeros_like(engy)
            # for i in range(pm.ntypes):
            #     itype = pm.atomType[i]
            #     engy_scaled[itypes == itype] = self.scalers[itype].\
            #         engy_scaler.transform(engy[itypes == itype])
        return engy

    def pre_feat(self, feat, itypes):
        feat_scaled = np.zeros(feat.shape)
        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            feat_scaled[itypes == itype] = self.scalers[itype].\
                                           feat_scaler.transform(feat[itypes == itype])
        return feat_scaled

    def pre_dfeat(self, dfeat, itypes, nblt):
        dfeat_scaled = np.zeros(dfeat.shape)
        natoms=dfeat.shape[0]
        max_nb=dfeat.shape[1]
        featnum=dfeat.shape[2]
       
        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            dfeat_scaled[(itypes.squeeze()[nblt-1] == itype) & (nblt>0)] = self.feat_as[itype][:,:,np.newaxis]\
                                            *dfeat[(itypes.squeeze()[nblt-1]==itype)&(nblt>0)]
        return dfeat_scaled


    def save2np(self, f_npfile):
        dsnp = []
        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            feat_scaler = self.scalers[itype].feat_scaler
            engy_scaler = self.scalers[itype].engy_scaler
            dsnp.append(np.array(feat_scaler.fr))
            dsnp.append(np.array(feat_scaler.a))
            dsnp.append(np.array(feat_scaler.b))
            dsnp.append(np.array(self.feat_as[itype]))   # 这个值不就是feat_scaler.a  多此一举？
            dsnp.append(np.array(engy_scaler.fr))
            dsnp.append(np.array(engy_scaler.a))
            dsnp.append(np.array(engy_scaler.b))
            dsnp.append(np.array(self.engy_as[itype]))
        dsnp = np.array(dsnp)
        np.save(f_npfile, dsnp)
        print('DataScaler.save2np to', f_npfile, dsnp.dtype, dsnp.shape)
        return

    def loadDSs_np(self, f_npfile):
        dsnp = np.load(f_npfile, allow_pickle=True)

        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            self.scalers[itype].feat_scaler.fr = np.asarray(dsnp[8*i+0])
            self.scalers[itype].feat_scaler.a  = np.asarray(dsnp[8*i+1])
            self.scalers[itype].feat_scaler.b  = np.asarray(dsnp[8*i+2])
            self.feat_as[itype]         = np.asarray(dsnp[8*i+3])
            self.scalers[itype].engy_scaler.fr = np.asarray(dsnp[8*i+4])
            self.scalers[itype].engy_scaler.a  = np.asarray(dsnp[8*i+5])
            self.scalers[itype].engy_scaler.b  = np.asarray(dsnp[8*i+6])
            self.engy_as[itype]         = np.asarray(dsnp[8*i+7])

        print('DataScaler.loadDS_np from', f_npfile, dsnp.dtype, dsnp.shape)
        #for i in range(dsnp.shape[0]):
        #    print("dsnp[i]",i, dsnp[i].shape)
        return
