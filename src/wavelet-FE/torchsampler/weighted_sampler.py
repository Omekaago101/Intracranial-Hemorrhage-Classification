from os import replace
from torch.utils.data.sampler import Sampler
import pandas as pd
import numpy as np

class ImbalancedDatasetSampler(Sampler):
    def __init__(self,dataset,class_probs,intra_calss_probs,num_sample=None):
        self.n = len(dataset) if num_sample is None else num_sample
        self.class_probs = class_probs
        self.intr_calss_probs = intra_calss_probs
        self.any_prop = class_probs[0]
        self.epidural_prop = class_probs[1]
        self.intraparenchymal_prop = class_probs[2]
        self.intraventricular_prop = class_probs[3]
        self.subarachnoid_prop = class_probs[4]
        self.subdural_prop = class_probs[5]
        
        #intra class probabilities
        self.any_1 = intra_calss_probs[0][0]
        self.any_0 = intra_calss_probs[0][1]
        
        self.epid_0 = intra_calss_probs[1][0]
        self.epid_1 = intra_calss_probs[1][1]
        
        self.intrap_0 = intra_calss_probs[2][0]
        self.intrap_1 = intra_calss_probs[2][1]
        
        self.intrav_0 = intra_calss_probs[3][0]
        self.intrav_1 = intra_calss_probs[3][1]
        
        self.subara_0 = intra_calss_probs[4][0]
        self.subara_1 = intra_calss_probs[4][1]
        
        self.subdural_0 = intra_calss_probs[5][0]
        self.subdural_1 = intra_calss_probs[5][1]
        
        self.any_1_indices,self.any_0_indices,self.epid_1_indices,self.epid_0_indices,\
        self.intrap_1_indices,self.intrap_0_indices,self.intrav_1_indices,\
        self.intrav_0_indices,self.subara_1_indices,self.subara_0_indices,\
        self.subdural_1_indices,self.subdural_0_indices = self.get_indices(dataset)
 
    def __len__(self):
        return self.n
    
    def __iter__(self):
        any_1 = np.random.choice(self.any_1_indices,int(self.n*self.any_1),replace=True)
        any_0 = np.random.choice(self.any_0_indices,int(self.n*self.any_0),replace=False)
        any_idx = np.hstack([any_1,any_0])
        any_class = np.random.choice(any_idx,int(self.n*self.any_prop),replace=False)
        
        epid_1 = np.random.choice(self.epid_1_indices,int(self.n*self.epid_1),replace=True)
        epid_0 = np.random.choice(self.epid_0_indices,int(self.n*self.epid_0),replace=False)
        epid_idx = np.hstack([epid_1,epid_0])
        epid_class = np.random.choice(epid_idx,int(self.n*self.epidural_prop),replace=True)
        
        intrap_1 = np.random.choice(self.intrap_1_indices,int(self.n*self.intrap_1),replace=True)
        intrap_0 = np.random.choice(self.intrap_0_indices,int(self.n*self.intrap_0),replace=False)
        intrap_idx = np.hstack([intrap_1,intrap_0])
        intrap_class = np.random.choice(intrap_idx,int(self.n*self.intraparenchymal_prop),replace=True)
        
        intrav_1 = np.random.choice(self.intrav_1_indices,int(self.n*self.intrav_1),replace=True)
        intrav_0 = np.random.choice(self.intrav_0_indices,int(self.n*self.intrav_0),replace=False)
        intrav_idx = np.hstack([intrav_1,intrav_0])
        intrav_class = np.random.choice(intrav_idx,int(self.n*self.intraventricular_prop),replace=True)
        
        subara_1 = np.random.choice(self.subara_1_indices,int(self.n*self.subara_1),replace=True)
        subara_0 = np.random.choice(self.subara_0_indices,int(self.n*self.subara_0),replace=False)
        subara_idx = np.hstack([subara_1,subara_0])
        subara_class = np.random.choice(subara_idx,int(self.n*self.subarachnoid_prop),replace=True)
        
        subdural_1 = np.random.choice(self.subdural_1_indices,int(self.n*self.subdural_1),replace=True)
        subdural_0 = np.random.choice(self.subdural_0_indices,int(self.n*self.subdural_0),replace=False)
        subdural_idx = np.hstack([subdural_1,subdural_0])
        subdural_class = np.random.choice(subdural_idx,int(self.n*self.subdural_prop),replace=True)
        
        idxs = np.hstack([any_class,epid_class,intrap_class,intrav_class,subara_class,subdural_class])
        np.random.shuffle(idxs)
        idxs = idxs[:self.n]
        return iter(idxs)
        pass
    
    def get_indices(self,dataset):
        # get the indices of the various classes
        any_1 = dataset[dataset['any']==1]
        any_0 = dataset[dataset['any']==0]
        any_1_indices = any_1.index
        any_0_indices = any_0.index
        
        epid_1 = dataset[dataset['epidural']==1]
        epid_0 = dataset[dataset['epidural']==0]
        epid_1_indices = epid_1.index
        epid_0_indices = epid_0.index
        
        intrap_1 = dataset[dataset['intraparenchymal']==1]
        intrap_0 = dataset[dataset['intraparenchymal']==0]
        intrap_1_indices = intrap_1.index
        intrap_0_indices = intrap_0.index
        
        intrav_1 = dataset[dataset['intraventricular']==1]
        intrav_0 = dataset[dataset['intraventricular']==0]
        intrav_1_indices = intrav_1.index
        intrav_0_indices = intrav_0.index
        
        subara_1 = dataset[dataset['subarachnoid']==1]
        subara_0 = dataset[dataset['subarachnoid']==0]
        subara_1_indices = subara_1.index
        subara_0_indices = subara_0.index
        
        subdural_1 = dataset[dataset['subdural']==1]
        subdural_0 = dataset[dataset['subdural']==0]
        subdural_1_indices = subdural_1.index
        subdural_0_indices = subdural_0.index
        
        return any_1_indices[:self.n],any_0_indices[:self.n],epid_1_indices[:self.n],\
            epid_0_indices[:self.n],intrap_1_indices[:self.n],intrap_0_indices[:self.n],\
            intrav_1_indices[:self.n],intrav_0_indices[:self.n],subara_1_indices[0:self.n],\
            subara_0_indices[:self.n],subdural_1_indices[:self.n],subdural_0_indices[:self.n]