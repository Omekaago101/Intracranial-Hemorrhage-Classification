import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CrossAttention():
    """Some Information about CrossAttention"""
    def __init__(self,attn_tensors):
        super(CrossAttention, self).__init__()
        self.attn_tensors = attn_tensors
    
    def cross_attention(self):
        crx_attentions = []
        for tensor in self.attn_tensors:
            for x in range(len(tensor)-1):
                crx_attentions.append(self.compute_cross_attention(tensor[0],tensor[x+1]))
        attentions = torch.cat(crx_attentions,1)
        print(attentions.shape)
        return attentions
    
    def compute_cross_attention(self,t1,t2):
        print(f'Tensor 1 shape: {t1.shape}')
        print(f'Tensor 2 shape: {t2.shape}')
        
        lui = rearrange(t1,"b c w h->b (w h) c")
        gui = rearrange(t1,"b c w h->b c (w h)")
    
        print(f'lui shape: {lui.shape}')
        print(f'gui shape: {gui.shape}')
        
        scale_factor = 0.002
        hvi = rearrange(t2,"b c w h->b (w h) c")
        uj = rearrange(t2,"b c w h->b c (w h)")
        
        print(f'hvi shape: {hvi.shape}')
        print(f'uj shape: {uj.shape}')
        
        hvj_uvi = torch.matmul(hvi,gui)
        
        print(f'hvj_uvi shape: {hvj_uvi.shape}')
        
        pij = F.softmax(hvj_uvi,1)
        
        print(f'pij shape: {pij.shape}')
        
        Ci = torch.matmul(pij,lui)  
        print(f'Ci shape after Mat Mul: {Ci.shape}')  
        
        Ci = torch.reshape(Ci,t1.shape)
        print(f'Ci shape after reshape: {Ci.shape}')  
        Ci = scale_factor*Ci
        
        Si = Ci+t1
        print(f'Si shape: {Si.shape}')  
        
        Cj = torch.matmul(uj,pij)
        print(f'Cj shape after Mat Mul: {Cj.shape}')  
        Cj = torch.reshape(Cj,t2.shape)
        print(f'Cj shape after reshape: {Cj.shape}')  
        Cj = scale_factor*Cj
        Sj = Cj+t2
        print(f'Sj shape: {Sj.shape}') 
        
        Zij = torch.cat([Si,Sj],1)
        return Zij