import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from nystrom_attention import Nystromformer
import gc
import collections

class ConvBlock(nn.Module):
    """Some Information about ConvBlock"""
    def __init__(self,channels,bn=False, activation=False,pooling=False,level=None,num_convs=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(channels[0],channels[1],kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(channels[1],128,kernel_size=3,stride=2,padding=1)
        #self.bn = nn.BatchNorm2d(out_ch)
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.relu = nn.ReLU()
        self.pooling = pooling
        self.activation = activation
        self.bn = bn
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.pooling:
            x = self.avg_pool(x)
        if self.activation:
            x = self.activation(x)
        return x

        
class Conv1x1(nn.Module):
    def __init__(self,channels,bn=False,activation=False):
            super(Conv1x1, self).__init__()
            self.conv1x1 = nn.Conv2d(channels[0],channels[1],kernel_size=1)
            self.batch_norm = nn.BatchNorm2d(channels[1])
            self.relu = nn.ReLU()
            self.activation = activation
            self.bn = bn
            
    def forward(self, x):
        x = self.conv1x1(x)
        if self.bn:
            x = self.batch_norm(x)
        if self.activation:
            x = self.relu(x)
        return x

class CrossTensorAttention(nn.Module):
    def __init__(self):
        super(CrossTensorAttention, self).__init__()  
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)      
        self.relu = nn.ReLU()
        self.nystrom = NystromSelfAttention()
        
    def forward(self, x):
        x = self.nystrom(x)
        return self.relu(self.avg_pool(x))
        
class NystromSelfAttention(nn.Module):
    def __init__(self):
        super(NystromSelfAttention, self).__init__()
        
    def forward(self, x):
        return self.calculate_nystrom_attn(x)
    
    def calculate_nystrom_attn(self,x):
        _c_,_w_,_h_ = x.shape[1],x.shape[2],x.shape[3]
        
        x = x.to(device="cuda")
        x = x.type(torch.cuda.FloatTensor)
        print(x.device,type(x),x.dtype)
        # mode embedding
        c_embd = Conv1x1([_c_,_c_]).to(device="cuda")
        w_embd = Conv1x1([_c_,_w_]).to(device="cuda")
        h_embd = Conv1x1([_c_,_h_]).to(device="cuda")
        
        x
        c_ = c_embd(x) # -> (b,c,w,h)
        w_ = w_embd(x) # -> (b,wc,w,h) wc is the new channel which is equal to the width of the input tensor
        h_ = h_embd(x) # -> (b,hc,w,h) hc is the new channel which is equal to the height of the input tensor
        
        del _c_,_w_,_h_
        gc.collect()
        
        # reshape embeddings
        c_rshp = rearrange(c_,"b c w h -> b c (h w)")
        w_rshp = rearrange(w_,"b c w h -> b w (c h)")
        h_rshp = rearrange(h_,"b c w h -> b h (c w)")
        
        # nystrom_attention for embeddings
        c_em_nys_attn = self.nystrom_attn(c_rshp.shape[2],c_rshp)
        w_em_nys_attn = self.nystrom_attn(w_rshp.shape[2],w_rshp)
        h_em_nys_attn = self.nystrom_attn(h_rshp.shape[2],h_rshp)
        
        del c_rshp,w_rshp,h_rshp
        gc.collect()
        
        # reshape attentions
        c_em_nys_attn = torch.reshape(c_em_nys_attn,(x.shape))
        w_em_nys_attn = torch.reshape(w_em_nys_attn,(w_.shape))
        h_em_nys_attn = torch.reshape(h_em_nys_attn,(h_.shape))
        
        del c_,w_,h_
        gc.collect()
        
        # increase the channels for the width and height embeddings
        incw_ch = Conv1x1([w_em_nys_attn.shape[1],128]).to(device="cuda") # increase channels for width embedding
        inch_ch = Conv1x1([h_em_nys_attn.shape[1],128]).to(device="cuda") # increase channels for height embedding
        w_em_nys_attn = incw_ch(w_em_nys_attn)
        h_em_nys_attn = inch_ch(h_em_nys_attn)
        
        del incw_ch,inch_ch
        gc.collect()
        
        # concatenate the tensors along the channel axis. This would increase the number of channels
        # 1x1 convolution would be used to reduce the channel size to 128
        x = torch.cat([c_em_nys_attn,w_em_nys_attn,h_em_nys_attn],dim=1)
        conv1x1 = Conv1x1([x.shape[1],128],bn=True,activation=True).to(device="cuda") 
        return conv1x1(x)
    
    def nystrom_attn(self,dim,input,mask=None):
        model = Nystromformer(
        dim = dim,
        dim_head = 16,
        heads = 8,
        depth = 4,
        num_landmarks = 32,
        pinv_iterations = 6
        ).to(device="cuda")
        x = model(input,mask=mask)
        return x