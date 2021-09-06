import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
#from nystrom_attention import Nystromformer
import gc
import collections
import torch.nn.functional as F

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

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()  
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)      
        self.relu = nn.ReLU()
        #self.layers = self._make_layer(channels=channels,level=level)
        #print(self.layers)
        
    def forward(self, x):
        x = self._self_attention(x)
        return x
    
    def _self_attention(self,x):
        lui = rearrange(x,"b c w h -> b c (w h)")
        gui = rearrange(x,"b c w h -> b c (w h)")
        hvi = rearrange(x,"b c w h -> b (w h) c")
        
        UiVi = torch.matmul(hvi,gui)
        attn_map = F.softmax(UiVi,1)
        
        Ai = torch.matmul(lui,attn_map)
        D = torch.reshape(Ai,x.shape)
        D = D*0.002
        
        Oi = x+D
        return Oi
    
    

if __name__ == '__main__':
    x = torch.rand((8,48,56,56))
    net = SelfAttention().to(device="cuda")
    y = net(x)
    print(y.shape)