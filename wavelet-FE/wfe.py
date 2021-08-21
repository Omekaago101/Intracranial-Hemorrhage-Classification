import pywt
import torch
import torch.nn as nn
import numpy as np
from nystrom_attention import NystromAttention
from einops import rearrange

class WaveletTransform(nn.Module):
    """Some Information about WaveletTransform"""
    def __init__(self,input):
        super(WaveletTransform, self).__init__()
        self.input = input

    def forward(self, x):
        coeffs = pywt.dwt2(x,'haar')
        LL, (LH, HL, HH) = coeffs
        LL = torch.tensor(LL)
        LH = torch.tensor(LH)
        HL = torch.tensor(HL)
        HH = torch.tensor(HH)
        x = torch.cat([LL,LH,HL,HH],1)
        return x,LL

class ConvBlock(nn.Module):
    """Some Information about ConvBlock"""
    def __init__(self,in_ch,out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=2,padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class Conv1x1(nn.Module):
    def __init__(self,in_ch,out_ch):
            super(Conv1x1, self).__init__()
            self.conv1x1 = nn.Conv2d(in_ch,out_ch,kernel_size=1)
            self.bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU()
    def forward(self, x):
        print(x.shape[1])
        x = self.relu(self.bn(self.conv1x1(x)))
        return x

class CrossTensorAttention(nn.Module):
    def __init__(self,input):
        super(CrossTensorAttention, self).__init__()
        
            
    
if __name__ == '__main__':
    img = torch.tensor(torch.rand((8,3,224,224)))
    
    model = WaveletTransform(img)
    
    #print(f'Input image shape: {img.shape}')
    
    x,LL= model(img)
    #print(x.shape)
    
    #print(f'After First Decomposition:{x.shape}')
    
    model = WaveletTransform(LL)
    x_1,LL_1= model(LL)
    #print(f'After Second Decomposition: {x_1.shape}')
    '''
    ra = torch.reshape(img,(None,img.shape[1],img.shape[2],img.shape[3]))
    print(ra.shape)
    '''
    real_img = Conv1x1(in_ch=x_1.shape[1],out_ch=64)
    c = real_img(x_1)
    k = nn.AdaptiveAvgPool2d(1)
    v = k(c)
    g = torch.permute(c,(0,3,2,1)) # BHWC
    g1 = torch.permute(c,(0,2,3,1)) #BWHC
    print(g.shape)
    #print(c.shape)
    gg = k(g)
    gg1 = k(g1)
    ggg = rearrange(gg,'b h w c -> b h (c w)')
    print(ggg.shape)
    gg = torch.permute(gg,(0,3,2,1)) #B
    gg1 = torch.permute(gg1,(0,3,2,1))
    print(gg1.shape)