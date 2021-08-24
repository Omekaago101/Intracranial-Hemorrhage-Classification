import pywt
import torch
import torch.nn as nn
import numpy as np
import gc
import collections

class WaveletTransform(nn.Module):
    """Some Information about WaveletTransform"""
    def __init__(self,channels,levels,num_convs):
        super(WaveletTransform, self).__init__()
        self.level2 = self._make_layer(channels[2],num_convs[2],levels[2])
        self.orig_input = self._make_layer(channels[0],num_convs[0],levels[0])
        self.level1 = self._make_layer(channels[1],num_convs[1],levels[1])

    def forward(self, x):
        orig_input = x
        
        # decomposition 1
        coeffs = pywt.dwt2(x,'haar')
        LL, (LH, HL, HH) = coeffs
        LL,LH,HL,HH = torch.tensor(LL),torch.tensor(LH),torch.tensor(HL),torch.tensor(HH)
        x_l1 = torch.cat([LL,LH,HL,HH],1)
        print(LL.shape)
        print(LH.shape)
        print(HL.shape)
        print(HH.shape)
        #del LL,LH,HL,HH,coeffs
        #gc.collect()
        
        # decomposition 2
        coeffs = pywt.dwt2(x_l1,'haar')
        LL1, (LH1, HL1, HH1) = coeffs
        LL1,LH1,HL1,HH1 = torch.tensor(LL1),torch.tensor(LH1),torch.tensor(HL1),torch.tensor(HH1)
        
        x_l2 = torch.cat([LL1,LH1,HL1,HH1],1)
        print(x_l2.shape)
        #del LL1,LH1,HL1,HH1,coeffs
        #gc.collect()
        
        # convolution operations
        orig_input = self.orig_input(orig_input)
        level1 = self.level1(x_l1)
        level2 = self.level2(x_l2)
        
        return orig_input,level1,level2
    
    def _make_layer(self,channels,num_convs,level):
        dict_layers = collections.OrderedDict()
        for i in range(num_convs):
            if level == 2 and i == 0:
                layer = nn.Conv2d(channels[i],channels[i+1],kernel_size=1)
                bn = nn.BatchNorm2d(channels[i+1])
                conv ='conv1x1_level_'+str(level)+'_'+str(i+1)
                bn_l = "bn_level_"+str(level)+'_'+str(i+1)
                dict_layers[conv] = layer
                dict_layers[bn_l] = bn
            elif level == 3 and i == 0:
                layer = nn.Conv2d(channels[i],channels[i+1],kernel_size=1)
                bn = nn.BatchNorm2d(channels[i+1])
                conv ='conv1x1_level_'+str(level)+'_'+str(i+1)
                bn_l = "bn_level_"+str(level)+'_'+str(i+1)
                dict_layers[conv] = layer
            else:
                layer = nn.Conv2d(channels[i],channels[i+1],kernel_size=3,stride=2,padding=1)
                bn = nn.BatchNorm2d(channels[i+1])
                conv ='conv_level_'+str(level)+'_'+str(i+1)
                bn_l = "bn_level_"+str(level)+'_'+str(i+1)
                dict_layers[conv] = layer
                dict_layers[bn_l] = bn
            
            if i == num_convs-1:
                relu = "relu_"+str(level)
                dict_layers[relu] = nn.ReLU()
        
        return nn.Sequential(dict_layers)