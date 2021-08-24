import torch, torch.nn as nn
import numpy as np
from wavelet_feature_ex import WaveletTransform
from cross_tensor_self_attn import CrossTensorAttention

class WaveletLeTransform(nn.Module):
    """Some Information about WaveletLeTransform"""
    def __init__(self,channels,decompose_level,num_convs):
        super(WaveletLeTransform, self).__init__()
        self.wavelet = WaveletTransform(channels,decompose_level,num_convs)
        self.cross_attention = CrossTensorAttention()
        
    def forward(self, x):
        orig_input,level1,level2 = self.wavelet(x)
        orig_input = self.cross_attention(orig_input)
        level1 = self.cross_attention(level1)
        level2 = self.cross_attention(level2)
        return orig_input,level1,level2
    
if __name__ == '__main__':
    c = torch.rand((8,3,224,224))
    print(c.dtype,c.device)
    
    channels = [[3,32,64,128],[12,32,64,128],[48,64,128]]
    num_convs = [3,3,2]
    level = [1,2,3]
    net = WaveletLeTransform(channels=channels,decompose_level=level,num_convs=num_convs)
    orig_input,level1,level2 = net(c)