import torch, torch.nn as nn
import numpy as np
from wavelet_feature_ex import WaveletTransform
from self_attention import SelfAttention
from cross_attention import CrossAttention
import collections
import gc
from models.levit import LeViT

class WaveletLeTransform(nn.Module):
    """Some Information about WaveletLeTransform"""
    def __init__(self,wavelet_ch,layer_channels,decompose_level):
        super(WaveletLeTransform, self).__init__()
        
        self.original_make_layer = self._make_layer(layer_channels[0],decompose_level[0])
        self.l2_make_layer = self._make_layer(layer_channels[1],decompose_level[1])
        self.l3_make_layer = self._make_layer(layer_channels[2],decompose_level[2])
        self.l4_make_layer = self._make_layer(layer_channels[3],decompose_level[3])
        
        self.wavelet = WaveletTransform(wavelet_ch)
        self.self_attention = SelfAttention()
        self.conv1x1 = nn.Conv2d(in_channels=3072,out_channels=1024,kernel_size=1)
        self.bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU()
        
        self
        
    def forward(self, x):
        # wavelet feature extraction
        orig_input,level2,level3,level4 = self.wavelet(x)
        
        #cross self attention of the original input
        orig_input = self.self_attention(orig_input)
        # convolution operations on the original input. Output shape (Bx256x14x14)
        orig_input = self.original_make_layer(orig_input)
        print(f'original->{orig_input.shape}')
        #cross self attention of first decomposition of wavelet feature extraction
        level2 = self.self_attention(level2)
        # convolution operations on level 2. Output shape (Bx256x14x14)
        level2 = self.l2_make_layer(level2)
        print(f'level 2->{level2.shape}')
        #cross self attention of 2nd decomposition of wavelet feature extraction
        level3 = self.self_attention(level3)
        # convolution operations on level 3. Output shape (Bx256x14x14)
        level3 = self.l3_make_layer(level3)
        print(f'level 3->{level3}')
        
        #cross self attention of 3rd decomposition of wavelet feature extraction
        level4 = self.self_attention(level4)
        # convolution operations on level 4. Output shape (Bx256x14x14)
        level4 = self.l4_make_layer(level4)
        print(f'level 4->{level4.shape}')
        
        tensor_list = [[orig_input,level2,level3,level4],[level2,level3,level4],[level3,level4]]
        
        del orig_input,level2,level3,level4
        gc.collect()
        
        # cross attention for the individual cross self tensor attention feature maps
        cross_attn = CrossAttention(tensor_list)
        crx_attn = cross_attn.cross_attention()
        
        # convolution operations to reduce feature map to 1024
        x = self.relu(self.bn(self.conv1x1(crx_attn)))
        return x
    
    def _make_layer(self,channels,level):
        dict_layers = collections.OrderedDict()
        if level == 1:
            dict_layers["'conv3x3'"] = nn.Conv2d(channels[0],channels[1],kernel_size=3,stride=2,padding=1)
            dict_layers["'avg_pool'"] = nn.AvgPool2d(kernel_size=2,stride=2)
            dict_layers["'bn1'"] = nn.BatchNorm2d(channels[1])
            dict_layers["'relu1'"] = nn.ReLU()
            dict_layers["'conv1x1'"] = nn.Conv2d(channels[1],channels[2],kernel_size=1)
            dict_layers["'bn2'"] = nn.BatchNorm2d(channels[2])
            dict_layers["'relu2'"] = nn.ReLU()
                           
        if level == 2:
            dict_layers["'con1x1'"] = nn.Conv2d(channels[0],channels[1],kernel_size=1)
            dict_layers["'conv3x3'"] = nn.Conv2d(channels[1],channels[2],kernel_size=3,stride=2,padding=1)
            dict_layers["'avg_pool'"] = nn.AvgPool2d(kernel_size=2,stride=2)
            dict_layers["'bn1'"] = nn.BatchNorm2d(channels[2])
            dict_layers["'relu1'"] = nn.ReLU()
            dict_layers["'con1x1_'"] = nn.Conv2d(channels[2],channels[3],kernel_size=1)
            dict_layers["'bn2'"] = nn.BatchNorm2d(channels[3])
            dict_layers["'relu2'"] = nn.ReLU()
            
        elif level == 3:
            dict_layers["'con1x1'"] = nn.Conv2d(channels[0],channels[1],kernel_size=1)
            dict_layers["'conv3x3'"] = nn.Conv2d(channels[1],channels[2],kernel_size=3,stride=2,padding=1)
            dict_layers["'avg_pool'"] = nn.AvgPool2d(kernel_size=2,stride=2)
            dict_layers["'bn1'"] = nn.BatchNorm2d(channels[2])
            dict_layers["'relu1'"] = nn.ReLU()
            dict_layers["'con1x1_'"] = nn.Conv2d(channels[2],channels[3],kernel_size=1)
            dict_layers["'bn2'"] = nn.BatchNorm2d(channels[3])
            dict_layers["'relu2'"] = nn.ReLU()
          
        elif level == 4:
            dict_layers["'con1x1'"] = nn.Conv2d(channels[0],channels[1],kernel_size=1)
            dict_layers["'avg_pool'"] = nn.AvgPool2d(kernel_size=2,stride=2)
            dict_layers["'bn1'"] = nn.BatchNorm2d(channels[1])
            dict_layers["'relu1'"] = nn.ReLU()
        
        return nn.Sequential(dict_layers)
    
if __name__ == '__main__':
    c = torch.rand((32,3,224,224))
    print(c.dtype,c.device)
    c = c.type(torch.cuda.FloatTensor)
    wavelet_channels = [[12,32],48,192]
    conv_channels = [[64,128,256],[32,64,128,256],[48,64,128,256],[192,256]]
    level = [1,2,3,4]
    net = WaveletLeTransform(wavelet_channels,conv_channels,level).to(device="cuda")
    x = net(c)
    print(x.shape)