import pywt
import torch
import torch.nn as nn
import numpy as np
import gc

class WaveletTransform(nn.Module):
    """Some Information about WaveletTransform"""
    def __init__(self,channel):
        super(WaveletTransform, self).__init__()
        self.level1_conv1x1 = self.conv1x1(channel[0])
        self.level2_conv1x1 = self.conv1x1(channel[1],level=2)
        self.level3_conv1x1 = self.conv1x1(channel[2],level=3)
        self.conv = nn.Conv2d(3,32,kernel_size=3,stride=2,padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv1x = nn.Conv2d(32,64,kernel_size=1)
        self.bn_orig64 = nn.BatchNorm2d(64)
        self.bn_orig32 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(device="cpu")
        orig_input = x
        print(f'Original Input Tensor Shape: {x.shape}')
        # decomposition 1
        coeffs = pywt.dwt2(x,'haar')
        LL, (LH, HL, HH) = coeffs
        LL,LH,HL,HH = torch.tensor(LL),torch.tensor(LH),torch.tensor(HL),torch.tensor(HH)
        x_l1 = torch.cat([LL,LH,HL,HH],1)
        print(f'Decomposition 1 shape: {x_l1.shape}')
        
        del LL,LH,HL,HH,coeffs
        gc.collect()
        
        # decomposition 2
        coeffs = pywt.dwt2(x_l1,'haar')
        LL1, (LH1, HL1, HH1) = coeffs
        LL1,LH1,HL1,HH1 = torch.tensor(LL1),torch.tensor(LH1),torch.tensor(HL1),torch.tensor(HH1)
        x_l2 = torch.cat([LL1,LH1,HL1,HH1],1)
        print(f'Decomposition 2 shape: {x_l2.shape}')
        
        del LL1,LH1,HL1,HH1,coeffs
        gc.collect()
        
        # decomposition 3
        coeffs = pywt.dwt2(x_l2,'haar')
        LL2, (LH2, HL2, HH2) = coeffs
        LL2, LH2, HL2, HH2 = torch.tensor(LL2),torch.tensor(LH2),torch.tensor(HL2),torch.tensor(HH2)
        x_l3 = torch.cat([LL2,LH2,HL2,HH2],1)
        print(f'Decomposition 3 shape: {x_l3.shape}')
        
        # convolution operations
        orig_input = orig_input.to(device="cuda")
        x_l1 = x_l1.to(device="cuda")
        x_l2 = x_l2.to(device="cuda")
        x_l3 = x_l3.to(device="cuda")
        
        orig_input = self.conv(orig_input)
        orig_input = self.relu(self.bn_orig32(self.avg_pool(orig_input)))
        orig_input = self.relu(self.bn_orig64(self.conv1x(orig_input)))
        level1 = self.level1_conv1x1(x_l1)
        level2 = self.level2_conv1x1(x_l2)
        level3 = self.level3_conv1x1(x_l3)
        
        print(f'Original input shape after convolution: {orig_input.shape}')
        print(f'Level shape after convolution: {level1.shape}')
        print(f'Level2 shape after convolution: {level2.shape}')
        print(f'Level3 shape after convolution: {level3.shape}')
        return orig_input,level1,level2,level3

    def conv1x1(self,channels,level=1):
        net = ""
        if level==1:
            net = nn.Sequential(
            nn.Conv2d(channels[0],channels[1],kernel_size=1),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU()
        )
        else:
            net = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU())
        return net
    
if __name__ == '__main__':
    x = torch.rand((8,3,224,224))
    channel = [[12,32],48]
    wl = WaveletTransform(channel).to(device="cuda")
    o,l1,l2 = wl(x)