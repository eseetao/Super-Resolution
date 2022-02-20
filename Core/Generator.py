import torch
import torch.nn as nn
import torch.nn.functional as F
from Backbone import GeneratorBlock

class Generator(nn.Module):
    '''
    Discriminates between HR and SR samples 
    Kwargs:

    '''
    def __init__(self,scale=4):
        super(Generator,self).__init__()
        self.scale = scale
        self.backbone_channels = 64
        self.feat_kernel_size=9
        self.feat_stride = 1
        self.dilation = 1
        self.nl = nn.PReLU()
        
        self.feat_extractor = nn.Conv2d(3,self.backbone_channels,kernel_size=self.feat_kernel_size,stride=self.feat_stride,padding=((self.feat_kernel_size-1)*self.dilation)//2,bias=False) 
        
        self.block1 = GeneratorBlock(kernel_size=3,in_channels=self.backbone_channels)
        self.block2 = GeneratorBlock(kernel_size=3,in_channels=self.backbone_channels)
        self.block3 = GeneratorBlock(kernel_size=3,in_channels=self.backbone_channels)
        self.block4 = GeneratorBlock(kernel_size=3,in_channels=self.backbone_channels)
        self.block5 = GeneratorBlock(kernel_size=3,in_channels=self.backbone_channels)
        self.block6 = GeneratorBlock(kernel_size=3,in_channels=self.backbone_channels)
        self.block7 = GeneratorBlock(kernel_size=3,in_channels=self.backbone_channels)

        self.feat_combine = nn.Conv2d(self.backbone_channels,self.backbone_channels,kernel_size=self.feat_kernel_size,stride=self.feat_stride,padding=((self.feat_kernel_size-1)*self.dilation)//2,bias=False) 
        self.BN1 = nn.BatchNorm2d(self.backbone_channels)

    def forward(self,x):

        feat = self.nl(self.feat_extractor(x))

        residual_feat = self.block1(feat)
        residual = self.block2(residual_feat)
        residual = self.block3(residual)
        residual = self.block4(residual)
        residual = self.block5(residual)
        residual = self.block6(residual)
        residual = self.block7(residual)

        return residual