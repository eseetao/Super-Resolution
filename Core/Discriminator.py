import torch
import torch.nn as nn
import torch.nn.functional as F
from Backbone import DiscriminatorBlock

class Discriminator(nn.Module):
    '''
    Discriminates between HR and SR samples 
    Kwargs:

    '''
    def __init__(self):
        super(Discriminator,self).__init__()
        
        self.backbone_channels = 64
        self.feat_kernel_size=3
        self.feat_stride = 1
        self.dilation = 1
        
        self.feat_extractor = nn.Conv2d(3,self.backbone_channels,kernel_size=self.feat_kernel_size,stride=self.feat_stride,padding=((self.feat_kernel_size-1)*self.dilation)//2,bias=False) 
        self.block1 = DiscriminatorBlock()
        self.block2 = DiscriminatorBlock()
        self.block3 = DiscriminatorBlock()
        self.block4 = DiscriminatorBlock()
        self.block5 = DiscriminatorBlock()
        self.block6 = DiscriminatorBlock()
        self.block7 = DiscriminatorBlock()
    
    def forward(self,x):
        feat = self.feat_extractor(x)
        return feat
