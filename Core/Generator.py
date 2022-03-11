import torch
import torch.nn as nn
import torch.nn.functional as F
from .Backbone import GeneratorBlock,UpsampleBlock
import math 

class Generator(nn.Module):
    '''
    Discriminates between HR and SR samples 
    Kwargs:
        scale: resolution to upsample the image by
        default: 4 (other option 8)

        block_depth: number of generator blocks used for backbone of generator
        defalut: 7
    '''
    def __init__(self,scale=4,block_depth=7):
        assert(scale==4 or scale==8),"Current model designed for upsampling by 4 or by 8"
        super(Generator,self).__init__()
        self.scale = scale
        self.backbone_channels = 64
        self.feat_kernel_size=9
        self.feat_stride = 1
        self.dilation = 1
        self.nl = nn.PReLU()
        
        self.feat_extractor = nn.Conv2d(3,self.backbone_channels,kernel_size=self.feat_kernel_size,stride=self.feat_stride,padding=((self.feat_kernel_size-1)*self.dilation)//2,bias=False) 
        
        generator_blocks = [GeneratorBlock(kernel_size=3,channels=self.backbone_channels) for _ in range(block_depth)]
        self.backbone = nn.Sequential(*generator_blocks)

        combine_feat_ks = 3
        self.combine_feat = nn.Conv2d(self.backbone_channels,self.backbone_channels,kernel_size=combine_feat_ks,stride=self.feat_stride,padding=((combine_feat_ks-1)*self.dilation)//2,bias=False) 
        self.BN1 = nn.BatchNorm2d(self.backbone_channels)

        upsample_rounds = math.floor(math.log2(scale))
        upsample_layers = [UpsampleBlock(self.backbone_channels) for _ in range(upsample_rounds)]
        self.Upsample = nn.Sequential(*upsample_layers)

        self.condense_to_image = nn.Conv2d(self.backbone_channels,3,kernel_size=self.feat_kernel_size,stride=self.feat_stride,padding=((self.feat_kernel_size-1)*self.dilation)//2,bias=False)

    def forward(self,x):

        feat = self.nl(self.feat_extractor(x))
        residual = self.backbone(feat)
        residual = feat + self.BN1(self.combine_feat(residual))
        upsampled = self.Upsample(residual)
        upsampled = self.condense_to_image(upsampled)
        return upsampled