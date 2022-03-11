import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorBlock(nn.Module):
    '''
    Defines Single Residual Block for Generator
    Kwargs:
        kernel_size: kernel size for conv blocks
        default: 3
        
        channels: number of channels for conv blocks
        default:64

        stride: stride for conv blocks
        defaults:1

        dilation: dilation for conv blocks
        default:1
    '''
    def __init__(self,kernel_size=3,channels=64,stride=1,dilation=1):
        super(GeneratorBlock,self).__init__()
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=kernel_size,stride=stride,padding=((kernel_size-1)*dilation)//2,bias=False) 
        self.BN1 = nn.BatchNorm2d(channels)
        self.nl = nn.PReLU()
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=kernel_size,stride=stride,padding=((kernel_size-1)*dilation)//2,bias=False) 
        self.BN2 = nn.BatchNorm2d(channels)
    
    def forward(self,x):
        residual = self.conv1(x)
        residual = self.BN1(residual)
        residual = self.nl(residual)
        residual = self.conv2(residual)
        residual = self.BN2(residual)
        return x+residual

class DiscriminatorBlock(nn.Module):
    '''
    Defines Template Block for Discriminator
    Kwargs:
        kernel_size: kernel size for conv blocks
        default: 3
        
        channels: number of channels for conv blocks
        default:64

        stride: stride for conv blocks
        defaults:1

        dilation: dilation for conv blocks
        default:1
    '''
    def __init__(self,kernel_size=3,in_channels=64,output_channels=64,stride=1,dilation=1):
        super(DiscriminatorBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,output_channels,kernel_size=kernel_size,stride=stride,padding=((kernel_size-1)*dilation)//2,bias=False) 
        self.BN1 = nn.BatchNorm2d(output_channels)
        self.nl = nn.LeakyReLU(0.2)
    
    def forward(self,x):
        residual = self.conv1(x)
        residual = self.BN1(residual)
        residual = self.nl(residual)
        return residual

class UpsampleBlock(nn.Module):
    def __init__(self, channels,kernel_size=3,stride=1,dilation=1):
        super(UpsampleBlock, self).__init__()
        self.upsample_conv= nn.Conv2d(channels, channels * 4, kernel_size=kernel_size,stride=stride,padding=((kernel_size-1)*dilation)//2,bias=False)
        self.shuffle = nn.PixelShuffle(2)
        self.nl = nn.PReLU()
        
    def forward(self, x):
        expanded = self.upsample_conv(x)
        upsampled = self.nl(self.shuffle(expanded))
        return upsampled   