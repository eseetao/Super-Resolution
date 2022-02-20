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
        residual = self.nl(residual)
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
    def __init__(self,kernel_size=3,channels=64,stride=1,dilation=1):
        super(DiscriminatorBlock,self).__init__()
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=kernel_size,stride=stride,padding=((kernel_size-1)*dilation)//2,bias=False) 
        self.BN1 = nn.BatchNorm2d(channels)
        self.nl = nn.LeakyReLU(0.2)
    
    def forward(self,x):
        residual = self.conv1(x)
        residual = self.BN1(residual)
        residual = self.nl(residual)
        return residual