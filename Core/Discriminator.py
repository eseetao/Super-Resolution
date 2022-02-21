import torch
import torch.nn as nn
import torch.nn.functional as F
from .Backbone import DiscriminatorBlock

class Discriminator(nn.Module):
    '''
    Discriminates between HR and SR samples 
    Args:
        image_height: height of input image
        image_width: width of input image
    '''
    def __init__(self,image_height,image_width):
        super(Discriminator,self).__init__()
        
        self.backbone_channels = 64
        self.feat_kernel_size=3
        self.feat_stride = 1
        self.dilation = 1
        self.nl = nn.LeakyReLU()
        
        self.feat_extractor = nn.Conv2d(3,self.backbone_channels,kernel_size=self.feat_kernel_size,stride=self.feat_stride,padding=((self.feat_kernel_size-1)*self.dilation)//2,bias=False) 
        
        self.block1 = DiscriminatorBlock(kernel_size=3,in_channels=64,output_channels=64,stride=2)
        self.block2 = DiscriminatorBlock(kernel_size=3,in_channels=64,output_channels=128,stride=1)
        self.block3 = DiscriminatorBlock(kernel_size=3,in_channels=128,output_channels=128,stride=2)
        self.block4 = DiscriminatorBlock(kernel_size=3,in_channels=128,output_channels=256,stride=1)
        self.block5 = DiscriminatorBlock(kernel_size=3,in_channels=256,output_channels=256,stride=2)
        self.block6 = DiscriminatorBlock(kernel_size=3,in_channels=256,output_channels=512,stride=1)
        self.block7 = DiscriminatorBlock(kernel_size=3,in_channels=512,output_channels=512,stride=2)

        # numer of features to vectorize
        # input feature shape: 512 channels, H/16, W/16
        #vectorized feature shape = 512*H*W/(16*16) = (512/256)*H*W = 2*H*W
        self.dense1 = nn.Linear(in_features= 2*(image_height*image_width),out_features=1024)
        self.dense2 = nn.Linear(in_features=1024,out_features=1)
        self.probability = nn.Sigmoid()

    
    def forward(self,x):
        #Extract features from input images
        feat = self.nl(self.feat_extractor(x))
        #Discriminator feature extractor blocks
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)
        feat = self.block4(feat)
        feat = self.block5(feat)
        feat = self.block6(feat)
        feat = self.block7(feat)
        #Flatten/vectorize input features
        feat = torch.flatten(feat,start_dim=1)
        feat = self.nl(self.dense1(feat))
        #Make predictionon data
        decision = self.probability(self.dense2(feat))
        return decision
