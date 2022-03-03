import os
import time
import argparse
import yaml

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
#from torch.utils.tensorboard import SummaryWriter

from Core.datasets import generate_dataloader
from Core import Discriminator, Generator, Loss

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='./configs/ImageNet.yaml', help="default config")


if __name__ == '__main__':
    args = parser.parse_args()
    #import pdb;pdb.set_trace()
    with open(args.cfg,'r') as f:
        config = yaml.full_load(f)
    
    train_loader,val_loader = generate_dataloader(config)
