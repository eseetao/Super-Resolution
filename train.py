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
from Core import Model
from Core.utils import create_folder

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='./configs/ImageNet.yaml', help="default config")


if __name__ == '__main__':
    args = parser.parse_args()
    #import pdb;pdb.set_trace()
    with open(args.cfg,'r') as f:
        config = yaml.full_load(f)
    config['device'] = torch.device("cuda:{}".format(config['device']) if torch.cuda.is_available() else "cpu")

    train_loader,val_loader = generate_dataloader(config)#dataloader for training and validation
    SRGAN = Model(config)#initialize model for training/validation
    create_folder(config['MODEL_SAVE_LOCATION'])#create folder for saving models after training
    best_loss = None#best generator loss

    for epoch in range(config['train']['epochs']):

        SRGAN.epoch_train(train_loader,epoch)#trains one epoch
        eval_loss = SRGAN.epoch_eval(val_loader,epoch)#evaluates validation data

        if epoch == 0 or best_loss>eval_loss:
            best_loss = eval_loss
            SRGAN.create_discriminator_checkpoint(epoch,os.path.join(config['MODEL_SAVE_LOCATION'],'discriminator','best_discriminator.pth'))
            SRGAN.create_generator_checkpoint(epoch,os.path.join(config['MODEL_SAVE_LOCATION'],'generator','best_generator.pth'))
        
        if epoch%config['save_interval']==0:
            SRGAN.create_discriminator_checkpoint(epoch,os.path.join(config['MODEL_SAVE_LOCATION'],'discriminator','epoch_{}.pth'.format(epoch)))
            SRGAN.create_generator_checkpoint(epoch,os.path.join(config['MODEL_SAVE_LOCATION'],'generator','epoch_{}.pth'.format(epoch)))    
        import pdb;pdb.set_trace()       