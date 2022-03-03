import torch
import torch.nn as nn
from torch.optim import Adam,lr_scheduler

from .Generator import Generator
from .Discriminator import Discriminator
from .Loss import ContentLoss

class Model:
    '''
    Wrapper to cover discriminator, generator, optimizers and schedulers. Class initializes and handles optimizers and schedulers during training.

    '''
    def __init__(self,config,mode='train'):
        #initialize encoder/decoder block
        self.discriminator = Discriminator(config['height'],config['width']).to(config['device'])
        self.generator = Generator(config['scale'],config['model']['block_depth']).to(config['device'])

        if mode != 'test':
            #initalize loss
            self.psnr_criterion = nn.MSELoss().to(config['device'])
            self.pixel_criterion = nn.MSELoss().to(config['device'])
            self.content_criterion = ContentLoss().to(config['device'])
            self.adversarial_criterion = nn.BCEWithLogitsLoss().to(config['device'])
            #initialize optimizer
            self.d_optimizer = Adam(self.discriminator.parameters(), config['train']['d_lr'], config['train']['d_beta'])
            self.g_optimizer = Adam(self.generator.parameters(), config['train']['g_lr'], config['train']['g_beta'])
            #initialize scheduler
            self.d_scheduler = lr_scheduler.StepLR(self.d_optimizer, config['train']['d_optimizer_step_size'], config['train']['d_optimizer_gamma'])
            self.g_scheduler = lr_scheduler.StepLR(self.g_optimizer, config['train']['g_optimizer_step_size'], config['train']['g_optimizer_gamma'])

        if config['resume']:
            self.restore_checkpoint(config,mode)
        
    def restore_checkpoint(self,config,mode):
        '''
        Restores checkpoint for given config file
        Args:
            config: config dictionary with config
        '''
        if config.resume_d_weight != "":
            # Get pretrained model state dict
            discriminator_checkpoint = torch.load(config['resume_d_weight'])
            self.discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'], strict=config.strict)
            self.discriminator.to(config['device'])
            if mode != 'test':
                #Extracts optimizer state
                self.d_optimizer.load_state_dict(discriminator_checkpoint['optimizer_state_dict'])
                #Extracts scheduler state
                self.d_scheduler.load_state_dict(discriminator_checkpoint['optimizer_state_dict'])            

        if config.resume_g_weight != "":
            # Get pretrained model state dict
            generator_checkpoint = torch.load(config['resume_g_weight'])
            self.generator.load_state_dict(generator_checkpoint['model_state_dict'], strict=config.strict)
            self.generator.to(config['device'])
            if mode!='test':
                #Extracts optimizer state
                self.g_optimizer.load_state_dict(generator_checkpoint['optimizer_state_dict'])
                #Extracts scheduler state
                self.g_scheduler.load_state_dict(generator_checkpoint['scheduler_state_dict'])
    
    def create_discriminator_checkpoint(self,epoch,PATH):
        '''
        Creates a checkpoint for discriminator at PATH
        Args:
            epoch: epoch at which discriminator is saved
            PATH: location where model is saved
        '''
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.discriminator.state_dict(),
                    'optimizer_state_dict': self.d_optimizer.state_dict(),
                    'scheduler_state_dict': self.d_scheduler.state_dict()
                    }, PATH)

    def create_generator_checkpoint(self,epoch,PATH):
        '''
        Creates a checkpoint for generator at PATH
        Args:
            epoch: epoch at which discriminator is saved
            PATH: location where model is saved
        '''
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': self.g_optimizer.state_dict(),
                    'scheduler_state_dict': self.g_scheduler.state_dict()
                    }, PATH)
    
    def epoch_train(self,dataloader,epoch):
        '''
        Trains model for 1 epoch through the dataset
        Args:
            dataloader: data to train model on
            epoch: epoch the model is currently training for
        '''
        self.generator.train()
        self.discriminator.train()
        return None

    def epoch_eval(self,dataloader,epoch):
        '''
        Evaluates model with dataloader for current epoch
        Args:
            dataloader: data to evaluate the model on
            epoch: epoch the model is currently training for
        '''
        self.generator.eval()
        self.discriminator.eval()
        return None

    def inference(self,dataloader):
        '''
        Runs inference with test data
        Args:
            dataloader: data to evaluate the model on
        '''
        self.generator.eval()
        self.discriminator.eval()
        return None