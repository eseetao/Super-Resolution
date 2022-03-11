from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.optim import Adam,lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from .Generator import Generator
from .Discriminator import Discriminator
from .Loss import ContentLoss

class Model:
    '''
    Wrapper to cover discriminator, generator, optimizers and schedulers. Class initializes and handles optimizers and schedulers during training.

    '''
    def __init__(self,config):
        #initialize encoder/decoder block
        self.discriminator = Discriminator(config['crop_size'],config['crop_size']).to(config['device'])
        self.generator = Generator(config['scale'],config['model']['block_depth']).to(config['device'])
        #save device for running ops on
        self.device = config['device']
        #initialize scaler for propagating losses
        self.scaler = amp.GradScaler()
        #Initialize summary writer to write status of scalars into
        self.writer = SummaryWriter(config['visualize_location'])
        #weights for combining losses
        self.pixel_loss_weight = config['model']['pixel_weight']
        self.content_loss_weight = config['model']['content_weight']
        self.adverserial_loss_weight = config['model']['adverserial_weight']

        if 'train' in config.keys():
            #initalize loss
            self.psnr_criterion = nn.MSELoss().to(config['device'])
            self.pixel_criterion = nn.MSELoss().to(config['device'])
            self.content_criterion = ContentLoss().to(config['device'])
            self.adversarial_criterion = nn.BCEWithLogitsLoss().to(config['device'])
            #initialize optimizer

            self.d_optimizer = Adam(self.discriminator.parameters(), config['train']['d_lr'], (0.9, 0.999))
            self.g_optimizer = Adam(self.generator.parameters(), config['train']['g_lr'],(0.9, 0.999))
            #initialize scheduler
            self.d_scheduler = lr_scheduler.StepLR(self.d_optimizer, config['train']['d_optimizer_step_size'], config['train']['d_optimizer_gamma'])
            self.g_scheduler = lr_scheduler.StepLR(self.g_optimizer, config['train']['g_optimizer_step_size'], config['train']['g_optimizer_gamma'])

        if 'resume' in config.keys():
            self.restore_checkpoint(config)
        
    def restore_checkpoint(self,config):
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
            if 'train' in config.keys():
                #Extracts optimizer state
                self.d_optimizer.load_state_dict(discriminator_checkpoint['optimizer_state_dict'])
                #Extracts scheduler state
                self.d_scheduler.load_state_dict(discriminator_checkpoint['optimizer_state_dict'])            

        if config.resume_g_weight != "":
            # Get pretrained model state dict
            generator_checkpoint = torch.load(config['resume_g_weight'])
            self.generator.load_state_dict(generator_checkpoint['model_state_dict'], strict=config.strict)
            self.generator.to(config['device'])
            if 'train' in config.keys():
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
    
    def discriminator_step(self,hr,sr):
        '''
        Computes and propagates loss for discriminator
        Args:   
            hr: high resolution samples
            sr: super resolution samples
        Returns:
            Discriminator_loss: returns loss of discriminator from current step in training 
        '''
        # Enabling optimizer for discriminator
        self.d_optimizer.zero_grad()
        # generate label for discriminator
        real_label = torch.full([hr.size(0), 1], 1.0, dtype=hr.dtype, device=self.device)
        fake_label = torch.full([hr.size(0), 1], 0.0, dtype=hr.dtype, device=self.device)
        #enabling gradients for discriminator
        for parameter in self.discriminator.parameters():
            parameter.requires_grad = True

        with amp.autocast():
            Discriminator_HR = self.discriminator(hr)
            Discriminator_HR_loss = self.adversarial_criterion(Discriminator_HR,real_label)  
        self.scaler.scale(Discriminator_HR_loss).backward()
        with amp.autocast():
            Discriminator_SR = self.discriminator(sr.detach())
            Discriminator_SR_loss = self.adversarial_criterion(Discriminator_SR, fake_label)
        self.scaler.scale(Discriminator_SR_loss).backward()
        self.scaler.step(self.d_optimizer)
        self.scaler.update()
        Discriminator_loss = Discriminator_HR_loss + Discriminator_SR_loss
        return Discriminator_loss

    def generator_step(self,sr,hr):
        '''
        Computes loss for generator and propagates loss into the network
        Args:
            sr: super resolution data
            hr: high resolution samples
        '''
        #enabling optimizer for generator
        self.g_optimizer.zero_grad()
        real_label = torch.full([hr.size(0), 1], 1.0, dtype=hr.dtype, device=self.device)
        #disabling gradients for discriminator
        #Because I dont trust myself and I dont know if it transfers from discriminator step
        for parameter in self.discriminator.parameters():
            parameter.requires_grad = False

        with amp.autocast():
            discriminator_output = self.discriminator(sr)
            pixel_loss = self.pixel_criterion(sr,hr)
            content_loss = self.content_criterion(sr,hr)
            adverserial_loss = self.adversarial_criterion(discriminator_output,real_label)
        generator_loss = (self.pixel_loss_weight*pixel_loss) + (self.adverserial_loss_weight*adverserial_loss) + (self.content_loss_weight*content_loss)
        
        self.scaler.scale(generator_loss).backward()
        # Update generator parameters
        self.scaler.step(self.g_optimizer)
        self.scaler.update()
        return generator_loss,pixel_loss,content_loss,adverserial_loss

    def epoch_train(self,dataloader,epoch):
        '''
        Trains model for 1 epoch through the dataset. Computes loss and propagates it backward for both generator and discriminator
        Args:
            dataloader: data to train model on
            epoch: epoch the model is currently training for
        '''
        self.generator.train()
        self.discriminator.train()

        generator_epoch_loss = 0
        adverserial_epoch_loss = 0
        content_epoch_loss = 0
        pixel_epoch_loss = 0
        discriminator_epoch_loss = 0
        count = 0

        for hr_tensor,lr_tensor in tqdm(dataloader):
            count = count + 1
            #transfer data to GPU for training
            hr = hr_tensor.to(self.device)
            lr = lr_tensor.to(self.device)
            #generate superresolution samples
            sr = self.generator(lr)
            # take discriminator step to update corresponding weights
            Discriminator_loss = self.discriminator_step(hr,sr)
            #computing loss
            discriminator_epoch_loss = discriminator_epoch_loss + Discriminator_loss.item()/lr.size(0)
            # take generator step to update corresponding weights
            Generator_loss,pixel_loss,content_loss,adverserial_loss = self.generator_step(hr,sr)
            #computing generator loss
            generator_epoch_loss = generator_epoch_loss + (Generator_loss.item()/lr.size(0))
            pixel_epoch_loss = pixel_epoch_loss + (pixel_loss.item()/lr.size(0))
            content_epoch_loss = content_epoch_loss + (content_loss.item()/lr.size(0))
            adverserial_epoch_loss = adverserial_epoch_loss  + (adverserial_loss.item()/lr.size(0))

        self.writer("Train/Generator_loss",generator_epoch_loss/count,epoch)
        self.writer("Train/Adverserial_loss",adverserial_epoch_loss/count,epoch)
        self.writer("Train/Content_loss",content_epoch_loss/count,epoch)
        self.writer("Train/Pixel_loss",pixel_epoch_loss/count,epoch)
        self.writer("Train/Discriminator_loss",discriminator_epoch_loss/count,epoch)

        print("Mean generator loss :{} \n".format(generator_epoch_loss/count))
        print("Mean discriminator loss :{} \n".format(discriminator_epoch_loss/count))

        self.d_scheduler.step()
        self.g_scheduler.step()

    @torch.no_grad()
    def epoch_eval(self,dataloader,epoch):
        '''
        Evaluates model with dataloader for current epoch during training
        Args:
            dataloader: data to evaluate the model on
            epoch: epoch the model is currently training for
        Returns:
            generator_loss : combination of pixel_loss,adverserial_loss and content_loss
        '''
        self.generator.eval()
        for hr_tensor,lr_tensor in tqdm(dataloader):
            count = count + 1
            #transfer data to GPU for training
            hr = hr_tensor.to(self.device)
            lr = lr_tensor.to(self.device)
            #generate superresolution samples
            sr = self.generator(lr)
            # take generator step to update corresponding weights
            Generator_loss,pixel_loss,content_loss,adverserial_loss = self.generator_step(hr,sr)
            #computing generator loss
            generator_epoch_loss = generator_epoch_loss + (Generator_loss.item()/lr.size(0))
            pixel_epoch_loss = pixel_epoch_loss + (pixel_loss.item()/lr.size(0))
            content_epoch_loss = content_epoch_loss + (content_loss.item()/lr.size(0))
            adverserial_epoch_loss = adverserial_epoch_loss  + (adverserial_loss.item()/lr.size(0))
        self.writer("Val/Generator_loss",generator_epoch_loss/count,epoch)
        self.writer("Val/Adverserial_loss",adverserial_epoch_loss/count,epoch)
        self.writer("Val/Content_loss",content_epoch_loss/count,epoch)
        self.writer("Val/Pixel_loss",pixel_epoch_loss/count,epoch)
        print("Mean generator loss :{} \n".format(generator_epoch_loss/count))
        return generator_epoch_loss/count

    @torch.no_grad()
    def inference(self,dataloader):
        '''
        Runs inference with test data
        Args:
            dataloader: data to test on. assumes that dataloader provided for inference does not share
        '''
        self.generator.eval()
        return None