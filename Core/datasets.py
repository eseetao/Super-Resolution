import torch
import os
from glob import glob
from torch.utils.data import dataset,dataloader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode as IMode
from PIL import Image

class ImageNet:
    '''
    class to generate data for super resolution task from ImageNet dataset
    Args:
        PATH: local location for ImageNet data
        image_size: sequence of image dimensions to crop from original image
    Kwargs:
        scale: downsampling ration to produce low resolution image for upsampling
        default: 4
        
        split: split of data used for train/val
        default:train
    '''
    def __init__(self,PATH,image_size,split="train",scale=4):
        super(ImageNet,self).__init__()
        self.file_list = sorted(glob(os.path.join(PATH,'CLS-LOC/{}/*/*.JPEG'.format(split))))
    
        if split == "train":
            self.hr_transforms = transforms.Compose([
                transforms.RandomCrop(image_size),
                transforms.RandomRotation([0, 90]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.hr_transforms = transforms.Compose([
                transforms.CenterCrop(image_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])   

        self.lr_transforms = transforms.Resize((image_size[0] // scale,image_size[1] // scale), interpolation=IMode.BICUBIC)     
    
    def __getitem__(self,index):
        #read image
        image = Image.open(self.file_list[index])
        #generate full res and low res images
        hr_tensor = self.hr_transforms(image)
        lr_tensor = self.lr_transforms(hr_tensor)
        return hr_tensor,lr_tensor

def generate_dataloader(configs):
    assert 'train' in configs.keys(),"config must have arguments for training(train) and validation(val)"
    if configs['dataset'] == "ImageNet":
        train_data = ImageNet(configs['PATH'],split='train',scale=configs['scale'])
        val_data = ImageNet(configs['PATH'],split='val',scale=configs['scale'])

    train_dataloader = dataloader(train_data,
                                batch_size=configs['PATH']['train']['batch_size'],
                                shuffle=True,
                                num_workers=configs['num_workers'],
                                pin_memory=True,
                                persistent_workers=True)

    val_dataloader = dataloader(val_data,
                                batch_size=configs['PATH']['val']['batch_size'],
                                shuffle=False,
                                num_workers=configs['num_workers'],
                                pin_memory=True,
                                persistent_workers=True)

    return train_dataloader,val_dataloader

       
