import torch
import os

from torchvision import datasets
from PIL import Image


print(os.getcwd())
PATH = '../../Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/'
# transform = transforms.Compose([])
dataset = datasets.ImageFolder(PATH)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=False) #shuffle=False because we don't necessarily care in SR task

