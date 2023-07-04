import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from helpers import NUM_WORKERS

import os

import random
import numpy as np

# Set the batch size
BATCH_SIZE = 32 #128, 64, 32
seed = 42   #42

# svhn 32x32
IMG_SIZE = 32 #32 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

"""
Augmentation Strategies:

    transforms.RandomApply(
        [transforms.RandomResizedCrop(IMG_SIZE)], 
        p=0.2),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2)],
        p=0.6),
    transforms.RandomGrayscale(p=0.2),

    #transforms.AutoAugment(policy),
    #transforms.RandomApply(
    #    [transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2)],
    #    p=0.4), #p=0.3
    #transforms.RandomGrayscale(p=0.1), #p=0.9
"""

policy = transforms.AutoAugmentPolicy.SVHN #[transforms.AutoAugmentPolicy.CIFAR10, transforms.AutoAugmentPolicy.IMAGENET, transforms.AutoAugmentPolicy.SVHN]
              
train_transform = transforms.Compose([
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)],
        p=0.4),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
])
    
test_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
])

train_set = datasets.SVHN(root='../data/SVHN', split='train', download=True)
val_data = datasets.SVHN(root='../data/SVHN', split='test', download=True)
#test_data = datasets.SVHN(root='../data/SVHN', split='extra', download=True)

train_data = [(train_transform(img), label) for img, label in train_set]
valid_data = [(test_transform(img), label) for img, label in val_data]
#test_data = [(test_transform(img), label) for img, label in test_data]

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, drop_last=True)
#test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, drop_last=True)

print(len(train_loader), len(valid_loader)) #,len(test_loader))

# Get a batch of images
image_batch, label_batch = next(iter(train_loader))

print(label_batch)
