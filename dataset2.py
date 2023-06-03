import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch import nn
from torchinfo import summary

import matplotlib.pyplot as plt

from helpers import NUM_WORKERS

import os
from os import getcwd

import random
import numpy as np

from sklearn.model_selection import train_test_split

# Set the batch size
BATCH_SIZE = 64 #64, 128
seed = 42   #42

# svhn 32x32
IMG_SIZE = 32 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

train_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
])

test_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
])

train_val_set = datasets.SVHN(root='../data/SVHN', split='train', download=True, transform=train_transform)
test_data = datasets.SVHN(root='../data/SVHN', split='test', download=True, transform=test_transform)

# Accessing labels for training/validation set
train_val_labels = train_val_set.labels

# Accessing labels for the test set
test_labels = test_data.labels

train_data, valid_data = train_test_split(train_val_set, 
                                          test_size=0.2,
                                          stratify=train_val_labels,
                                          random_state=seed)
#train_data, valid_data = torch.utils.data.random_split(train_val_set, [54943, 18314])  # 3/4 of train set for training 1/4 for validation
#classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)


print(len(train_loader), len(valid_loader), len(test_loader))

# Get a batch of images
image_batch, label_batch = next(iter(train_loader))

print(label_batch)