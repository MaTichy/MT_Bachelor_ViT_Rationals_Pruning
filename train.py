from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
import lightning as pl

from vit_loader import vit_loader

from torchinfo import summary

# change depending on dataset: for tiny images: dataset and for svhn: dataset2
from dataset2 import train_data, test_data, valid_data, train_loader, test_loader, valid_loader, seed, BATCH_SIZE


# Hyperparameters
# Training settings
epochs = 50 #20
lr = 3e-5 #3e-5
gamma=0.7
gamma=0.7

# when changing numbers the hyperparameters of the models have to be adjustet in a matter fit for the dataset
model = vit_loader("simple") # "simple" or "efficient"


# loss function
loss = nn.CrossEntropyLoss()

"""
# for tiny adjust weight decay original 0.3
# Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper 
optimizer = torch.optim.Adam(params=model.parameters(), 
                             lr=3e-5, # Base LR from Table 3 for ViT-* ImageNet-1k (3e-3 eigentlich)
                             betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                             weight_decay=0.3) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k
"""




# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

trainer = pl.Trainer(max_epochs=epochs)
trainer.fit(model, train_loader, valid_loader)