from __future__ import print_function
import datetime
from itertools import chain
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vit_loader import vit_loader
from dataset2 import train_loader, valid_loader, seed, seed_everything



device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(seed)

# Hyperparameters
# Training settings
epochs = 20 #20
lr = 3e-6 #3e-5
gamma = 0.7 #0.7

model = torch.load('/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/structural_pruned_2023-07-07_16-25-35.pth') #torch.load('/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/structural_pruned_2023-07-07_15-27-55.pth') #torch.load('/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/structural_pruned_2023-07-04_20-13-41.pth')

model = model.to(device)

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

# create a SummaryWriter for TensorBoard
writer = SummaryWriter('pytorch_logs')

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch+1}")
    for step, (data, label) in progress_bar:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

        progress_bar.set_postfix({'step_train_loss': loss.item(), 'step_train_acc': acc.item()})
    
    writer.add_scalar('Train Loss', epoch_loss.item(), epoch * len(train_loader) + step)
    writer.add_scalar('Train Accuracy', epoch_accuracy.item(), epoch * len(train_loader) + step)

    progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Validation Epoch {epoch+1}")
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for step, (data, label) in progress_bar:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

            progress_bar.set_postfix({'step_val_loss': val_loss.item(), 'step_val_acc': acc.item()})

    writer.add_scalar('Validation Loss', epoch_val_loss.item(), epoch)
    writer.add_scalar('Validation Accuracy', epoch_val_accuracy.item(), epoch)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

writer.close()

# torch.save(model)
if not os.path.exists('pruned_models'):
    os.makedirs('pruned_models')

my_timezone = pytz.timezone('Europe/Berlin')  
now = datetime.now(my_timezone)  
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  

# Save model in the 'pruned_models' directory with a unique name
torch.save(model, f'pruned_models/structured_retrain_{timestamp}.pth')