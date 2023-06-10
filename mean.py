import torch
from torchvision import datasets, transforms

from dataset2 import train_data
import numpy as np

# Initialize lists to store all image data
data_r = []
data_g = []
data_b = []

# Iterate over the dataset and store pixel values
for image, _ in train_data:
    data_r.append(image[0,:,:])
    data_g.append(image[1,:,:])
    data_b.append(image[2,:,:])

# Concatenate all information in a single tensor
data_r = torch.cat(data_r)
data_g = torch.cat(data_g)
data_b = torch.cat(data_b)

# Calculate the mean and standard deviation
mean_r = torch.mean(data_r)
mean_g = torch.mean(data_g)
mean_b = torch.mean(data_b)

std_r = torch.std(data_r)
std_g = torch.std(data_g)
std_b = torch.std(data_b)

print('Mean:', [mean_r.item(), mean_g.item(), mean_b.item()])
print('Std:', [std_r.item(), std_g.item(), std_b.item()])