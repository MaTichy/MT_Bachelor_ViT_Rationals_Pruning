import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset2 import valid_loader, seed, seed_everything

seed_everything(seed)

# In evaluation normalization of valid_loader OFF


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load('/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/structural_pruned_2023-07-04_20-13-41.pth')

model.to(device)

test_loader = valid_loader

# Set the model to evaluation mode
model.eval()

# Create the directory if it doesn't exist
if not os.path.exists('evaluation_img'):
    os.makedirs('evaluation_img')

# Get one batch of test data
data, target = next(iter(test_loader))

# Move the data to the device
data = data.to(device)
target = target.to(device)

# Disable gradient computation
with torch.no_grad():
    # Forward pass
    output = model(data)

    # Get the predicted class for each image in the batch
    pred = output.argmax(dim=1, keepdim=True)

    # Save the first image in the batch and its predicted class
    plt.imshow(np.transpose(data[0].cpu().numpy(), (1, 2, 0)))
    plt.title(f'Predicted class: {pred[0].item()}, Target class: {target[0].item()}')
    plt.savefig(f'evaluation_img/image_0.png')

    # If you want to save all images in the batch, you can do so in a loop:
    for i in range(data.shape[0]):
        plt.figure()
        plt.imshow(np.transpose(data[i].cpu().numpy(), (1, 2, 0)))
        plt.title(f'Predicted class: {pred[i].item()}, Target class: {target[i].item()}')
        plt.savefig(f'evaluation_img/image_{i}.png')
        plt.close()
