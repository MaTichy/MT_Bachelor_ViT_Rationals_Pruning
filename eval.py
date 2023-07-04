import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset2 import valid_loader

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


"""
# Get one batch of test data
data, target = next(iter(test_loader))

# Only take the first image from the batch
data_single = data[0].unsqueeze(0)  # Add an extra dimension because the model expects batches
target_single = target[0].unsqueeze(0)

# If you're using a GPU
data_single, target_single = data_single.to(device), target_single.to(device)  # Assuming device is your device, either "cpu" or "cuda"

# Disable gradient computation
with torch.no_grad():
    # Forward pass
    output = model(data_single)

    # Get the predicted class if the output is not already in that form
    pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability

    # Convert the data to a format suitable for display
    img = data_single.cpu().numpy()[0][0]  # Adjust this if your images are not single-channel

    # Display the image and prediction
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted: {pred.item()}, Actual: {target_single.item()}')
    plt.savefig('output.png')
"""