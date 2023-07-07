import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import lightning as pl
import math
from datetime import datetime  
import pytz  

from simpleViT_structural_pruning import SimpleViT
from vit_loader import vit_loader
from dataset2 import seed, seed_everything

seed_everything(seed)

model_pruned = torch.load('/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/model_2023-07-04_17-18-01.pth')

#Create new model
new_model = SimpleViT(
        image_size = 32,
        patch_size = 4, 
        num_classes = 10,
        dim = 1024, 
        depth = 6, 
        heads = 12, 
        mlp_dim = 2048
)
# Get state dict from pruned model
pruned_state_dict = model_pruned.state_dict()

# Create new state dict
new_state_dict = {}

# For each entry in the pruned state dict...
for key, value in pruned_state_dict.items():
    # Check if the key exists in the state dictionary of the new model
    if key in new_model.state_dict():
        new_key = key
    elif key.endswith('_orig'):
        # If the key does not exist and it ends with '_orig', remove the suffix to access the original parameter
        new_key = key[:-5]
    else:
        continue  # If the key does not exist and it does not end with '_orig', skip this iteration

    # Get the non-zero weights
    non_zero_weights = value[value != 0]

    # Create a new tensor of the appropriate shape
    new_tensor = torch.zeros(new_model.state_dict()[new_key].shape)

    # Calculate the number of zeros needed to match the number of elements in new_tensor
    num_zeros = new_tensor.numel() - non_zero_weights.numel()

    if num_zeros > 0:
        # Create a tensor of zeros with the calculated number
        zeros = torch.zeros(num_zeros, device=non_zero_weights.device)

        # Concatenate non_zero_weights with the zeros tensor
        padded_non_zero_weights = torch.cat([non_zero_weights, zeros])
    else:
        # If num_zeros is less than or equal to zero, slice non_zero_weights to match the size of new_tensor
        padded_non_zero_weights = non_zero_weights[:new_tensor.numel()]

    # Reshape padded_non_zero_weights to match the shape of new_tensor
    reshaped_non_zero_weights = padded_non_zero_weights.view(*new_tensor.shape)

    # Assign reshaped_non_zero_weights to new_tensor
    new_tensor = reshaped_non_zero_weights

    # Add the new tensor to the new state dict
    new_state_dict[new_key] = new_tensor

# Load the new state dict into the new model
new_model.load_state_dict(new_state_dict)

my_timezone = pytz.timezone('Europe/Berlin')  
now = datetime.now(my_timezone)  
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  

torch.save(new_model, f'pruned_models/structural_pruned_{timestamp}.pth') 
