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

for name, module in model_pruned.named_modules():
        if "transformer.layers" in name and (".net.1" in name or ".net.3" in name) and isinstance(module, nn.Linear):
            module = prune.remove(module, name='weight')

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

"""
cleaned_state_dict = {k: v for k, v in pruned_state_dict.items() if '_orig' not in k and '_mask' not in k}

for key in pruned_state_dict.keys():
    if '_orig' in key:
        # extract original key name by removing _orig suffix
        orig_key = key.replace('_orig', '')
        
        if orig_key + '_mask' in pruned_state_dict:
            # apply the mask to the original weights
            pruned_weight = pruned_state_dict[key] * pruned_state_dict[orig_key + '_mask']
            
            # put the pruned weights into the cleaned_state_dict
            cleaned_state_dict[orig_key] = pruned_weight
        else:
            print(f'Mask not found for {orig_key}')
"""
# Replace pruned_state_dict with the cleaned version
#pruned_state_dict = cleaned_state_dict

# Create new state dict
new_state_dict = {}

# Process each layer in the pruned model
for key, value in pruned_state_dict.items():
    if "transformer.layers" in key and (".net.1" in key or ".net.3" in key):
        
        if "weight" in key:
            # Average every 2 parameters
            averaged_values = value.view(-1, 2).mean(dim=1)

            # Determine the shape based on layer
            if ".net.1" in key:
                expected_shape = (512, 1024)
            elif ".net.3" in key:
                expected_shape = (1024, 512)
                
            # Check if the averaged_values tensor is smaller than the expected shape
            if averaged_values.numel() < torch.prod(torch.tensor(expected_shape)):
                # If so, pad with zeros
                padding = torch.zeros(torch.prod(torch.tensor(expected_shape)) - averaged_values.numel())
                averaged_values = torch.cat([averaged_values, padding])
            elif averaged_values.numel() > torch.prod(torch.tensor(expected_shape)):
                # If too many, slice off the extras
                averaged_values = averaged_values[:torch.prod(torch.tensor(expected_shape))]

            # Assign the averaged weights to this layer
            new_state_dict[key] = averaged_values.view(*expected_shape)
        
        elif "bias" in key:
            if ".net.1" in key:
                expected_shape = (512,)
            elif ".net.3" in key:
                expected_shape = (1024,)
            
            # Copy the bias from the pruned model
            new_state_dict[key] = value[:torch.prod(torch.tensor(expected_shape))].clone()
                
    else:
        # For all other layers, directly copy the weights
        if key in new_model.state_dict() and value.shape == new_model.state_dict()[key].shape:
            new_state_dict[key] = value.clone()

# Load the new state dict into the new model
new_model.load_state_dict(new_state_dict)

my_timezone = pytz.timezone('Europe/Berlin')  
now = datetime.now(my_timezone)  
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  

torch.save(new_model, f'pruned_models/structural_pruned_{timestamp}.pth')


"""
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
    # If the current layer is a linear layer within the transformer block
    if "transformer.layers" in key and (".net.1" in key or ".net.3" in key) and len(value.shape) == 2:
        # Compress the weights of this layer
        non_zero_weights = value[value != 0]
        non_zero_weights = non_zero_weights.view(-1)
        
        # Re-construct the weight tensor while maintaining the original number of rows
        reshaped_weights = []
        for i in range(value.shape[0]):
            non_zero_row = non_zero_weights[i*value.shape[1]:(i+1)*value.shape[1]]
            num_zeros = value.shape[1] - len(non_zero_row)
            if num_zeros > 0:
                zero_padding = torch.zeros(num_zeros, device=non_zero_weights.device)
                non_zero_row = torch.cat([non_zero_row, zero_padding])
            reshaped_weights.append(non_zero_row)

        reshaped_weights = torch.stack(reshaped_weights)
        new_state_dict[key] = reshaped_weights
    else:
        # For all other layers, just copy the weights without compressing
        new_state_dict[key] = copy.deepcopy(value)

# Load the new state dict into the new model
new_model.load_state_dict(new_state_dict)

my_timezone = pytz.timezone('Europe/Berlin')  
now = datetime.now(my_timezone)  
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  

torch.save(new_model, f'pruned_models/structural_pruned_{timestamp}.pth')

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

# Clean up pruned model's state_dict
cleaned_state_dict = {k: v for k, v in pruned_state_dict.items() if not (k.endswith('_orig') or k.endswith('_mask'))}

# Replace pruned_state_dict with the cleaned version
pruned_state_dict = cleaned_state_dict

# Create new state dict
new_state_dict = {}

# Flatten and average every 2 parameters in the pruned model
flattened = pruned_state_dict.view(-1)
averaged_weights = flattened.reshape(-1, 2).mean(dim=1)

# Assign the averaged weights to the new model
i = 0

for key, value in pruned_state_dict.items():
    if "transformer.layers" in key and (".net.1" in key or ".net.3" in key):
        if "weight" in key:
            # The existing code to handle these layers
            # Calculate the number of weights this layer needs
            num_weights = value.numel()

            # Get the weights for this layer
            layer_weights = averaged_weights[i:i+num_weights]

            # Check if we have enough weights left
            if len(layer_weights) < num_weights:
                # If not, pad with zeros
                padding = torch.zeros(num_weights - len(layer_weights))
                layer_weights = torch.cat([layer_weights, padding])
            elif len(layer_weights) > num_weights:
                # If too many, slice off the extras
                layer_weights = layer_weights[:num_weights]

            # Assign the weights to this layer
            new_state_dict[key] = layer_weights.view(*value.shape)

            # Move the index
            i += num_weights

        elif "bias" in key:
            # Now handling bias terms similarly to the weights
            num_biases = value.numel()
            
            # Get the biases for this layer
            layer_biases = averaged_weights[i:i+num_biases]
            
            # Check if we have enough biases left
            if len(layer_biases) < num_biases:
                # If not, pad with zeros
                padding = torch.zeros(num_biases - len(layer_biases))
                layer_biases = torch.cat([layer_biases, padding])
            elif len(layer_biases) > num_biases:
                # If too many, slice off the extras
                layer_biases = layer_biases[:num_biases]

            # Assign the biases to this layer
            new_state_dict[key] = layer_biases.view(*value.shape)

            # Move the index
            i += num_biases
            
    else:
        # For all other layers, directly copy the weights
        if key in new_model.state_dict() and value.shape == new_model.state_dict()[key].shape:
            new_state_dict[key] = value.clone()

# Load the new state dict into the new model
new_model.load_state_dict(new_state_dict)

my_timezone = pytz.timezone('Europe/Berlin')  
now = datetime.now(my_timezone)  
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  

torch.save(new_model, f'pruned_models/2structural_pruned_{timestamp}.pth') 


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

# Create a new state dictionary for the pruned model with compressed weights
compressed_state_dict = {}

# For each entry in the pruned state dict...
for key, value in pruned_state_dict.items():
    # Skip the keys related to unpruned original parameters.
    if key.endswith('_orig'):
        continue

    # Get the non-zero weights
    non_zero_weights = value[value != 0]

    # Re-construct the weight tensor while maintaining the original number of rows
    compressed_weights = []
    for row in non_zero_weights:
        non_zero_row = row[row != 0]
        compressed_weights.append(non_zero_row)

    compressed_weights_tensor = torch.stack(compressed_weights)
    
    # Assign the compressed weights to the compressed state dict
    compressed_state_dict[key] = compressed_weights_tensor

# Now compare the compressed_state_dict with the new model's state_dict
new_state_dict = {}

# For each entry in the compressed state dict...
for key, value in compressed_state_dict.items():
    # Make sure the key exists in the state dictionary of the new model
    if key not in new_model.state_dict():
        continue

    # Calculate the number of zeros needed to match the shape of the corresponding tensor in the new model
    num_zeros = new_model.state_dict()[key].numel() - value.numel()
    if num_zeros > 0:
        zero_padding = torch.zeros(num_zeros, device=value.device)
        padded_weights = torch.cat([value.view(-1), zero_padding]).view(new_model.state_dict()[key].shape)
    else:
        padded_weights = value[:new_model.state_dict()[key].shape[0], :new_model.state_dict()[key].shape[1]]

    # Assign padded_weights to the new state dict
    new_state_dict[key] = padded_weights

# Update the new model's state_dict
new_model.load_state_dict(new_state_dict)

my_timezone = pytz.timezone('Europe/Berlin')  
now = datetime.now(my_timezone)  
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  

torch.save(new_model, f'pruned_models/2structural_pruned_{timestamp}.pth') 
"""