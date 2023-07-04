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
#from dataset2 import seed, seed_everything, train_loader, valid_loader

model_pruned = torch.load('/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/model_2023-07-04_17-18-01.pth')

# Create new model
#new_model = SimpleViT(new_architecture)
new_model = SimpleViT(
        image_size = 32, # 32, 64 svhn / 64 tiny ;Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
        patch_size = 4, # 4 for 32, 16 for 224 svhn / 8 tiny, ;Number of patches. image_size must be divisible by patch_size. The number of patches is:  n = (image_size // patch_size) ** 2 and n must be greater than 16.
        num_classes = 10, #10, 200 Number of classes to classify.
        dim = 1024, #1024, Last dimension of output tensor after linear transformation nn.Linear(..., dim).
        depth = 6, # 6 Number of Transformer blocks. 9
        heads = 12, # 16 Number of heads in Multi-head Attention layer. 12
        mlp_dim = 2048 # 2048 Dimension of the MLP (FeedForward) layer.
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

# Load the new state dict into the new model
new_model.load_state_dict(new_state_dict)

my_timezone = pytz.timezone('Europe/Berlin')  
now = datetime.now(my_timezone)  
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  

torch.save(new_model, f'pruned_models/structural_pruned_{timestamp}.pth') 

"""
import torch  
import torch.nn as nn 
import copy  
from datetime import datetime  
import pytz  
from simpleViT import Attention, FeedForward  

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  

model_pruned = torch.load("/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/model_2023-06-27_10-57-58.pth")  # Loading a pre-trained pruned model from a file

model = model_pruned.to(device)  # Moving the loaded model to the specified device (CUDA or CPU) for computation

# count the non-zero weights in a layer:
def count_nonzero_weights(layer):  
     # Returning the count of non-zero weights as an integer
    return (layer.weight.data != 0).sum().item()

# Create a new layer with a reduced number of parameters:
def create_new_layer(old_layer, num_nonzero_weights):  
    # Creating a new linear layer with reduced parameters
    new_layer = nn.Linear(old_layer.in_features, num_nonzero_weights, bias=old_layer.bias is not None)  
    # Creating a mask of non-zero weights in the old layer
    mask = old_layer.weight.data.abs() > 0 
    # Copying the non-zero weights from the old layer to the new layer
    new_layer.weight.data = old_layer.weight.data[mask].clone()  
    
    return new_layer 

def create_new_model(old_model): 
    # Creating a deep copy of the old model
    new_model = copy.deepcopy(old_model)  # Creating a deep copy of the old model
    for name, module in old_model.named_modules():  # Iterating over the modules in the old model
        parent_module = None  # Initializing a variable to store the parent module
        # Checking if the module is a linear layer within the "transformer.layers" namespace
        if "transformer.layers" in name and isinstance(module, nn.Linear):  
            # Counting the number of non-zero weights in the module
            num_nonzero_weights = count_nonzero_weights(module)  
            # Creating a new layer with reduced parameters
            new_layer = create_new_layer(module, num_nonzero_weights)  
            # Splitting the module name to get the parent and child names
            parent_name, child_name = name.rsplit('.', 1) 
            # Getting the parent module from the new model
            parent_module = dict(new_model.named_modules())[parent_name] 
            # Replacing the old layer with the new layer in the new model
            setattr(parent_module, child_name, new_layer)  
            # Checking if the parent module is of type Attention
            if isinstance(parent_module, Attention): 
                # Adjusting the dimensions of the to_qkv linear layer
                parent_module.to_qkv = nn.Linear(parent_module.to_qkv.in_features, num_nonzero_weights * 3, bias=False) 
                # Printing the number of non-zero weights
                print(num_nonzero_weights)
                # Printing the output features of the to_out linear layer
                print(parent_module.to_out.out_features) 
                # Adjusting the dimensions of the to_out linear layer
                parent_module.to_out = nn.Linear(num_nonzero_weights, parent_module.to_out.out_features, bias=False)  
                # Printing the output features of the to_out linear layer
                print(parent_module.to_out.out_features) 
            elif isinstance(parent_module, FeedForward): 
                # Adjusting the dimensions of the second linear layer in the FeedForward module
                parent_module.net[1] = nn.Linear(parent_module.net[1].in_features, num_nonzero_weights, bias=True)  
                # Adjusting the dimensions of the fourth linear layer in the FeedForward module
                parent_module.net[3] = nn.Linear(num_nonzero_weights, parent_module.net[3].out_features, bias=True)  
    return new_model  

new_model = create_new_model(model)  

my_timezone = pytz.timezone('Europe/Berlin')  
now = datetime.now(my_timezone)  
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  

torch.save(new_model, f'pruned_models/structural_pruned_{timestamp}.pth') 








import torch
import torch.nn as nn
import copy
from datetime import datetime
import pytz
from simpleViT import Attention, FeedForward

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_pruned = torch.load("/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/model_2023-06-27_10-57-58.pth")
model = model_pruned.to(device)

# count the non-zero weights in a layer:
def count_nonzero_weights(layer):
    #
    return (layer.weight.data != 0).sum().item()

# Create a new layer with a reduced number of parameters:
def create_new_layer(old_layer, num_nonzero_weights):
    #
    new_layer = nn.Linear(old_layer.in_features, num_nonzero_weights, bias=old_layer.bias is not None)
    # Copy the non-zero weights from the old layer to the new layer
    mask = old_layer.weight.data.abs() > 0
    #
    new_layer.weight.data = old_layer.weight.data[mask].clone()
    return new_layer

def create_new_model(old_model):
    #
    new_model = copy.deepcopy(old_model)
    #
    for name, module in old_model.named_modules():
        #
        parent_module = None
        #
        if "transformer.layers" in name and isinstance(module, nn.Linear):
            #
            num_nonzero_weights = count_nonzero_weights(module)
            #
            new_layer = create_new_layer(module, num_nonzero_weights)
            # Replace the old layer with the new layer in the new model
            parent_name, child_name = name.rsplit('.', 1)
            #
            parent_module = dict(new_model.named_modules())[parent_name]
            #
            setattr(parent_module, child_name, new_layer)
            # Adjust the structure of the parent module
            if isinstance(parent_module, Attention):
                #
                parent_module.to_qkv = nn.Linear(parent_module.to_qkv.in_features, num_nonzero_weights * 3, bias=False)
                #
                print(num_nonzero_weights)
                #
                print(parent_module.to_out.out_features)
                #
                parent_module.to_out = nn.Linear(num_nonzero_weights, parent_module.to_out.out_features, bias=False)
                #
                print(parent_module.to_out.out_features) 
                #
            elif isinstance(parent_module, FeedForward):
                #
                parent_module.net[1] = nn.Linear(parent_module.net[1].in_features, num_nonzero_weights, bias=True)
                #
                parent_module.net[3] = nn.Linear(num_nonzero_weights, parent_module.net[3].out_features, bias=True)
    return new_model

# Create a new model with the reduced dimensions
new_model = create_new_model(model)

my_timezone = pytz.timezone('Europe/Berlin')
now = datetime.now(my_timezone)
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# Save your model in the 'pruned_models' directory with a unique name
torch.save(new_model, f'pruned_models/structural_pruned_{timestamp}.pth')



import torch
import torch.nn as nn
import copy
from datetime import datetime
import pytz
from simpleViT import Attention, FeedForward

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_pruned = torch.load("/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/model_2023-06-27_10-57-58.pth")
model = model_pruned.to(device)


#count the non-zero weights in a layer:

def count_nonzero_weights(layer):
    return (layer.weight.data != 0).sum().item()

#Create a new layer with a reduced number of parameters:
def create_new_layer(old_layer, num_nonzero_weights):
    new_layer = nn.Linear(num_nonzero_weights, old_layer.out_features)
    # Copy the non-zero weights from the old layer to the new layer
    mask = old_layer.weight.data.abs() > 0
    new_layer.weight.data = old_layer.weight.data[mask].clone()
    return new_layer


#Adjust the input size of a layer:
def adjust_layer_input_size(old_layer, new_input_size):
    new_layer = nn.Linear(new_input_size, old_layer.out_features)
    return new_layer

def create_new_model(old_model):
    new_model = copy.deepcopy(old_model)
    for name, module in old_model.named_modules():
        if "transformer.layers" in name and isinstance(module, nn.Linear):
            num_nonzero_weights = count_nonzero_weights(module)
            new_layer = create_new_layer(module, num_nonzero_weights)
            # Replace the old layer with the new layer in the new model
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(new_model.named_modules())[parent_name]
            setattr(parent_module, child_name, new_layer)
            # Adjust the structure of the parent module
            if isinstance(parent_module, Attention):
                parent_module.to_qkv = nn.Linear(parent_module.to_qkv.in_features, num_nonzero_weights * 3, bias=False)
                parent_module.to_out = nn.Linear(num_nonzero_weights, parent_module.to_out.out_features, bias=False)
            elif isinstance(parent_module, FeedForward):
                parent_module.net[1] = nn.Linear(parent_module.net[1].in_features, num_nonzero_weights, bias=True)
                parent_module.net[3] = nn.Linear(num_nonzero_weights, parent_module.net[3].out_features, bias=True)
    return new_model

# Create a new model with the reduced dimensions
new_model = create_new_model(model)

my_timezone = pytz.timezone('Europe/Berlin')
now = datetime.now(my_timezone)
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# Save your model in the 'pruned_models' directory with a unique name
torch.save(new_model, f'pruned_models/structural_pruned_{timestamp}.pth')


#Create a new model with the reduced dimensions:
def create_new_model(old_model):
    new_model = copy.deepcopy(old_model)
    for i, (name, layer) in enumerate(old_model.named_modules()):
        if "transformer.layers" in name and isinstance(layer, nn.Linear):
            num_nonzero_weights = count_nonzero_weights(layer)
            new_layer = create_new_layer(layer, num_nonzero_weights)
            # Replace the old layer with the new layer in the new model
            setattr(new_model, name, new_layer)
            # Adjust the input size of the next layer
            if i < len(old_model._modules) - 1:  # if not the last layer
                next_name, next_layer = list(old_model.named_modules())[i+1]
                if isinstance(next_layer, nn.Linear):
                    adjusted_next_layer = adjust_layer_input_size(next_layer, num_nonzero_weights)
                    setattr(new_model, next_name, adjusted_next_layer)
    return new_model
 
from simplify import simplify, fuse, remove
import torch
from torch import nn
from datetime import datetime
import pytz

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_pruned = torch.load("/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/model_2023-06-27_10-57-58.pth")
model = model_pruned .to(device)

@torch.no_grad()
def propagate_bias(model: nn.Module, x: torch.Tensor) -> nn.Module:

    x = x.to(device)

    @torch.no_grad()
    def propagate_biases_hook(module, input, output, name=None):
        
        PyTorch hook used to propagate the biases of pruned neurons to following non-pruned layers.
        

        if isinstance(module, nn.Linear):
            nonzero_weight_rows = (module.weight.abs().sum(dim=1) != 0)

            bias_feature_maps = output[0].clone()

            if getattr(module, 'bias', None) is not None:
                module.bias.data = bias_feature_maps.mean(dim=0)
            else:
                module.register_parameter('bias', nn.Parameter(bias_feature_maps.mean(dim=0)))

            # Create a mask that matches the output tensor dimensions
            mask = nonzero_weight_rows.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(output.shape)

            output = output * mask.float() # convert mask to float and multiply instead of indexing

        else:
            raise ValueError(f'Unsupported module type: {module}')

        return output

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handle = module.register_forward_hook(lambda m, i, o, n=name: propagate_biases_hook(m, i, o, n))
            handles.append(handle)

    zeros = torch.zeros_like(x).to(device)
    model(zeros)

    for h in handles:
        h.remove()
    
    return model

#helper

@torch.no_grad()
def remove_zeroed(model: nn.Module) -> nn.Module:
    for name, module in model.named_modules():
        if "transformer.layers" in name and isinstance(module, nn.Linear):
            nonzero_weight_rows = (module.weight.abs().sum(dim=1) != 0)

            # Prune weights and update out_features
            module.weight = nn.Parameter(module.weight[nonzero_weight_rows])
            module.out_features = module.weight.shape[0]

            # Set biases of pruned neurons to zero
            if module.bias is not None:
                bias_mask = nonzero_weight_rows.float().to(module.bias.device)
                module.bias = nn.Parameter(module.bias * bias_mask)

    return model


layer_tuples = [
    ('transformer.layers.0.0.norm', 'transformer.layers.0.1.net.0'),
    ('transformer.layers.0.1.net.1', 'transformer.layers.0.1.net.3'),
    ('transformer.layers.1.0.norm', 'transformer.layers.1.1.net.0'),
    ('transformer.layers.1.1.net.1', 'transformer.layers.1.1.net.3'),
    ('transformer.layers.2.0.norm', 'transformer.layers.2.1.net.0'),
    ('transformer.layers.2.1.net.1', 'transformer.layers.2.1.net.3'),
    ('transformer.layers.3.0.norm', 'transformer.layers.3.1.net.0'),
    ('transformer.layers.3.1.net.1', 'transformer.layers.3.1.net.3'),
    ('transformer.layers.4.0.norm', 'transformer.layers.4.1.net.0'),
    ('transformer.layers.4.1.net.1', 'transformer.layers.4.1.net.3'),
    ('transformer.layers.5.0.norm', 'transformer.layers.5.1.net.0'),
    ('transformer.layers.5.1.net.1', 'transformer.layers.5.1.net.3'),
]

model_pruned = fuse(model_pruned, bn_folding=layer_tuples)

model_pruned=propagate_bias(model_pruned, x=torch.zeros(32, 3, 32, 32))

model_pruned=remove_zeroed(model_pruned)

my_timezone = pytz.timezone('Europe/Berlin')
now = datetime.now(my_timezone)
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# Save your model in the 'pruned_models' directory with a unique name
torch.save(model_pruned, f'pruned_models/structural_pruned_{timestamp}.pth')



@torch.no_grad()
def remove_zeroed(model: nn.Module, x: torch.Tensor) -> nn.Module:
    @torch.no_grad()
    def __remove_nan(module, input):
        nan_idx = torch.isnan(input[0])
        new_input = input[0].clone()
        new_input[~nan_idx] = 0
        new_input[nan_idx] = 1
        return (new_input, *input[1:])
    
    x = x.to(device)

    @torch.no_grad()
    def __remove_zeroed_channels_hook(module, input, output, name):
        input = input[0][0]  # get first item of batch

        # Compute non-zero input channels indices
        nonzero_input_idx = ~(input.view(input.shape[0], -1).sum(dim=1) == 0)

        if isinstance(module, nn.Linear):
            # Compute non-zero weight indices
            nonzero_weight_idx = ~(module.weight.sum(dim=0) == 0)
            # Get the intersection of non-zero indices from the input and weights
            nonzero_idx = nonzero_input_idx & nonzero_weight_idx
            
            module.weight = nn.Parameter(module.weight[nonzero_idx, :])
            module.in_features = module.weight.shape[1]
            
            # Remove weight channels
            module.weight = nn.Parameter(module.weight[:, nonzero_idx])

            output = torch.zeros_like(output)
            output[:, nonzero_idx] = float('nan')

            if getattr(module, 'bias', None) is not None:
                module.bias = nn.Parameter(module.bias[nonzero_idx])
            
            module.out_features = module.weight.shape[0]

            return output

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handle = module.register_forward_pre_hook(__remove_nan)
            handles.append(handle)
            handle = module.register_forward_hook(lambda m, i, o, n=name: __remove_zeroed_channels_hook(m, i, o, n))
            handles.append(handle)

    zeros = torch.zeros_like(x).to(device)
    model(zeros)

    for h in handles:
        h.remove()

    return model



    @torch.no_grad()
def propagate_bias(model: nn.Module, x: torch.Tensor) -> nn.Module:

    x = x.to(device)

    @torch.no_grad()
    def __remove_nan(module, input):
        
        PyTorch hook that removes nans from input.
        
        module.register_buffer("pruned_input", ~torch.isnan(input[0][0].view(input[0][0].shape[0], -1).sum(dim=1)))
        if torch.isnan(input[0]).sum() > 0:
            input[0][torch.isnan(input[0])] = 0
        return input
            
    @torch.no_grad()
    def propagate_biases_hook(module, input, output, name=None):
        
        PyTorch hook used to propagate the biases of pruned neurons to following non-pruned layers.
        

        if isinstance(module, nn.Linear):
            bias_feature_maps = output[0].clone()

            if getattr(module, 'bias', None) is not None:
                module.bias.data = bias_feature_maps.mean(dim=0)
            else:
                module.register_parameter('bias', nn.Parameter(bias_feature_maps))
                
        else:
            raise ValueError(f'Unsupported module type: {module}')

        pruned_channels = module.weight.sum(dim=1) == 0
        output[~pruned_channels[None, :].expand_as(output)] *= float('nan')

        if getattr(module, 'bias', None) is not None:
            module.bias.data.mul_(~pruned_channels)
        
        return output

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handle = module.register_forward_pre_hook(__remove_nan)
            handles.append(handle)
            handle = module.register_forward_hook(lambda m, i, o, n=name: propagate_biases_hook(m, i, o, n))
            handles.append(handle)

    zeros = torch.zeros_like(x).to(device)
    model(zeros)

    for h in handles:
        h.remove()
    
    return model
"""