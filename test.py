import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import lightning as pl

from vit_loader import vit_loader
#from dataset2 import seed, seed_everything, train_loader, valid_loader

import math

def calculate_pruning_percentage(total_prune_percentage, iterations):
    remaining_percentage = 1 - total_prune_percentage
    prune_percentage_per_iteration = 1 - math.pow(remaining_percentage, 1/iterations)
    return prune_percentage_per_iteration

# Example usage:
total_prune_percentage = 0.86  # 86% total pruning
iterations = 3
prune_percentage_per_iteration = calculate_pruning_percentage(total_prune_percentage, iterations)

print(prune_percentage_per_iteration)

"""
model = vit_loader("simple") # "simple" or "efficient"

#2. Trained model that converges - val_acc = 90%
trained_model_path = "/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/lightning_logs/version_193/checkpoints/epoch=4-step=11445.ckpt" # "/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/lightning_logs/version_192/checkpoints/epoch=48-step=112161.ckpt"
trained_model = model.load_from_checkpoint(checkpoint_path=trained_model_path)


overall_pruning = 0.84  # The overall proportion of weights to prune
pruning_iterations = 3  # The number of pruning iterations

initial_prune_ratio = (2 * overall_pruning) / (pruning_iterations * (pruning_iterations + 1))

print(initial_prune_ratio)

#for name, module in trained_model.named_modules():
#    if "transformer.layers" in name and (".net.1" in name or ".net.3" in name) and isinstance(module, nn.Linear):
#        print(f"Found a linear layer in a FeedForward block of the Transformer: {name}")
    #if "transformer.layers" in name and "FeedForward.net" in module and isinstance(module, nn.Linear):
    #    print(f"Found a linear layer in a FeedForward block of the Transformer: {name}")

#print(trained_model.named_modules())
"""
"""
seed_everything(seed)

# Hyperparameters
epochs = 1
initial_prune_percentage = 0.2
final_sparsity_level = 0.8
pruning_iterations = 4

# Calculate prune ratio decay per iteration
prune_ratio_decay = (final_sparsity_level / initial_prune_percentage) ** (1 / (pruning_iterations - 1))

# Calculate prune ratio for the first iteration
prune_ratio = 1 - (1 - initial_prune_percentage) ** (1 / pruning_iterations)

# 1. Randomly initialize the model parameters
model = vit_loader("simple")

# Create a copy of the model for reinitialization
model_copy = copy.deepcopy(model)

# 2. Train model until it converges
trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=False, limit_train_batches=0.1, limit_val_batches=0.1)
trainer.fit(model, train_loader, valid_loader)

# 3./ 4. Iterative pruning and reinitialization
for iteration in range(pruning_iterations):
    # Get feedforward module
    feedforward_module = model.transformer.layers[0][1]
    
    # Access linear layers
    linear_layer_1 = feedforward_module.net[1]
    linear_layer_2 = feedforward_module.net[3]

    # Prune the linear layers
    prune.l1_unstructured(linear_layer_1, name="weight", amount=prune_ratio)
    prune.l1_unstructured(linear_layer_2, name="weight", amount=prune_ratio)

    # Remove the pruned connections from the linear layers
    linear_layer_1 = prune.remove(linear_layer_1, "weight")
    linear_layer_2 = prune.remove(linear_layer_2, "weight")

    # Reinitialize the pruned weights
    linear_layer_1.weight.data = model_copy.transformer.layers[0][1].net[1].weight.data.clone()
    linear_layer_2.weight.data = model_copy.transformer.layers[0][1].net[3].weight.data.clone()

    # Update the model with the pruned and reinitialized linear layers
    model.transformer.layers[0][1].net[1] = linear_layer_1
    model.transformer.layers[0][1].net[3] = linear_layer_2

    # Train the pruned model
    trainer.fit(model, train_loader, valid_loader)

    # Create a copy of the model for the next iteration
    model_copy = copy.deepcopy(model)

    # Update prune ratio for the next iteration
    prune_ratio *= prune_ratio_decay
"""
"""
import copy
import torch
import torch.nn.utils.prune as prune
import lightning as pl
from vit_loader import vit_loader
from dataset2 import seed, seed_everything, train_loader, valid_loader

seed_everything(seed)

# Hyperparameters
epochs = 1
prune_ratio = 0.8

# 1. Randomly initialize the model parameters
model = vit_loader("simple")

# Create a copy of the model
model_copy = copy.deepcopy(model)

# 2. Train the model until it converges
trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=False, limit_train_batches=0.1, limit_val_batches=0.1)
trainer.fit(model, train_loader, valid_loader)

# 3. Prune the model to remove connections that have low weights
feedforward_module = model.transformer.layers[0][1]
linear_layer_1 = feedforward_module.net[1]
linear_layer_2 = feedforward_module.net[3]

# Prune the linear layers
prune.l1_unstructured(linear_layer_1, name="weight", amount=prune_ratio)
prune.l1_unstructured(linear_layer_2, name="weight", amount=prune_ratio)

# Remove the pruned connections from the linear layers
linear_layer_1 = prune.remove(linear_layer_1, "weight")
linear_layer_2 = prune.remove(linear_layer_2, "weight")

# 4. Update the model with the pruned linear layers
model.transformer.layers[0][1].net[1] = linear_layer_1
model.transformer.layers[0][1].net[3] = linear_layer_2

# 5. Reinitialize non-pruned weights in the pruned model with original weights
pruned_model = copy.deepcopy(model)

# Get the state dictionaries of the original model and pruned model
model_state_dict = model_copy.state_dict()
pruned_state_dict = pruned_model.state_dict()

# Update the pruned model's state dictionary with the non-pruned weights from the original model
for name, param in model_state_dict.items():
    if name in pruned_state_dict and param.dim() != 0:
        pruned_state_dict[name].copy_(param)

# Load the updated state dictionary into the pruned model
pruned_model.load_state_dict(pruned_state_dict)

# 6. Train the pruned model
trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=False, limit_train_batches=0.25, limit_val_batches=0.25)
trainer.fit(pruned_model, train_loader, valid_loader)

import copy
import torch
import torch.nn.utils.prune as prune
import lightning as pl
from vit_loader import vit_loader
from dataset2 import seed, seed_everything, train_loader, valid_loader

seed_everything(seed)

# Hyperparameters
epochs = 1
prune_ratio = 0.6

# 1. Randomly initialize the model parameters
model = vit_loader("simple")

# Create a copy of the model
model_copy = copy.deepcopy(model)

# 2. Train the model until it converges
trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=False, limit_train_batches=0.1, limit_val_batches=0.1)
trainer.fit(model, train_loader, valid_loader)

# 3. Prune the model to remove connections that have low weights
feedforward_module = model.transformer.layers[0][1]
linear_layer_1 = feedforward_module.net[1]
linear_layer_2 = feedforward_module.net[3]

# Prune the linear layers
prune.l1_unstructured(linear_layer_1, name="weight", amount=prune_ratio)
prune.l1_unstructured(linear_layer_2, name="weight", amount=prune_ratio)

# Remove the pruned connections from the linear layers
linear_layer_1 = prune.remove(linear_layer_1, "weight")
linear_layer_2 = prune.remove(linear_layer_2, "weight")

# 4. Update the model with the pruned linear layers
model.transformer.layers[0][1].net[1] = linear_layer_1
model.transformer.layers[0][1].net[3] = linear_layer_2

# 5. Reinitialize non-pruned weights in the pruned model with original weights
pruned_model = copy.deepcopy(model)
pruned_state_dict = pruned_model.state_dict()
model_state_dict = model_copy.state_dict()

# Update the pruned state dictionary with the non-pruned weights from the original model
for name, param in model_state_dict.items():
    if name in pruned_state_dict:
        pruned_state_dict[name] = param

pruned_model.load_state_dict(pruned_state_dict)

# 6. Train the pruned model
trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=False, limit_train_batches=0.25, limit_val_batches=0.25)
trainer.fit(pruned_model, train_loader, valid_loader)



#train_and_prune(model, train_loader, valid_loader, epochs)
def compute_amount(epoch):
    # the sum of all returned values need to be smaller than 1
    if epoch == 1:
        return 0.5

    elif epoch == 3:
        return 0.25

    elif 4 < epoch < 6:
        return 0.01

trainer = pl.Trainer(max_epochs=epochs, callbacks=[ModelPruning("l1_unstructured", amount=compute_amount, lottery_ticket_hypothesis=True)])
trainer.fit(model, train_loader, valid_loader)

"""

"""
def check_pruned_layer(layer):
    params = {param_name for param_name, _ in layer.named_parameters() if "weight" in param_name}
    expected_params = {"weight_orig"}
    print(f"Params: {params}, Expected Params: {expected_params}")

    return params == expected_params


def check_pruned_layer(module):
    expected_params = {"weight_orig"}
    found_params = set()

    # Helper function to recursively traverse the module and its children
    def traverse_module(module):
        # Traverse all parameters of the module
        for param_name, _ in module.named_parameters():
            # Check if it's a weight parameter
            if "weight" in param_name:
                # Split the parameter name by "."
                names = param_name.split(".")
                # Get the name of the last parameter (should be 'weight' or 'weight_orig')
                last_name = names[-1]
                # Add the found parameter to the set
                found_params.add(last_name)

        # Recursively traverse the children modules
        for name, child_module in module.named_children():
            traverse_module(child_module)

    # Start traversing the module
    traverse_module(module)

    print(f"Params: {found_params}, Expected Params: {expected_params}")

    return found_params == expected_params
"""


#lth

"""
import copy
import lightning as pl

from vit_loader import vit_loader

import numpy
import torch 
from torchinfo import summary

import torch.nn as nn

from torch.nn.utils.prune import l1_unstructured, random_unstructured

# change depending on dataset: for tiny images: dataset and for svhn: dataset2
from dataset2 import train_loader, valid_loader, seed, seed_everything

from lightning.pytorch.callbacks import ModelPruning

seed_everything(seed)
# Hyperparameters
# Training settings
epochs = 5 #20
prune_ratio=0.5


# helpers for pruning

def prune_layer(module, prune_ratio=prune_ratio, method="l1"):
    # Check if the module is of type nn.Linear
    if isinstance(module, nn.Linear):
        layer = module.weight
        l1_unstructured(module, name='weight', amount=prune_ratio)
    else:
        print("No nn.Linear layer found.")
        return
    
    # now the print statement is inside the condition where layer is defined.
    #print(f"Pruning layer: {layer}, prune_ratio: {prune_ratio}, method: {method}")

def prune_model_global(model, prune_ratio=prune_ratio, method="l1"):
    if isinstance(prune_ratio, float):
        prune_ratios = [prune_ratio] * len(model.transformer.layers)
    elif isinstance(prune_ratio, list):
        if(len(prune_ratio) != len(model.transformer.layers)): 
            raise ValueError("Prune ratio list must have the same length as the number of layers")
        prune_ratios = prune_ratio
    else:
        raise TypeError
    
    for prune_ratio, transformer_layer in zip(prune_ratios, model.transformer.layers):
        # recursively prune all submodules
        prune_recursive(transformer_layer, prune_ratio, method)

def prune_recursive(module, prune_ratio, method):
    # check for submodules
    for name, sub_module in module.named_children():
        # if submodule, recursively check it as well
        prune_recursive(sub_module, prune_ratio, method)
    # if this module is nn.Linear, prune it
    if isinstance(module, nn.Linear):
        prune_layer(module, prune_ratio, method)

def check_pruned_layer(module):
    expected_params = {"weight_orig"}
    found_params = set()

    # Helper function to recursively traverse the module and its children
    def traverse_module(module):
        # Check if it's an instance of nn.Linear
        if isinstance(module, nn.Linear):
            # Check if the weight_orig parameter exists
            if hasattr(module, "weight_orig"):
                found_params.add("weight_orig")

        # Recursively traverse the children modules
        for name, child_module in module.named_children():
            traverse_module(child_module)

    # Start traversing the module
    traverse_module(module)

    print(f"Params: {found_params}, Expected Params: {expected_params}")

    return found_params == expected_params

def reinit_layers(module, model_copy):
    for name, sub_module in module.named_children():
        if len(list(sub_module.children())) > 0:
            # recursively go to the sub_modules
            reinit_layers(sub_module, model_copy)
        else:
            if hasattr(sub_module, 'weight'):
                is_pruned = check_pruned_layer(sub_module)
                # get parameters of interest
                if is_pruned:
                    # get the corresponding layer in the copy of the model
                    copy_sub_module = dict(model_copy.named_modules())[name]

                    # replace the pruned weights with the initial weights
                    sub_module.weight_orig.data = copy.deepcopy(copy_sub_module.weight.data)

def reinit_model(model):
    for layer in model.transformer.layers:
        reinit_layers(layer)

def copy_weights_layers(layers_unpruned, layers_pruned):
    assert check_pruned_layer(layers_pruned)
    assert not check_pruned_layer(layers_unpruned)

    with torch.no_grad():
        layers_pruned.weight_orig.copy_(layers_unpruned.weight)
        layers_pruned.bias_orig.copy_(layers_unpruned.bias)
    
def copy_weights_model(model_unpruned, model_pruned):
    zipped = zip(model_unpruned.layers, model_pruned.layers)
    for layer_unpruned, layer_pruned in zipped:
        copy_weights_layers(layer_unpruned, layer_pruned)

def compute_stats(model):
    stats = {}
    total_params = 0
    total_pruned_params = 0

    for layer_ix, layers in enumerate(model.transformer.layers):
        assert check_pruned_layer(layers)

        layer_params = {param_name for param_name, _ in layers[0].named_parameters() if "weight" in param_name}
        weight_mask = layers[0].weight_mask

        params = len(layer_params)
        pruned_params = (weight_mask == 0).sum()

        total_params += params
        total_pruned_params += pruned_params

        stats[f"layer_{layer_ix}_total_params"] = params
        stats[f"layer_{layer_ix}_pruned_params"] = pruned_params
        stats[f"layer_{layer_ix}_pruned_ratio"] = pruned_params / params

    stats["total_params"] = total_params
    stats["total_pruned_params"] = total_pruned_params
    stats["total_pruned_ratio"] = total_pruned_params / total_params

    return stats

#Pie

model = vit_loader("simple") # "simple" or "efficient"

#safe initial state
#initial_state_dict = copy.deepcopy(model.state_dict())

def train_and_prune(model, train_loader, valid_loader, epochs, prune_ratio=0.6, method="l1"):

    #copy the model
    model_copy = copy.deepcopy(model)
    model_copy.load_state_dict(model.state_dict())

    #train model until it converges 
    trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=False)
    trainer.fit(model, train_loader, valid_loader)

    #prune the model to remove connections that have low weights
    prune_model_global(model, prune_ratio, method)

    #reinitialize the pruned model
    reinit_model(model, model_copy)

    #train the pruned model
    trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=False)
    trainer.fit(model, train_loader, valid_loader)

    #compute stats
    #stats = compute_stats(model)

    return model #, stats

train_and_prune(model, train_loader, valid_loader, epochs)
"""

#version_number_trainer_prune=trainer_prune.logger.version
#epoch_number_trainer_prune=trainer_prune.current_epoch
#global_step_trainer_prune=trainer_prune.global_step

#last_model_path_final = current_path + f"/lightning_logs/version_{version_number_trainer_prune}/checkpoints/epoch_{epoch_number_trainer_prune}-step_{global_step_trainer_prune}.ckpt"
#model_pruned_final = model.load_from_checkpoint(checkpoint_path=last_model_path_final)

# Compute statistics
#final_stats = compute_stats(model_pruned_final)

# final train pruned model 
#trainer_final = pl.Trainer(max_epochs=epochs, fast_dev_run=False, limit_train_batches=0.5, limit_val_batches=0.5)
#trainer_final.fit(model_pruned_final, train_loader, valid_loader)


# prune the new model first due to mismatch of unpruned and pruned layers: Unexpected key(s) in state_dict: "to_patch_embedding.2.weight_mask", "transformer.layers.0.0.to_qkv.weight_mask", 
#for name, module in model.named_modules():
#    if isinstance(module, nn.Linear):
#        prune.l1_unstructured(module, name='weight', amount=0)

#pruned_model_path ="/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/lightning_logs/version_196/checkpoints/epoch=0-step=457.ckpt"
#model_pruned_final = model.load_from_checkpoint(checkpoint_path=pruned_model_path)