import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import lightning as pl
from vit_loader import vit_loader
from dataset2 import seed, seed_everything, train_loader, valid_loader

seed_everything(seed)

#helpers

def compute_stats(model):
    stats = {}
    total_params = 0
    total_pruned_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_mask = module.weight_mask

            params = module.weight.numel()
            pruned_params = (weight_mask == 0).sum().item()

            total_params += params
            total_pruned_params += pruned_params

            stats[f"{name}_total_params"] = params
            stats[f"{name}_pruned_params"] = pruned_params
            stats[f"{name}_pruned_ratio"] = pruned_params / params

    stats["total_params"] = total_params
    stats["total_pruned_params"] = total_pruned_params
    stats["total_pruned_ratio"] = total_pruned_params / total_params

    return stats

# Hyperparameters
epochs = 3
prune_ratio = 0.2
pruning_iterations = 3
prune_ratio_decay = (1 - prune_ratio) / pruning_iterations

# 1. Randomly initialize the model parameters
model = vit_loader("simple")

# Create a copy of the model for reinitialization
model_copy = copy.deepcopy(model)

# 2. Train model until it converges
trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=False, limit_train_batches=0.2, limit_val_batches=0.2)
trainer.fit(model, train_loader, valid_loader)

trained_model_path = trainer.checkpoint_callback.last_model_path
trained_model = model.load_from_checkpoint(checkpoint_path=trained_model_path)

trainer_prune=pl.Trainer(max_epochs=1, fast_dev_run=False, limit_train_batches=0.2, limit_val_batches=0.2)

# 3. Iterative pruning and reinitialization
for iteration in range(pruning_iterations):
    # Prune the model weights
    for name, module in trained_model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_ratio)

            # Get the original module (before training) from the copied model
            original_module = dict(model_copy.named_modules())[name]

            # Reinitialize the pruned weights
            mask = module.weight_mask
            # This will replace weights where mask == 0 (i.e., the pruned weights) with their original values
            module.weight.data = torch.where(mask != 0, original_module.weight.data, module.weight.data)

    # Train the pruned model
    trainer_prune.fit(trained_model, train_loader, valid_loader)

    # Update the prune ratio for the next iteration
    prune_ratio += prune_ratio_decay

last_model_path_final = trainer_prune.checkpoint_callback.last_model_path
model_pruned_final = model.load_from_checkpoint(checkpoint_path=last_model_path_final)

# Compute statistics
#final_stats = compute_stats(model_pruned_final)

# final train pruned model 
trainer_final = pl.Trainer(max_epochs=epochs, fast_dev_run=False, limit_train_batches=0.5, limit_val_batches=0.5)
trainer_final.fit(model_pruned_final, train_loader, valid_loader)

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