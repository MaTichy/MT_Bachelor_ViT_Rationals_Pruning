
import lightning as pl

from vit_loader import vit_loader

import numpy

from torchinfo import summary

import torch.nn as nn

from torch.nn.utils.prune import l1_unstructured, random_unstructured

# change depending on dataset: for tiny images: dataset and for svhn: dataset2
from dataset2 import train_data, test_data, valid_data, train_loader, test_loader, valid_loader, seed, BATCH_SIZE


# Hyperparameters
# Training settings
epochs = 20 #20


# helpers for pruning
def get_pruneable_layers(model):
    pruneable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            pruneable_layers.append(name)
    return pruneable_layers

def get_pruneable_layer(model, layer_name):
    module = model
    for attr in layer_name.split("."):
        module = getattr(module, attr)
    return module

def prune_layer(layer, prune_ratio=0.2, method="l1"):
    if method == "l1":
        prune_func = l1_unstructured
    elif method == "random":
        prune_func = random_unstructured
    else:
        raise ValueError
    prune_func(layer, name="weight", amount=prune_ratio)
    prune_func(layer, name="bias", amount=prune_ratio)

def prune_model_global(model, prune_ratio=0.2, method="l1"):
    if isinstance(prune_ratio, float):
        prune_ratios = [prune_ratio] * len(model.layers)
    elif isinstance(prune_ratio, list):
        if(len(prune_ratio) != len(model.layers)):
            raise ValueError("Prune ratio list must have the same length as the number of layers")

        prune_ratios = prune_ratio
    else:
        raise TypeError
    
    for prune_ration, model in zip(prune_ratios, model.layers):
        prune_model(model, prune_ratio, method)

def check_pruned_layers(layers):
    params = {param_name for param_name, _ in layers.named_parameters() if "weight" in param_name}
    expected_params = {"weight_orig", "bias_orig"}

    return params == expected_params

def get_pruned_weights(layers):
    pruned_weights = []
    for name, module in layers.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            pruned_weights.append(module.weight.data.cpu().numpy())
    return pruned_weights

def get_pruned_weights_global(model):
    pruned_weights = []
    for layer in model.layers:
        pruned_weights.extend(get_pruned_weights(layer))
    return pruned_weights

def get_weights(layers):
    weights = []
    for name, module in layers.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            weights.append(module.weight.data.cpu().numpy())
    return weights

def get_weights_global(model):
    weights = []
    for layer in model.layers:
        weights.extend(get_weights(layer))
    return weights

def get_masks(model):
    masks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            masks.append(module.weight_mask.cpu().numpy())
    return masks

def reinit_layers(layers):
    is_pruned = check_pruned_layers(layers)

    #get parameters of interest
    if is_pruned:
        weight = layers.weight_orig
        bias = layers.bias_orig
    else:
        weight = layers.weight
        bias = layers.bias

    #reinitialize weight
    torch.nn.init.normal_(weight, mean=0, std=0.1)

    #reinitialize bias
    torch.nn.init.constant_(bias, 0)

def reinit_model(model):
    for layer in model.layers:
        reinit_layers(layer)

def copy_weights_layers(layers_unpruned, layers_pruned):
    assert check_pruned_layers(layers_pruned)
    assert not check_pruned_layers(layers_unpruned)

    with torch.no_grad():
        layers_pruned.weight_orig.copy_(layers_unpruned.weight)
        layers_pruned.bias_orig.copy_(layers_unpruned.bias)
    
def copy_weights_model(model_unpruned, model_pruned):
    zipped = zip(model_unpruned.layers, model_pruned.layers)
    for layer_unpruned, layer_pruned in zipped:
        copy_weights_layers(layer_unpruned, layer_pruned)

def compute_stats(weights):
    stats = {}
    total_params = 0
    total_pruned_params = 0

    for layer_ix, layers in enumarate(model.layers):
        assert check_pruned_layers(layers)

        weight_mask = layers.weight_mask
        bias_mask = layers.bias_mask

        params = weight_mask.nume1() + bias_mask.nume1()
        pruned_params = (weight_mask==0).sum() + (bias_mask==0).sum()

        total_params += params
        total_pruned_params += pruned_params

        stats[f"layer_{layer_ix}_total_params"] = params
        stats[f"layer_{layer_ix}_pruned_params"] = pruned_params
        stats[f"layer_{layer_ix}_pruned_ratio"] = pruned_params / params

    stats["total_params"] = total_params
    stats["total_pruned_params"] = total_pruned_params
    stats["total_pruned_ratio"] = total_pruned_params / total_params

    return stats

    

# 1. randomly initialize the model parameters
#torch.nn.init.normal_(model.weight, mean=0, std=0.1)


#2. train model until it converges 
#trainer = pl.Trainer(max_epochs=epochs)
#trainer.fit(model, train_loader, valid_loader)


#3. prune the model to remove connections that have low weights
#l1_unstructured(model, "weight", amount=0.5)
# 4. To extract the winning ticket, reset the weights of the remaining portion of the network to their values from (1) - the initializations they received before training began.

# 5. To evaluate whether the resulting network at step (4) is indeed a winning ticket, train the pruned, untrained network and examine its convergence behavior and accuracy.

model = vit_loader("simple") # "simple" or "efficient"

def train_and_prune(model, train_loader, valid_loader, epochs, prune_ratio=0.2, method="l1"):
    
    #copy the model
    model_copy = model

    model_copy.load_state_dict(model.state_dict())

    #train model until it converges 
    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model, train_loader, valid_loader)

    #prune the model to remove connections that have low weights
    prune_model_global(model, prune_ratio, method)

    #reinitialize the pruned model
    reinit_model(model)

    #train the pruned model
    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model, train_loader, valid_loader)

    #compute stats
    stats = compute_stats(model)

    return model, stats

train_and_prune(model, train_loader, valid_loader, epochs)