import copy
import pytz
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import lightning as pl
from datetime import datetime
from vit_loader import vit_loader
from dataset2 import seed, seed_everything, train_loader, valid_loader

import os
import math

#utils
current_path = os.getcwd()

seed_everything(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
l1_unstructured, the L1-norm (which is simply the sum of the absolute values of the elements) is used as 
a measure of magnitude. The method prunes a specified fraction of the parameters with the smallest L1-norm.

The gradients of the pruned weights (which are set to zero) will automatically be zero during the backward pass of the training process. This is because the backward pass computes the gradients as the derivative of the loss with respect to the weights. Since the pruned weights are zero, their gradients will also be zero.
"""


"""
Lottery Ticket Hypothesis - Iterative Pruning:
1. Random Initialization: They started with a randomly initialized neural network.
2. Training: They trained this network for a certain number of iterations.
3. Pruning: After training, they pruned a certain percentage of the smallest weights in the network. This was done by setting the corresponding elements in the mask to zero.
4. Resetting: After pruning, they reset the weights of the remaining (unpruned) part of the network to their values from the initial (random) initialization. This was done by element-wise multiplying the weights with the mask.
5. Iteration: They repeated the above steps (training, pruning, resetting) for a certain number of iterations, each time pruning more weights.
"""

#Model path
train_model_path_input = '/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/lightning_logs/version_228/checkpoints/epoch=6-step=16023.ckpt' #"/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/lightning_logs/version_193/checkpoints/epoch=4-step=11445.ckpt" # "/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/lightning_logs/version_192/checkpoints/epoch=48-step=112161.ckpt"

#Total prune percentage
total_prune_percentage_input = 0.8002
#Number of pruning iterations
pruning_iterations_input = 3

#Choose model
#1. random Initialisation
model = vit_loader("simple") # "simple" or "efficient"

# Create a copy of the model for reinitialization
model_copy = copy.deepcopy(model)

# Hyperparameter
epochs = 1

def calculate_pruning_percentage(total_prune_percentage, iterations):
    remaining_percentage = 1 - total_prune_percentage
    prune_percentage_per_iteration = 1 - remaining_percentage ** (1 / iterations) #remaining_percentage*iterations
    return prune_percentage_per_iteration

total_prune_percentage = total_prune_percentage_input
pruning_iterations = pruning_iterations_input
prune_ratio = calculate_pruning_percentage(total_prune_percentage, pruning_iterations)

#2. Trained model that stopped training when val_loss doesnt improve anymore 
trained_model_path = train_model_path_input 
trained_model = model.load_from_checkpoint(checkpoint_path=trained_model_path)

# 3. Iterative pruning and reinitialization
for iteration in range(pruning_iterations):
    trainer_prune=pl.Trainer(max_epochs=1, fast_dev_run=False) # limit_train_batches=0.2, limit_val_batches=0.2, enable_checkpointing=True
    # Prune the model weights
    for name, module in trained_model.named_modules():
        if "transformer.layers" in name and (".net.1" in name or ".net.3" in name) and isinstance(module, nn.Linear):
        #if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_ratio)

            # Get the original module (before training) from the copied model
            original_module = dict(model_copy.named_modules())[name]

            module.weight.data = module.weight.data.to(device)
            original_module.weight.data = original_module.weight.data.to(device)

            # create mask
            mask = module.weight_mask
            mask = mask.to(device)
            # Reinitialize weights where mask != 0 (i.e., the unpruned weights) with their original random values before training, the pruned weights remain 0
            module.weight.data = torch.where(mask != 0, original_module.weight.data, module.weight.data)

    # Train the pruned model
    trainer_prune.fit(trained_model, train_loader, valid_loader)


if not os.path.exists('pruned_models'):
    os.makedirs('pruned_models')

my_timezone = pytz.timezone('Europe/Berlin')  
now = datetime.now(my_timezone)  
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  

# Save your model in the 'pruned_models' directory with a unique name
torch.save(trained_model, f'pruned_models/model_{timestamp}.pth')