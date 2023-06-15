import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import lightning as pl
from datetime import datetime
from vit_loader import vit_loader
from dataset2 import seed, seed_everything, train_loader, valid_loader

import os

#utils
current_path = os.getcwd()

seed_everything(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
l1_unstructured, the L1-norm (which is simply the sum of the absolute values of the elements) is used as 
a measure of magnitude. The method prunes a specified fraction of the parameters with the smallest L1-norm.
"""

"""
1. Random Initialization: They started with a randomly initialized neural network.
2. Training: They trained this network for a certain number of iterations.
3. Pruning: After training, they pruned a certain percentage of the smallest weights in the network. This was done by setting the corresponding elements in the mask to zero.
4. Resetting: After pruning, they reset the weights of the remaining (unpruned) part of the network to their values from the initial (random) initialization. This was done by element-wise multiplying the weights with the mask.
5. Iteration: They repeated the above steps (training, pruning, resetting) for a certain number of iterations, each time pruning more weights.
"""

# choose model
#1. random Initialisation
model = vit_loader("simple") # "simple" or "efficient"

# Create a copy of the model for reinitialization
model_copy = copy.deepcopy(model)

# Hyperparameters
epochs = 1
prune_ratio = 0.2
pruning_iterations = 3
prune_ratio_decay = (1 - prune_ratio) / pruning_iterations


#2. Trained model that converges - val_acc = 90%
trained_model_path = "/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/lightning_logs/version_193/checkpoints/epoch=4-step=11445.ckpt" # "/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/lightning_logs/version_192/checkpoints/epoch=48-step=112161.ckpt"
trained_model = model.load_from_checkpoint(checkpoint_path=trained_model_path)

# 3. Iterative pruning and reinitialization
for iteration in range(pruning_iterations):
    trainer_prune=pl.Trainer(max_epochs=1, fast_dev_run=False) # limit_train_batches=0.2, limit_val_batches=0.2, enable_checkpointing=True
    # Prune the model weights
    for name, module in trained_model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_ratio)

            # Get the original module (before training) from the copied model
            original_module = dict(model_copy.named_modules())[name]

            module.weight.data = module.weight.data.to(device)
            original_module.weight.data = original_module.weight.data.to(device)

            # Reinitialize the pruned weights
            mask = module.weight_mask
            mask = mask.to(device)
            # This will replace weights where mask == 0 (i.e., the pruned weights) with their original values
            module.weight.data = torch.where(mask != 0, original_module.weight.data, module.weight.data)

    # Train the pruned model
    trainer_prune.fit(trained_model, train_loader, valid_loader)

    # Update the prune ratio for the next iteration
    prune_ratio += prune_ratio_decay


if not os.path.exists('pruned_models'):
    os.makedirs('pruned_models')

# Get the current date and time
now = datetime.now()

# Format the date and time as a string
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# Save your model in the 'pruned_models' directory with a unique name
torch.save(trained_model, f'pruned_models/model_{timestamp}.pth')