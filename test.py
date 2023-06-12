import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import lightning as pl

from vit_loader import vit_loader
from dataset2 import seed, seed_everything, train_loader, valid_loader

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