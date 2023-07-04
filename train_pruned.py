import torch.nn as nn
import torch
import lightning as pl
from datetime import datetime
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from dataset2 import train_loader, valid_loader, seed, seed_everything
from vit_loader import vit_loader

#model
model = vit_loader('simple')

#seed
seed_everything(seed)

#epochs
epochs = 15

def compute_stats(model):
    total_params = 0
    total_pruned_params = 0

    for name, module in model.named_modules():
        if "transformer.layers" in name and (".net.1" in name or ".net.3" in name) and isinstance(module, nn.Linear):
            #if isinstance(module, nn.Linear):
                total_params += torch.numel(module.weight.data)
                total_pruned_params += torch.sum(module.weight_mask == 0).item()

    stats = {
        "total_params": total_params,
        "total_pruned_params": total_pruned_params,
        "total_pruned_ratio": total_pruned_params / total_params,
    }

    return print(stats)

#initialize with pruned model from lth.py
model_pruned_final = torch.load("/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/structural_pruned_2023-07-04_20-13-41.pth") # "/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/model_2023-06-27_10-57-58.pth" "/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/model_2023-06-26_20-12-58.pth" #"/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/model_2023-06-15_11-26-16.pth"  #"/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/model_2023-06-13_10-07-01.pth")

# learning rate reinit
#model_pruned_final.hparams.lr = 4e-6 

#compute_stats(model_pruned_final)

# 4. final train pruned model with unpruned weights initialized from model_copy at start when initializing the model
trainer_final = pl.Trainer(max_epochs=epochs, fast_dev_run=False,  callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=2)]) # , limit_train_batches=0.5, limit_val_batches=0.5

#train
trainer_final.fit(model_pruned_final, train_loader, valid_loader)

# Get the current date and time
now = datetime.now()

# Format the date and time as a string
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# Save your model in the 'pruned_models' directory with a unique name
torch.save(model_pruned_final, f'pruned_models/retrained_{timestamp}.pth')