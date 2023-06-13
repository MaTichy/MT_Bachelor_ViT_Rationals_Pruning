import torch.nn as nn
import torch
import lightning as pl
import torch.nn.utils.prune as prune

from dataset2 import train_loader, valid_loader, seed, seed_everything
from vit_loader import vit_loader

#model
model = vit_loader('simple')

#seed
seed_everything(seed)

#epochs
epochs = 5

def compute_stats(model):
    total_params = 0
    total_pruned_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_params += torch.numel(module.weight.data)
            total_pruned_params += torch.sum(module.weight_mask == 0).item()

    stats = {
        "total_params": total_params,
        "total_pruned_params": total_pruned_params,
        "total_pruned_ratio": total_pruned_params / total_params,
    }

    return stats

model_pruned_final = torch.load("/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/model_2023-06-13_10-07-01.pth")

#compute_stats(model_pruned_final)

# final train pruned model 
trainer_final = pl.Trainer(max_epochs=epochs, fast_dev_run=False, limit_train_batches=0.5, limit_val_batches=0.5)
trainer_final.fit(model_pruned_final, train_loader, valid_loader)
