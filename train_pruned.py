import torch.nn as nn
import torch
import lightning as pl
import torch.nn.utils.prune as prune

from dataset2 import train_loader, valid_loader, seed, seed_everything
from train import model

seed_everything(seed)

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

#
epochs = 5

pruned_model_path ="/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/lightning_logs/version_196/checkpoints/epoch=0-step=457.ckpt"

model_pruned_final = model.load_from_checkpoint(checkpoint_path=pruned_model_path)

#compute_stats(model_pruned_final)

# final train pruned model 
trainer_final = pl.Trainer(max_epochs=epochs, fast_dev_run=False, limit_train_batches=0.5, limit_val_batches=0.5)
trainer_final.fit(model_pruned_final, train_loader, valid_loader)
