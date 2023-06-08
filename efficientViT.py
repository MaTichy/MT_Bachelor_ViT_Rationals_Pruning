import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from rationals import RationalsModel
import lightning as pl

from torchmetrics import Accuracy  

from warmupScheduler import LinearWarmupCosineAnnealingLR

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT(pl.LightningModule):
    def __init__(self, *, image_size, patch_size, num_classes, dim, transformer, pool = 'cls', channels = 3, lr):
        super().__init__()

        # new PL attributes: 
        self.train_acc = Accuracy(task="multiclass", num_classes=10, top_k=1) 
        self.valid_acc = Accuracy(task="multiclass", num_classes=10, top_k=1) 
        #self.test_acc = Accuracy() 

        self.lr=lr 
         
        image_size_h, image_size_w = pair(image_size)
        assert image_size_h % patch_size == 0 and image_size_w % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = (image_size_h // patch_size) * (image_size_w // patch_size)
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape 

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, fused=True) #weight_decay=0.003, fused=True !!lr: 3e-4!!

        return optimizer
    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, fused=True) # 3e-5, 3e-4, 3e-3, 4e-5, 4e-4, 4e-3, 5e-4, 0,0001! ... weight_decay=0.005
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=2, warmup_start_lr=5e-4, eta_min=1e-4, max_epochs=12) #Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr and base_lr followed by a cosine annealing schedule between base_lr and eta_min.
        
        return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler
        }
    """
    def training_step(self, batch, batch_idx):

        # Loop through data loader data batches
        x,y = batch #X

        # 1. Forward pass
        y_pred = self(x) #X

        # define loss
        loss_fn = nn.CrossEntropyLoss()
        # 2. Calculate loss
        loss = loss_fn(y_pred, y)

        # Compute accuracy
        preds = torch.argmax(y_pred, dim=1)
        #acc = torch.mean((preds == y).float())
        self.train_acc.update(preds, y) 

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc.compute(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        # Turn on inference context manager
        #with torch.inference_mode():
        # Loop through DataLoader batches
        x,y = batch #X

        # 1. Forward pass
        val_pred_logits = self(x) #X

        # define loss
        loss_fn = nn.CrossEntropyLoss()
        # 2. Calculate and accumulate loss
        loss = loss_fn(val_pred_logits, y)

        # Compute accuracy
        preds = torch.argmax(val_pred_logits, dim=1)
        self.valid_acc.update(preds, y) 
        #acc = torch.mean((preds == y).float())

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        #self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.valid_acc.compute(), prog_bar=True, logger=True) 

        return loss
    