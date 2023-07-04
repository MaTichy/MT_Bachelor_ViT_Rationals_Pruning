import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from rationals import RationalsModel
import lightning as pl

from torchmetrics import Accuracy  

from warmupScheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import StepLR

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT(pl.LightningModule):
    def __init__(self, *, image_size, patch_size, num_classes, dim, transformer, pool = 'cls', channels = 3, lr):
        super().__init__()

        # new PL attributes: 
        self.train_acc = Accuracy(task="multiclass", num_classes=10, top_k=1) 
        self.valid_acc = Accuracy(task="multiclass", num_classes=10, top_k=1) 
        #self.test_acc = Accuracy() 

        #set learning rate
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
    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0) #weight_decay=0.003, fused=True !!lr: 3e-4!!
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

        return [optimizer],[scheduler]
    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, fused=True) # optimal lr: 0.0001445439770745928 (lr_find) eta_min= 1e-4, 4e-5, 2.7542287033381663e-04, warmup_start_lr=1e-4
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, warmup_start_lr=1e-5, eta_min=5e-4, max_epochs=30) #Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr and base_lr followed by a cosine annealing schedule between base_lr and eta_min.
        
        return [optimizer],[scheduler]
    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0) #weight_decay=0.003, fused=True !!lr: 3e-4!!
        scheduler = StepLR(optimizer, step_size=4, gamma=0.7)

        return [optimizer],[scheduler]
    """
    
    def training_step(self, batch, batch_idx):

        # Loop through data loader data batches
        x,y = batch #X

        # Forward pass
        y_pred = self(x) #X

        # define loss
        loss_fn = nn.CrossEntropyLoss()
        # Calculate loss
        loss = loss_fn(y_pred, y)

        # Compute accuracy
        preds = torch.argmax(y_pred, dim=1)
        self.train_acc.update(preds, y) 

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc.compute(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        # Loop through DataLoader batches
        x,y = batch 

        # Forward pass
        val_pred_logits = self(x)

        # define loss
        loss_fn = nn.CrossEntropyLoss()
        # Calculate and accumulate loss
        loss = loss_fn(val_pred_logits, y)

        # Compute accuracy
        preds = torch.argmax(val_pred_logits, dim=1)
        self.valid_acc.update(preds, y) 

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.valid_acc.compute(), prog_bar=True, logger=True) 

        return loss
    