import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

from rationals import RationalsModel

import lightning as pl

from torchmetrics import Accuracy  

from warmupScheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import StepLR


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForward(pl.LightningModule):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.PReLU(), #nn.PReLU(), RationalsModel(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        #print(f'Input shape1: {x.shape}')
        return self.net(x)

class Attention(pl.LightningModule):
    def __init__(self, dim, heads=8, dim_head=64): 
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(pl.LightningModule):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            #print(f'Input shape2: {x.shape}')
            x = attn(x) + x
            #print(f'Output shape3: {x.shape}')
            x = ff(x) + x
            #print(f'Output shape4: {x.shape}')
        return x

class simple_ViT(pl.LightningModule):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, lr):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.lr=lr

        # new PL attributes: 
        self.train_acc = Accuracy(task="multiclass", num_classes=10, top_k=1) 
        self.val_acc = Accuracy(task="multiclass", num_classes=10, top_k=1) 
        #self.test_acc = Accuracy()  

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0) #weight_decay=0.003, fused=True !!lr: 3e-4!!
        scheduler = StepLR(optimizer, step_size=2, gamma=0.7)

        return [optimizer],[scheduler]
    """
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, fused=True) # 3e-5, 3e-4, 3e-3, 4e-5, 4e-4, 4e-3, 5e-5, ...  weight_decay=0.05,
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, warmup_start_lr=5e-3, eta_min=3e-4, max_epochs=35) #Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr and base_lr followed by a cosine annealing schedule between base_lr and eta_min.
        
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
        self.val_acc.update(preds, y) 
        #acc = torch.mean((preds == y).float())

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        #self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_acc.compute(), prog_bar=True, logger=True) 

        return loss

