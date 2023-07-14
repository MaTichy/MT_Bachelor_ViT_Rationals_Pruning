import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

import lightning as pl

from torchmetrics import Accuracy

from warmupScheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import StepLR, LambdaLR, _LRScheduler

from dataset2 import seed, seed_everything

seed_everything(seed)

#Parameter

scheduler_string = 'StepLR' #'WarmupStepLR' 'Cosine'

#StepLR               #gamma=0.6, 0.7 rationals: StepLR(optimizer, step_size=1, gamma=0.7), WarmupStepLR(optimizer, warmup_epochs=1, start_lr_warmup=4e-5, step_size=1, gamma=0.7) -> v_num242 lr simple: 5e-5, rationals, "gelu", coefficients=True
step_size_steplr=2
gamma_steplr=0.7

#WarmupStepLR
warmup_epochs=1
start_lr_warmup=5e-4
step_size_warmup=1 
gamma_warmup=0.7

#LinearCosineAnnealing
warmup_epochs_cosine=3
warmup_start_lr_cosine=5e-5
eta_min_cosine=1e-6
max_epochs_cosine=35

#Accuracy Attributes train/ val
num_classes_train=10 #10, 200
top_k_train=1

num_classes_val=10 #10
top_k_val=1

# helpers
class WarmupStepLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, step_size, start_lr_warmup, gamma, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.step_size = step_size
        self.start_lr_warmup = start_lr_warmup
        self.gamma = gamma
        self.lr_decay = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma, last_epoch=last_epoch)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.start_lr_warmup + ((base_lr - self.start_lr_warmup) / self.warmup_epochs) * self.last_epoch for base_lr in self.base_lrs]
        else:
            return self.lr_decay.get_last_lr()
    
    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            super().step(epoch)
        else:
            self.lr_decay.step(epoch)

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

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, activation):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        #print(f'Input shape1: {x.shape}')
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64): 
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)  # dim AUCH HIER WEGEN OPTUNA MUSS input tensor matchen 

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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, activation):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim, activation)
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
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, lr, activation):
        super().__init__()

        self.save_hyperparameters(ignore=['activation'])
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.lr=lr

        # new PL attributes: 
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes_train, top_k=top_k_train) #top_k=1 means only if the correct output is given the answer is right, top_k=3 would mean the answer has to be in top 3 output of model for correct response
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes_val, top_k=top_k_val) 
        #self.test_acc = Accuracy()  

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),  #dim ACHTUNG GEÄNDERT FÜR OPTUNA
            nn.LayerNorm(dim),          # dim AUCH HIER
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, activation)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) # nn.Activations: lr: 3e-4
        if scheduler_string == 'StepLR':
            scheduler = StepLR(optimizer, step_size=step_size_steplr, gamma=gamma_steplr) 
        elif scheduler_string == 'WarmupStepLR':
            scheduler = WarmupStepLR(optimizer, warmup_epochs=warmup_epochs, start_lr_warmup=start_lr_warmup, step_size=step_size_warmup, gamma=gamma_warmup)
        else: 
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs_cosine, warmup_start_lr=warmup_start_lr_cosine, eta_min=eta_min_cosine, max_epochs=max_epochs_cosine)

        return [optimizer],[scheduler]
 
    def training_step(self, batch, batch_idx):

        # Loop through data loader data batches
        x,y = batch 

        # 1. Forward pass
        y_pred = self(x) 

        # define loss
        loss_fn = nn.CrossEntropyLoss()
        # 2. Calculate loss
        loss = loss_fn(y_pred, y)

        # Compute accuracy
        preds = torch.argmax(y_pred, dim=1)
        self.train_acc.update(preds, y) 

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc.compute(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # logg gradients
        """
        Turn OFF for optuna, becasue of CSV logger doesnt have self.logger.experiment.add_histogram
        
        if self.global_step % 2000 == 0:
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    # Log the gradients of linear layers
                    for param_name, param in module.named_parameters():
                        if param.grad is not None:
                            self.logger.experiment.add_histogram(f'gradients/{name}/{param_name}', param.grad, self.global_step)
        """
        return loss

    def validation_step(self, batch, batch_idx):

        # Loop through DataLoader batches
        x,y = batch 

        # 1. Forward pass
        val_pred_logits = self(x) 

        # define loss
        loss_fn = nn.CrossEntropyLoss()
        # 2. Calculate and accumulate loss
        loss = loss_fn(val_pred_logits, y)

        # Compute accuracy
        preds = torch.argmax(val_pred_logits, dim=1)
        self.val_acc.update(preds, y) 

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_acc.compute(), prog_bar=True, logger=True) 
        self.val_loss = loss

        return loss
    
        
        

