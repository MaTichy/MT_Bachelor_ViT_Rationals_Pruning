import lightning as pl
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.tuner.tuning import Tuner

from vit_loader import vit_loader

# change depending on dataset: for tiny images: dataset and for svhn: dataset2, dataset3 = svhn as LightningModule
from dataset2 import train_loader, valid_loader #test_loader,
from dataset2 import seed_everything, seed
#from dataset3 import SVHNDataModule

#svhn_datamodule = SVHNDataModule()

#seed
seed_everything(seed)

# set epochs
epochs = 40 #20

# choose model
model = vit_loader("efficient") # "simple" or "efficient"

# lightning Trainer 
trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=False, accelerator='auto') # precision="16-mixed" callbacks=[EarlyStopping(monitor="val_loss", mode="min")]  limit_train_batches=0.25, limit_val_batches=0.25, , callbacks=[StochasticWeightAveraging(swa_lrs=2e-4)]

tuner = Tuner(trainer)
# Run learning rate finder
lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=valid_loader, attr_name='lr')

# Plot with
fig = lr_finder.plot(suggest=True)
fig.show()

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()

