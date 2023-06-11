import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.tuner.tuning import Tuner

from vit_loader import vit_loader

# change depending on dataset: for tiny images: dataset and for svhn: dataset2, dataset3 = svhn as LightningModule
from dataset2 import train_loader, valid_loader #test_loader,
from dataset2 import seed_everything, seed
#from dataset3 import SVHNDataModule

# set Tensorboard debugger
# tf.debugging.experimental.enable_dump_debug_info("../lightning_logs/", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

#svhn_datamodule = SVHNDataModule()

#seed
seed_everything(seed)

# set epochs
epochs = 50 #20

# choose model
model = vit_loader("efficient") # "simple" or "efficient"

# lightning Trainer 
trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=False, callbacks=[StochasticWeightAveraging(swa_lrs=2e-5)]) # precision="16-mixed" callbacks=[EarlyStopping(monitor="val_loss", mode="min")], callbacks=[StochasticWeightAveraging(swa_lrs=2e-5)], accelerator='auto' imit_train_batches=0.7, limit_val_batches=0.7,
trainer.fit(model, train_loader, valid_loader) # train_loader, valid_loader / svhn_datamodule 
