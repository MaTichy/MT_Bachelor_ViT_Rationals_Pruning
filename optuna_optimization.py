
import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from simpleViT import simple_ViT  
from dataset2 import train_data, valid_data, NUM_WORKERS  
from rationals import RationalsModel  
import pandas as pd
import torch
import glob
import os


seed_everything(42)

# torch.save(model)
if not os.path.exists('optuna_logs'):
    os.makedirs('optuna_logs')

def objective(trial):

    # Hyperparameters for the RationalsModel
    n = trial.suggest_int('n', 2, 8)
    m = trial.suggest_int('m', 2, 8)
    function = trial.suggest_categorical('function', ["relu", "sigmoid", "tanh", "leaky_relu", "swish", "gelu"])


    rational_model = RationalsModel(n=n, m=m, function=function, use_coefficients=False)

    patch_sizes = [4, 8]

    # Suggest a value from the list
    patch_size_op = trial.suggest_categorical('patch_size', patch_sizes)

    """
    I changed the simpleViT in a way that the first linear layer has the dimension of 1024, this way the model will always suit the input tensor.
    So that the dim parameter here doesnt change these first linear layers 
    After that however any dimension can be valid so this part will be part of the optimization process
    """

    # Calculate the 'mlp_dim' parameter based on the 'dim' parameter
    mlp_dim_op = trial.suggest_int('dim', 512, 2048)

    # Define a neural network using pytorch lightning.
    model = simple_ViT(
        image_size=32,
        patch_size=patch_size_op,
        num_classes=10,
        dim=1024,
        depth=trial.suggest_int('depth', 2, 8),
        heads=trial.suggest_int('heads', 6, 12),
        mlp_dim=mlp_dim_op,
        lr=trial.suggest_float('lr', 1e-5, 1e-2),
        activation=rational_model
    )

        
    # Optimized batch size
    BATCH_SIZE = trial.suggest_int('BATCH_SIZE', 16, 128)
    #BATCH_SIZE = 128

    # Create new data loaders with the optimized batch size
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, drop_last=True)

    print(len(train_loader), len(valid_loader))

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')

    logger = CSVLogger('optuna_logs/', name='optuna_exp')

    # Define a trainer
    trainer = Trainer(
        logger=logger,
        max_epochs=20,  # You may want to optimize the number of epochs as well
        callbacks=[checkpoint_callback, 
                   EarlyStopping(monitor='val_loss', mode='min', patience=2), 
                   PyTorchLightningPruningCallback(trial, monitor='val_loss')],
    )
    try:
        # Train the model
        trainer.fit(model, train_loader, valid_loader)

        # Get the logger and version
        logger = trainer.logger
        version = logger.version

        logs = pd.read_csv(f'optuna_logs/optuna_exp/version_{version}/metrics.csv')
        
        val_loss = logs['val_loss']

        # Convert the 'val_loss' column to numeric, turning errors into NaN
        val_loss = pd.to_numeric(val_loss, errors='coerce')

        # Handle NaN values (e.g., by replacing them with a high loss value)
        val_loss = val_loss.fillna(1e6)

        # Get the final validation loss
        final_val_loss = val_loss.dropna().iloc[-1] 
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"WARNING: Caught out of memory error at trial {trial.number}. Skipping trial.")
            raise optuna.TrialPruned()  # skip this trial
        else:
            raise e 
    # Return the validation accuracy
    return final_val_loss

study = optuna.create_study(direction='minimize', study_name='Hypa Tuna', pruner=optuna.pruners.MedianPruner(), storage='sqlite:///optuna_results.db')
study.optimize(objective, n_trials=150)