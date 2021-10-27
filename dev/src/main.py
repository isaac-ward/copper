import os

import numpy as np
import torch 
import pytorch_lightning as pl
import pytorch_lightning.loggers
import wandb

import data 
import data.ds
import data.dl
import data.dm

import models

# Prevents the prompt
os.environ["WANDB_API_KEY"] = "a6cdd4fa5fdf946057a6e2825ad0e9addbeed513"

# Execute when the module is not initialized from an import statement
if __name__ == '__main__':

    # Where are we?
    folder_repo = os.path.normpath(os.getcwd())
    folder_data = f'{folder_repo}/data/'

    # Set the seeds
    random_seed = 42
    pl.utilities.seed.seed_everything(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Check our devies
    print(f'{torch.cuda.device_count()} CUDA devices found')
    print(f'CUDA available = {torch.cuda.is_available()}')

    train_fraction = [ 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001 ]

    for tf in train_fraction:

        # Get a logger 
        wandb_logger = pytorch_lightning.loggers.WandbLogger(project='copper', log_model=False, name=f"fraction={tf}")

        # Get the data
        #ds_train, ds_val, ds_test = data.ds.get_train_val_test_datasets(folder_data)
        #dl_train, dl_val, dl_test = data.dl.get_train_val_test_dataloaders(folder_data, batch_size=64)
        dm_turbulence = data.dm.TurbulenceDataModule(folder_data, batch_size=64, train_fraction=1.0)

        # Get the model
        model_baseline = models.BaselineModel(lr=0.00001)

        # Set up the trainer
        trainer = pl.Trainer(
            max_epochs=12, 
            precision=16,
            gpus=1, 
            accelerator='dp',
            num_sanity_val_steps=2, 
            check_val_every_n_epoch=1,
            log_every_n_steps=1,    # Need this to see logged results at every step
            terminate_on_nan=True,
            gradient_clip_val=0.5,  # Good range is [-1,1]. Prevents one minibatch from ruining the train
            logger=wandb_logger)

        # Run the experiment
        trainer.fit(model_baseline, dm_turbulence)

        # This run is complete
        wandb.finish()