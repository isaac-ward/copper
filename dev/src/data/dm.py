import pytorch_lightning as pl

import data 
import data.dl

class TurbulenceDataModule(pl.LightningDataModule):
    def __init__(self, folder_data, batch_size, train_fraction):
        super().__init__()
        self.folder_data = folder_data
        self.batch_size  = batch_size
        self.train_fraction = train_fraction

    def setup(self, stage):
        # Get the required dataloaders
        self.dl_train, self.dl_val, self.dl_test = data.dl.get_train_val_test_dataloaders(self.folder_data, self.batch_size, self.train_fraction)

    def train_dataloader(self):
        return self.dl_train

    def val_dataloader(self):
        return self.dl_val

    def test_dataloader(self):
        return self.dl_test

    def teardown(self, stage):
        pass