from torch.utils.data import DataLoader

import data 
import data.ds

import os

def get_train_val_test_dataloaders(folder_data, batch_size, train_fraction):
    
    # Load the datasets
    ds_train, ds_val, ds_test = data.ds.get_train_val_test_datasets(folder_data, train_fraction)

    def _instantiate_dl(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

    # Now get the dataloaders - note they're already shuffled
    return \
        _instantiate_dl(ds_train), \
        _instantiate_dl(ds_val), \
        _instantiate_dl(ds_test)