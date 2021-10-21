from torch.utils.data import DataLoader

import data 
import data.ds

def get_train_val_test_dataloaders(folder_data, batch_size):
    
    # Load the datasets
    ds_train, ds_val, ds_test = data.ds.get_train_val_test_datasets(folder_data)

    # Now get the dataloaders - note they're already shuffled
    return \
        DataLoader(ds_train, batch_size=batch_size, shuffle=False), \
        DataLoader(ds_val,   batch_size=batch_size, shuffle=False), \
        DataLoader(ds_test,  batch_size=batch_size, shuffle=False)