import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

import numpy as np

def get_train_val_test_datasets(data_folder):
    # Data from this set is stored in an npz file
    def load_npz_file(filename):
        data = np.load(f'{data_folder}/{filename}')
        train, test = data['arr_0'], data['arr_1']
        return train, test
    
    # Load the full data
    xs = load_npz_file('PSI1.npz')
    ys = load_npz_file('PSI2.npz')

    print(xs[0].shape)

    # Return the trio of datasets
    return \
        DatasetTurbulence(xs, ys, 'train'), \
        DatasetTurbulence(xs, ys, 'val'), \
        DatasetTurbulence(xs, ys, 'test')


class DatasetTurbulence(Dataset):
    def __init__(self, xs, ys, split):#, train_fraction=1.0):

        # Unwrap the loaded raw data
        x, x_test = xs
        y, y_test = ys

        # Reshape it as needed
        x       = x.reshape(-1,64,64,1)
        x_test  = x_test.reshape(-1,64,64,1)

        # Split into validation and training
        val_ratio   = 0.2
        ind_shuffle = np.random.randint(0, y.shape[0], size=int(y.shape[0] * val_ratio)) 

        # Set aside the validation set
        x_val = x[ind_shuffle,:,:,:]
        y_val = y[ind_shuffle]

        # Set aside the training set
        x_train = np.delete(x, ind_shuffle, axis=0)
        y_train = np.delete(y, ind_shuffle)

        # Store what we need based on the split
        if split == 'train':
            # Sometimes we may only want a fraction of the training data. Its shuffled
            # so we can just take the first part
            n = len(self.x)
            n_fraction = int(train_fraction * n)
            self.x = x_train[:n_fraction]
            self.y = y_train[:n_fraction]

        elif split == 'val':
            self.x = x_val
            self.y = y_val

        elif split == 'test':
            self.x = x_test
            self.y = y_test

        assert len(self.x) == len(self.y), f'Dataset error: length of x ({len(self.x)}) not equal to length of y ({len(self.x)})'

        # Announce useful information
        print(f"{split} split:")
        print(f"\tx.shape={self.x.shape}")
        print(f"\ty.shape={self.y.shape}")
        print(f"\tfraction of data retained={train_fraction}")

        """
        # Don't do this - its a regression problem!
        values, counts = np.unique(self.y, return_counts=True)
        for i in range(len(values)):
            print(f"\ty={values[i]} has n={counts[i]} instances")
        """

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return the grayscale image and label
        return self.x[idx], self.y[idx]