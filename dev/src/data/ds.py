import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

import wandb

import numpy as np

def get_train_val_test_datasets(data_folder, train_fraction):
    # Data from this set is stored in an npz file
    def load_npz_file(filename):
        data = np.load(f'{data_folder}/{filename}')
        train, test = data['arr_0'], data['arr_1']

        # Convert to float32
        train = np.float32(train)
        test = np.float32(test)

        return train, test
    
    # Load the full data
    xs = load_npz_file('PSI1.npz')
    ys = load_npz_file('PSI2.npz')

    # Return the trio of datasets
    return \
        DatasetTurbulence(xs, ys, 'train', train_fraction=train_fraction), \
        DatasetTurbulence(xs, ys, 'val'), \
        DatasetTurbulence(xs, ys, 'test')


class DatasetTurbulence(Dataset):
    def __init__(self, xs, ys, split, train_fraction=1.0):

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
            n = len(x_train)
            n_fraction = int(train_fraction * n)
            self.x = x_train[:n_fraction]
            self.y = y_train[:n_fraction]

            # Plot histogram
            wandb.init()
            data = [[_] for _ in self.y]
            table = wandb.Table(data=data, columns=["targets"])
            wandb.log({'histogram': wandb.plot.histogram(table, "targets",
                title="Label Distribution")})

        elif split == 'val':
            self.x = x_val
            self.y = y_val

        elif split == 'test':
            self.x = x_test
            self.y = y_test

        assert len(self.x) == len(self.y), f'Dataset error: length of x ({len(self.x)}) not equal to length of y ({len(self.x)})'

        # PyTorch generally expects the order of axes to be (3,h,w)
        self.x = np.repeat(self.x, 3, -1)
        self.x = self.x.swapaxes(1, 3)
        self.x = self.x.swapaxes(2, 3)

        # Get rid of unecessary dimension
        self.y = np.reshape(self.y, (self.y.shape[0],))

        # Announce useful information
        print(f"{split} split:")
        print(f"\tx.shape={self.x.shape}")
        print(f"\tx.dtype={self.x.dtype}")
        print(f"\ty.shape={self.y.shape}")
        print(f"\ty.dtype={self.y.dtype}")
        print(f"\tfraction of data returned={train_fraction}")

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