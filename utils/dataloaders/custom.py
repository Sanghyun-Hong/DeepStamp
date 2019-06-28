"""
    Load the custom dataset (watermarked dataset)
"""
import os
import torch
import pickle
import random
import numpy as np
from PIL import Image

# torch/torchvision module
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, location, train=True, transform=None):
        self.location = location
        self.train    = train
        self.transform= transform

        ############
        # Load the train data
        ############
        if self.train:
            # initialize holders
            self.train_data   = []
            self.train_labels = []

            train_files = [os.path.join(self.location, each_file) \
                           for each_file in os.listdir(self.location) \
                           if 'train' in each_file]

            # : read the files and load the train data
            for each_file in train_files:
                train_data = pickle.load(open(each_file, 'rb'))
                self.train_data.append(train_data['data'])
                self.train_labels += train_data['labels']

            # : correct the dimensions
            self.train_data = np.concatenate(self.train_data)
            # self.store_sample(self.train_data, 80, 'custom_train_samples.png')
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            print ('CustomDataset: read the train data - {}'.format(self.train_data.shape))

        ############
        # Load the valid data
        ############
        else:

            # initialize holders
            self.valid_data   = []
            self.valid_labels = []

            valid_files = [os.path.join(self.location, each_file) \
                           for each_file in os.listdir(self.location) \
                           if 'valid' in each_file]

            # : read the files and load the valid data
            for each_file in valid_files:
                valid_data = pickle.load(open(each_file, 'rb'))
                self.valid_data.append(valid_data['data'])
                self.valid_labels += valid_data['labels']

            # : correct the dimensions
            self.valid_data = np.concatenate(self.valid_data)
            # self.store_sample(self.valid_data, 80, 'custom_valid_samples.png')
            self.valid_data = self.valid_data.transpose((0, 2, 3, 1))  # convert to HWC
            print ('CustomDataset: read the valid data - {}'.format(self.valid_data.shape))


    def __len__(self):
        """
            Return the number of instances in a dataset
        """
        if self.train:
            return len(self.train_data)
        else:
            return len(self.valid_data)


    def __getitem__(self, idx):
        """
            Return (data, label) where label is index of the label class
        """
        # pick (data, label) at an index
        if self.train:
            data, label = self.train_data[idx], self.train_labels[idx]
        else:
            data, label = self.valid_data[idx], self.valid_labels[idx]
        # return a PIL images
        data = Image.fromarray(data)
        # Notes:
        #  - ToTensor only works with Python 1d list, thus,
        #    we need to convert the 2d list into a numpy 2d array
        #  - Also, the 'int -> float32' conversion is required to compute loss
        label = np.array(label).astype(np.float32)
        if self.transform:
            data = self.transform(data)
        return (data, label)

    def store_sample(self, data, ssize, filename):
        data_size    = data.shape[0]
        data_samples = data[np.random.randint(data_size, size=ssize)]
        data_samples = data_samples / 255.0
        torchvision.utils.save_image(torch.from_numpy(data_samples), \
                                     filename, normalize=False, scale_each=False)
        # done.


###########################
#    Main: functions
###########################
if __name__ == '__main__':
    print ('this is a test version.')
    # Fin.
