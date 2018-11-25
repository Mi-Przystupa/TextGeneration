import errno
import os
from functools import reduce

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

def custom_collate_fn(batch):
    #assumed only 2 sets of lists in batch: data and label
    data = [None] * len(batch)
    lengths = [None] * len(batch)
    label = [None] * len(batch)

    for i, b in enumerate(sorted(batch, key=lambda x: len(x[0]), reverse=True)):
        data[i] = torch.tensor(b[0]).long()
        lengths[i] = len(b[0])
        label[i] = b[1]

    label = torch.stack(label)

    return [data, lengths, label]

class SentenceDataSet(Dataset):

    def __init__(self, X, y):
        if not (y is None):
            assert(len(X) == len(y))
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        x = self.X[index]#torch.LongTensor(self.X[index]) 
        if not (self.y is None):
            y = self.y[index]
        else:
            y = []
        return (x, y)

class TextDataSet(torch.utils.data.Dataset):

    def __init__(self, filename='tokenized_dataset.npy'):
        print('load data')
        data = np.load(filename).item()
        self.X = data['review_tokens']
        self.y = data['label']


    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
