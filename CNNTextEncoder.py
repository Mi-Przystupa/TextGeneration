import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class CNNTextEncoder(nn.Module):

    def __init__(self,
            kernels,
            filters,
            embed_dim,
            p=.5, hidden_dim=300,
            outputs=2,
            batch_first=True,
            input_y=0):
        super(CNNTextEncoder, self).__init__()
        assert(len(kernels) == len(filters))
        assert(len(kernels) == 3)
        assert(len(filters) == 3)

        self.batch_first = batch_first
        self.batch_size = 1

        #parameter configs
        self.conv1 = nn.Conv2d(1, filters[0], (kernels[0], embed_dim))
        self.conv2 = nn.Conv2d(1, filters[1], (kernels[1], embed_dim))
        self.conv3 = nn.Conv2d(1, filters[2], (kernels[2], embed_dim))

        self.fully_connected = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(np.sum(filters) + input_y, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, outputs))

    def forward(self, X, lengths, y=None):
        """
        :param X:
        :param lengths: 
        :param y: 
        :return: 
        """
        #X: batch x seq x embeddings tensor
        #lengths the lengths of sequences in padded sequence
        self.batch_size = len(X)
        #Get embeddings
        X = X.unsqueeze(1)

        X1 = F.relu(self.conv1(X))
        X2 = F.relu(self.conv2(X))
        X3 = F.relu(self.conv3(X))

        X1 = F.adaptive_max_pool2d(X1, (1,1)).squeeze(2).squeeze(2)
        X2 = F.adaptive_max_pool2d(X2, (1,1)).squeeze(2).squeeze(2)
        X3 = F.adaptive_max_pool2d(X3, (1,1)).squeeze(2).squeeze(2)

        if y is not None:

            X = torch.cat([X1, X2, X3, y], dim=1)
        else:
            X = torch.cat([X1, X2, X3], dim=1)

        X = F.relu(X)
        X = self.fully_connected(X)

        return X
