import torch.nn as nn
import torch.nn.functional as F
import torch

class CNNTextEncoder(nn.Module):

    def __init__(self,
            kernels,
            filters,
            num_embed,
            embed_dim,
            p=.5, hidden_dim=300,
            padding_idx=0,
            outputs=2,
            batch_first=True,
            input_y=0,
            dropout=0.3):
        super(CNNTextEncoder, self).__init__()
        assert(len(kernels) == len(filters))
        assert(len(kernels) == 3)
        assert(len(filters) == 3)

        self.batch_first = batch_first
        self.padding_idx = padding_idx
        self.batch_size = 1

        #parameter configs
        self.embedding = nn.Embedding(num_embed, embed_dim, padding_idx)
        self.conv1 = nn.Conv2d(1, filters[0], (kernels[0], embed_dim))
        self.conv2 = nn.Conv2d(1, filters[1], (kernels[1], embed_dim))
        self.conv3 = nn.Conv2d(1, filters[2], (kernels[2], embed_dim))

        self.fully_connected = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(np.sum(filters) + input_y, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, outputs))

    def load_embeddings(self, weights, padding_index, freeze=False, sparse=False):
        #replaces existing embeddings with a pretrained one
        self.padding_index = padding_index
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=freeze, sparse=sparse)

    def forward(self, X, lengths, y=None):

        self.batch_size = len(X)
        X = pad_sequence(X, batch_first=self.batch_first,
                padding_value=self.padding_idx)
        #Get embeddings
        X = self.embedding(X)
        X = X.unsqueeze(1)

        X1 = F.relu(self.conv1(X))
        X2 = F.relu(self.conv2(X))
        X3 = F.relu(self.conv3(X))

        X1 = F.adaptive_max_pool2d(X1, (1,1)).squeeze(2).squeeze(2)
        X2 = F.adaptive_max_pool2d(X2, (1,1)).squeeze(2).squeeze(2)
        X3 = F.adaptive_max_pool2d(X3, (1,1)).squeeze(2).squeeze(2)

        if y is not None:
            print('we have labels')
            print(y)

            print(y.size())
            print(X1.size())
            X = torch.cat([X1, X2, X3, y],dim=1)
        else:
            X = torch.cat([X1, X2, X3],dim=1)

        print(X.size())
        X = F.relu(X)
        X = self.fully_connected(X)

        return X
