import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class SeqRNN(nn.Module):
    def __init__(self,
            embed_dim,
            hidden_size=300,
            num_rnn_layers=1,
            outputs=2,
            batch_size=4,
            batchfirst=True,
            dropout = 0.3
            ):
        super(SeqRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.batch_first=batchfirst
        self.use_cuda = torch.cuda.is_available()
        self.batch_size = batch_size

        #Layers
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, \
                batch_first=batchfirst, num_layers=self.num_rnn_layers)
        self.out = nn.Linear(embed_dim, outputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden=None, y=None):
        #input = input.unqueeze(0)
        self.batch_size = len(input)
        if hidden is None:
            hidden = self.initHidden()

        input = self.dropout(input)
        output, hidden = self.rnn(input, hidden)

        prediction = self.out(F.relu(output.squeeze(0)))

        return prediction, hidden

    def initHidden(self):
        result = torch.zeros(self.num_rnn_layers, self.batch_size, self.hidden_size)
        if self.use_cuda:
            return result.cuda()
        else:
            return result

    def load_embeddings(self, weights, padding_index, freeze=False, sparse=False):
        #replaces existing embeddings with a pretrained one
        self.padding_index = padding_index
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=freeze, sparse=sparse)