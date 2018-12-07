import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import pyro
import pyro.distributions as dist
from collections import OrderedDict
import numpy as np

#Insert model here
class SeqRNN(nn.Module):
    def __init__(self,
            num_embed,
            embed_dim,
            hidden_size=300,
            num_rnn_layers=1,
            outputs=2,
            padding_idx=0,
            batch_size=4,
            batchfirst=True
            ):
        super(SeqRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.batch_first=batchfirst
        self.use_cuda = torch.cuda.is_available()
        self.padding_idx=padding_idx
        self.batch_size = batch_size

        #Layers
        self.embedding = nn.Embedding(num_embed,embed_dim , padding_idx)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, 
                batch_first=batchfirst, num_layers=self.num_rnn_layers)
        self.output = nn.Linear(embed_dim, outputs)
        

    def forward(self, X, lengths, y=None):
        #length corresponds to length of entries in X, expected to be sorted from longest to shortest
        #assumes X is not padded 
        #since X has to be a list, dimensions of X are batch_size
        self.batch_size = len(X) #lol, there's no reason this needs to be a field hahah. 
        hidden = self.initHidden()
        X = pad_sequence(X, batch_first=self.batch_first, padding_value=self.padding_idx) 

        #Get embeddings
        X = self.embedding(X)
        X = pack_padded_sequence(X, lengths, batch_first=self.batch_first)
        #output, lengths = pad_packed_sequence(embedded, batch_first=self.batch_first)

        X, hidden = self.gru(X, hidden)
        X, lengths = pad_packed_sequence(X, batch_first=self.batch_first)
        X = F.relu(X)
        X = X.contiguous()

        #This was for classification originally, now it should be for sequences
        #temp = torch.zeros(self.batch_size, X.size()[2])
        #if self.use_cuda:
        #    temp = temp.cuda()


        #for i in range(0, self.batch_size):
        #    l = lengths[i]
        #    temp[i] = X[i,l - 1,:]
        #X = temp

        X = self.output(X)
        return X, hidden

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
            input_y=0):
        super(CNNTextEncoder, self).__init__()
        assert(len(kernels) == len(filters))
        assert(len(kernels) == 3)
        assert(len(filters) == 3)

        self.batch_first = batch_first
        self.padding_idx = padding_idx

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


def get_MLP(self, input_size, hidden_layers, output_size, o_activation=None):
    #creates a network which accepts inputs_size
    #hidden_layers can be a list which we will iterate over
    #output_size expected output size of network
    layers = [('input', nn.Linear(input_size, hidden_layers[0]))]
    for i, size in enumerate(hidden_layers[1:-1]):
        layers = layers + [('hidden{}'.format(i), nn.Linear(size, hidden_layers[i+1])), 
                ('activation{}'.format(i), nn.Softplus())
                ]
    layers = layers + [('output', nn.Linear(hidden_layers[-1], output_size))]
    if not (o_activation is None):
        layers = layers + [('o_activation', o_activation)]


    MLP = nn.Sequential(OrderedDict(layers))
    return MLP



class TextSSVAE(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised variational auto-encoder on the MNIST image dataset
    :param output_size: size of the tensor representing the class label (10 for MNIST since
                        we represent the class labels as a one-hot vector with 10 components)
    :param input_size: size of the tensor representing the image (28*28 = 784 for our MNIST dataset
                       since we flatten the images and scale the pixels to be in [0,1])
    :param z_dim: size of the tensor representing the latent random variable z
                  (handwriting style for our MNIST dataset)
    :param hidden_layers: a tuple (or list) of MLP layers to be used in the neural networks
                          representing the parameters of the distributions in our model
    :param use_cuda: use GPUs for faster training
    :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
    """
    def __init__(self,
            embed_dim=300,
            z_dim=100, kernels=[3,4,5],
            filters=[100,100,100], hidden_size = 300,
            padding_index = 0,
            num_rnn_layers=1, 
            config_enum=None, use_cuda=False,
            aux_loss_multiplier=None,
            vocab_size = 100,
            output_size = 2,
            embed_model='word2vec.model.wv.vectors.npy'):

        super(TextSSVAE, self).__init__()

        # initialize the class with all arguments provided to the constructor
        self.embed_dim = embed_dim
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.kernels = kernels
        self.filters = filters
        self.padding_index = padding_index
        self.num_rnn_layers = num_rnn_layers

        #loss and performance parameters
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier
        try:
            embed_matrix = torch.from_numpy(np.load(embed_model)).float()
        except (Exception, e):
            embed_matrix = None
        # define and instantiate the neural networks representing
        # the paramters of various distributions in the model
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.setup_networks(embed_matrix)

    def setup_networks(self, embed_matrix):

        z_dim = self.z_dim
        hidden_size = self.hidden_size
        num_rnn_layers = self.num_rnn_layers

        # define the neural networks used later in the model and the guide.
        # these networks are MLPs (multi-layered perceptrons or simple feed-forward networks)
        # where the provided activation parameter is used on every linear layer except
        # for the output layer where we use the provided output_activation parameter

        #self.encoder_y = self.get_MLP(self.input_size, 
        #        self.hidden_layers, self.output_size, nn.Softmax(dim=1))
        self.encoder_y = CNNTextEncoder( self.kernels,
            self.filters,
            self.vocab_size,
            self.embed_dim,
            p=.5, hidden_dim=hidden_size,
            padding_idx=0,
            outputs=2,
            batch_first=True)

        if embed_matrix is not None:
            self.encoder_y.load_embeddings(embed_matrix, self.padding_index)


        # a split in the final layer's size is used for multiple outputs
        # and potentially applying separate activation functions on them
        # e.g. in this network the final output is of size [z_dim,z_dim]
        # to produce loc and scale, and apply different activations [None,Exp] on them
        
        #self.encoder_z = self.get_MLP(self.input_size + self.output_size,
        #        self.hidden_layers, 2 * z_dim, nn.Softplus())
        print('encoder_z inputs y is fixed, please fix')
        self.encoder_z = CNNTextEncoder( self.kernels,
            self.filters,
            self.vocab_size,
            self.embed_dim,
            outputs=2 * self.z_dim,
            p=.5, hidden_dim= hidden_size,
            padding_idx=self.padding_index,
            batch_first=True,
            input_y=2)

        if embed_matrix is not None:
            self.encoder_z.load_embeddings(embed_matrix, self.padding_index)

        self.encoder_z_mean = nn.Linear(2 * z_dim, z_dim)
        self.encoder_z_var = nn.Sequential(
                nn.Linear(2 * z_dim, z_dim),
                nn.Softplus(beta=2.7)) #probably should use exp function...oh well

        #self.decoder = self.get_MLP(z_dim + self.output_size, 
        #        self.hidden_layers, self.input_size, nn.Sigmoid())
        self.decoder = SeqRNN( self.vocab_size, self.embed_dim,
            hidden_size=self.hidden_size, num_rnn_layers=1, outputs=self.vocab_size) 

        if embed_matrix is not None:
            self.decoder.load_embeddings(embed_matrix, self.padding_index)

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def model(self, xs, lengths, ys=None):
        """
        The model corresponds to the following generative process:
        p(z) = normal(0,I)              # handwriting style (latent)
        p(y|x) = categorical(I/10.)     # which digit (semi-supervised)
        p(x|y,z) = bernoulli(loc(y,z))   # an image
        loc is given by a neural network  `decoder`
        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        if self.use_cuda:
            #xs = xs.cuda()
            lengths = lengths.cuda()
            if ys is not None:
                ys = ys.cuda()
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("text_ss_vae", self)

        batch_size = len(xs)#xs.size(0)
        with pyro.iarange("data"):

            # sample the handwriting style from the constant prior distribution
            print('original used new_zeros...not sure wut it is')
            prior_loc = torch.zeros([batch_size, self.z_dim])
            prior_scale = torch.ones([batch_size, self.z_dim])
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).independent(1))

            # if the label y (which digit to write) is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
            alpha_prior = torch.ones([batch_size, self.output_size]) / (1.0 * self.output_size)
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)

            # finally, score the image (x) using the handwriting style (z) and
            # the class label y (which digit to write) against the
            # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
            # where `decoder` is a neural network
            #inputs = torch.cat([zs, ys], dim=1)
            print('model')
            loc = self.decoder.forward(zs,lengths, y=ys)
            
            pyro.sample("x", dist.Categorical(loc).independent(1), obs=xs)
            # return the loc so we can visualize it later
            return loc

    def guide(self, xs, lengths, ys=None):
        """
        The guide corresponds to the following:
        q(y|x) = categorical(alpha(x))              # infer digit from an image
        q(z|x,y) = normal(loc(x,y),scale(x,y))       # infer handwriting style from an image and the digit
        loc, scale are given by a neural network `encoder_z`
        alpha is given by a neural network `encoder_y`
        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        if self.use_cuda:
            xs = [ x.cuda() for x in xs]
            lengths = lengths.cuda()
            if ys is not None:
                ys = ys.cuda()
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent

        with pyro.iarange("data"):

            # if the class label (the digit) is not supervised, sample
            # (and score) the digit with the variational distribution
            # q(y|x) = categorical(alpha(x))
            if ys is None:
                alpha = self.encoder_y.forward(xs, lengths)

                ys = pyro.sample("y", dist.OneHotCategorical(F.softmax(alpha,dim=1)))

            # sample (and score) the latent handwriting-style with the variational
            # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            #inputs = torch.cat([xs, ys], dim=1)
            print('Guide')
            print(ys.size())
            output = self.encoder_z.forward(xs, lengths, y=ys)
            loc = self.encoder_z_mean(output)
            scale = self.encoder_z_var(output)

            #loc, scale = self.encoder_z.forward([xs, ys])
            pyro.sample("z", dist.Normal(loc, scale).independent(1))

    def classifier(self, xs, lengths):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for the image(s)
        if self.use_cuda:
            #xs = xs.cuda()
            lengths = lengths.cuda()
            if ys is not None:
                ys = ys.cuda()
        alpha = self.encoder_y.forward(xs,lengths)

        # get the index (digit) that corresponds to
        # the maximum predicted class probability
        res, ind = torch.topk(alpha, 1)

        # convert the digit(s) to one-hot tensor(s)
        ys = xs.new_zeros(alpha.size())
        ys = ys.scatter_(1, ind, 1.0)
        return ys

    def model_classify(self, xs, lengths, ys=None):
        """
        this model is used to add an auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        if self.use_cuda:
            #xs = xs.cuda()
            lengths = lengths.cuda()
            if ys is not None:
                ys = ys.cuda()
        # register all pytorch (sub)modules with pyro
        pyro.module("text_ss_vae", self)

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.iarange("data"):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if ys is not None:
                alpha = self.encoder_y.forward(xs,lengths)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None):
        """
        dummy guide function to accompany model_classify in inference
        """
#pass
