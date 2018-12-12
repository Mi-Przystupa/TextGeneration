import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import pyro
import pyro.distributions as dist
from collections import OrderedDict
import numpy as np
from CNNTextEncoder import CNNTextEncoder
from SeqRNN import SeqRNN
from gensim.models import Word2Vec

def get_MLP(input_size, hidden_layers, output_size, o_activation=None):
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


class Exp(nn.Module):
    """
    custom module I stole from pyro...although i mean it's not like super profound
    """

    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, val):
        return torch.exp(val)


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
            z_dim=300, kernels=[3,4,5],
            filters=[100,100,100], hidden_size = 300,
            num_rnn_layers=1,
            config_enum=None, use_cuda=False,
            aux_loss_multiplier=None,
            output_size = 2,
            embed_model='word2vec.model.wv.vectors.npy'):

        super(TextSSVAE, self).__init__()

        # initialize the class with all arguments provided to the constructor
        self.embed_dim = embed_dim
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.kernels = kernels
        self.filters = filters
        self.num_rnn_layers = num_rnn_layers

        #loss and performance parameters
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier
        self.w2v_model = Word2Vec.load('word2vec.model').wv
        self.padding_idx = self.w2v_model.vocab.get('<PADDING>').index
        self.vocab_size = len(self.w2v_model.vectors)

        try:
            embed_matrix = torch.from_numpy(np.load(embed_model)).float()
        except:
            embed_matrix = torch.from_numpy(self.w2v_model.syn0).float()
        # define and instantiate the neural networks representing
        # the paramters of various distributions in the model

        self.output_size = output_size
        self.embeddings = None
        self.encoder_y = None
        self.encoder_z = None
        self.decoder = None

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

        self.embeddings = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)

        if embed_matrix is not None:
            self.embeddings.from_pretrained(embed_matrix, freeze=False,sparse=False)
        self.encoder_y = CNNTextEncoder( self.kernels,
            self.filters,
            self.embed_dim,
            p=0.00,
            hidden_dim=hidden_size,
            outputs=2,
            batch_first=True, input_y=0, output_activation=F.softmax)

        # a split in the final layer's size is used for multiple outputs
        # and potentially applying separate activation functions on them
        # e.g. in this network the final output is of size [z_dim,z_dim]
        # to produce loc and scale, and apply different activations [None,Exp] on them
        
        print('encoder_z inputs y is fixed, please fix')
        self.encoder_z = CNNTextEncoder(self.kernels,
            self.filters,
            self.embed_dim,
            p=0.00,
            outputs= self.z_dim,
             hidden_dim= hidden_size,
            batch_first=True,
            input_y=2)

        self.encoder_z_mean = nn.Linear(z_dim, z_dim)
        self.encoder_z_var = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                Exp()) #probably should use exp function...oh well

        self.decoder = SeqRNN(self.embed_dim, hidden_size=self.hidden_size, num_rnn_layers=num_rnn_layers, outputs= self.vocab_size, y_inputs=2)

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
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("text_ss_vae", self)

        batch_size = len(xs)#xs.size(0)
        with pyro.plate("data"):

            # sample the  style from the constant prior distribution
            prior_loc = torch.zeros([batch_size, self.z_dim])
            prior_scale = torch.ones([batch_size, self.z_dim])
            if self.use_cuda:
                prior_loc = prior_loc.cuda()
                prior_scale = prior_scale.cuda()

            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).independent(1))

            # if the label y (which digit to write) is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
            alpha_prior = torch.ones([batch_size, self.output_size]) / (1.0 * self.output_size)
            if self.use_cuda:
                alpha_prior = alpha_prior.cuda()

            if ys is not None:
                ys = torch.stack(ys)
                if self.use_cuda:
                    ys = ys.cuda()
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)

            # finally, score the image (x) using the handwriting style (z) and
            # the class label y (which digit to write) against the
            # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
            # where `decoder` is a neural network

            if len(ys.size()) > 2:
                slices = []
                for zs_marginal, y_marginal in zip(zs,ys):
                    decoded_output = self.decode_sentence(zs_marginal, y_marginal, lengths, batch_size)
                    slices.append(decoded_output)
                decoded_output = torch.cat(slices,dim=-1)
            else:
                decoded_output = self.decode_sentence(zs, ys, lengths, batch_size)

            xs = pad_sequence(xs, batch_first=True, padding_value=self.padding_idx)

            if self.use_cuda:
                xs = xs.cuda()
            xs = xs[:,0:torch.max(lengths)]
            pyro.sample("x", dist.Categorical(logits=decoded_output).to_event(1), obs=xs)
            # return the loc so we can visualize it later
            return decoded_output

    def guide(self, xs, lengths, ys=None):
        """
        The guide corresponds to the following:
        q(y|x) = categorical(alpha(x))              # infer digit from an image
        q(z|x,y) = normal(loc(x,y),scale(x,y))       # infer handwriting style from an image and the digit
        loc, scale are given by a neural network `encoder_z`
        alpha is given by a neural network `encoder_y`
        :param xs: a batch of sequences represented as indices in embedding matrix
        :param lengths: length of each of the sequences in xs
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        xs = pad_sequence(xs, batch_first=True, padding_value=self.padding_idx)
        if self.use_cuda:
            xs = xs.cuda()
            lengths = lengths.cuda()

        #Get embeddings
        xs = self.embeddings(xs)
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            # if the class label (the digit) is not supervised, sample
            # (and score) the digit with the variational distribution
            # q(y|x) = categorical(alpha(x))
            if ys is None:
                alpha = self.encoder_y.forward(xs, lengths)
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))
            else:
                ys = torch.stack(ys)
                if self.use_cuda:
                    ys = ys.cuda()

            # sample (and score) the latent handwriting-style with the variational
            # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            #inputs = torch.cat([xs, ys], dim=1)
            output = self.encoder_z.forward(xs, lengths, y=ys)
            loc = self.encoder_z_mean(output)
            scale = self.encoder_z_var(output)

            #loc, scale = self.encoder_z.forward([xs, ys])
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def classifier(self, xs, lengths):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for the image(s)
        xs = pad_sequence(xs, batch_first=True, padding_value=self.padding_idx)
        if self.use_cuda:
            xs = xs.cuda()
            lengths = lengths.cuda()

        #Get embeddings
        xs = self.embeddings(xs)

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
        xs = pad_sequence(xs, batch_first=True, padding_value=self.padding_idx)
        if ys is not None:
            ys = torch.stack(ys)

        if self.use_cuda:
            xs = xs.cuda()
            lengths = lengths.cuda()
            if ys is not None:
                ys = ys.cuda()
        #Get embeddings
        xs = self.embeddings(xs)

        # register all pytorch (sub)modules with pyro
        pyro.module("text_ss_vae", self)

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if ys is not None:
                alpha = self.encoder_y.forward(xs,lengths)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, lengths, ys=None):
        """
        dummy guide function to accompany model_classify in inference

        """

    def decode_sentence(self, init_hidden, ys, lengths, batch_size):
        length_hack = torch.max(lengths)

        decoder_output = torch.zeros(batch_size, length_hack, self.vocab_size) #torch.max(lengths))

        value = self.w2v_model.vocab.get('<SOS>')
        SOS = value.index
        inputs = torch.tensor([[SOS] for _ in lengths])
        if self.use_cuda:
            inputs = inputs.cuda()
            decoder_output = decoder_output.cuda()

        inputs = self.embeddings(inputs)
        if len(ys.size()) > 2:
            print('break point')

        hidden = torch.cat([init_hidden, ys], dim=1)
        hidden = hidden.unsqueeze(0)
        for t in range(1, length_hack):
            output, hidden = self.decoder.forward(inputs, hidden)
            decoder_output[:,t, :] = output.squeeze(1)
            top1 = output.max(2)[1]
            inputs = self.embeddings(top1)
            #hidden = torch.cat([hidden, ys], dim=1)

        return decoder_output

#pass
