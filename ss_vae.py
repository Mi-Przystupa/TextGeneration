import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from collections import OrderedDict

#Insert model here


class SSVAE(nn.Module):
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
    def __init__(self, output_size=10, input_size=784, z_dim=50, hidden_layers=(500,),
                 config_enum=None, use_cuda=False, aux_loss_multiplier=None):

        super(SSVAE, self).__init__()

        # initialize the class with all arguments provided to the constructor
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier

        # define and instantiate the neural networks representing
        # the paramters of various distributions in the model
        self.setup_networks()
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

    def setup_networks(self):

        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers

        # define the neural networks used later in the model and the guide.
        # these networks are MLPs (multi-layered perceptrons or simple feed-forward networks)
        # where the provided activation parameter is used on every linear layer except
        # for the output layer where we use the provided output_activation parameter
        self.encoder_y = self.get_MLP(self.input_size, 
                self.hidden_layers, self.output_size, nn.Softmax(dim=1))

        #self.encoder_y = MLP([self.input_size] + hidden_sizes + [self.output_size],
        #                     activation=nn.Softplus,
        #                     output_activation=nn.Softmax,
        #                     allow_broadcast=self.allow_broadcast,
        #                     use_cuda=self.use_cuda)

        # a split in the final layer's size is used for multiple outputs
        # and potentially applying separate activation functions on them
        # e.g. in this network the final output is of size [z_dim,z_dim]
        # to produce loc and scale, and apply different activations [None,Exp] on them
        self.encoder_z = self.get_MLP(self.input_size + self.output_size,
                self.hidden_layers, 2 * z_dim, nn.Softplus())
        self.encoder_z_mean = nn.Linear(2 * z_dim, z_dim)
        self.encoder_z_var = nn.Sequential(
                nn.Linear(2 * z_dim, z_dim),
                nn.Softplus(beta=2.7)) #probably should use exp function...oh well

        #self.encoder_z = MLP([self.input_size + self.output_size] +
        #                     hidden_sizes + [[z_dim, z_dim]],
        #                     activation=nn.Softplus,
        #                     output_activation=[None, Exp],
        #                     allow_broadcast=self.allow_broadcast,
        #                     use_cuda=self.use_cuda)
        self.decoder = self.get_MLP(z_dim + self.output_size, 
                self.hidden_layers, self.input_size, nn.Sigmoid())
        #self.decoder = MLP([z_dim + self.output_size] +
        #                   hidden_sizes + [self.input_size],
        #                   activation=nn.Softplus,
        #                   output_activation=nn.Sigmoid,
        #                   allow_broadcast=self.allow_broadcast,
        #                   use_cuda=self.use_cuda)

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def model(self, xs, ys=None):
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
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)

        batch_size = xs.size(0)
        with pyro.iarange("data"):

            # sample the handwriting style from the constant prior distribution
            prior_loc = xs.new_zeros([batch_size, self.z_dim])
            prior_scale = xs.new_ones([batch_size, self.z_dim])
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).independent(1))

            # if the label y (which digit to write) is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
            alpha_prior = xs.new_ones([batch_size, self.output_size]) / (1.0 * self.output_size)
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)

            # finally, score the image (x) using the handwriting style (z) and
            # the class label y (which digit to write) against the
            # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
            # where `decoder` is a neural network
            inputs = torch.cat([zs, ys], dim=1)
            loc = self.decoder.forward(inputs)
            
            pyro.sample("x", dist.Bernoulli(loc).independent(1), obs=xs)
            # return the loc so we can visualize it later
            return loc

    def guide(self, xs, ys=None):
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
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.iarange("data"):

            # if the class label (the digit) is not supervised, sample
            # (and score) the digit with the variational distribution
            # q(y|x) = categorical(alpha(x))
            if ys is None:
                alpha = self.encoder_y.forward(xs)
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))

            # sample (and score) the latent handwriting-style with the variational
            # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            inputs = torch.cat([xs, ys], dim=1)
            output = self.encoder_z.forward(inputs)
            loc = self.encoder_z_mean(output)
            scale = self.encoder_z_var(output)

            #loc, scale = self.encoder_z.forward([xs, ys])
            pyro.sample("z", dist.Normal(loc, scale).independent(1))

    def classifier(self, xs):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for the image(s)
        alpha = self.encoder_y.forward(xs)

        # get the index (digit) that corresponds to
        # the maximum predicted class probability
        res, ind = torch.topk(alpha, 1)

        # convert the digit(s) to one-hot tensor(s)
        ys = xs.new_zeros(alpha.size())
        ys = ys.scatter_(1, ind, 1.0)
        return ys

    def model_classify(self, xs, ys=None):
        """
        this model is used to add an auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.iarange("data"):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if ys is not None:
                alpha = self.encoder_y.forward(xs)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None):
        """
        dummy guide function to accompany model_classify in inference
        """
#pass
