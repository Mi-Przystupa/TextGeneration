import errno
import os
from functools import reduce

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class TextDataSet(torch.utils.data.Dataset):

    def __init__(self, train = True, root='tokenized_dataaset.npy'):
        print('load data')
        data = np.load(root).item()
        self.train_data = data['review_tokens']
        self.train_labels = data['label']

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        return self.train_data[index], self.train_labels[index]





# This file contains utilities for caching, transforming and splitting MNIST data
# efficiently. By default, a PyTorch DataLoader will apply the transform every epoch
# we avoid this by caching the data early on in MNISTCached class

def fn_x_imdb(sup_x, unsup_x, use_cuda):
    if use_cuda:
        sup_x = sup_x.cuda()
        unsup_x = unsup_x.cuda()

    return sup_x, unsup_x
def fn_y_imdb(y, use_cuda):
    y_tensor = torch.tensor(y, dtype=torch.long)
    yp = torch.zeros(y_tensor.size(0), 2)

    # transform the label y (integer between 0 and 9) to a one-hot
    yp = yp.scatter_(1, y_tensor.view(-1, 1), 1.0)
    if use_cuda:
        yp = yp.cuda()
    return yp


def get_ss_indices_per_class(y, sup_per_class):
    y_tensor = torch.tensor(y)

    # number of indices to consider
    n_idxs = y_tensor.size()[0]


    # calculate the indices per class
    idxs_per_class = {j: [] for j in range(10)}

    # for each index identify the class and add the index to the right class
    for i in range(n_idxs):
        curr_y = y_tensor[i]
        for j in range(10):
            if curr_y[j] == 1:
                idxs_per_class[j].append(i)
                break

    idxs_sup = []
    idxs_unsup = []
    for j in range(10):
        np.random.shuffle(idxs_per_class[j])
        idxs_sup.extend(idxs_per_class[j][:sup_per_class])
        idxs_unsup.extend(idxs_per_class[j][sup_per_class:len(idxs_per_class[j])])

    return idxs_sup, idxs_unsup


def split_sup_unsup_valid(unsup_x, unsup_y, sup_x, sup_y, validation_num=10000):
    """
    helper function for splitting the data into supervised, un-supervised and validation parts
    :param X: images
    :param y: labels (digits)
    :param sup_num: what number of examples is supervised
    :param validation_num: what number of last examples to use for validation
    :return: splits of data by sup_num number of supervised examples
    """

    # validation set is the last 10,000 examples
    X_valid = sup_x[-validation_num:]
    y_valid = sup_y[-validation_num:]

    X_sup = sup_x[0:-validation_num]
    y_sup = sup_y[0:-validation_num]

    return X_sup, y_sup, unsup_x, unsup_y, X_valid, y_valid


def print_distribution_labels(y):
    """
    helper function for printing the distribution of class labels in a dataset
    :param y: tensor of class labels given as one-hots
    :return: a dictionary of counts for each label from y
    """
    counts = {j: 0 for j in range(10)}
    for i in range(y.size()[0]):
        for j in range(10):
            if y[i][j] == 1:
                counts[j] += 1
                break
    print(counts)


class IMDBCached(TextDataSet):
    """
    a wrapper around MNIST to load and cache the transformed data
    once at the beginning of the inference
    """

    # static class variables for caching training data
    train_data_size = 50000
    train_data_sup, train_labels_sup = None, None
    train_data_unsup, train_labels_unsup = None, None
    validation_size = 10000
    data_valid, labels_valid = None, None
    test_size = 10000

    def __init__(self, mode, sup_num, use_cuda=True, *args, **kwargs):
        super(IMDBCached, self).__init__(train=mode in ["sup", "unsup", "valid"], *args, **kwargs)

        # transformations on MNIST data (normalization and one-hot conversion for labels)
        def transform_x(sup_x, unsup_x, use_cuda):
            return fn_x_imdb(sup_x, unsup_x, use_cuda)

        def target_transform(y, use_cuda):
            return fn_y_imdb(y, use_cuda)

        def separating_sup_unsup(x, y):
            y_array = np.asarray(y)
            x_array = np.asarray(x)
            unsup_indices = np.where(y_array == -1)
            sup_indices = np.where(y_array != -1)
            unsup_x = x_array[unsup_indices]
            unsup_y = y_array[unsup_indices]

            sup_x = x_array[sup_indices]
            sup_y = y_array[sup_indices]
            return unsup_x, unsup_y, sup_x, sup_y

        self.mode = mode
        self.unsup_x, self.unsup_y, self.sup_x, self.sup_y = separating_sup_unsup(self.train_data, self.train_labels)

        assert mode in ["sup", "unsup", "test", "valid"], "invalid train/test option values"

        if mode in ["sup", "unsup", "valid"]:

            # transform the training data if transformations are provided

            if target_transform is not None:
                self.sup_y = (target_transform(self.sup_y, use_cuda))

            if IMDBCached.train_data_sup is None:
                if sup_num is None:
                    assert mode == "unsup"
                    IMDBCached.train_data_unsup, IMDBCached.train_labels_unsup = \
                        self.train_data, self.train_labels
                else:
                    IMDBCached.train_data_sup, IMDBCached.train_labels_sup, \
                    IMDBCached.train_data_unsup, IMDBCached.train_labels_unsup, \
                    IMDBCached.data_valid, IMDBCached.labels_valid = \
                        split_sup_unsup_valid(self.unsup_x, self.unsup_y, self.sup_x, self.sup_y, sup_num)

            if mode == "sup":
                self.train_data, self.train_labels = IMDBCached.train_data_sup, IMDBCached.train_labels_sup
            elif mode == "unsup":
                self.train_data = IMDBCached.train_data_unsup

                # making sure that the unsupervised labels are not available to inference
                self.train_labels = (torch.Tensor(
                    IMDBCached.train_labels_unsup.shape[0]).view(-1, 1)) * np.nan
            else:
                self.train_data, self.train_labels = IMDBCached.data_valid, IMDBCached.labels_valid

        #else:
            # transform the testing data if transformations are provided
         #   if transform is not None:
          #      self.test_data = (transform(self.test_data.float()))
           # if target_transform is not None:
            #    self.test_labels = (target_transform(self.test_labels))

    def __getitem__(self, index):
        """
        :param index: Index or slice object
        :returns tuple: (image, target) where target is index of the target class.
        """
        if self.mode in ["sup", "unsup", "valid"]:
            img, target = self.train_data[index], self.train_labels[index]
        elif self.mode == "test":
            img, target = self.test_data[index], self.test_labels[index]
        else:
            assert False, "invalid mode: {}".format(self.mode)
        return img, target


def my_collate(batch):
    sequences = [torch.tensor(item[0]) for item in batch]
    lengths = torch.tensor([len(x) for x in sequences]).long()
    labels = [item[1] for item in batch]
    return [sequences, lengths, labels]


def setup_data_loaders(dataset, use_cuda, batch_size, sup_num=None, **kwargs):
    """
        helper function for setting up pytorch data loaders for a semi-supervised dataset
    :param dataset: the data to use
    :param use_cuda: use GPU(s) for training
    :param batch_size: size of a batch of data to output when iterating over the data loaders
    :param sup_num: number of supervised data examples
    :param root: where on the filesystem should the dataset be
    :param download: download the dataset (if it doesn't exist already)
    :param kwargs: other params for the pytorch data loader
    :return: three data loaders: (supervised data for training, un-supervised data for training,
                                  supervised data for testing)
    """
    # instantiate the dataset as training/testing sets
    if 'num_workers' not in kwargs:
        kwargs = {'num_workers': 0, 'pin_memory': False}

    cached_data = {}
    loaders = {}
    for mode in ["unsup", "test", "sup", "valid"]:
        if sup_num is None and mode == "sup":
            # in this special case, we do not want "sup" and "valid" data loaders
            return loaders["unsup"], loaders["test"]
        cached_data[mode] = dataset(mode=mode,
                                    sup_num=sup_num, use_cuda=use_cuda)
        loaders[mode] = DataLoader(cached_data[mode], batch_size=batch_size, shuffle=True, collate_fn=my_collate,  **kwargs)

    return loaders


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


EXAMPLE_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
DATA_DIR = os.path.join(EXAMPLE_DIR, 'data')
RESULTS_DIR = os.path.join(EXAMPLE_DIR, 'results')


if __name__ == '__main__':
    data_loaders = setup_data_loaders(IMDBCached, False, 200, sup_num=3000)
    sup_iter = iter(data_loaders["sup"])
    for i in range(10):
        (xs, ys) = next(sup_iter)

    print('hi')
