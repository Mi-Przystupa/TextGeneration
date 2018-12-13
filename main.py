import argparse

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.contrib.examples.util import print_and_log
import pyro.poutine as poutine
from pyro.infer import SVI, JitTrace_ELBO, JitTraceEnum_ELBO, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
#from mnist_cached import MNISTCached, mkdir_p, setup_data_loaders
from text_ss_vae import TextSSVAE
from IMDB_cached_New import setup_data_loaders, IMDBCached
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

nltk.download('punkt')
def run_inference_for_epoch(data_loaders, losses, periodic_interval_batches):
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """
    num_losses = len(losses)

    # compute number of batches for an epoch
    sup_batches = len(data_loaders["sup"])
    unsup_batches = len(data_loaders["unsup"])
    batches_per_epoch = sup_batches + unsup_batches

    # initialize variables to store loss values
    epoch_losses_sup = [0.] * num_losses
    epoch_losses_unsup = [0.] * num_losses

    # setup the iterators for training data loaders
    sup_iter = iter(data_loaders["sup"])
    unsup_iter = iter(data_loaders["unsup"])

    # count the number of supervised batches seen in this epoch
    ctr_sup = 0
    for i in range(batches_per_epoch):

        # whether this batch is supervised or not
        is_supervised = (i % periodic_interval_batches == 1) and ctr_sup < sup_batches

        # extract the corresponding batch
        if is_supervised:
            try:
                (xs, lengths, ys) = next(sup_iter)
                ctr_sup += 1
            except StopIteration:
                continue
        else:
            try:
                (xs, lengths, ys) = next(unsup_iter)
            except StopIteration:
                break

        # run the inference for each loss with supervised or un-supervised
        # data as arguments
        for loss_id in range(num_losses):
            if is_supervised:
                new_loss = losses[loss_id].step(xs, lengths, ys)
                epoch_losses_sup[loss_id] += new_loss
            else:
                new_loss = losses[loss_id].step(xs, lengths)
                epoch_losses_unsup[loss_id] += new_loss

    # return the values of all losses
    return epoch_losses_sup, epoch_losses_unsup


def get_accuracy(data_loader, classifier_fn, batch_size):
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions, actuals = [], []

    # use the appropriate data loader
    for (xs,lengths, ys) in data_loader:
        # use classification function to compute all predictions for each batch
        predictions.append(classifier_fn(xs,lengths))
        actuals.append(ys)

    # compute the number of accurate predictions
    accurate_preds = 0
    for pred, act in zip(predictions, actuals):
        for i in range(pred.size(0)):
            v = torch.sum(pred[i] == act[i])
            accurate_preds += (v.item() == 2)

    # calculate the accuracy between 0 and 1
    accuracy = (accurate_preds * 1.0) / (len(predictions) * batch_size)
    return accuracy

def create_sentences(indicies, word_model,word_limit=100):
    sentences = []
    for entry in indicies:
        sentence = ""
        for i, e in enumerate(entry):
            if i > word_limit:
                break
            sentence = sentence + " " + word_model.index2word[e]
        sentences = sentences + [sentence]
    return sentences


def generateSentences(data_loader, generator_fn, word_model, sentiment=0):
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions, actuals = [], []
    total_bleu = 0.
    # use the appropriate data loader
    for (xs,lengths, ys) in data_loader:

        conditioned_feature = [torch.zeros(2) for i in lengths]
        for i, e in enumerate(conditioned_feature):
            conditioned_feature[i][sentiment] = 1
        # use classification function to compute all predictions for each batch
        generated = generator_fn(xs, lengths,conditioned_feature)
        generated = pyro.sample("generated_sentence", dist.Categorical(logits=generated).independent(1))
        predictions.append(generated)
        actuals.append(xs)


    # compute the number of accurate predictions
    sentences = []
    word_limit = 50
    for pred, act in zip(predictions, actuals):
        pred_sentence = create_sentences(pred, word_model, word_limit=word_limit)
        act_sentence = create_sentences(act, word_model, word_limit=word_limit)
        blue_score = compute_bleu(act_sentence, pred_sentence)
        total_bleu += blue_score
        sentences = {'pred': pred_sentence, 'act': act_sentence, 'bleu': blue_score}
        break

    return sentences, total_bleu
def compute_bleu(reference_sentence, predicted_sentence):
    """
    Given a reference sentence, and a predicted sentence, compute the BLEU similary between them.
    """
    bleu = 0
    for act, pred in zip(reference_sentence, predicted_sentence):
        reference_tokenized = word_tokenize(act)
        predicted_tokenized = word_tokenize(pred)
        bleu += sentence_bleu([reference_tokenized], predicted_tokenized)
    return bleu

def main():
    """
    run inference for SS-VAE
    :param args: arguments for SS-VAE
    :return: None
    """
    pyro.set_rng_seed(12345)
    cuda = True
    # batch_size: number of images (and labels) to be considered in a batch
    ss_vae = TextSSVAE(
            embed_dim=300,
            z_dim=300, kernels=[3,4,5],
            filters=[100,100,100], hidden_size = 300,
            num_rnn_layers = 1,
            config_enum="parallel", use_cuda=cuda,
            aux_loss_multiplier=46
            )

    ss_vae = ss_vae.cuda()

    try:
        pyro.get_param_store().load('pyro_param_store.store')
        print('successfully loaded param store, remove file from directory if undesired')
    except Exception:
        print("failed to load param store, starting over")

    try:
        ss_vae.load_state_dict(torch.load('ss_vae_model.pth'))
        print('successfully loaded model parameters, remove file from directory if undesired')
    except Exception:
        print("failed to load model parameters")

    # setup the optimizer
    adam_params = {"lr": 1e-4,"betas": (0.9, 0.999), "weight_decay": 0.01}
    optimizer = Adam(adam_params)

    # set up the loss(es) for inference. wrapping the guide in config_enumerate builds the loss as a sum
    # by enumerating each class label for the sampled discrete categorical distribution in the model
    jit = False
    guide = config_enumerate(ss_vae.guide, "parallel", expand=True)
    elbo = (JitTraceEnum_ELBO if jit else TraceEnum_ELBO)()

    loss_basic = SVI(ss_vae.model, guide, optimizer, loss=elbo)

    # build a list of all losses considered
    losses = [loss_basic]

    # aux_loss: whether to use the auxiliary loss from NIPS 14 paper (Kingma et al)
    aux_loss = True
    if aux_loss:
        elbo = JitTrace_ELBO() if jit else Trace_ELBO()
        loss_aux = SVI(ss_vae.model_classify, ss_vae.guide_classify, optimizer, loss=elbo)
        losses.append(loss_aux)
    
    batch_size = 32
    valid_num = 100
    train_data_size = 3409
    sup_num = 1163
    try:
        # setup the logger if a filename is provided
        logger = open('./tmp.log', "w") if './tmp.log' else None
        data_loaders = setup_data_loaders(IMDBCached, cuda, batch_size=32, sup_num=valid_num)

        # how often would a supervised batch be encountered during inference
        # e.g. if sup_num is 3000, we would have every 16th = int(50000/3000) batch supervised
        # until we have traversed through the all supervised batches
        periodic_interval_batches = int(train_data_size / (1.0 * sup_num))

        # number of unsupervised examples
        unsup_num =  train_data_size - sup_num

        # initializing local variables to maintain the best validation accuracy
        # seen across epochs over the supervised training set
        # and the corresponding testing set and the state of the networks
        best_valid_acc, corresponding_test_acc = 0.0, 0.0

        # run inference for a certain number of epochs
        num_epochs = 200
        sup_loss_log = []
        unsup_loss_log = []


        for i in range(0, num_epochs):
            # get the losses for an epoch
            epoch_losses_sup, epoch_losses_unsup = \
                run_inference_for_epoch(data_loaders, losses, periodic_interval_batches)

            # compute average epoch losses i.e. losses per example
            avg_epoch_losses_sup = map(lambda v: v / sup_num, epoch_losses_sup)
            avg_epoch_losses_unsup = map(lambda v: v / unsup_num, epoch_losses_unsup)
            
            sup_loss_log.append(avg_epoch_losses_sup)
            unsup_loss_log.append(avg_epoch_losses_unsup)
            
            # store the loss and validation/testing accuracies in the logfile
            str_loss_sup = " ".join(map(str, avg_epoch_losses_sup))
            str_loss_unsup = " ".join(map(str, avg_epoch_losses_unsup))

            str_print = "{} epoch: avg losses {}".format(i, "{} {}".format(str_loss_sup, str_loss_unsup))
            ss_vae.eval()
            validation_accuracy = get_accuracy(data_loaders["valid"], ss_vae.classifier, batch_size)
            str_print += " validation accuracy {}".format(validation_accuracy)

            # this test accuracy is only for logging, this is not used
            # to make any decisions during training
            test_accuracy = get_accuracy(data_loaders["test"], ss_vae.classifier, batch_size)
            str_print += " test accuracy {}".format(test_accuracy)
            ss_vae.train()
            torch.save(ss_vae.state_dict(), 'ss_vae_model.pth')
            pyro.get_param_store().save('pyro_param_store.store')


            # update the best validation accuracy and the corresponding
            # testing accuracy and the state of the parent module (including the networks)
            if best_valid_acc < validation_accuracy:
                best_valid_acc = validation_accuracy
                corresponding_test_acc = test_accuracy
            if i % 10 == 0:
                neg_sentences, neg_bleu = generateSentences(data_loaders["test"], ss_vae.model, ss_vae.w2v_model, sentiment=0)
                pos_sentences, pos_bleu = generateSentences(data_loaders["test"], ss_vae.model, ss_vae.w2v_model, sentiment=1)
                str_print += " neg_bleu {}".format(neg_bleu)
                str_print += " pos_bleu {}".format(pos_bleu)
                pd.DataFrame.from_dict(pos_sentences).to_csv('positive_sentences.csv', encoding='utf-8')
                pd.DataFrame.from_dict(neg_sentences).to_csv('negative_sentences.csv', encoding='utf-8')

                cond_neg_sentences, neg_bleu = generateSentences(data_loaders["test"], ss_vae.conditioned_generation, ss_vae.w2v_model, sentiment=0)
                cond_pos_sentences, pos_bleu = generateSentences(data_loaders["test"], ss_vae.conditioned_generation, ss_vae.w2v_model, sentiment=1)
                pd.DataFrame.from_dict(cond_pos_sentences).to_csv('cond_positive_sentences.csv', encoding='utf-8')
                pd.DataFrame.from_dict(cond_neg_sentences).to_csv('cond_negative_sentences.csv', encoding='utf-8')
                str_print += "cond_neg_bleu {}".format(neg_bleu)
                str_print += "cond_pos_bleu {}".format(pos_bleu)


            print_and_log(logger, str_print)
        
        np.save("avg_loss_sup", np.asarray(sup_loss_log))
        np.save("avg_loss_unsup", np.asarray(unsup_loss_log))
        ss_vae.eval()
        final_test_accuracy = get_accuracy(data_loaders["test"], ss_vae.classifier, batch_size)
        print_and_log(logger, "best validation accuracy {} corresponding testing accuracy {} "
                      "last testing accuracy {}".format(best_valid_acc, corresponding_test_acc, final_test_accuracy))

    finally:
        # close the logger file object if we opened it earlier
        logfile = True
        if logfile:
            logger.close()

if __name__ == "__main__":
    main()
    print("hello world")
