#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: jackharding, akashrajkn
"""

import torch
import torch.distributions as distributions
import torch.optim as optim

from BSG_Net import BSG_Net
from preprocess import *
from utils import *
from bsg_parameters import *


def divergence_closed_form(mu_1, sigma_1, mu_2, sigma_2):
    '''
    Closed form of the KL divergence
    '''
    posterior = distributions.MultivariateNormal(mu_1, torch.diag(sigma_1 ** 2))
    prior = distributions.MultivariateNormal(mu_2, torch.diag(sigma_2 **2))
    kl_z = torch.distributions.kl.kl_divergence(posterior, prior)

    #kl_z = (-0.5 + torch.log(sigma_2) - torch.log(sigma_1) + (0.5 * (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (sigma_2 ** 2))).sum()

    return kl_z

def elbo(categorical, mu_1, sigma_1, mu_2, sigma_2, words_pair):
    '''
    Loss function
    '''
    center_word = words_pair[0]
    context_words = words_pair[1]

    kl_term = divergence_closed_form(mu_1, sigma_1, mu_2, sigma_2) * len(context_words)

    likelihood_term = 0
    for context_word in context_words:
        context_word_idx = global_w2i[context_word]
        likelihood_term += categorical[context_word_idx]

    return (kl_term - likelihood_term) / len(context_words)


def train_model(model, data):
    # for BSG prior: we need to 'learn' these
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        # print('Epoch number: ', epoch)
        losses = []
        for x in data:
            optimizer.zero_grad()
            approximated, mu, sigma, p_mean, p_sigma = model(x[0], x[1])

            loss = elbo(approximated, mu, sigma, p_mean, p_sigma, x)

            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        print("AVERAGE LOSS IN REGION {} was {}".format(epoch, sum(losses)/len(losses)))


def save_model(model, vocab_size, interrupted=False):

    filename = '../models/bsg.e-{}.v-{}'.format(str(EPOCHS), str(vocab_size))

    if interrupted:
        filename += '-interrupted'
    filename += '.model'

    model_file = open(filename, 'wb+')
    pickle.dump(model, model_file)


def load_model(filepath):
    model = pickle.load(open(filepath, 'rb'))
    return model


if __name__ == '__main__':
    interrupt = False
    # load data:
    data = load_data_from_file('../data/processed/english-french_large/training.en', CONTEXT_SIZE)

    create_vocabulary('../data/processed/english-french_large/training.en')

    v_size = len(global_w2i.keys())
    model = BSG_Net(v_size, EMBEDDING_DIMENSION)

    try:
        train_model(model, data)
    except KeyboardInterrupt:
        save_model(model, v_size, interrupted=True)
        interrupt = True

    if not interrupt:
        save_model(model, v_size)
