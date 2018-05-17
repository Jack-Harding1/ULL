#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: jackharding, akashrajkn
"""
import pickle
import time

import torch
import torch.distributions as distributions
import torch.optim as optim

from BayesianSkipGram import *
from preprocess import *
from utils import *
from BayesianSkipGram_parameters import *


def divergence_closed_form(mu_1, sigma_1, mu_2, sigma_2):
    '''
    Computes the KL divergence between two Gaussians
    @param mu_1
    @param sigma_1
    @param mu_2
    @param sigma_2

    @return kl_z: KL divergence
    '''
    posterior = distributions.MultivariateNormal(mu_1, torch.diag(sigma_1 ** 2))
    prior = distributions.MultivariateNormal(mu_2, torch.diag(sigma_2 ** 2))
    kl_z = torch.distributions.kl.kl_divergence(posterior, prior)

    # kl_z = (-0.5 + torch.log(sigma_2) - torch.log(sigma_1) + (0.5 * (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (sigma_2 ** 2))).sum()
    return kl_z

def elbo(categorical, mu_1, sigma_1, mu_2, sigma_2, words_pair):
    '''
    Loss function

    @param categorical: predicted
    @param mu_1
    @param sigma_1
    @param mu_2
    @param sigma_2
    @param words_pair: input (center_word, context_words) pair for forward pass

    @return negative elbo
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

    print('- Start training')
    previous_loss = 0
    for epoch in range(EPOCHS):
        current_loss = 0
        start_time = time.time()
        for x in data:
            optimizer.zero_grad()
            approximated, mu, sigma, p_mean, p_sigma = model(x[0], x[1])
            loss = elbo(approximated, mu, sigma, p_mean, p_sigma, x)
            loss.backward()
            current_loss += loss.item()
            optimizer.step()

        elapsed_time = time.time() - start_time
        print("   - AVERAGE LOSS IN EPOCH {} was {}".format(epoch, current_loss/len(data)))
        print("     Time taken: {}".format(str(elapsed_time)))

        if abs(previous_loss - current_loss) < THRESHOLD:
            break
        previous_loss = current_loss

def save_model(model, vocab_size, interrupted=False):
    '''
    Save trained model. Interrupted models are appended with "-interrupted" key
    @param model: model to be saved
    @param vocab_size
    @param interrupted: If interrupted from Keyboard (Crtl + C)
    '''
    filename = '../models/bsg-{}_ed-{}'.format(str(DOWNSAMPLE_DATA), str(EMBEDDING_DIMENSION))

    if interrupted:
        filename += '-interrupted'
    filename += '.model'

    model_file = open(filename, 'wb+')
    pickle.dump(model, model_file)


def load_model(filepath):
    model = pickle.load(open(filepath, 'rb'))
    return model


if __name__ == '__main__':

    print('MODEL, training-set: {}, embedding_dimension: {}'.format(str(DOWNSAMPLE_DATA), str(EMBEDDING_DIMENSION)))

    interrupt = False
    # load data:
    filepath = '../data/processed/english-french_large/training-{}.en'.format(DOWNSAMPLE_DATA)
    data = load_data_from_file(filepath, CONTEXT_SIZE)

    create_vocabulary(filepath)
    print('- Created Vocabulary')
    v_size = len(global_w2i.keys())
    # Initialize model
    model = BayesianSkipGram(v_size, EMBEDDING_DIMENSION)

    print('- Initalized model')

    try:
        train_model(model, data)
        print('- Model Trained')
    except KeyboardInterrupt:
        print('- Process interrupted')
        save_model(model, v_size, interrupted=True)
        interrupt = True

    if not interrupt:
        save_model(model, v_size)
    print('- Model saved')
