#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: jackharding, akashrajkn
"""

import torch
import torch.optim as optim

from BSG_Net import BSG_Net
from preprocess import *
from utils import *
from bsg_parameters import *


def divergence_closed_form(mu_1, sigma_1, mu_2, sigma_2):
    '''
    Closed form of the KL divergence
    '''
    return -0.5 + torch.log(sigma_2 / sigma_1) + (0.5 * (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (sigma_2 ** 2))

def elbo(approximated, mu_1, sigma_1, mu_2, sigma_2, words_pair):
    '''
    Loss function
    '''
    center_word = words_pair[0]
    context_words = words_pair[1]

    kl_term = divergence_closed_form(mu_1, sigma_1, mu_2, sigma_2)

    other_term = 0
    for i, context_word in enumerate(context_words):
        other_term += torch.log(approximated[i])

    return (other_term - kl_term)


def train_model(model, data):
    # for BSG prior: we need to 'learn' these
    # optimizer = optim.Adam([model.fc1.parameters(), model.fc2.parameters(), model.fc3.parameters(),
    #                         model.fc4.parameters(), model.re1.parameters(), model.p_mean, model.p_sigma],
    #                        lr=LEARNING_RATE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []

    for epoch in range(EPOCHS):
        print('Epoch number: ', epoch)

        for x in data:
            optimizer.zero_grad()
            approximated, mu, sigma, p_mean, p_sigma = model(x[0], x[1])

            loss = elbo(approximated, mu, sigma, p_mean, p_sigma, x)

            loss.backward()
            print(loss)
            optimizer.step()


if __name__ == '__main__':

    # load data:
    data = load_data_from_file('../data/processed/english-french_small/dev.en', CONTEXT_SIZE)
    # create vocabulary
    create_vocabulary('../data/processed/english-french_small/dev.en')
    # Initialize model
    model = BSG_Net(VOCABULARY_SIZE, EMBEDDING_DIMENSION)

    # print(model.parameters())

    train_model(model, data)
