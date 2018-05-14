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
    
#    print(mu_1)
#    print(mu_2)
#    
#    print(sigma_1)
#    print(sigma_2)
    # print(torch.log(sigma_2) - torch.log(sigma_1))
    # print((mu_1 - mu_2) ** 2)
    # print((sigma_2 ** 2))
    # print(sigma_1 ** 2)

    
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

    # print('mu_1: ', mu_1.shape)
    # print('sigma_1: ', sigma_1.shape)
    # print('mu_2: ', mu_2.shape)
    # print('sigma_2', sigma_2.shape)


    kl_term = divergence_closed_form(mu_1, sigma_1, mu_2, sigma_2) * len(context_words)
    
    likelihood_term = 0

    for context_word in context_words:
        context_word_idx = global_w2i[context_word]
        likelihood_term += torch.log(categorical[context_word_idx])

    return kl_term - likelihood_term


def train_model(model, data):
    # for BSG prior: we need to 'learn' these
    # optimizer = optim.Adam([model.fc1.parameters(), model.fc2.parameters(), model.fc3.parameters(),
    #                         model.fc4.parameters(), model.re1.parameters(), model.p_mean, model.p_sigma],
    #                        lr=LEARNING_RATE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        i = 0
        print('Epoch number: ', epoch)
        losses = []
        for x in data:
            optimizer.zero_grad()
            approximated, mu, sigma, p_mean, p_sigma = model(x[0], x[1])

            loss = elbo(approximated, mu, sigma, p_mean, p_sigma, x)
#            print("-------------")
#            print(loss.item())
#            print("-------------")
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        print("AVERAGE LOSS IN REGION {} was {}".format(i, sum(losses)/len(losses)))

if __name__ == '__main__':

    # load data:
    data = load_data_from_file('../data/processed/english-french_small/dev.en', CONTEXT_SIZE)
    #create vocabulary
    create_vocabulary('../data/processed/english-french_small/dev.en')
    
    # load data:
#    data = load_data_from_file('../data/processed/english-french_large/training.en', CONTEXT_SIZE)
#    # create vocabulary
#    create_vocabulary('../data/processed/english-french_large/training.en')
#    # Initialize model
    model = BSG_Net(VOCABULARY_SIZE, EMBEDDING_DIMENSION)

    #print(model.parameters().data)

    train_model(model, data)
