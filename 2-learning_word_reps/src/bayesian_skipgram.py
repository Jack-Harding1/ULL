import datetime
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class BSG_Net(nn.Module):

    def __init__(self, vocabulary_size, embedding_dimension=20):
        super(BSG_Net, self).__init__()

        self.embedding_dimension = embedding_dimension
        # Initialize epsilon here to speed it up
        self.epsilon = distb.MultivariateNormal(torch.zeros(embedding_dimension), torch.eye(embedding_dimension))

        # inference network
        self.fc1 = nn.Linear(vocabulary_size, embedding_dimension, bias=True)  # word embedding. TODO: use a different dimension?
        self.fc2 = nn.Linear(embedding_dimension * 2, embedding_dimension * 2, bias=True)
        self.fc3 = nn.Linear(embedding_dimension * 2, embedding_dimension, bias=True)
        self.fc4 = nn.Linear(embedding_dimension * 2, embedding_dimension, bias=True)
        # for reparameterization
        self.re1 = nn.Linear(embedding_dimension, vocabulary_size, bias=True)

        # for BSG prior: we need to 'learn' these
        self.p_mean = nn.Parameter(torch.empty(embedding_dimension, vocabulary_size).uniform_(-1, 1), requires_grad=True)
        self.p_sigma = nn.Parameter(torch.empty(embedding_dimension, vocabulary_size).uniform_(-1, 1))


    def forward(self, center_word, context_words):
        '''
        TODO: Each data point 'x' is { center_word: [context_words] }
              Currently we are repeating center_word for each of the context_words
        '''
        context_representation = torch.zeros(self.embedding_dimension * 2)
        center_word_embedding = self.fc1(one_hot(center_word))

        for i, context_word in enumerate(context_words):
            context_word_embedding = self.fc1(one_hot(context_word))

            concatenated = torch.cat([center_word, context_word], dim=0)
            context_representation += F.relu(self.fc2(concatenated))

        ## Deprecated
        # for pair in x:
        #     center_word = self.fc1(onehot(pair[0]))
        #     context_word = self.fc1(onehot(pair[1]))
        #
        #     concatenated = torch.cat([center_word, context_word], dim=0)
        #     concatenated = F.relu(self.fc2(concatenated))
        #     context_representation += concatenated

        mu = self.fc3(context_representation)
        sigma = F.softplus(self.fc4(context_representation))
        sigma = sigma ** 2
        # Kingma-Welling trick
        z = mu + self.epsilon.sample() * sigma

        approximated = F.softmax(self.re1(z), dim=0)

        p_mean = torch.matmul(self.p_mean, center_word)
        p_sigma = F.softplus(torch.matmul(self.p_sigma, center_word))
        p_sigma = prior_sigma ** 2

        return approximated, mu, sigma, p_mean, p_sigma
