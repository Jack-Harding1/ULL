#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: jackharding, akashrajkn
"""
import numpy as np

import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

from utils import one_hot
from torch.autograd import Variable


class BayesianSkipGram(nn.Module):

    def __init__(self, vocabulary_size, embedding_dimension=20):
        super(BayesianSkipGram, self).__init__()

        self.embedding_dimension = embedding_dimension
        # Initialize epsilon here to speed it up
        self.epsilon = distributions.MultivariateNormal(torch.zeros(embedding_dimension), torch.eye(embedding_dimension))
        # inference network
        self.fc1 = nn.Linear(vocabulary_size, embedding_dimension, bias=True)  # word embedding. TODO: use a different dimension?
        self.fc2 = nn.Linear(embedding_dimension * 2, embedding_dimension * 2, bias=True)
        self.fc3 = nn.Linear(embedding_dimension * 2, embedding_dimension, bias=True)
        self.fc4 = nn.Linear(embedding_dimension * 2, embedding_dimension, bias=True)

        #to obtain categorical distribution from reparameterized sample
        self.re1 = nn.Linear(embedding_dimension, vocabulary_size, bias=True)

        # for BSG prior: we need to 'learn' these: X
        self.p_mean = nn.Linear(vocabulary_size, embedding_dimension, bias = True)
        self.p_sigma = nn.Linear(vocabulary_size, embedding_dimension, bias = True)

        self.fc1 = self.fc1.cuda()
        self.fc2 = self.fc2.cuda()
        self.fc3 = self.fc3.cuda()
        self.fc4 = self.fc4.cuda()
        self.re1 = self.re1.cuda()
        self.p_mean = self.p_mean.cuda()
        self.p_sigma = self.p_sigma.cuda()

    def forward(self, center_word, context_words):
        '''
        Forward pass
        @param center_word
        @param context_words: list of words
        '''
        center_word_1hot = one_hot(center_word)
        center_word_embedding = self.fc1(center_word_1hot)

        context_representation = torch.zeros(self.embedding_dimension * 2).cuda()

        for i, context_word in enumerate(context_words):
            context_word_embedding = self.fc1(one_hot(context_word))

            concatenated = torch.cat([context_word_embedding, center_word_embedding], dim=0)
            context_representation += F.relu(self.fc2(concatenated))

        mu = self.fc3(context_representation)
        sigma = F.softplus(self.fc4(context_representation))

        # Kingma-Welling reparameterization trick
        epsilon_noise = self.epsilon.sample().cuda()
        reparameterized_sample = mu + (epsilon_noise * sigma)
        categorical_distribution = F.softmax(self.re1(reparameterized_sample), dim=0)

        p_mean = self.p_mean(center_word_1hot)
        p_sigma = F.softplus(self.p_sigma(center_word_1hot))

        return categorical_distribution, mu, sigma, p_mean, p_sigma
