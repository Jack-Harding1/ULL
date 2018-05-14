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


class BSG_Net(nn.Module):

    def __init__(self, vocabulary_size, embedding_dimension=20):
        super(BSG_Net, self).__init__()

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
        # self.p_mean = nn.Parameter(torch.empty(embedding_dimension, vocabulary_size).uniform_(-1, 1), requires_grad=True)
        # self.p_sigma = nn.Parameter(torch.empty(embedding_dimension, vocabulary_size).uniform_(-1, 1), requires_grad=True)
        self.p_mean = nn.Linear(vocabulary_size, embedding_dimension, bias = True)
        self.p_sigma = nn.Linear(vocabulary_size, embedding_dimension, bias = True)

    def forward(self, center_word, context_words):
        '''
        TODO: Each data point 'x' is { center_word: [context_words] }
              Currently we are repeating center_word for each of the context_words
        '''
        center_word_1hot = one_hot(center_word)
        context_representation = Variable(torch.zeros(self.embedding_dimension * 2))
        center_word_embedding = Variable(self.fc1(center_word_1hot))

        for i, context_word in enumerate(context_words):
            context_word_embedding = self.fc1(one_hot(context_word))

            concatenated = torch.cat([context_word_embedding, center_word_embedding], dim=0)
            context_representation += F.relu(self.fc2(concatenated))
            
        mu = self.fc3(context_representation)
        sigma = F.softplus(self.fc4(context_representation))
        
        # Kingma-Welling trick
        epsilon_noise = self.epsilon.sample()
        reparameterized_sample = mu + (epsilon_noise * sigma)
        categorical_distribution = F.softmax(self.re1(reparameterized_sample), dim=0)

        # print("center_word_embedding: ", center_word_embedding.shape)
        # print("p_mean: ", self.p_mean.shape)

        # p_mean = torch.matmul(self.p_mean, one_hot(center_word))
        # p_sigma = F.softplus(torch.matmul(self.p_sigma, one_hot(center_word)))
        # p_sigma = p_sigma ** 2

        # sadly there is no direct conversion from float to long tensor :/
        #p_mean = self.p_mean(torch.LongTensor(np.array(center_word_1hot)))
        #p_sigma = F.softplus(self.p_sigma(torch.LongTensor(np.array(center_word_1hot))))
        
        p_mean = self.p_mean(center_word_1hot)
        p_sigma = F.softplus(self.p_sigma(center_word_1hot))
        

        return categorical_distribution, mu, sigma, p_mean, p_sigma
