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

    def __init__(self, vocabulary_size, embedding_dimension=20, gpu=False):
        super(BayesianSkipGram, self).__init__()

        self.embedding_dimension = embedding_dimension
        # Initialize epsilon here to speed it up
        self.epsilon = distributions.MultivariateNormal(torch.zeros(embedding_dimension), torch.eye(embedding_dimension))

        # inference network
        # TODO: use word2vec to initialize? :P
        self.embeddings = nn.Embedding(vocabulary_size, embedding_dimension)
        self.affine = nn.Linear(embedding_dimension * 2, embedding_dimension * 2)
        self.mu = nn.Linear(embedding_dimension * 2, embedding_dimension)
        self.sigma = nn.Linear(embedding_dimension * 2, embedding_dimension)

        # reparameterization
        self.reparameterization = nn.Linear(embedding_dimension, vocabulary_size)

        # generative network
        self.mu_generative = nn.Embedding(vocabulary_size, embedding_dimension)
        self.sigma_generative = nn.Embedding(vocabulary_size, embedding_dimension)

        if gpu:
            pass

    def forward(self, center_word, context_words):
        '''
        Forward pass
        @param center_word: one hot encoding of center word - tensor
        @param context_word: list of one hot encoding of context words - tensor
        '''

        num_context_words = context_words.size(1)
        center_word_stack = center_word.repeat(num_context_words, 1)

        center_embeds  = self.embeddings(center_word_stack.t())
        context_embeds = self.embeddings(context_words.t())
        concatenated_embeds = torch.cat([center_embeds, context_embeds], dim=-1)
        after_activation = F.relu(self.affine(concatenated_embeds)).sum(dim=1)

        mean = self.mu(after_activation)
        std  = F.softplus(self.sigma(after_activation))

        # Reparameterization trick
        z = mean + torch.mul(self.epsilon.sample(), std)
        categorical_distribution = F.softmax(self.reparameterization(z), dim=0)

        mu_gen  = self.mu_generative(center_word)
        std_gen = F.softplus(self.sigma_generative(center_word))

        return categorical_distribution, mean, std, mu_gen, std_gen
