#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 21:08:52 2018

@author: jackharding
"""

import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F

class EmbedAlign(nn.Module):

    def __init__(self, vocabulary_size_l1, vocabulary_size_l2, embedding_dimension):
        super(EmbedAlign, self).__init__()

        # generative network
        self.embeddings = nn.Embedding(vocabulary_size_l1, embedding_dimension)
        self.fc_l1 = nn.Linear(embedding_dimension, vocabulary_size_l1)
        self.fc_l2 = nn.Linear(embedding_dimension, vocabulary_size_l2)

        # inference network
        self.fc1 = nn.Linear(embedding_dimension * 2, embedding_dimension)
        self.fc2 = nn.Linear(embedding_dimension * 2, embedding_dimension)
        self.reparametrization = distributions.MultivariateNormal(torch.zeros(embedding_dimension), torch.eye(embedding_dimension))

        # loss function
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, l1_sentence, l2_sentence):
        '''
        Forward pass
        @param l1_sentence: (here english sentence)
        @param l2_sentence: (here french sentence)
        '''
        # deterministic embedding for english words
        embedded_l1 = self.embeddings(l1_sentence)

        # taking into account padding
        # credit to Mario Giulianelli and Florian Mohnert for this way of doing padding
        # (also for their help on this part of the model in general)
        words_in_l1 = torch.sign(l1_sentence).float()
        words_in_l2 = torch.sign(l2_sentence).float()
        l1_sent_lengths = torch.sum(words_in_l1, dim=1)
        l1_sent_lengths = torch.unsqueeze(l1_sent_lengths, dim=1)

        # finding a context by summing over the context words in the sentence
        # this is essentially the alternative suggested in the blog post
        sums = torch.sum(embedded_l1, dim=1)
        sums = sums.unsqueeze(1).repeat(1, l1_sentence.size()[1], 1)
        context = (sums - embedded_l1) / (l2_sentence.size()[1] - 1)
        h = torch.cat((embedded_l1, context), dim=2)

        # inference forward pass
        location = self.fc1(h)
        scale = F.softplus(self.fc2(h))
        # Kingma-Welling reparametrization trick
        epsilon = self.reparametrization.sample()
        z = location + (epsilon * scale)

        logits_l1 = self.fc_l1(z)
        cat_l2 = F.softmax(self.fc_l2(z), dim=2)

        # creating a uniform distribution over possible alignments for the french words
        uniform_probs = torch.unsqueeze(words_in_l1 / l1_sent_lengths.float(), dim=1)
        uniform_probs = uniform_probs.repeat(1, l2_sentence.size()[1], 1)
        # taking into account the uniform alignments for the MC estimate
        adjusted_probabilities = torch.bmm(uniform_probs, cat_l2)

        # calculating the likelihood term across the whole batch
        likelihood_l1 = torch.sum(self.cross_entropy(logits_l1.permute([0,2,1]), l1_sentence) * words_in_l1)
        likelihood_l2 = torch.sum(self.cross_entropy(adjusted_probabilities.permute([0,2,1]), l2_sentence) * words_in_l2)

        # Calculating the KL divergence "KL(q(Z|x) || N(0, I))" using the closed form expression
        kl = -0.5 * (1 + torch.log(scale**2) - (location**2) - (scale**2))
        kl = torch.sum(kl, dim=2)
        kl = torch.sum(kl * words_in_l1, dim=1)
        kl = torch.sum(kl, dim=0)

        # returning the loss, the negative ELBO
        loss = likelihood_l1 + likelihood_l2 + kl
        return loss
