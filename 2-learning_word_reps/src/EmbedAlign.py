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


class EmbedAlign(nn.Module):

    def __init__(self, vocabulary_size_1, vocabulary_size_2, embedding_dimension=20, gpu=False):
        super(EmbedAlign, self).__init__()

        # Initialize epsilon here to speed it up
        self.epsilon = distributions.MultivariateNormal(torch.zeros(embedding_dimension), torch.eye(embedding_dimension))

        self.embedding_dimension = embedding_dimension
        self.vocabulary_size_1 = vocabulary_size_1
        self.vocabulary_size_2 = vocabulary_size_2

        self.embeddings = nn.Embedding(vocabulary_size_1, embedding_dimension)
        # Holy bram's hyperparameters
        self.BiLSTM = nn.LSTM(embedding_dimension, embedding_dimension, num_layers=3, dropout=0.33, bidirectional=True)

        # inference
        self.inference_mu_1 = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.inference_mu_2 = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.inference_sigma_1 = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.inference_sigma_2 = nn.Linear(self.embedding_dimension, self.embedding_dimension)

        self.english_affine_1 = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.english_affine_2 = nn.Linear(self.embedding_dimension, self.vocabulary_size_1)
        self.french_affine_1 = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.french_affine_2 = nn.Linear(self.embedding_dimension, self.vocabulary_size_2)

        # not sure about this.
        self.softmax_approximation_english = nn.Embedding(self.vocabulary_size_1, self.embedding_dimension)
        self.softmax_approximation_french = nn.Embedding(self.vocabulary_size_2, self.embedding_dimension)

        # TODO: I have to make my gpu do the hard work :P
        if gpu:
            pass

    def forward(self, en, fr, en_neg, fr_neg):
        '''
        Forward pass
        '''
        center_embedding = self.embeddings(en)
        center_embedding_neg = self.embeddings(en_neg)

        # en = en[:, None, :]  # add an empty y-dimension, because that's how LSTM takes its input
        # fr = fr[:, None, :]

        # calculate loss
        log_sum_x, kl  = self.loss_function(center_embedding, en)
        alignment, log_sum_y = self.generate_alignments(en, fr, fr_neg, zs)

        return -1 * (log_sum_x + log_sum_y - kl)

    def divergence_closed_form(mu_1, sigma_1, mu_2, sigma_2):
        '''
        Closed form of the KL divergence
        '''
        posterior = distributions.MultivariateNormal(mu_1, torch.diag(sigma_1 ** 2))
        prior = distributions.MultivariateNormal(mu_2, torch.diag(sigma_2 ** 2))
        kl_z = torch.distributions.kl.kl_divergence(posterior, prior)

        #kl_z = (-0.5 + torch.log(sigma_2) - torch.log(sigma_1) + (0.5 * (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (sigma_2 ** 2))).sum()

        return kl_z

    def approximation(self, data_point, data_point_negative, z, embed):
        '''
        Does the softmax approximation
        @param data_point
        @param data_point_negative
        @param z
        @param embed

        @return softmax_approximation
        '''
        top = torch.exp(torch.matmul(z, embed(data_point)))
        down = torch.exp(torch.matmul(z, embed(data_point_negative).t()))

        softmax_approximation = top / (top + down.sum())

        return softmax_approximation

    def loss_function(center_embedding, context):
        '''
        The loss function for embed align
        @param center_embedding
        @param context
        '''
        intermediate = (torch.randn(2, 1, self.embedding_dimension), torch.randn(2, 1, self.embedding_dimension))

        sum_log_x = torch.zeros(1)
        kl_total = torch.zeros(1)

        zs = torch.zeros(context.size(0), self.embedding_dimension)
        mus = torch.zeros(context.size(0), self.embedding_dimension)
        sigmas = torch.zeros(context.size(0), self.embedding_dimension)

        for i, (k, val) in enumerate(zip(center_embedding, context)):
            _, intermediate = self.BiLSTM(k.view(1, 1, -1), intermediate)
            # TODO: make this more efficient
            mu_1 = self.inference_mu_2(F.relu(self.inference_mu_1(intermediate[0][0] + intermediate[0][1])))
            sigma_1 = F.softplus(self.inference_sigma_2(F.relu(self.inference_sigma_1(intermediate[0][0] + intermediate[0][1]))))
            z = mu_1 + torch.mul(self.epsilon.sample(), sigma_1)

            mu_2 = torch.zeros(self.embedding_dimension)
            sigma_2 = torch.ones(self.embedding_dimension)

            # TODO: move this entire funciton outside
            kl_loss = self.divergence_closed_form(mu, sigma, mu_2, sigma_2)
            kl_total += kl_loss

            sum_log_x += self.approximation(val, en_neg, z, self.softmax_approximation_english)

            # Store all values
            mus[i] = mu_1
            sigmas[i] = sigma_1
            zs[i] = z

        return sum_log_x, kl_total

    def generate_alignments(self, en, fr, fr_neg, zs):
        '''
        Returns the alignments for english-french
        @param en
        @param
        '''
        size_en = en.size(0)
        size_fr = fr.size(0)

        alignment = torch.zeros(size_fr)
        sum_logs = 0

        for k, val in enumerate(fr):
            align_max_val = -1
            align_idx = -1

            # TODO: this is probably not efficient
            for i in range(size_en):

                temp = self.approximation(val, fr_neg, zs[a_m], self.softmax_approximation_french)

                if align_max_val < temp:
                    align_idx = i
                    align_max_val = temp
                else:
                    temp_val = (1 / size_en) * temp
                    sum_logs += temp_val

            alignment[k] = align_idx

        return alignment, sum_logs
