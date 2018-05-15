#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 21:08:52 2018

@author: jackharding
"""

import torch
from torch.nn.functional import logsigmoid
from torch.nn import Embedding, Linear, CrossEntropyLoss, Softmax, Softplus
from torch.nn.modules.module import Module
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
import torch.distributions as distributions

class Embed_Align(Module):

    def __init__(self, vocab_size_l1, vocab_size_l2, embedding_dimension, context):

        super(Embed_Align, self).__init__()

        self.context = context

        self.vocab_size_l1 = vocab_size_l1
        self.vocab_size_l2 = vocab_size_l2
        self.embed_dim = embedding_dimension

        if self.context:
            self.h_dim = 2 * embedding_dimension # 'd' in comments
        else:
            self.h_dim = embedding_dimension

        #generative model
        self.embeddings = Embedding(vocab_size_l1, embedding_dimension)
        self.affine_l1 = Linear(embedding_dimension, vocab_size_l1)
        self.affine_l2 = Linear(embedding_dimension, vocab_size_l2)
        
        self.softmax = Softmax(dim=2)
        self.cross_entropy = CrossEntropyLoss()
        
        #inference model
        self.fc1 = Linear(self.h_dim, embedding_dimension)
        self.fc2 = Linear(self.h_dim, embedding_dimension)
        self.softplus = Softplus()
        self.reparametrization = distributions.MultivariateNormal(torch.zeros(embedding_dimension), torch.eye(embedding_dimension))


    def forward(self, batch_l1, batch_l2):

        batch_size = batch_l1.size()[0]  # 'b' in comments
        m = batch_l1.size()[1]  # longest English sentence in batch
        n = batch_l2.size()[1]  # longest French sentence in batch

        embedded_l1 = self.embeddings(batch_l1)  # [b,m,d]

        l1_mask = torch.sign(batch_l1).float()
        l2_mask = torch.sign(batch_l2).float()

        l1_sent_lengths = torch.sum(l1_mask, dim=1)
        l1_sent_lengths = torch.unsqueeze(l1_sent_lengths, dim=1)
        # l1_sent_lengths = l1_sent_lengths.repeat(1, l1_mask.size()[1])


        align_probs = l1_mask / l1_sent_lengths.float()
        align_probs = torch.unsqueeze(align_probs, dim=1)
        align_probs = align_probs.repeat(1, n, 1)

        if self.context:
            sums = torch.sum(embedded_l1, dim=1)
            sums = sums.unsqueeze(1).repeat(1, m, 1)
            context = sums - embedded_l1
            context /= m - 1
            h = torch.cat((embedded_l1, context), dim=2)  # [b,m,2d]
        else:
            h = embedded_l1  # [b,m,d]

        #inference forward pass
        z_mean = self.fc1(h)
        z_var= self.softplus(self.fc2(h))

        epsilon = self.reparametrization.sample()

        z = z_mean + (epsilon * z_var)  # make sure z is [b,m,d]

        logits_l1 = self.affine_l1(z)  # [b,m,V_l1]
        cat_l1 = self.softmax(logits_l1)

        logits_l2 = self.affine_l2(z)
        cat_l2 = self.softmax(logits_l2)


        # print(cat_l2)
        p_l2_zx = torch.bmm(align_probs, cat_l2)  # [b,n,V_l2]

        cross_entropy_l1 = self.cross_entropy(logits_l1.permute([0,2,1]), batch_l1)  # [b,m]
        cross_entropy_l1 = torch.sum(cross_entropy_l1 * l1_mask, dim=1)  # [b]
        cross_entropy_l1 = torch.mean(cross_entropy_l1, dim=0)  # []

        cross_entropy_l2 = self.cross_entropy(p_l2_zx.permute([0,2,1]), batch_l2)
        cross_entropy_l2 = torch.sum(cross_entropy_l2 * l2_mask, dim=1)  # [b]
        cross_entropy_l2 = torch.mean(cross_entropy_l2, dim=0)  # []


        z_var_squared = z_var*z_var
        # KL(q(Z|x) || N(0, I))
        kl_z = -0.5 * (1 + torch.log(z_var_squared) - z_mean*z_mean - z_var_squared)  # [b,m,d]
        kl_z = torch.sum(kl_z, dim=2)  # [b, m]
        kl_z = torch.sum(kl_z * l1_mask, dim=1)  # [b]
        kl_z = torch.mean(kl_z, dim=0)  # []

        loss = cross_entropy_l1 + cross_entropy_l2 + kl_z

        return loss