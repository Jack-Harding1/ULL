#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:56:26 2018

@author: jackharding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from utils import *
from Skipgram_Parameters import *

class Skipgram_Net(nn.Module):

    def __init__(self, vocabulary_size, embedding_dimension=20):
        super(Skipgram_Net, self).__init__()
        
        self.center_embedding = nn.Linear(vocabulary_size, embedding_dimension, bias=True)
        self.output = nn.Linear(embedding_dimension, vocabulary_size, bias=True)

        self.vocabulary_size = vocabulary_size
        
    def forward(self, center_word):
        #takes in a batch of center words idx and context words idx, returns the nll-loss
        
        input_to_network = torch.zeros(BATCH_SIZE, self.vocabulary_size)
        for idx in range(BATCH_SIZE):
            input_to_network[idx, center_word[idx]] = 1.0

        center_word_embedding = self.center_embedding(input_to_network)
        output = self.output(center_word_embedding)
        prediction = F.log_softmax(output, dim=1)
        return prediction
        
        