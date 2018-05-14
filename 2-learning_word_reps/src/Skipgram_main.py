#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:55:44 2018

@author: jackharding
"""

import torch
import torch.distributions as distributions
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from Skipgram_Net import Skipgram_Net
from preprocess import *
from utils import *
from Skipgram_Parameters import *

def train_model(model, data):

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.NLLLoss(reduce=False)

    for epoch in range(EPOCHS):
        print('Epoch number: ', epoch)
        losses = []
        for x in data[:5]:
            x_values = [global_w2i[pair[0]] for pair in x]
            y_values = [global_w2i[pair[1]] for pair in x]
            
            prediction = model(x_values)
            target = torch.tensor(y_values, dtype=torch.long)
            loss = loss_fn(prediction, target)
            
            losses.append(loss.mean().item())
            
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
        print("AVERAGE LOSS IN EPOCH {} was {}".format(epoch, sum(losses)/len(losses)))

if __name__ == '__main__':

    # load data:
    data = load_skipgram_data_from_file('../data/processed/english-french_small/dev.en', CONTEXT_SIZE)
    data = make_batches(data, BATCH_SIZE)
    #create vocabulary
    create_vocabulary('../data/processed/english-french_small/dev.en')
    
    # load data:
#    data = load_skipgram_data_from_file('../data/processed/english-french_large/training.en', CONTEXT_SIZE)
#    # create vocabulary
#    create_vocabulary('../data/processed/english-french_large/training.en')
#    # Initialize model
    model = Skipgram_Net(VOCABULARY_SIZE, EMBEDDING_DIMENSION)
    train_model(model, data)