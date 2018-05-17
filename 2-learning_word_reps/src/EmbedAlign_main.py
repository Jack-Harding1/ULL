#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 21:09:50 2018

@author: jackharding
"""
import torch
import torch.distributions as distributions
import torch.optim as optim

from preprocess import *
from utils import *
from EmbedAlign_parameters import *
from EmbedAlign import EmbedAlign


def train_embed_align(model, l1_data, l2_data):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    batches = list(zip(l1_data, l2_data))
    shuffle(batches)

    for epoch in range(EPOCHS):
        overall_loss = 0
        model.train()

        losses = []
        for batch_en, batch_fr in batches:

            optimizer.zero_grad()
            loss = model(batch_en, batch_fr)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print("AVERAGE LOSS IN REGION {} was {}".format(epoch, sum(losses) / len(losses)))

if __name__ == '__main__':
    # load data:
    l1_data, l2_data = load_embed_align_data_from_file(WRITE_FILEPATH, global_w2i_l1, global_w2i_l2, BATCH_SIZE)
    # create vocabulary
    create_embed_align_vocabulary(WRITE_FILEPATH)
    # Initialize model
    model = EmbedAlign(len(global_w2i_l1.keys()), len(global_w2i_l2.keys()), EMBEDDING_DIMENSION, False)

    train_embed_align(model, l1_data, l2_data)
