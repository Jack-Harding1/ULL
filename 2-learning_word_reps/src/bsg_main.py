#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: jackharding, akashrajkn
"""

from BSG_Net import BSG_Net
from preprocess import *
from utils import *
from bsg_parameters import *


def train_model(model):
    pass


if __name__ == '__main__':

    # load data:
    data = load_data_from_file('../data/processed/english-french_small/dev.en', CONTEXT_SIZE)
    # create vocabulary
    create_vocabulary('../data/processed/english-french_small/dev.en')
    # Initialize model
    model = BSG_Net(VOCABULARY_SIZE, EMBEDDING_DIMENSION)

    train_model(model)
