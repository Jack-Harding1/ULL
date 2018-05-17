#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:57:51 2018

@author: jackharding
"""

# FOR DEVELOPMENT
READ_FILEPATH = '../data/english-french_small/dev.en'
WRITE_FILEPATH = '../data/processed/english-french_small/dev.en'

# # FOR TRAINING
# READ_FILEPATH = '../data/english-french_large/training.en'
# WRITE_FILEPATH = '../data/processed/english-french_large/training.en'

REMOVE_STOP_WORDS = True
REMOVE_CAPITALS = True
DOWNSIZE_VOCABULARY = True
VOCABULARY_SIZE = 250
UNK_KEYWORD = '<unk>'

EMBEDDING_DIMENSION = 200
CONTEXT_SIZE = 5
EPOCHS = 5000
LEARNING_RATE = 0.0001
BATCH_SIZE = 10
