#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 21:10:35 2018

@author: jackharding
"""
#FOR DEVELOPMENT
READ_FILEPATH = ['../data/english-french_small/dev.en', '../data/english-french_small/dev.fr']
WRITE_FILEPATH = ['../data/processed/embed_align/english-french_small/dev.en', '../data/processed/embed_align/english-french_small/dev.fr']

#FOR TRAINING
#READ_FILEPATH = ['../data/english-french_large/training.en', '../data/english-french_large/training.fr']
#WRITE_FILEPATH = ['../data/processed/english-french_large/training.en', '../data/processed/english-french_large/training.fr']

REMOVE_STOP_WORDS = True
REMOVE_CAPITALS = True
DOWNSIZE_VOCABULARY = True
VOCABULARY_SIZE = 200
UNK_KEYWORD = '<unk>'

EMBEDDING_DIMENSION = 100
CONTEXT_SIZE = 5
EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 5