#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: jackharding, akashrajkn
"""

#FOR DEVELOPMENT
#READ_FILEPATH = '../data/english-french_small/dev.en'
#WRITE_FILEPATH = '../data/processed/english-french_small/dev.en'

#FOR TRAINING
#READ_FILEPATH = '../data/english-french_large/training.en'
#WRITE_FILEPATH = '../data/processed/english-french_large/training.en'

REMOVE_STOP_WORDS = True
REMOVE_CAPITALS = True
DOWNSIZE_VOCABULARY = True
VOCABULARY_SIZE = 5000
UNK_KEYWORD = '<unk>'

EMBEDDING_DIMENSION = 100
CONTEXT_SIZE = 5
EPOCHS = 100
LEARNING_RATE = 0.0001
