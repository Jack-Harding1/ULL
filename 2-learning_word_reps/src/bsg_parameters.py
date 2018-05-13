#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: jackharding, akashrajkn
"""

READ_FILEPATH = '../data/english-french_small/dev.en'
WRITE_FILEPATH = '../data/processed/english-french_small/dev.en'

REMOVE_STOP_WORDS = True
REMOVE_CAPITALS = True
DOWNSIZE_VOCABULARY = True
VOCABULARY_SIZE = 200
UNK_KEYWORD = '<unk>'

EMBEDDING_DIMENSION = 50
CONTEXT_SIZE = 3
EPOCHS = 100
LEARNING_RATE = 0.0001
