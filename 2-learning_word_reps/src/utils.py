#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: jackharding, akashrajkn
"""

import torch
import numpy as np

from Skipgram_Parameters import *


# TODO: I don't like global variables: Probably create a class to do all vocabulary stuff?
global_w2i = dict()
global_i2w = dict()


def create_vocabulary(filepath):
    '''
    Creates vocabulary and w2i and i2w dictionaries
    @param filepath
    '''
    global global_w2i
    global global_i2w

    with open(filepath, 'r') as f:
        data = f.read().splitlines()

    vocabulary = []
    for sentence in data:
        words = sentence.split()
        for word in words:
            if word not in vocabulary:
                vocabulary.append(word)

    for idx, word in enumerate(vocabulary):
        global_i2w[idx] = word
        global_w2i[word] = idx

def one_hot(word):
    '''
    One-hot vector representation of word
    @param word
    @param vocab_size: vocabulary size
    @return one-hot pytorch vector
    '''
    onehot = torch.zeros(VOCABULARY_SIZE)
    onehot[global_w2i[word]] = 1.0

    return onehot

def _load_data(data, context_size):
    '''
    converts data into required format

    @param: data (list of sentences)
    @param: context_size
    @return: [(center_word, [context_words]), (center_word, [context_words]), ...]
    '''
    X = []
    for sentence in data:
        words = sentence.split()
        if len(words) == 1:  # ignore sentences of length 1.
            continue

        for idx, word in enumerate(words):
            center_word = word

            context_words = []
            for j in range(max(0, idx - context_size), min(len(words), idx + context_size + 1)):
                if j == idx:  # center_word is included in this range, ignore that
                    continue
                context_words.append(words[j])
            X.append((center_word, context_words))

    return X

def _load_skipgram_data(data, context_size):
    '''
    converts data into required format

    @param: data (list of sentences)
    @param: context_size
    @return: [(center_word, [context_words]), (center_word, [context_words]), ...]
    '''
    X = []
    for sentence in data:
        words = sentence.split()
        if len(words) == 1:  # ignore sentences of length 1.
            continue

        for idx, word in enumerate(words):
            center_word = word

            context_words = []
            for j in range(max(0, idx - context_size), min(len(words), idx + context_size + 1)):
                if j == idx:  # center_word is included in this range, ignore that
                    continue
                X.append((center_word, words[j]))

    return X

def load_data_from_file(filepath, context_size):
    '''
    loads data from filepath and converts it into required format
    '''
    with open(filepath, 'r') as f:
        data = f.read().splitlines()

    return _load_data(data, context_size)

def load_skipgram_data_from_file(filepath, context_size):
    '''
    loads skipgram data from filepath and converts it into required format
    '''
    with open(filepath, 'r') as f:
        data = f.read().splitlines()

    return _load_skipgram_data(data, context_size)

def make_batches(data, batch_size):

    new_data = []
    num_samples = len(data)
    print(num_samples)
    for idx in range(num_samples // batch_size):
        batch = data[(idx)*batch_size : (idx+1)*batch_size]
        new_data.append(batch)
    
    return new_data
