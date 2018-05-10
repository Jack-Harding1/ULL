#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: jackharding, akashrajkn
"""

import os

from collections import Counter
from nltk.corpus import stopwords
from time import sleep

from bsg_parameters import *


def get_most_occuring_words(data, num):
    '''
    Return the top 'num' most occuring words in 'data'
      @param data: list of strings. Each string is a sentence
      @param num: number of most occuring words
      @return list of most occuring words
    '''
    all_words = []
    for sentence in data:
        words = sentence.split()
        all_words += words
    y = Counter(all_words).most_common(num)

    return [x[0] for x in y]

def downsize_vocabulary(data):
    '''
    Decrease the vocabulary size.
      @param data: list of strings. Each string is a sentence
      @return processed_data
    '''

    most_occuring = get_most_occuring_words(data, VOCABULARY_SIZE)

    processed_data = []
    for sentence in data:
        words = sentence.split()

        new_sentence = []
        for word in words:
            if word in most_occuring:
                new_sentence.append(word)
            else:
                new_sentence.append(UNK_KEYWORD)

        new_sentence = ' '.join(new_sentence)
        processed_data.append(new_sentence)

    return processed_data

def remove_stop_words_and_punctuation(data):
    '''
    Remove nltk stop words.
    Remove only punctuation marks that are not associated with words.
    For example '.' has to go since it is one of the most occuring "word" but 'can't' stays.
      @param data: list of strings. Each string is a sentence
      @return processed_data
    '''
    stop_words = list(stopwords.words('english'))
    to_remove = ['.', ',', ':', ';', '$', '?']

    processed_data = []
    for sentence in data:
        words = sentence.split()
        new_sentence = [w for w in words if not w in stop_words]
        new_sentence = [w for w in words if not w in to_remove]
        new_sentence = ' '.join(new_sentence)
        processed_data.append(new_sentence)

    return processed_data

def convert_to_lowercase(data):
    '''
    Convert data to lowercase
      @param data: list of strings. Each string is a sentence
      @return processed_data
    '''
    processed_data = []
    for sentence  in data:
        words = sentence.split()
        new_sentence = [x.lower() for x in words]
        new_sentence = ' '.join(new_sentence)
        processed_data.append(new_sentence)

    return processed_data

if __name__ == '__main__':
    # Read data
    with open(READ_FILEPATH, 'r') as f:
        data = f.read().splitlines()

    if REMOVE_STOP_WORDS:
        data = remove_stop_words_and_punctuation(data)

    if REMOVE_CAPITALS:
        data = convert_to_lowercase(data)

    if DOWNSIZE_VOCABULARY:
        data = downsize_vocabulary(data)

    # Save data
    if not os.path.exists('../data/processed'):
        os.makedirs('../data/processed')
        sleep(0.5)
    if not os.path.exists('../data/processed/english-french_small'):
        os.makedirs('../data/processed/english-french_small')
        sleep(0.5)

    data = '\n'.join(data)

    with open(WRITE_FILEPATH, 'w+') as f:
        f.write(data)
