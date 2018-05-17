#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: jackharding, akashrajkn
"""

import os
import random

from collections import Counter
from nltk.corpus import stopwords
from time import sleep

from bsg_parameters import *
from utils import process_lst_gold_file


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

    unique = list(set(all_words))

    # if len(unique) < num:
    #     y = Counter(all_words).most_common(num)

    print("vocab: {}, most: {}".format(str(len(unique)), str(num)))

    y = Counter(all_words).most_common(num)

    return [x[0] for x in y]

def downsize_vocabulary(data, lst_words=None):
    '''
    Decrease the vocabulary size.
      @param data: list of strings. Each string is a sentence
      @return processed_data
    '''

    most_occuring = get_most_occuring_words(data, VOCABULARY_SIZE - 1)

    print('lst_words: {}'.format(len(list(set(lst_words)))))

    if lst_words is not None:
        most_occuring = list(set(lst_words)) + list(set(most_occuring))
        most_occuring = list(set(most_occuring))
        # most_occuring = most_occuring[:VOCABULARY_SIZE - 1]
        print('final: {}'.format(len(most_occuring)))

    # if lst_words is not None:
    #     idx = len(most_occuring) - 1
    #     for w in lst_words:
    #         if w not in most_occuring:
    #             # print(idx)
    #             most_occuring[idx] = w
    #             idx -= 1

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

def remove_stop_words_and_punctuation(data, lst_words=None):
    '''
    Remove nltk stop words.
    Remove only punctuation marks that are not associated with words.
    For example '.' has to go since it is one of the most occuring "word" but 'can't' stays.
      @param data: list of strings. Each string is a sentence
      @return processed_data
    '''
    stop_words = list(stopwords.words('english'))
    to_remove = ['.', ',', ':', ';', '$', '?']

    if lst_words is not None:
        for w in lst_words:
            if w in stop_words:
                stop_words.remove(w)

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

    lst_gold = process_lst_gold_file()
    lst_words = []
    for w in lst_gold.keys():
        lst_words.append(w)
        lst_words += lst_gold[w]
    lst_words = set(lst_words)

    if DOWNSAMPLE_DATA > 0:
        samples = random.sample(range(len(data)), DOWNSAMPLE_DATA)
        data = [data[x] for x in samples]

        # save this dataset for further use
        save_data = '\n'.join(data)
        with open('../data/english-french_large/training-{}.en'.format(str(DOWNSAMPLE_DATA)), 'w+') as f:
            f.write(save_data)

    if REMOVE_STOP_WORDS:
        data = remove_stop_words_and_punctuation(data)

    if REMOVE_CAPITALS:
        data = convert_to_lowercase(data)

    if DOWNSIZE_VOCABULARY:
        data = downsize_vocabulary(data, lst_words)

    # Save data
    if not os.path.exists('../data/processed'):
        os.makedirs('../data/processed')
        sleep(0.5)
    if not os.path.exists('../data/processed/english-french_large'):
        os.makedirs('../data/processed/english-french_large')
        sleep(0.5)

    data = '\n'.join(data)
    save_path = '../data/processed/english-french_large/training-{}.en'.format(str(DOWNSAMPLE_DATA))
    with open(save_path, 'w+') as f:
        f.write(data)
