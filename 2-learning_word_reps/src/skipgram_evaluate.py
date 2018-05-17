import collections
import pickle

import numpy as np
import torch
import torch.distributions as distributions

from Skipgram_Net import *
from utils import *
from Skipgram_Parameters import *


def perform_lst(embeddings, gold, lst_test, n_zeros=0):
    '''
    For the model run the Lexical Substitution task

    @param embeddings: word embeddings of the skipgram model
    @param gold: dictionary of target words and similar words
    @param test: list of dictionaries,
                 {
                     'center_full': '',
                     'center_word': '',
                     'sentence_id': '',
                     'center_position': '',
                     'sentence': ''
                 }
    '''

    skip_count = 0
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    scores = collections.defaultdict(list)
    vocabulary = global_w2i.keys()
    # During training we extended the vocabulary with n_zeros.
    # They are not used (Required for compatibility)
    zeros = torch.zeros(n_zeros)  # 410 for the other model

    for t in lst_test:
        center_full = t['center_full']
        center_word = t['center_word']
        center_position = t['center_position']
        sentence = t['sentence']
        sentence_id = t['sentence_id']

        if center_word not in vocabulary:
            # FIXME: Ideally, this shouldn't happen.
            skip_count += 1
            continue

        center_word_embedding = embeddings(torch.cat([one_hot(center_word).cpu(), zeros]))
        context_words = gold[center_word]

        for w in context_words:
            if w not in vocabulary:
                continue

            w_1hot = torch.cat([one_hot(w).cpu(), zeros])
            w_encoding = embeddings(w_1hot)
            score = cos(center_word_embedding, w_encoding)
            scores[center_full, sentence_id].append((w, score.item()))
    # print(skip_count)
    create_out_file(lst_test, scores, '../results/sg-full_scores.out')


if __name__ == '__main__':

    full_vocab = True

    print('- Created Vocabulary')
    if full_vocab:
        create_vocabulary('../data/processed/english-french_large/training-full.en')
        model_filepath = '../models/skipgram_full-interrupted.model'
        n_zeros = 29
    else:
        create_vocabulary('../data/processed/english-french_large/training-sg-5000.en')
        model_filepath = '../models/skipgram_5000-interrupted.model'
        n_zeros = 410

    print('- Load model')
    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)

    print('- Start evaluation')
    # TODO: lst_test.preprocessed has a different format -_-
    lst_test = process_lst_test_file()
    lst_gold = process_lst_gold_file()
    perform_lst(model.center_embedding, lst_gold, lst_test, n_zeros=29)
