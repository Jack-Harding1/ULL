from __future__ import absolute_import, division, unicode_literals

import json
import os
import sys
import numpy as np
import logging
import sklearn
import senteval
import tensorflow as tf
import logging
from collections import defaultdict
import dill
import dgm4nlp
import gensim
import random
import pickle


class dotdict(dict):
    '''
    dot.notation access to dictionary attributes
    '''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


params_senteval = {
    'task_path': '',
    'usepytorch': False,
    'kfold': 10,
    'model': None
}
params_senteval = dotdict(params_senteval)


def sentence_embedding(word_embeds, sentence, word_counts, rule='MEAN'):
    '''
    defines the type of sentence embedding
    @param word_embeds: word embeddings - np array of arrays
    @param rule: type of sentence embedding
    @return sentence_embedding
    '''
    if rule == 'MEAN':
        return np.mean(word_embeds, axis=0)

    if rule == 'SUM':
        return np.sum(word_embeds, axis=0)

    if rule == 'MULTIPLY':
        sentence_embedding = word_embeds[0]
        num_words = len(word_embeds)

        for i in range(1, num_words):
            sentence_embedding = np.multiply(sentence_embedding, word_embeds[i])

        return sentence_embedding

    if rule == 'RANDOM':
        num_words = len(word_embeds)
        rand_nums = [random.uniform(0, 1) for i in range(num_words)]

        sentence_embedding = 0
        for i in range(num_words):
            sentence_embedding += (word_embeds[i] * rand_nums[i])

        sentence_embedding /= sum(rand_nums)

        return sentence_embedding

    if rule == 'WEIGHTED':
        num_words = len(word_embeds)

        counts = [word_counts[word] for word in sentence]
        div_counts = []
        for c in counts:
            if c == 0:
                div_counts.append(1)
            else:
                div_counts.append(1 / c)

        sum_counts = sum(div_counts)
        counts_normalized = [x / sum_counts for x in div_counts]

        sentence_embedding = 0
        for i in range(num_words):
            sentence_embedding += (word_embeds[i] * counts_normalized[i])
        return sentence_embedding

    return 0


def prepare(params, samples):
    '''
    loads word2vec model
    '''
    params.model = gensim.models.Word2Vec.load('../models/skipgram/{}'.format(params.sg_model))
    return


def batcher(params, batch):
    '''
    batcher method for SentEval
    '''
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        word_embeds = [params.model.wv[word] for word in sent if word in params.model]
        if len(word_embeds) == 0:
            word_embeds = [params.model.wv['.']]
        sent_vec = sentence_embedding(word_embeds, sent, params.word_counts, params.sentence_embedding_rule)
        if np.isnan(sent_vec.sum()):
            sent_vec = np.nan_to_num(sent_vec)
        embeddings.append(sent_vec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    params_senteval.task_path = '../../../SentEval/data'
    params_senteval.kfold = 10

    params_senteval.word_counts = pickle.load(open('../data/europarl/vocabulary_processed.pkl', 'rb'))

    transfer_tasks = ['BigramShift', 'CR', 'Depth', 'MPQA', 'MR', 'MRPC', 'SUBJ', 'SST2', 'STS16', 'SST5',
                      'SICKEntailment', 'SubjNumber', 'Tense', 'TREC', 'Length', 'SICKRelatedness', 'WordContent',
                      'TopConstituents', 'ObjNumber', 'OddManOut', 'CoordinationInversion', 'ImageCaptionRetrieval',
                      'STSBenchmark']

    for filename in os.listdir('../models/skipgram'):
        if filename.endswith('.model'):
            resultfile = filename.split('.')[0]
            params_senteval.sg_model = filename

            for rule in ['MEAN', 'SUM', 'WEIGHTED', 'RANDOM']:

                print('-------------------------------------------------')
                print('  {} - {}  '.format(resultfile, rule))

                params_senteval.sentence_embedding_rule = rule
                se = senteval.engine.SE(params_senteval, batcher, prepare)
                results = se.eval(transfer_tasks)

                with open('../results/{}-{}.json'.format(resultfile, rule), 'w+') as f:
                    f.write(json.dumps(results, indent=4))
                print('-------------------------------------------------')
