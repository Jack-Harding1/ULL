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


def sentence_embedding(word_embeds, rule='MEAN'):
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
        sent_vec = sentence_embedding(word_embeds, params.sentence_embedding_rule)
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

    transfer_tasks = ['BigramShift', 'CR', 'Depth', 'MPQA', 'MR', 'MRPC', 'SUBJ', 'SST2', 'STS16', 'SST5',
                      'SICKEntailment', 'SubjNumber', 'Tense', 'TREC']

    for filename in os.listdir('../models/skipgram'):
        if filename.endswith('.model'):
            resultfile = filename.split('.')[0]
            params_senteval.sg_model = filename

            for rule in ['MEAN', 'SUM']:

                print('-------------------------------------------------')
                print('  {}  '.format(resultfile))

                params_senteval.sentence_embedding_rule = rule
                se = senteval.engine.SE(params_senteval, batcher, prepare)
                results = se.eval(transfer_tasks)

                with open('../results/{}-{}.json'.format(resultfile, rule), 'w+') as f:
                    f.write(json.dumps(results, indent=4))
                print('-------------------------------------------------')
