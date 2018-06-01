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


class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class EmbeddingExtractor:
    """
    This will compute a forward pass with the inference model of EmbedAlign and
        give you the variational mean for each L1 word in the batch.

    Note that this takes monolingual L1 sentences only (at this point we have a traiend EmbedAlign model
        which dispenses with L2 sentences).

    You don't really want to touch anything in this class.
    """

    def __init__(self, graph_file, ckpt_path, config=None):
        g1 = tf.Graph()
        self.meta_graph = graph_file
        self.ckpt_path = ckpt_path

        self.softmax_approximation = 'botev-batch' #default
        with g1.as_default():
            self.sess = tf.Session(config=config, graph=g1)
            # load architecture computational graph
            self.new_saver = tf.train.import_meta_graph(self.meta_graph)
            # restore checkpoint
            self.new_saver.restore(self.sess, self.ckpt_path) #tf.train.latest_checkpoint(
            self.graph = g1  #tf.get_default_graph()
            # retrieve input variable
            self.x = self.graph.get_tensor_by_name("X:0")
            # retrieve training switch variable (True:trianing, False:Test)
            self.training_phase = self.graph.get_tensor_by_name("training_phase:0")
            #self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")

    def get_z_embedding_batch(self, x_batch):
        """
        :param x_batch: is np array of shape [batch_size, longest_sentence] containing the unique ids of words

        :returns: [batch_size, longest_sentence, z_dim]
        """
        # Retrieve embeddings from latent variable Z
        # we can sempale several n_samples, default 1
        try:
            z_mean = self.graph.get_tensor_by_name("z:0")

            feed_dict = {
                self.x: x_batch,
                self.training_phase: False,
                #self.keep_prob: 1.

            }
            z_rep_values = self.sess.run(z_mean, feed_dict=feed_dict)
        except:
            raise ValueError('tensor Z not in graph!')
        return z_rep_values


params_senteval = {
    'task_path': '',
    'usepytorch': False,
    'kfold': 10,
    'ckpt_path': '',
    'tok_path': '',
    'extractor': None,
    'tks1': None
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
    """
    In this example we are going to load a tensorflow model,
    we open a dictionary with the indices of tokens and the computation graph
    """
    params.extractor = EmbeddingExtractor(
        graph_file='%s.meta'%(params.ckpt_path),
        ckpt_path=params.ckpt_path,
        config=None #run in cpu
    )

    # load tokenizer from training
    params.tks1 = dill.load(open(params.tok_path, 'rb'))
    return

def batcher(params, batch):
    """
    At this point batch is a python list containing sentences. Each sentence is a list of tokens (each token a string).
    The code below will take care of converting this to unique ids that EmbedAlign can understand.

    This function should return a single vector representation per sentence in the batch.
    In this example we use the average of word embeddings (as predicted by EmbedAlign) as a sentence representation.

    In this method you can do mini-batching or you can process sentences 1 at a time (batches of size 1).
    We choose to do it 1 sentence at a time to avoid having to deal with masking.

    This should not be too slow, and it also saves memory.
    """
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
    for sent in batch:
        x1 = params.tks1[0].to_sequences([(' '.join(sent))])
        z_batch1 = params.extractor.get_z_embedding_batch(x_batch=x1)
        sent_vec = sentence_embedding(z_batch1, params.sentence_embedding_rule)
        if np.isnan(sent_vec.sum()):
            sent_vec = np.nan_to_num(sent_vec)
        embeddings.append(sent_vec)
    embeddings = np.vstack(embeddings)
    return embeddings


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":

    # transfer_tasks = ['BigramShift', 'CR', 'Depth', 'MPQA', 'MR', 'MRPC', 'SUBJ', 'SST2', 'STS16', 'SST5',
    #                   'SICKEntailment', 'SubjNumber', 'Tense', 'TREC']

    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC', 'SICKEntailment', 'STS14']

    params_senteval.task_path = '../../../SentEval/data/'
    params_senteval.ckpt_path = '../models/ull-practical3-embedalign/model.best.validation.aer.ckpt'
    params_senteval.tok_path = '../models/ull-practical3-embedalign/tokenizer.pickle'
    params_senteval.kfold = 10

    for rule in ['MEAN', 'SUM']:
        print('-------------------------------------------------')
        print('  {}  '.format(rule))

        params_senteval.sentence_embedding_rule = rule
        se = senteval.engine.SE(params_senteval, batcher, prepare)

        results = se.eval(transfer_tasks)
        with open('../results/embed_align-{}.json'.format(rule), 'w+') as f:
            f.write(json.dumps(results, indent=4))
        print('-------------------------------------------------')
