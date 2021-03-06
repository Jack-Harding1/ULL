import pickle

from collections import defaultdict

from BayesianSkipGram import *
from utils import *
from BayesianSkipGram_main import divergence_closed_form
from BayesianSkipGram_parameters import *


def perform_lst(model, gold, test):
    '''
    For the model run the Lexical Substitution task

    @param model
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
    mus = collections.defaultdict(list)
    priors = collections.defaultdict(list)
    posteriors = collections.defaultdict(list)
    vocabulary = global_w2i.keys()
    counter = 0

    for t in test:
        print('counter: {}'.format(counter))
        counter += 1

        center_full = t['center_full']
        center_word = t['center_word']
        center_position = t['center_position']
        sentence = t['sentence']
        sentence_id = t['sentence_id']

        if center_word not in vocabulary:
            skip_count += 1
            continue

        context_words = get_context_words(center_position, sentence, CONTEXT_SIZE, vocabulary)
        _, mu, sigma, p_mean, p_sigma = model(center_word, context_words)

        for w in gold.keys():
            if w not in vocabulary:
                continue

            _, mu_g, sigma_g, p_mean_g, p_sigma_g = model(w, context_words)

            mu_score = cos(mu_g, p_mean_g)
            prior = -1 * divergence_closed_form(mu, sigma, p_mean_g, p_sigma_g)
            posterior = -1 * divergence_closed_form(mu, sigma, mu_g, sigma_g)
            mus[center_full, sentence_id].append((w, mu_score.item()))
            priors[center_full, sentence_id].append((w, prior.item()))
            posteriors[center_full, sentence_id].append((w, posterior.item()))

    create_out_file(test, mus, '../results/bsg-300-random-mu_scores.out')
    create_out_file(test, posteriors, '../results/bsg-300-random-post_scores.out')
    create_out_file(test, priors, '../results/bsg-300-random-prior_scores.out')


if __name__ == '__main__':

    filepath = '../data/processed/english-french_large/training-300.en'
    create_vocabulary(filepath)

    with open('../models/bsg-300.model', 'rb') as f:
        model = pickle.load(f)

    test = process_lst_test_file()
    gold = process_lst_gold_file()
    perform_lst(model, gold, test)
