from collections import defaultdict
import pickle

import numpy as np
import torch
import torch.distributions as distributions

from EmbedAlign import *
from utils import *
from EmbedAlign_parameters import *


def divergence_closed_form(sigma_1, sigma_2, mu_1, mu_2):
    '''
    Computes the KL divergence between two Gaussians
    @param mu_1
    @param sigma_1
    @param mu_2
    @param sigma_2

    @return kl_z: KL divergence
    '''
    cov1 = np.diagflat(sigma_1 ** 2)
    cov2 = np.diagflat(sigma_2 ** 2)

    kl = 0.5 * (np.trace(np.matmul(np.linalg.inv(cov2), cov1)) \
             + np.matmul(np.matmul(np.transpose(mu_2 - mu_1), np.linalg.inv(cov2)), (mu_2 - mu_1)) - cov1.shape[0] \
             + np.log(np.linalg.det(cov2)) - np.log(np.linalg.det(cov1)))

    return kl

def perform_lst(embeddings, gold, lst_test):
    skip_count = 0

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    scores = collections.defaultdict(list)
    vocabulary = global_w2i.keys()


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

    print("Skip count: {}".format(skip_count))

    create_out_file(lst_test, scores, '../results/sg-full_scores.out')

def scores_divergence(embeds_locs, embeds_scales):
    '''
    Returns kl scores
    '''
    t_loc = embeds_locs[0]
    t_scale = embeds_scales[0]

    alternatives_locs = list(embeds_locs[1:])
    alternatives_scales = list(embeds_scales[1:])

    scores = [divergence_closed_form(alternatives_scales[a], t_scale, alternatives_locs[a], t_loc) for a in range(len(alternatives_locs))]

    return scores

def get_vectors(model_path, task_path, candidates_dict, word2index, threshold):

    model = torch.load(model_path)

    # Retrieve parameters
    embeddings = model['embeddings.weight']
    mean_W = model['fc1.weight']
    var_W = model['fc2.weight']
    mean_b = model['fc1.bias']
    var_b = model['fc2.bias']
    softplus = Softplus()

    with open(task_path, 'r') as f_in:
        lines = f_in.readlines()

    target2means        = defaultdict(list)
    target2vars         = defaultdict(list)
    target2strings      = defaultdict(list)
    target2sentIDs      = defaultdict(list)
    target2alternatives = defaultdict(list)

    skip_count = 0
    for line in lines:
        target, sentID, target_position, context = line.split('\t')
        target_word = target.split('.')[0]

        context_ids = [word2index[w] for w in context.split() if w in word2index]  # might be empty
        try:
            target_id = word2index[target_word]
        except KeyError:
            # target word not in dictionary, skip it
            skip_count += 1
            continue

        alternatives = candidates_dict[target_word]
        alternative_count = 0
        good_alternatives = []
        alternative_ids = []

        for a in alternatives:
            try:
                alternative_ids += [word2index[a]]
                good_alternatives += [a]
                alternative_count += 1
            except KeyError:
                # alternative word not in dictionary
                pass

        if alternative_count < threshold:
            skip_count += 1
            continue

        context_embeds = torch.stack([embeddings[i] for i in context_ids])
        context_avg = torch.mean(context_embeds, dim=0)
        context_avg = context_avg.repeat(alternative_count+1, 1)
        context_avg = torch.tensor(context_avg)

        embeds = [embeddings[w] for w in [target_id] + alternative_ids]
        embeds = torch.stack(embeds)

        # h = torch.cat((embeds, context_avg), dim=1)
        h = embeds
        mean_vecs = h @ torch.t(mean_W)# + mean_b
        var_vecs = h @ torch.t(var_W) #+ var_b
        var_vecs = softplus(var_vecs)

        target2means[target].append(mean_vecs.numpy())
        target2vars[target].append(var_vecs.numpy())
        target2strings[target].append(target)
        target2sentIDs[target].append(sentID)
        target2alternatives[target].append(good_alternatives)

    return target2means, target2vars, target2strings, target2sentIDs, target2alternatives, skip_count


if __name__ == '__main__':

    create_vocabulary('../data/processed/english-french_large/Archive/training.en')

    # TODO: lst_test.preprocessed has a different format :/
    test = process_lst_test_file()
    gold = process_lst_gold_file()
    # perform_lst(model, gold, test)

    target2locs, target2scales, target2str, target2sentIDs, targe2alt, skip_count = get_vectors(
        '../models/embed_align-5001.torch-model', '../data/lexical_substitution/lst_test.preprocessed',
        gold, global_w2i, 10)

    with open('../results/lst.out', 'w') as f_out:
        skipped_entries = 0

        for target in target2locs.keys():
            for locs_matrix, scales_matrix, target_str, sentID, alt in zip(target2locs[target], target2scales[target], target2str[target], target2sentIDs[target], targe2alt[target]):

                # TODO: replace this with utils create_out_file function
                print('RANKED\t{} {}'.format(target_str, sentID), file=f_out, end='')
                scores = scores_divergence(locs_matrix, scales_matrix)
                words_and_scores = list(zip(alt, scores))
                words_and_scores.sort(key=lambda t: t[1], reverse=False)
                for w, s in words_and_scores:
                    print('\t{} {}'.format(w, s), file=f_out, end='')
                print(file=f_out)

    # For compatibility
    with open('../results/lst.out', 'r') as f:
        out_data = f.read().splitlines()

    existing = []
    for line in out_data:
        words = line.split()
        existing.append((words[1], words[2]))

    _all = []
    for _item in test:
        _all.append((_item['center_full'], _item['sentence_id']))

    appended_output = out_data

    for _item in _all:
        if _item not in existing:
            appended_output.append('RANKED\t{} {}'.format(_item[0], _item[1]))

    with open('../results/appended_lst.out', 'w+') as f:
        f.write('\n'.join(appended_output))
