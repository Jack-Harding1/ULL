#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 20:20:18 2018

@author: jackharding
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict

def run_experiment_one():
    word_embeddings_file = open("../data/deps.words", "r")
    test_file = open("../data/2000_nouns_sorted.txt", "r")
    cluster_file = "../results/Dependency_Clusters.pdf"
    return PCA_reduction(word_embeddings_file, test_file, cluster_file)

def run_experiment_two():
    word_embeddings_file = open("../data/bow2.words", "r")
    test_file = open("../data/2000_nouns_sorted.txt", "r")
    cluster_file = "../resultsBow2_Clusters.pdf"
    return PCA_reduction(word_embeddings_file, test_file, cluster_file)

def run_experiment_three():
    word_embeddings_file = open("../data/bow5.words", "r")
    test_file = open("../data/2000_nouns_sorted.txt", "r")
    cluster_file = "../resultsBow5_Clusters.pdf"
    return PCA_reduction(word_embeddings_file, test_file, cluster_file)

def PCA_reduction(word_embeddings, test_file, cluster_file):
    word_embedding_dictionary = defaultdict()
    word_embedding_list = word_embeddings.read().split("\n")
    for embedding in word_embedding_list:
        embedding_list = embedding.split(" ")
        word_embedding_dictionary[embedding_list[0]] = [float(i) for i in embedding_list[1:]]

    top_embeddings_list = []
    top_word_list = []
    test_list = test_file.read().split("\n")
    for line in test_list:
        test_word = line.strip()
        if test_word in word_embedding_dictionary.keys() and len(word_embedding_dictionary[test_word]) == 300:
            top_embeddings_list.append(word_embedding_dictionary[test_word])
            top_word_list.append(test_word)

    top_embeddings_list = np.array(top_embeddings_list)
    pca = PCA(n_components=2, random_state = 0)
    pca_result = pca.fit_transform(top_embeddings_list)
    x = []
    y = []
    for value in pca_result:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(25, 25))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(top_word_list[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(cluster_file, bbox_inches='tight')


print(run_experiment_one())
print(run_experiment_two())
print(run_experiment_three())
