#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 20:20:18 2018

@author: jackharding
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def run_experiment_one():
    word_embeddings_file = open("deps.words", "r")
    return PCA_reduction(word_embeddings_file)
    
def run_experiment_two():
    word_embeddings_file = open("bow2.words", "r")
    return PCA_reduction(word_embeddings_file)
    
def run_experiment_three():
    word_embeddings_file = open("bow5.words", "r")
    return PCA_reduction(word_embeddings_file)

def PCA_reduction(word_embeddings):
    word_embedding_list = word_embeddings.read().split("\n")
    word_embedding_targets = []
    word_embeddings = np.zeros(shape=(len(word_embedding_list), len(word_embedding_list[0].split(" ")) - 1))
    for row, embedding in enumerate(word_embedding_list):
        embedding_list = embedding.split(" ")
        if len(embedding_list) == len(word_embedding_list[0].split(" ")):
            word_embedding_targets.append(embedding_list[0])
            word_embeddings[row] = [float(i) for i in embedding_list[1:]]
    
    feat_cols = ["dimension"+str(i) for i in range(word_embeddings.shape[1])]
    df = pd.DataFrame(word_embeddings, columns=feat_cols)
    df['label'] = pd.Series(word_embedding_targets)
    df['label'] = df['label'].apply(lambda i: str(i))
    
    print(df.loc[:3000, :])


    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


    
print(run_experiment_one())