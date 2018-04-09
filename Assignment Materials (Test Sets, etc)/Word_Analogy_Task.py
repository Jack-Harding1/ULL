#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:35:03 2018

@author: jackharding
"""
from scipy.spatial.distance import cosine
from collections import defaultdict
import numpy as np

def run_experiment_one():
    word_embeddings_file = open("deps.words", "r")
    test_file = open("Analogy-Benchmark.txt", "r")
    return run_analogy_test(word_embeddings_file, test_file)
    
def run_experiment_two():
    word_embeddings_file = open("bow2.words", "r")
    test_file = open("Analogy-Benchmark.txt", "r")
    return run_analogy_test(word_embeddings_file, test_file)
    
def run_experiment_three():
    word_embeddings_file = open("bow5.words", "r")
    test_file = open("Analogy-Benchmark.txt", "r")
    return run_analogy_test(word_embeddings_file, test_file)

def run_analogy_test(word_embeddings, test_file):
    
    word_embedding_dictionary = defaultdict()
    word_embedding_list = word_embeddings.read().split("\n")
    for embedding in word_embedding_list:
        embedding_list = embedding.split(" ")
        word_embedding_values = [float(i) for i in embedding_list[1:]]
        normalised_values = word_embedding_values / np.linalg.norm(word_embedding_values)
        word_embedding_dictionary[embedding_list[0]] = normalised_values
    
    test_list = test_file.read().split("\n")
    score = 0
    total_count = 0
    for line in test_list[1:]:
        line_list = line.split(" ")
        if len(line_list) == 4:
            a = line_list[0]
            a_star = line_list[1]
            b = line_list[2]
            b_star = line_list[3]
            if a in word_embedding_dictionary.keys() and a_star in word_embedding_dictionary.keys() and b in word_embedding_dictionary.keys():
                difference = list(np.subtract(word_embedding_dictionary[a_star], word_embedding_dictionary[a]))
                predicted_b_star = list(np.add(word_embedding_dictionary[b], difference))
                predicted_b_star = predicted_b_star / np.linalg.norm(predicted_b_star)
                
                maximum_similarity = -1
                closest_word = " "
                for word in word_embedding_dictionary.keys():
                    if word == b:
                        pass
                    elif len(word_embedding_dictionary[word]) == 300:
                        similarity = 1 - cosine(word_embedding_dictionary[word], predicted_b_star)
                        if similarity > maximum_similarity:
                            maximum_similarity = similarity
                            closest_word = word
                total_count += 1
                if closest_word == b_star:
                    score += 1
    return score / total_count
    

print(run_experiment_one())
print(run_experiment_two())
print(run_experiment_three())