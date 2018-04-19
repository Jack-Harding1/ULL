#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:35:03 2018

@author: jackharding
"""
from scipy.spatial.distance import cosine
from collections import defaultdict
import numpy as np
import os

#def run_experiment_one():
#    word_embeddings_file = open("deps.words", "r")
#    for filename in os.listdir("Analogies/"):
#        if filename.endswith(".txt"):
#            print(filename)
#        test_file = open("Analogies/" + filename, "r") 
#        result = run_analogy_test(word_embeddings_file, test_file)
#        print("Score of model {} was {} on part {}".format("deps", result[0], filename))
#        print("MRR of model {} was {} on part {}".format("deps", result[1], filename))
#        test_file.close()

#def run_experiment_two():
#    word_embeddings_file = open("bow2.words", "r")
#    for filename in os.listdir("Analogies/"):
#        if not filename.endswith(".txt"):
#            pass
#        test_file = open("Analogies/" + filename, "r") 
#        result = run_analogy_test(word_embeddings_file, test_file)
#        print("Score of model {} was {} on part {}".format("deps", result[0], filename))
#        print("MRR of model {} was {} on part {}".format("deps", result[1], filename))    
#def run_experiment_three():
#    word_embeddings_file = open("bow5.words", "r")
#    for filename in os.listdir("Analogies/"):
#        if not filename.endswith(".txt"):
#            pass
#        test_file = open("Analogies/" + filename, "r")
#        result = run_analogy_test(word_embeddings_file, test_file)
#        print("Score of model {} was {} on part {}".format("deps", result[0], filename))
#        print("MRR of model {} was {} on part {}".format("deps", result[1], filename))

def run_analogy_test(word_embeddings, test_file):
    
    word_embedding_list = word_embeddings.read().split("\n")
    word_embedding_names = []
    word_embeddings = []
    for embedding in word_embedding_list:
        embedding_list = embedding.split(" ")
        word_embedding_name = embedding_list[0]
        word_embedding_value = [float(i) for i in embedding_list[1:]]
        normalised_values = word_embedding_value / np.linalg.norm(word_embedding_value)
        if len(list(normalised_values)) == 300:
            word_embedding_names.append(word_embedding_name)
            word_embeddings.append(list(normalised_values))
    
    word_embeddings_matrix = np.array(word_embeddings)
    
    testing_file = open("Analogies/" + test_file, "r")
    test_list = testing_file.read().split("\n")
    testing_file.close()
    score = 0
    total_count = 0
    MRR = []
    
    for line in test_list:
        line_list = line.split(" ")
        if len(line_list) == 4:
            a = line_list[0]
            a_star = line_list[1]
            b = line_list[2]
            b_star = line_list[3]
            if a in word_embedding_names and a_star in word_embedding_names and b in word_embedding_names and b_star in word_embedding_names:
                total_count += 1
                a_index = word_embedding_names.index(a)
                a_star_index = word_embedding_names.index(a_star)
                b_index = word_embedding_names.index(b)
                
                difference = list(np.subtract(word_embeddings[a_star_index], word_embeddings[a_index]))
                predicted_b_star = list(np.add(word_embeddings[b_index], difference))
                predicted_b_star = predicted_b_star / np.linalg.norm(predicted_b_star)
                
                distances = list(word_embeddings_matrix.dot(predicted_b_star))
                word_names = word_embedding_names.copy()
                
                indices_for_deletion = [a_index, a_star_index, b_index]
                for idx in sorted(indices_for_deletion, reverse = True):
                    del(distances[idx])
                    del(word_names[idx])

                prediction = distances.index(max(distances))
                closest_word = word_names[prediction]
                
                b_star_index = word_names.index(b_star)
                correct_distance = distances[b_star_index]
                distances.sort(reverse = True)
                ranking = distances.index(correct_distance)
                MRR.append(1 / (ranking + 1))
                
                print("Analogy from ({} to {}) to ({} to {}): prediction was {}".format(a, a_star, b, b_star, closest_word))
                if closest_word == b_star:
                    score += 1
                    
    if total_count != 0:                
        return (score / total_count), (sum(MRR)/len(MRR))
    return "NOT IN DATA SET", "NOT IN DATA SET"
    
word_embeddings_file = open("deps.words", "r")    
print(run_analogy_test(word_embeddings_file, "gram8-plural.txt"))
print(run_analogy_test(word_embeddings_file, "gram5-present-participle.txt"))
print(run_analogy_test(word_embeddings_file, "gram1-adjective-to-adverb.txt"))
print(run_analogy_test(word_embeddings_file, "family.txt"))
"gram8-plural.txt"
"gram3-comparative.txt"
"gram2-opposite.txt"
"gram4-superlative.txt"
"capital-world.txt"
"city-in-state.txt"
"gram9-plural-verbs.txt"
"gram7-past-tense.txt"
"capital-common-countries.txt"
"gram6-nationality-adjective.txt"
"currency.txt"