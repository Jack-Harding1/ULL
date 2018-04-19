#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 18:43:18 2018

@author: jackharding
"""
from scipy.spatial.distance import cosine
from collections import defaultdict
from scipy.stats.stats import spearmanr
from scipy.stats.stats import pearsonr

def run_experiment_one():
    word_embeddings_file = open("deps.words", "r")
    return create_embedding_dict(word_embeddings_file)
    
def run_experiment_two():
    word_embeddings_file = open("bow2.words", "r")
    return create_embedding_dict(word_embeddings_file)
    
def run_experiment_three():
    word_embeddings_file = open("bow5.words", "r")
    return create_embedding_dict(word_embeddings_file)
    
def create_embedding_dict(word_embeddings):
    
    word_embedding_dictionary = defaultdict()
    word_embedding_list = word_embeddings.read().split("\n")
    for embedding in word_embedding_list:
        embedding_list = embedding.split(" ")
        word_embedding_dictionary[embedding_list[0]] = [float(i) for i in embedding_list[1:]]    
    return word_embedding_dictionary

def run_qualitative_MEN_test(word_embedding_dicts, test_file):   
    test_list = test_file.read().split("\n")
    
    deps = word_embedding_dicts[0]
    bow2 = word_embedding_dicts[1]
    bow5 = word_embedding_dicts[2]
    
    first_ten_MEN = []
    second_ten_MEN = []
    third_ten_MEN = []
    fourth_ten_MEN = []
    fifth_ten_MEN = []
    
    first_ten_deps = []
    second_ten_deps = []
    third_ten_deps = []
    fourth_ten_deps = []
    fifth_ten_deps = []
    
    first_ten_bow2 = []
    second_ten_bow2 = []
    third_ten_bow2 = []
    fourth_ten_bow2 = []
    fifth_ten_bow2 = []
    
    first_ten_bow5 = []
    second_ten_bow5 = []
    third_ten_bow5 = []
    fourth_ten_bow5 = []
    fifth_ten_bow5 = []
    
    for line in test_list[1:]:
        line_list = line.split(" ")
        if len(line_list) == 3:
            word1 = line_list[0]
            word2 = line_list[1]
            if word1 in deps.keys() and word2 in deps.keys():
                deps_cosine_score = 1 - cosine(deps[word1], deps[word2])
                bow2_cosine_score = 1 - cosine(bow2[word1], bow2[word2])
                bow5_cosine_score = 1 - cosine(bow5[word1], bow5[word2])
                MEN_score = float(line_list[2])
                
#                if bow5_cosine_score - bow2_cosine_score > 0.2:
#                    print("_____________________________________________________________________________________")
#                    print("MEN SCORE", MEN_score)
#                    print("Dependency similarity between {} and {} is {}".format(word1, word2, deps_cosine_score))
#                    print("Bow2 similarity between {} and {} is {}".format(word1, word2, bow2_cosine_score))
#                    print("Bow5 similarity between {} and {} is {}".format(word1, word2, bow5_cosine_score))
                    
                if MEN_score <= 10:
                    first_ten_MEN.append(MEN_score)
                    first_ten_deps.append(deps_cosine_score)
                    first_ten_bow2.append(bow2_cosine_score)
                    first_ten_bow5.append(bow5_cosine_score)
                    pass
                
                elif MEN_score <= 20:
                    second_ten_MEN.append(MEN_score)
                    second_ten_deps.append(deps_cosine_score)
                    second_ten_bow2.append(bow2_cosine_score)
                    second_ten_bow5.append(bow5_cosine_score)
                    pass
                
                elif MEN_score <= 30:
                    third_ten_MEN.append(MEN_score)
                    third_ten_deps.append(deps_cosine_score)
                    third_ten_bow2.append(bow2_cosine_score)
                    third_ten_bow5.append(bow5_cosine_score)
                    pass
                
                elif MEN_score <= 40:
                    fourth_ten_MEN.append(MEN_score)
                    fourth_ten_deps.append(deps_cosine_score)
                    fourth_ten_bow2.append(bow2_cosine_score)
                    fourth_ten_bow5.append(bow5_cosine_score)
                    pass
                
                elif MEN_score <= 50:
                    fifth_ten_MEN.append(MEN_score)
                    fifth_ten_deps.append(deps_cosine_score)
                    fifth_ten_bow2.append(bow2_cosine_score)
                    fifth_ten_bow5.append(bow5_cosine_score)
                    pass
    
    first_deps_spearman = spearmanr(first_ten_deps, first_ten_MEN)
    first_deps_pearson = pearsonr(first_ten_deps, first_ten_MEN)                 
    first_bow2_spearman = spearmanr(first_ten_bow2, first_ten_MEN)
    first_bow2_pearson = pearsonr(first_ten_bow2, first_ten_MEN)                
    first_bow5_spearman = spearmanr(first_ten_bow5, first_ten_MEN)
    first_bow5_pearson = pearsonr(first_ten_bow5, first_ten_MEN)
    
    second_deps_spearman = spearmanr(second_ten_deps, second_ten_MEN)
    second_deps_pearson = pearsonr(second_ten_deps, second_ten_MEN)                 
    second_bow2_spearman = spearmanr(second_ten_bow2, second_ten_MEN)
    second_bow2_pearson = pearsonr(second_ten_bow2, second_ten_MEN)                
    second_bow5_spearman = spearmanr(second_ten_bow5, second_ten_MEN)
    second_bow5_pearson = pearsonr(second_ten_bow5, second_ten_MEN)
    
    third_deps_spearman = spearmanr(third_ten_deps, third_ten_MEN)
    third_deps_pearson = pearsonr(third_ten_deps, third_ten_MEN)                 
    third_bow2_spearman = spearmanr(third_ten_bow2, third_ten_MEN)
    third_bow2_pearson = pearsonr(third_ten_bow2, third_ten_MEN)                
    third_bow5_spearman = spearmanr(third_ten_bow5, third_ten_MEN)
    third_bow5_pearson = pearsonr(third_ten_bow5, third_ten_MEN)
    
    fourth_deps_spearman = spearmanr(fourth_ten_deps, fourth_ten_MEN)
    fourth_deps_pearson = pearsonr(fourth_ten_deps, fourth_ten_MEN)                 
    fourth_bow2_spearman = spearmanr(fourth_ten_bow2, fourth_ten_MEN)
    fourth_bow2_pearson = pearsonr(fourth_ten_bow2, fourth_ten_MEN)                
    fourth_bow5_spearman = spearmanr(fourth_ten_bow5, fourth_ten_MEN)
    fourth_bow5_pearson = pearsonr(fourth_ten_bow5, fourth_ten_MEN)
    
    fifth_deps_spearman = spearmanr(fifth_ten_deps, fifth_ten_MEN)
    fifth_deps_pearson = pearsonr(fifth_ten_deps, fifth_ten_MEN)                 
    fifth_bow2_spearman = spearmanr(fifth_ten_bow2, fifth_ten_MEN)
    fifth_bow2_pearson = pearsonr(fifth_ten_bow2, fifth_ten_MEN)                
    fifth_bow5_spearman = spearmanr(fifth_ten_bow5, fifth_ten_MEN)
    fifth_bow5_pearson = pearsonr(fifth_ten_bow5, fifth_ten_MEN)
    
    print("_______________________________________________________________")
    print("DEPS SCORES")
    print(first_deps_spearman, first_deps_pearson)
    print(second_deps_spearman, second_deps_pearson)
    print(third_deps_spearman, third_deps_pearson)
    print(fourth_deps_spearman, fourth_deps_pearson)
    print(fifth_deps_spearman, fifth_deps_pearson)
    print("_______________________________________________________________")
    
    print("_______________________________________________________________")
    print("BOW2 SCORES")
    print(first_bow2_spearman, first_bow2_pearson)
    print(second_bow2_spearman, second_bow2_pearson)
    print(third_bow2_spearman, third_bow2_pearson)
    print(fourth_bow2_spearman, fourth_bow2_pearson)
    print(fifth_bow2_spearman, fifth_bow2_pearson)
    print("_______________________________________________________________")
    
    print("_______________________________________________________________")
    print("BOW5 SCORES")
    print(first_bow5_spearman, first_bow5_pearson)
    print(second_bow5_spearman, second_bow5_pearson)
    print(third_bow5_spearman, third_bow5_pearson)
    print(fourth_bow5_spearman, fourth_bow5_pearson)
    print(fifth_bow5_spearman, fifth_bow5_pearson)
    print("_______________________________________________________________")

    
    return

deps_dict = run_experiment_one()
bow2_dict = run_experiment_two()
bow5_dict = run_experiment_three()
word_embeddings = [deps_dict, bow2_dict, bow5_dict]
test_file = open("MEN_dataset_natural_form_full", "r")
qualitative = run_qualitative_MEN_test(word_embeddings, test_file)