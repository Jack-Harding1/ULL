from scipy.spatial.distance import cosine
from scipy.stats.stats import spearmanr
from scipy.stats.stats import pearsonr
from collections import defaultdict


def run_experiment_one():
    word_embeddings_file = open("../data/deps.words", "r")
    test_file = open("../data/SimLex-999.txt", "r")
    return run_simlex_similarity_test(word_embeddings_file, test_file)


def run_experiment_two():
    word_embeddings_file = open("../data/bow2.words", "r")
    test_file = open("../data/SimLex-999.txt", "r")
    return run_simlex_similarity_test(word_embeddings_file, test_file)


def run_experiment_three():
    word_embeddings_file = open("../data/bow5.words", "r")
    test_file = open("../data/SimLex-999.txt", "r")
    return run_simlex_similarity_test(word_embeddings_file, test_file)


def run_simlex_similarity_test(word_embeddings, test_file):

    word_embedding_dictionary = defaultdict()
    word_embedding_list = word_embeddings.read().split("\n")
    for embedding in word_embedding_list:
        embedding_list = embedding.split(" ")
        word_embedding_dictionary[embedding_list[0]] = [float(i) for i in embedding_list[1:]]

    embeddings_score_list = []
    simlex_score_list = []
    test_list = test_file.read().split("\n")

    for line in test_list[1:]:
        line_list = line.split("\t")
        if len(line_list) == 10:
            word1 = line_list[0]
            word2 = line_list[1]
            if word1 in word_embedding_dictionary.keys() and word2 in word_embedding_dictionary.keys():
                cosine_score = 1 - cosine(word_embedding_dictionary[word1], word_embedding_dictionary[word2])
                embeddings_score_list.append(cosine_score)
                simlex_score_list.append(float(line_list[3]))
    return(spearmanr(embeddings_score_list, simlex_score_list))


print(run_experiment_one())
print(run_experiment_two())
print(run_experiment_three())


def run_experiment_oneb():
    word_embeddings_file = open("../data/deps.words", "r")
    test_file = open("../data/SimLex-999.txt", "r")
    return run_simlex_similarity_testb(word_embeddings_file, test_file)


def run_experiment_twob():
    word_embeddings_file = open("../data/bow2.words", "r")
    test_file = open("../data/SimLex-999.txt", "r")
    return run_simlex_similarity_testb(word_embeddings_file, test_file)


def run_experiment_threeb():
    word_embeddings_file = open("../data/bow5.words", "r")
    test_file = open("../data/SimLex-999.txt", "r")
    return run_simlex_similarity_testb(word_embeddings_file, test_file)


def run_simlex_similarity_testb(word_embeddings, test_file):

    word_embedding_dictionary = defaultdict()
    word_embedding_list = word_embeddings.read().split("\n")
    for embedding in word_embedding_list:
        embedding_list = embedding.split(" ")
        word_embedding_dictionary[embedding_list[0]] = [float(i) for i in embedding_list[1:]]

    embeddings_score_list = []
    simlex_score_list = []
    test_list = test_file.read().split("\n")

    for line in test_list[1:]:
        line_list = line.split("\t")
        if len(line_list) == 10:
            word1 = line_list[0]
            word2 = line_list[1]
            if word1 in word_embedding_dictionary.keys() and word2 in word_embedding_dictionary.keys():
                cosine_score = 1 - cosine(word_embedding_dictionary[word1], word_embedding_dictionary[word2])
                embeddings_score_list.append(cosine_score)
                simlex_score_list.append(float(line_list[3]))
    return(pearsonr(embeddings_score_list, simlex_score_list))


print(run_experiment_oneb())
print(run_experiment_twob())
print(run_experiment_threeb())
