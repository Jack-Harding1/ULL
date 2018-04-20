import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
import pickle

from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def read_nouns(path=None):
    if path is None:
        path = './2000_nouns_sorted.txt'

    with open(path, 'r') as f:
        nouns = f.read().splitlines()
    return nouns


def get_embeddings_list(word_embeddings, test_file):
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

    return top_embeddings_list, top_word_list


def plot_clusters(kmeans_model, reduced_data, clusters):
    # x = []
    # y = []
    # for value in reduced_data:
    #     x.append(value[0])
    #     y.append(value[1])

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans_model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    # for i in range(len(x)):
    #     plt.annotate(embeddings_array[i],
    #                  xy=(x[i], y[i]),
    #                  xytext=(5, 2),
    #                  textcoords='offset points',
    #                  ha='right',
    #                  va='bottom')
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans_model.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the word vectors dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('./results/clusters_{}.png'.format(clusters), bbox_inches='tight')


def cluster_word_vectors(save_plot=False):
    '''
    Function performs K-means clustering on word representations
    '''
    # Hyper parameters
    clusters = [2, 5, 10, 30, 33, 45, 75, 100, 150, 175, 200, 300, 400, 500, 822, 1000]
    word_representations = ['bow5.words']

    test_file = open('2000_nouns_sorted.txt', 'r')

    print('----------- Start K-Means -----------')
    for rep in word_representations:
        print('   Word Representation: {}'.format(rep))
        word_embeddings_file = open(rep, 'r')

        if os.path.exists('./models/embeddings_{}'.format(rep)):
            embeddings = pickle.load(open('./models/embeddings_{}'.format(rep), 'rb'))
            words = pickle.load(open('./models/words_{}'.format(rep), 'rb'))
        else:
            embeddings, words = get_embeddings_list(word_embeddings_file, test_file)
            # write
            pickle.dump(embeddings, open('./models/embeddings_{}'.format(rep), 'wb'))
            pickle.dump(words, open('./models/words_{}'.format(rep), 'wb'))

        embeddings_array = np.array(embeddings)

        for cluster in clusters:
            print('      Clusters: {}'.format(cluster))
            kmeans_model = KMeans(n_clusters=cluster, random_state=0)
            kmeans_fit = kmeans_model.fit(embeddings_array)

            # Save the model
            pickle.dump(kmeans_model, open('./models/kmeans/{}_{}'.format(rep, cluster), 'wb'))
            if save_plot:
                plot_clusters(kmeans_model, reduced_data, clusters)
    print('-------------------------------------')


def generate_figures(model, rep='default', c_num=0):
    test_file = open('2000_nouns_sorted.txt', 'r')
    word_embeddings_file = open(rep, 'r')

    if os.path.exists('./models/embeddings_{}'.format(rep)):
        embeddings = pickle.load(open('./models/embeddings_{}'.format(rep), 'rb'))
        words = pickle.load(open('./models/words_{}'.format(rep), 'rb'))
    else:
        embeddings, words = get_embeddings_list(word_embeddings_file, test_file)
        # write
        pickle.dump(embeddings, open('./models/embeddings_{}'.format(rep), 'wb'))
        pickle.dump(words, open('./models/words_{}'.format(rep), 'wb'))

    embeddings = np.array(embeddings)

    pca = PCA(n_components=2, random_state = 0)
    pca_result = pca.fit_transform(embeddings)

    clusters = np.arange(model.n_clusters)

    # Create figure
    plt.figure(figsize=(25, 25))
    colors = cm.rainbow(np.linspace(0, 1, model.n_clusters))

    for cluster, color in zip(clusters, colors):
        x = []
        y = []
        index = 0
        for value in pca_result:

            if model.labels_[index] == cluster:
                x.append(value[0])
                y.append(value[1])
            index += 1

        for i in range(len(x)):
            plt.scatter(x[i], y[i], color=color)
            plt.annotate(words[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

    plt.savefig('./results/kmeans/{}_{}.pdf'.format(rep, c_num), bbox_inches='tight')
    plt.close()

def generate_csv(model, rep='default', c_num=0):

    words = pickle.load(open('./models/words_{}'.format(rep), 'rb'))
    clusters = np.arange(model.n_clusters)
    labels = model.labels_

    output = defaultdict(list)

    for i, word in enumerate(words):
        output[labels[i]].append(word)

    output_df = None

    for key, val in output.items():
        if val is None:
            val = []
        df = pd.DataFrame({key: val})

        if output_df is None:
            output_df = df
        else:
            output_df = pd.concat([output_df, df], ignore_index=True, axis=1)

    output_df.to_csv('./results/kmeans/{}_{}.csv'.format(rep, c_num), sep=',', na_rep='')


# cluster_word_vectors()
for file in os.listdir("./models/kmeans/"):
    filepath = os.path.join("./models/kmeans/", file)
    details = file.split('_')

    print("------ {} ------".format(file))
    model = pickle.load(open(filepath, 'rb'))

    if not os.path.exists('./results/kmeans/{}.pdf'.format(file)):
        generate_figures(model, details[0], int(details[1]))

    if not os.path.exists('./results/kmeans/{}.csv'.format(file)):
        generate_csv(model, details[0], int(details[1]))
