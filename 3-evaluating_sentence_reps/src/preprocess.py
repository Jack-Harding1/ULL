import os
import pickle

from collections import defaultdict


UNK_KEYWORD = '<unk>'


def read_training_data(filepath=None, lowercase=False):
    '''
    Reads training data and returns a list of sentences. Each sentence is a list of words.
    @param filepath

    @return data: list of lists
    '''
    if filepath is None:
        filepath = '../data/europarl/europarl-v7.fr-en.en'

    print(' - Start reading data from: {}'.format(filepath))

    with open(filepath, 'r') as f:
        data = f.read().splitlines()

    if lowercase:
        data = [[word.lower() for word in sentence.split()] for sentence in data]
    else:
        data = [sentence.split() for sentence in data]

    return data


def process_data(data):
    '''
    Pre processing. Words with frequency 1 are replaced with '<unk>' keyword.
    @param data: list of sentences. Each sentence is a list of words
    @return processed_data: list of lists
    '''
    print(' - Start processing the data')

    vocabulary = defaultdict(int)

    for sentence in data:
        for word in sentence:
            vocabulary[word] += 1

    processed_data = []

    total = len(vocabulary.keys())
    print('   - Total vocab: {}'.format(str(total)))

    unknowns = 0

    for sentence in data:
        processed_sentence = []
        # if len(sentence) > 50:
        #     continue

        for word in sentence:
            if vocabulary[word] == 1:
                processed_sentence.append(UNK_KEYWORD)
                unknowns += 1
            else:
                processed_sentence.append(word)
        if len(processed_sentence) > 0:
            new_sentence = ' '.join(processed_sentence)
            processed_data.append(new_sentence)

    print('   - Processed vocab: {}'.format(str(total - unknowns + 1)))

    return processed_data


def save_processed_data(data, filepath):
    '''
    Save processed data to file
    @param data: list of sentences
    '''
    print(' - Writing processed data to file')

    data = '\n'.join(data)
    with open(filepath, 'w+') as f:
        f.write(data)


if __name__ == '__main__':

    print('-------------------')
    print('Begin preprocessing')

    data = read_training_data()
    data = process_data(data)
    save_path = '../data/europarl/training_processed.en'
    save_processed_data(data, save_path)

    print(' *** Lowercase *** ')

    data = read_training_data(lowercase=True)
    data = process_data(data)
    save_path = '../data/europarl/training_processed_lowercase.en'
    save_processed_data(data, save_path)

    print(' Done')
    print('-------------------')
