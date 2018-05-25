import gensim
import os
import pickle
import time

from preprocess import read_training_data


EPOCHS = [5, 10]
FEATURE_SIZE = [100]
WINDOW = [3, 5]
NEGATIVE = [5] # Can be anywhere between 5-20.
LOWERCASE = False


def train_skipgram(data, epochs, feature_size, window, negative):
    '''
    Trains a skipgram models.
    For documentation, check https://radimrehurek.com/gensim/models/word2vec.html

    @param data: input sentences
    @return model: trained skipgram model
    '''
    print('     Training model')
    model = gensim.models.Word2Vec(sentences=data, sg=1, size=feature_size, window=window, min_count=1,
                                   max_vocab_size=None, workers=4, negative=negative, iter=2)

    model.train(sentences=data, epochs=epochs, total_examples=len(data))

    return model


if __name__ == '__main__':

    print('-------------------')

    if not os.path.exists('../models/skipgram'):
        os.makedirs('../models/skipgram')
        sleep(0.5)

    if LOWERCASE:
        filepath = '../data/europarl/training_processed_lowercase.en'
    else:
        filepath = '../data/europarl/training_processed.en'

    data = read_training_data(filepath=filepath)

    for negative in NEGATIVE:
        for feature_size in FEATURE_SIZE:
            for window in WINDOW:
                for epochs in EPOCHS:

                    decorator = 'Negative_{}-Features{}-Window_{}-Epochs_{}'.format(negative, feature_size, window, epochs)
                    if LOWERCASE:
                        decorator += '-lowercase'

                    print('   - {}'.format(decorator))
                    start_time = time.time()
                    model = train_skipgram(data, epochs, feature_size, window, negative)
                    print('     %s seconds' % (time.time() - start_time))

                    print('Saving to file: SG-{}.model'.format(decorator))
                    model.save('../models/skipgram/SG-{}.model'.format(decorator))

    print('-------------------')
