def _load_data(data, context_size):
    '''
    converts data into required format

    @param: data (list of sentences)
    @param: context_size
    @return: [(center_word, [context_words]), (center_word, [context_words]), ...]
    '''
    X = []

    for sentence in data:
        words = sentence.split()
        if len(words) == 1:  # ignore sentences of length 1.
            continue

        for idx, word in enumerate(words):
            center_word = word

            context_words = []
            for j in range(max(0, idx - context_size), min(len(words), idx + context_size + 1)):
                if j == idx:  # center_word is included in this range, ignore that
                    continue
                context_words.append(words[j])
            X.append((center_word, context_words))

    return X

def load_data_from_file(filepath, context_size):
    '''
    loads data from filepath and converts it into required format
    '''

    with open(filepath, 'r') as f:
        data = f.read().splitlines()

    return _load_data(data, context_size)
