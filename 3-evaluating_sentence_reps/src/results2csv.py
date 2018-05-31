import json
import os

tasks = ['BigramShift', 'CR', 'Depth', 'MPQA', 'MR', 'MRPC', 'SUBJ', 'SST2', 'SST5',
         'SICKEntailment', 'SubjNumber', 'Tense', 'TREC']
# STS16 does not have acc


def process_data(data):
    '''
    @param data: model results as a dictionary
    '''
    response = []
    for task in tasks:
        response.append(str(data[task]['acc']))

    return response


if __name__ == '__main__':

    model_results = ','.join(['model', 'embed_type'] + tasks)
    model_results += '\n'

    for filename in os.listdir('../results/skipgram'):
        if filename.endswith('.json'):
            filepath = '../results/skipgram/' + filename
            with open(filepath, 'r') as f:
                data = json.load(f)

            model_name = filename.split('.')[0]
            model_results += (model_name + ' ,')

            if 'MEAN' in filename:
                model_results += 'MEAN ,'
            elif 'SUM' in filename:
                model_results += 'SUM ,'

            model_results += ' ,'.join(process_data(data))
            model_results += '\n'

    with open('../results/skipgram_task_acc.csv', 'w+') as f:
        f.write(model_results)
