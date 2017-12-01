
from nltk.metrics.distance import edit_distance

import numpy as np
import cPickle as p

def corpus_evaluation():
    def pronoun_count(data):
        num = 0
        for reference in data:
            if reference['refex'] in ['he', 'his', 'him', 'she', 'hers', 'her', 'it', 'its', 'we', 'our', 'ours', 'they', 'theirs', 'them']:
                num += 1
        return num
    train_data = p.load(open('data/train/data.cPickle'))
    dev_data = p.load(open('data/dev/data.cPickle'))
    test_data = p.load(open('data/test/data.cPickle'))

    train_entities = set(map(lambda x: x['entity'], train_data))
    dev_entities = set(map(lambda x: x['entity'], dev_data))
    test_entities = set(map(lambda x: x['entity'], test_data))

    print 'Train entities: ', len(list(train_entities))
    print 'Dev entities: ', len(list(dev_entities))
    print 'Test entities: ', len(list(test_entities))

    dev_intersect = train_entities.intersection(dev_entities)
    print 'Train intersect dev: ', str(len(list(dev_intersect)))

    test_intersect = train_entities.intersection(test_entities)
    print 'Train intersect test: ', str(len(list(test_intersect)))

def evaluate(y_real, y_pred):
    edit_distances = []
    pronoun_num, pronoun_dem = 0.0, 0.0
    num, dem = 0.0, 0.0
    wrong = []

    for real, pred in zip(y_real, y_pred):
        real = real.replace('eos', '').strip()
        pred = pred.replace('eos', '').strip()

        edit_distances.append(edit_distance(real, pred.strip()))

        if pred.strip() == real:
            num += 1
        else:
            wrong.append({'real': real, 'pred':pred})
        dem += 1

        if real in ['he', 'his', 'him',
                    'she', 'her', 'hers',
                    'it', 'its', 'we', 'us', 'our', 'ours',
                    'they', 'them', 'their', 'theirs']:
            # print(real)
            if pred.strip() == real:
                pronoun_num += 1
            pronoun_dem += 1

    print('ACCURACY: ', str(round(num/dem, 4)))
    print('DISTANCE: ', str(round(np.mean(edit_distances), 4)))
    print('PRONOUN ACCURACY: ', str(round(pronoun_num/pronoun_dem, 4)))
    return wrong

if __name__ == '__main__':
    references = p.load(open('baseline/baseline_names.cPickle'))
    y_pred = map(lambda x: x['y_pred'], references)

    y_real = map(lambda x: x['y_real'].lower(), references)
    print 'BASELINE: ONLY NAMES'
    evaluate(y_real, y_pred)

    references = p.load(open('baseline/reg/result.cPickle'))
    y_pred = map(lambda x: x['realization'], references)

    y_real = map(lambda x: x['refex'].lower(), references)
    print 'BASELINE:'
    evaluate(y_real, y_pred)

    with open('data/att/results/test_best_1_300_512_512_2_False_1/0') as f:
        y_pred = f.read().split('\n')

    with open('data/test/refex.txt') as f:
        y_real = f.read().split('\n')
    print 'MODEL: test_best_1_300_512_512_2_False_1'
    evaluate(y_real, y_pred)

    with open('data/att/results/test_best_1_300_512_512_3_False_1/0') as f:
        y_pred = f.read().split('\n')

    with open('data/test/refex.txt') as f:
        y_real = f.read().split('\n')
    print 'MODEL: test_best_1_300_512_512_3_False_1'
    wrong = evaluate(y_real, y_pred)

    for e in wrong:
        print 'REAL: ', e['real']
        print 'PRED: ', e['pred']
        print 10 * '-'

    corpus_evaluation()