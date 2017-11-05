
from nltk.metrics.distance import edit_distance

import numpy as np

def evaluate(y_real, y_pred):
    edit_distances = []
    pronoun_num, pronoun_dem = 0.0, 0.0
    num, dem = 0.0, 0.0

    for real, pred in zip(y_real, y_pred):
        real = real.replace('eos', '').strip()

        edit_distances.append(edit_distance(real, pred.strip()))

        if pred.strip() == real:
            num += 1
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

if __name__ == '__main__':
    with open('data/results/baseline_names.txt') as f:
        y_pred = f.read().split('\n')

    with open('data/dev/refex.txt') as f:
        y_real = f.read().split('\n')
    print 'BASELINE: '
    evaluate(y_real, y_pred)


    with open('data/results/dev_1_256_1024_1024_0.2') as f:
        y_pred = f.read().split('\n')

    with open('data/dev/refex.txt') as f:
        y_real = f.read().split('\n')
    print 'MODEL: '
    evaluate(y_real, y_pred)