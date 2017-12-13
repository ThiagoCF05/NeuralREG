__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 12/12/2017
Description:
    Evaluation script to obtain accuracy, pronoun accuracy, string edit distance and BLEU scores.

    UPDATE CONSTANT PATHS:
        ORIGINAL
        ONLYNAMES
        FERREIRA
        CATT
        HIERATT
"""

from nltk.metrics.distance import edit_distance

import numpy as np
import cPickle as p
import os

# ORIGINAL
ORIGINAL = 'data/test/data.cPickle'
# ONLY NAMES RESULTS PATH
ONLYNAMES = 'baseline/baseline_names.cPickle'
# FERREIRA RESULTS PATH
FERREIRA = 'baseline/reg/result.cPickle'
# NEURAL-CATT RESULTS PATH
CATT = 'data/att/results/test_best_1_300_512_512_3_False_5/0'
# NEURAL-HIERATT RESULTS PATH
HIERATT = 'data/hier/results/test_best_1_300_512_512_3_False_5/0'

original = p.load(open(ORIGINAL))
y_real = map(lambda x: x['refex'].lower(), original)

# ONLY NAMES RESULTS AND GOLD-STANDARDS
only = p.load(open(ONLYNAMES))
y_only = map(lambda x: x['y_pred'], only)

# FERREIRA ET AL., 2016 RESULTS
ferreira = p.load(open(FERREIRA))
# RESULTS ARE SAVE IN A DIFFERENT ORDER THAN OTHERS
y_real_ferreira = map(lambda x: x['refex'].lower(), ferreira)
y_ferreira = map(lambda x: x['realization'].lower(), ferreira)

# NEURAL CATT RESULTS
with open(CATT) as f:
    y_catt = f.read().decode('utf-8').lower().split('\n')

# NEURAL HIERATT RESULTS
with open(HIERATT) as f:
    y_hieratt = f.read().decode('utf-8').lower().split('\n')

def evaluate(y_real, y_pred):
    '''
    Accuracy, String Edit Distance and Pronoun Accuracy for the models
    :param y_real:
    :param y_pred:
    :return:
    '''
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

        if real.lower() in ['he', 'his', 'him',
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

def generate_text(data, y_pred):
    originals = []
    templates = []

    for i, reference in enumerate(data):
        reference['pred'] = y_pred[i]

    text_ids = sorted(list(set(map(lambda x: x['text_id'], data))))
    for text_id in text_ids:
        references = filter(lambda x: x['text_id'] == text_id, data)
        references = sorted(references, key=lambda x: x['general_pos'])

        text = references[0]['text'].lower()
        template = references[0]['pre_context'] + ' ' + references[0]['entity'] + ' ' + references[0]['pos_context']

        for reference in references:
            entity = reference['entity']

            refex = reference['pred'].replace('eos', '').strip()
            template = template.replace(entity, refex, 1)

        originals.append(text)
        templates.append(template.replace('_', ' ').replace('eos', '').strip())

    with open('reference', 'w') as f:
        f.write('\n'.join(originals).lower().encode('utf-8'))

    with open('output', 'w') as f:
        f.write('\n'.join(templates).lower().encode('utf-8'))

    os.system('perl multi-bleu.perl reference < output')

    os.remove('reference')
    os.remove('output')

if __name__ == '__main__':
    # ONLY NAMES ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY
    # print 'ONLY NAMES'
    # evaluate(y_real, y_only)
    # print '\n'
    # generate_text(original, y_only)
    # print 10 * '-'
    #
    # # FERREIRA ET AL., 2016 - ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY
    # print 'FERREIRA ET AL. 2016:'
    # evaluate(y_real_ferreira, y_ferreira)
    # print '\n'
    # generate_text(ferreira, y_ferreira)
    # print 10 * '-'

    # ATT
    print 'NEURAL CATT'
    evaluate(y_real, y_catt)
    print '\n'
    generate_text(original, y_catt)
    print 10 * '-'

    # HIER
    print 'NEURAL HIERATT'
    evaluate(y_real, y_hieratt)
    print '\n'
    generate_text(original, y_hieratt)
    print 10 * '-'