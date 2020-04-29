__author__ = 'thiagocastroferreira'

import json

"""
Author: Thiago Castro Ferreira
Date: 02/02/2018
Description:
    Scripts used to surface realized the texts of the human evaluation in the HTML format

    PYTHON VERSION :2.7

    DEPENDENCIES:
        Numpy
"""

__author__ = ''

import os

DATA_PATH = '../data/v1.5/'
EVAL_PATH = '../eval/stats/beta/v1.5/'
OUTPUT_PATH = '../eval/stats/beta/v1.5/'

# ORIGINAL
ORIGINAL = os.path.join(DATA_PATH, 'gold-ids.json')

# REFERENCES
REFS = EVAL_PATH + 'refs.txt'
# ONLY NAMES RESULTS PATH
ONLYNAMES = EVAL_PATH + 'only.txt'
# ATTENTION ACL NAMES RESULTS PATH
ATTENTION_ACL = EVAL_PATH + 'attacl.txt'
# ATTENTION COPY NAMES RESULTS FOLDER
ATTENTION_COPY = EVAL_PATH + 'attcopy.txt'
# PROFILEREG
PROFILEREG = EVAL_PATH + 'profilereg.txt'


def load_models():
    original = json.load(open(ORIGINAL, encoding='utf-8'))
    # y_original = map(lambda x: x['refex'], original)
    y_original = []
    for i, row in enumerate(original):
        refex = [w.lower() for w in row['refex']]
        refex = ' '.join(refex).strip()
        y_original.append(refex)

    # REFS
    with open(REFS, encoding='utf-8') as f:
        y_refs = f.read().lower().split('\n')

    # ONLY NAMES RESULTS
    # only = json.load(open(ONLYNAMES, encoding='utf-8'))
    # y_only = list(map(lambda x: x['y_pred'], only))

    with open(ONLYNAMES, encoding='utf-8') as f:
        y_only = f.read().lower().split('\n')

    # ATTENTION_ACL RESULTS
    with open(ATTENTION_ACL, encoding='utf-8') as f:
        y_attacl = f.read().lower().split('\n')

    # ATTENTION COPY RESULTS
    with open(ATTENTION_COPY, encoding='utf-8') as f:
        y_attcopy = f.read().lower().split('\n')

    # PROFILEREG RESULTS
    with open(PROFILEREG, encoding='utf-8') as f:
        y_profilereg = f.read().lower().split('\n')

    return original, y_original, y_only, y_attacl, y_attcopy, y_profilereg, y_refs


def save_trials(originals, only, attacl, attcopy, profilereg):
    if not os.path.exists('trials/beta/'):
        os.mkdir('trials/beta/')

    trials = []
    for i, original in enumerate(originals):
        trial = {
            'original': original,
            'only': only[i],
            'attacl': attacl[i],
            'attcopy': attcopy[i],
            'profilereg': profilereg[i]
        }
        trials.append(trial)

        with open(os.path.join(OUTPUT_PATH, 'evaluation_c.csv'), 'w') as f:
            f.write(original + ';' + only[i] + ';' + attacl[i] + ';' + attcopy[i] + ';' + attcopy[i])
            f.write('\n')

        with open(os.path.join(OUTPUT_PATH, 'evaluation_r.csv'), 'w') as f:
            f.write(original + '\n')
            f.write(only[i] + '\n')
            f.write(attacl[i] + '\n')
            f.write(attcopy[i] + '\n')
            f.write(attcopy[i] + '\n')
            f.write('\n')

    return trials


def generate_texts(data, y_pred, fname):
    templates = []

    for i, reference in enumerate(data):
        reference['pred'] = y_pred[i].lower().strip()

    for i, reference in enumerate(data):
        entity = reference['entity'].lower() + ' '
        refex = reference['pred'].replace('eos', '').strip()
        pre_context = ' '.join(reference['pre_context']).replace('eos', '').lower().strip()
        pos_context = ' '.join(reference['pos_context']).replace('eos', '').lower().strip()

        template = pre_context + ' ' + entity + ' ' + pos_context
        template = template.replace(entity, refex + ' ')

        templates.append(template.replace('_', ' ').replace('~', ' ').replace('eos', '').strip())

    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write('\n'.join(templates).lower().encode('utf-8'))

    return templates


if __name__ == '__main__':
    original, y_original, y_only, y_attacl, y_attcopy, y_profilereg, y_refs = load_models()

    if not os.path.exists('texts/beta/'):
        os.mkdir('texts/beta')

    trials = save_trials(y_refs, y_only, y_attacl, y_attcopy, y_profilereg)

    json.dump(trials, open(os.path.join('texts/beta/', 'trials.json'), 'w'))