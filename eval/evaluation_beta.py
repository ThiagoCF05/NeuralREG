__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 12/12/2017
Description:
    Automatic evaluation scripts to obtain accuracy, pronoun measures, string edit distance and BLEU scores.
    Moreover, it saves .csv files to test statistica significance in R.

    PYTHON VERSION :3

    DEPENDENCIES:
        NLTK:           http://www.nltk.org/
        SciKit learn:   http://scikit-learn.org/stable/

    UPDATE CONSTANT PATHS:
        ORIGINAL
        ONLYNAMES
        ATTENTION
        ATTENTION_ACL
        PROFILEREG

        MULTIBLEU
"""

from nltk.metrics.distance import edit_distance
from sklearn.metrics import classification_report

import codecs
import json
import nltk
import numpy as np
import os

DATA_PATH = '../data/v1.5/'
EVAL_PATH = 'data/'
OUTPUT_PATH = 'stats/beta/v1.5/'

# ORIGINAL
ORIGINAL = os.path.join(DATA_PATH, 'test.json')

# ONLY NAMES RESULTS PATH
ONLYNAMES = EVAL_PATH + 'onlynames/results/onlynames.json'
# ATTENTION ACL NAMES RESULTS PATH
ATTENTION_ACL = EVAL_PATH + 'attention_acl/results/test_1'
# ATTENTION COPY NAMES RESULTS FOLDER
ATTENTION_COPY = EVAL_PATH + 'attention/results/test_1'
# PROFILEREG
PROFILEREG = EVAL_PATH + 'profilereg/results/preds.txt'

# TEST INFO
TEST_INFO = DATA_PATH + 'test_info.json'

MULTIBLEU = '../eval/multi-bleu.perl'


def load_models():
    original = json.load(open(ORIGINAL, encoding='utf-8'))
    test_info = json.load(open(TEST_INFO, encoding='utf-8'))
    for i, row in enumerate(original):
        original[i]['eid'] = test_info[i]['eid']
        original[i]['lid'] = test_info[i]['lid']
        original[i]['category'] = test_info[i]['category']
        original[i]['text'] = ' '.join(test_info[i]['targets'][0]['output'])
    del test_info

    # y_original = map(lambda x: x['refex'], original)
    y_original = []
    for i, row in enumerate(original):
        refex = ' '.join(row['refex']).lower().strip()
        y_original.append(refex)

    # ONLY NAMES RESULTS
    only = json.load(open(ONLYNAMES, encoding='utf-8'))
    y_only = list(map(lambda x: ' '.join(nltk.word_tokenize(x['y_pred'])).lower().strip(), only))
    # SEEN and UNSEEN
    # y_only = [' '.join(nltk.word_tokenize(w.lower().strip())) for w in only]

    # ATTENTION_ACL RESULTS
    with open(ATTENTION_ACL, encoding='utf-8') as f:
        y_attacl = [' '.join(nltk.word_tokenize(w.strip())) for w in f.read().lower().split('\n')]

    # ATTENTION COPY RESULTS
    with open(ATTENTION_COPY, encoding='utf-8') as f:
        y_attcopy = [' '.join(nltk.word_tokenize(w.strip())) for w in f.read().lower().split('\n')]

    # PROFILEREG RESULTS
    with open(PROFILEREG, encoding='utf-8') as f:
        y_profilereg = [' '.join(nltk.word_tokenize(w.strip())) for w in f.read().lower().split('\n')]

    return original, y_original, y_only, y_attacl, y_attcopy, y_profilereg


def evaluate_references(y_real, y_pred):
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

    pron_acc = []
    pron_real, pron_pred = [], []

    for real, pred in zip(y_real, y_pred):
        real = real.replace('eos', '').strip()
        pred = pred.replace('eos', '').strip()

        edit_distances.append(edit_distance(real, pred.strip()))

        if pred.strip() == real:
            num += 1
        else:
            wrong.append({'real': real, 'pred': pred})
        dem += 1

        if real.lower() in ['he', 'his', 'him',
                            'she', 'her', 'hers',
                            'it', 'its', 'we', 'us', 'our', 'ours',
                            'they', 'them', 'their', 'theirs']:
            pron_real.append('pronoun')

            if pred.strip() == real:
                pronoun_num += 1
                pron_acc.append(1)
            else:
                pron_acc.append(0)
            pronoun_dem += 1
        else:
            pron_real.append('non_pronoun')

        if pred.lower().strip() in ['he', 'his', 'him',
                                    'she', 'her', 'hers',
                                    'it', 'its', 'we', 'us', 'our', 'ours',
                                    'they', 'them', 'their', 'theirs']:
            pron_pred.append('pronoun')
        else:
            pron_pred.append('non_pronoun')

    print('ACCURACY: ', str(round(num / dem, 4)))
    print('DISTANCE: ', str(round(np.mean(edit_distances), 4)))
    print('PRONOUN ACCURACY: ', str(round(pronoun_num / pronoun_dem, 4)))
    print('\n')

    print(classification_report(pron_real, pron_pred))
    return wrong, edit_distances, pron_acc


def domain_evaluate(y_real, y_pred, info):
    '''
    Accuracy, String Edit Distance and Pronoun Accuracy for the models per domain
    :param y_real:
    :param y_pred:
    :param info:
    :return:
    '''
    domains = {}
    for i in range(len(y_real)):
        domain = info[i]['category']
        if domain not in domains:
            domains[domain] = {'num': 0.0, 'dem': 0.00000001, 'pronoun_num': 0.0, 'pronoun_dem': 0.00000001,
                               'distance': []}

        real = y_real[i].replace('eos', '').strip()
        pred = y_pred[i].replace('eos', '').strip()

        domains[domain]['distance'].append(edit_distance(real, pred.strip()))

        if pred.strip() == real:
            domains[domain]['num'] += 1
        domains[domain]['dem'] += 1

        if real.lower() in ['he', 'his', 'him',
                            'she', 'her', 'hers',
                            'it', 'its', 'we', 'us', 'our', 'ours',
                            'they', 'them', 'their', 'theirs']:
            # print(real)
            if pred.strip() == real:
                domains[domain]['pronoun_num'] += 1
            domains[domain]['pronoun_dem'] += 1

    for domain in domains:
        print(domain.upper())
        print('ACCURACY: ', str(round(domains[domain]['num'] / domains[domain]['dem'], 4)))
        print('DISTANCE: ', str(round(np.mean(domains[domain]['distance']), 4)))
        print('PRONOUN ACCURACY: ', str(round(domains[domain]['pronoun_num'] / domains[domain]['pronoun_dem'], 4)))


def evaluate_text(data, y_pred):
    originals = []
    templates = []
    text_acc = []

    for i, reference in enumerate(data):
        reference['pred'] = y_pred[i]

    text_ids = [w['eid'] for w in data]
    text_ids = sorted(list(set(text_ids)))

    for text_id in text_ids:
        eid = text_id
        references = [w for w in data if w['eid'] == eid]
        references = sorted(references, key=lambda x: len(x['pre_context']))

        pos_context = ' '.join(references[0]['pos_context']).strip()
        pre_context = ' '.join(references[0]['pre_context']).strip()

        text = references[0]['text'].lower()
        template = pre_context + ' ' + references[0]['entity'].strip() + ' ' + pos_context

        for reference in references:
            entity = reference['entity'] + ' '

            refex = '~'.join(reference['pred'].replace('eos', '').strip().split()) + ' '
            template = template.replace(entity, refex, 1)

        template = template.lower().replace('^', ' ').replace('@', ' ').replace('"', '').replace('_', ' ').replace('~', ' ').replace('eos', '').strip()
        originals.append(text)
        templates.append(template)

    # Original accuracy
    num, dem = 0, 0
    # Original accuracy
    for original, template in zip(originals, templates):
        if original.lower().replace('@', '') == template.lower().replace('@', ''):
            num += 1
            text_acc.append(1)
        else:
            text_acc.append(0)
        dem += 1

    with codecs.open('reference', 'w', encoding='utf8') as f:
        f.write('\n'.join(originals).lower())

    with codecs.open('output', 'w', encoding='utf8') as f:
        f.write('\n'.join(templates).lower())

    os.system('perl ' + MULTIBLEU + ' reference < output')
    os.remove('reference')
    os.remove('output')

    with codecs.open(os.path.join(OUTPUT_PATH, 'refs.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(originals).lower())

    return originals, templates, num, dem, text_acc


def model_report(model_name, original, y_real, y_pred):
    print(model_name)
    wrong, edit_distances, pron_acc = evaluate_references(y_real, y_pred)
    print('\n')
    originals, templates, num, dem, text_acc = evaluate_text(original, y_pred)
    print('TEXT ACCURACY: ', str(round(float(num) / dem, 4)), str(num), str(dem))
    print('\n')
    # print('DOMAIN ACCURACY:')
    # domain_evaluate(y_real, y_pred, original)
    # print(10 * '-')

    return originals, templates, edit_distances, pron_acc, text_acc


def run():
    original, y_real, y_only, y_attacl, y_attcopy, y_profilereg = load_models()

    # ONLY - NAMES ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY
    originals, templates, only_distances, only_pron_acc, only_text_acc = model_report('ONLY NAMES', original, y_real,
                                                                                      y_only)
    with codecs.open(os.path.join(OUTPUT_PATH, 'only.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(templates).lower())

    only_ref_acc = []
    for real, pred in zip(y_real, y_only):
        if real.replace('eos', '').strip() == pred.replace('eos', '').strip():
            only_ref_acc.append(1)
        else:
            only_ref_acc.append(0)

    # ATTENTION ACL - NAMES ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY
    originals, templates, attacl_distances, attacl_pron_acc, attacl_text_acc = model_report('ATTENTION ACL', original,
                                                                                            y_real, y_attacl)
    with codecs.open(os.path.join(OUTPUT_PATH, 'attacl.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(templates).lower())

    attacl_ref_acc = []
    for real, pred in zip(y_real, y_attacl):
        if real.replace('eos', '').strip() == pred.replace('eos', '').strip():
            attacl_ref_acc.append(1)
        else:
            attacl_ref_acc.append(0)

    # ATTENTION COPY - NAMES ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY

    originals, templates, attcopy_distances, attcopy_pron_acc, attcopy_text_acc = model_report('ATTENTION COPY',
                                                                                               original, y_real,
                                                                                               y_attcopy)
    with codecs.open(os.path.join(OUTPUT_PATH, 'attcopy.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(templates).lower())

    attcopy_ref_acc = []
    for real, pred in zip(y_real, y_attcopy):
        if real.replace('eos', '').strip() == pred.replace('eos', '').strip():
            attcopy_ref_acc.append(1)
        else:
            attcopy_ref_acc.append(0)

    # PROFILEREG - ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY
    originals, templates, profilereg_distances, profilereg_pron_acc, profilereg_text_acc = model_report(
        'PROFILEREG', original, y_real, y_profilereg)

    with codecs.open(os.path.join(OUTPUT_PATH, 'profilereg.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(templates).lower())

    profilereg_ref_acc = []
    for real, pred in zip(y_real, y_profilereg):
        if real.replace('eos', '').strip() == pred.replace('eos', '').strip():
            profilereg_ref_acc.append(1)
        else:
            profilereg_ref_acc.append(0)

    # Save files to perform statistical tests in R
    # Reference accuracy file
    resp = np.arange(1, len(y_real) + 1)
    ref_acc = np.concatenate(
        [[resp], [only_ref_acc], [attacl_ref_acc], [attcopy_ref_acc], [profilereg_ref_acc]])
    ref_acc = ref_acc.transpose().tolist()

    with open(os.path.join(OUTPUT_PATH, 'r_ref_acc.csv'), 'w') as f:
        f.write('resp;onlynames;attentionacl;attentioncopy;profilereg\n')
        for row in ref_acc:
            f.write(';'.join(map(lambda x: str(x), row)))
            f.write('\n')

    # Pronoun accuracy
    resp = np.arange(1, len(attcopy_pron_acc) + 1)
    pron_acc = np.concatenate(
        [[resp], [only_pron_acc], [attacl_pron_acc], [attcopy_pron_acc], [profilereg_pron_acc]])
    pron_acc = pron_acc.transpose().tolist()

    with open(os.path.join(OUTPUT_PATH, 'r_pron_acc.csv'), 'w') as f:
        f.write('resp;onlynames;attentionacl;attentioncopy;profilereg\n')
        for row in pron_acc:
            f.write(';'.join(map(lambda x: str(x), row)))
            f.write('\n')

    # Text accuracy
    resp = np.arange(1, len(attcopy_text_acc) + 1)
    pron_acc = np.concatenate(
        [[resp], [only_text_acc], [attacl_text_acc], [attcopy_text_acc], [profilereg_text_acc]])
    pron_acc = pron_acc.transpose().tolist()

    with open(os.path.join(OUTPUT_PATH, 'r_text_acc.csv'), 'w') as f:
        f.write('resp;onlynames;attentionacl;attentioncopy;profilereg\n')
        for row in pron_acc:
            f.write(';'.join(map(lambda x: str(x), row)))
            f.write('\n')

    # String edit distance
    resp = np.arange(1, len(y_real) + 1)
    r_distances = np.concatenate(
        [[resp], [only_distances], [attacl_distances], [attcopy_distances], [profilereg_distances]])
    r_distances = r_distances.transpose().tolist()

    with open(os.path.join(OUTPUT_PATH, 'r_distances.csv'), 'w') as f:
        f.write('resp;onlynames;attentionacl;attentioncopy;profilereg\n')
        for row in r_distances:
            f.write(';'.join(map(lambda x: str(x), row)))
            f.write('\n')


if __name__ == '__main__':
    run()
