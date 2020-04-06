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
        ATTENTION
        ATTENTION_ACL
        ATTENTION_PRECONTEXT
        ATTENTION_COPY_CONTEXT
        ATTENTION_COPY_PRECONTEXT
        ATTENTION_COPY_POSCONTEXT

        MULTIBLEU
"""

from nltk.metrics.distance import edit_distance
from sklearn.metrics import classification_report

import codecs
import json
import numpy as np
import os

DATA_PATH = '../data/v1.5/'
EVAL_PATH = 'coling/'
OUTPUT_PATH = 'stats/coling/v1.5/ablation/test'

# ORIGINAL
ORIGINAL = os.path.join(DATA_PATH, 'test.json')

# ONLY NAMES RESULTS PATH
ONLYNAMES = 'data/onlynames/results/onlynames.json'
# ATTENTION ACL NAMES RESULTS PATH
ATTENTION_ACL = EVAL_PATH + 'attention_acl/results/test_1'
# ATTENTION COPY NAMES RESULTS FOLDER
ATTENTION_COPY = EVAL_PATH + 'attention/results/test_1'
# ATTENTION PRECONTEXT RESULTS FOLDER
ATTENTION_PRECONTEXT = EVAL_PATH + 'attention_precontext/results/test_1'
# ATTENTION COPY CONTEXT RESULTS FOLDER
ATTENTION_COPY_CONTEXT = EVAL_PATH + 'attention_copy_context/results/test_1'
# ATTENTION COPY PRECONTEXT RESULTS FOLDER
ATTENTION_COPY_PRECONTEXT = EVAL_PATH + 'attention_copy_precontext/results/test_1'
# ATTENTION COPY POSCONTEXT RESULTS FOLDER
ATTENTION_COPY_POSCONTEXT = EVAL_PATH + 'attention_copy_poscontext/results/test_1'

MULTIBLEU = '../eval/multi-bleu.perl'


def load_models():
    original = json.load(open(ORIGINAL, encoding='utf-8'))

    # ONLY NAMES RESULTS
    only = json.load(open(ONLYNAMES, encoding='utf-8'))

    # ATTENTION_ACL RESULTS
    with open(ATTENTION_ACL, encoding='utf-8') as f:
        y_attacl = f.read().lower().split('\n')

    # ATTENTION COPY RESULTS
    with open(ATTENTION_COPY, encoding='utf-8') as f:
        y_attcopy = f.read().lower().split('\n')

    # NEURAL ATTENTION_PRECONTEXT RESULTS
    with open(ATTENTION_PRECONTEXT, encoding='utf-8') as f:
        y_attpre = f.read().lower().split('\n')

    # NEURAL ATTENTION_COPY_CONTEXT RESULTS
    with open(ATTENTION_COPY_CONTEXT, encoding='utf-8') as f:
        y_attctxt = f.read().lower().split('\n')

    # NEURAL ATTENTION_COPY_PRECONTEXT RESULTS
    with open(ATTENTION_COPY_PRECONTEXT, encoding='utf-8') as f:
        y_attcpre = f.read().lower().split('\n')

    # NEURAL ATTENTION_COPY_POSCONTEXT RESULTS
    with open(ATTENTION_COPY_POSCONTEXT, encoding='utf-8') as f:
        y_attcpos = f.read().lower().split('\n')

    return original, only, y_attacl, y_attcopy, y_attpre, y_attctxt, y_attcpre, y_attcpos


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

    for i, row in enumerate(data):
        pre_context = ' '.join(row['pre_context']).lower().replace('_', ' ').replace('~', ' ').replace('eos',
                                                                                                       '').strip()
        pos_context = ' '.join(row['pos_context']).lower().replace('_', ' ').replace('~', ' ').replace('eos',
                                                                                                       '').strip()
        refex = ' '.join(row['refex']).replace('_', ' ').lower().replace('~', ' ').replace('eos', '').strip()
        reference = row['pred'].strip()

        text = pre_context + ' ' + refex + ' ' + pos_context
        template = pre_context + ' ' + reference + ' ' + pos_context

        originals.append(text.strip())
        templates.append(template.strip())

    # Original accuracy
    _num = filter(lambda x: x[0].lower().replace('@', '') == x[1].lower().replace('@', ''), zip(originals, templates))
    num = len(list(_num))

    dem = len(originals)
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
    print('DOMAIN ACCURACY:')
    domain_evaluate(y_real, y_pred, original)
    print(10 * '-')

    return originals, templates, edit_distances, pron_acc, text_acc


def run():
    original, only, y_attacl, y_attcopy, y_attprectxt, y_attctxt, y_attcpre, y_attcpos = load_models()

    y_real = []
    for i, row in enumerate(original):
        refex = [w.lower() for w in row['refex']]
        refex = ' '.join(refex).strip()
        y_real.append(refex)

    y_only = list(map(lambda x: x['y_pred'].lower().strip(), only))

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

    # ATTENTION PRE CONTEXT - ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY
    originals, templates, attprectxt_distances, attprectxt_pron_acc, attprectxt_text_acc = model_report(
        'ATTENTION PRECONTEXT', original, y_real, y_attprectxt)

    with codecs.open(os.path.join(OUTPUT_PATH, 'attprectxt.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(templates).lower())

    attprectxt_ref_acc = []
    for real, pred in zip(y_real, y_attprectxt):
        if real.replace('eos', '').strip() == pred.replace('eos', '').strip():
            attprectxt_ref_acc.append(1)
        else:
            attprectxt_ref_acc.append(0)

    # ATTENTION_COPY_CONTEXT - ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY
    originals, templates, attctxt_distances, attctxt_pron_acc, attctxt_text_acc = model_report('ATTENTION COPY CONTEXT',
                                                                                               original, y_real,
                                                                                               y_attctxt)
    with codecs.open(os.path.join(OUTPUT_PATH, 'attctxt.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(templates).lower())

    attctxt_ref_acc = []
    for real, pred in zip(y_real, y_attctxt):
        if real.replace('eos', '').strip() == pred.replace('eos', '').strip():
            attctxt_ref_acc.append(1)
        else:
            attctxt_ref_acc.append(0)

    # ATTENTION COPY PRE CONTEXT - ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY
    originals, templates, attcpre_distances, attcpre_pron_acc, attcpre_text_acc = model_report(
        'ATTENTION COPY PRECONTEXT', original, y_real, y_attcpre)

    with codecs.open(os.path.join(OUTPUT_PATH, 'attcpre.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(templates).lower())

    attcpre_ref_acc = []
    for real, pred in zip(y_real, y_attcpre):
        if real.replace('eos', '').strip() == pred.replace('eos', '').strip():
            attcpre_ref_acc.append(1)
        else:
            attcpre_ref_acc.append(0)

    # ATTENTION COPY POS CONTEXT - ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY
    originals, templates, attcpos_distances, attcpos_pron_acc, attcpos_text_acc = model_report(
        'ATTENTION COPY POSCONTEXT', original, y_real, y_attcpos)

    with codecs.open(os.path.join(OUTPUT_PATH, 'attcpos.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(templates).lower())

    attcpos_ref_acc = []
    for real, pred in zip(y_real, y_attcpos):
        if real.replace('eos', '').strip() == pred.replace('eos', '').strip():
            attcpos_ref_acc.append(1)
        else:
            attcpos_ref_acc.append(0)

    # Save files to perform statistical tests in R
    # Reference accuracy file
    resp = np.arange(1, len(y_real) + 1)
    ref_acc = np.concatenate(
        [[resp], [attacl_ref_acc], [attcopy_ref_acc], [attprectxt_ref_acc], [attctxt_ref_acc], [attcpre_ref_acc], [attcpos_ref_acc]])
    ref_acc = ref_acc.transpose().tolist()

    with open(os.path.join(OUTPUT_PATH, 'r_ref_acc.csv'), 'w') as f:
        f.write('resp;attacl;attcopy;attprectxt;attctxt;attcpre;attcpos\n')
        for row in ref_acc:
            f.write(';'.join(map(lambda x: str(x), row)))
            f.write('\n')

    # Pronoun accuracy
    resp = np.arange(1, len(attcopy_pron_acc) + 1)
    pron_acc = np.concatenate(
        [[resp], [attacl_pron_acc], [attcopy_pron_acc], [attprectxt_pron_acc], [attctxt_pron_acc], [attcpre_pron_acc], [attcpos_pron_acc]])
    pron_acc = pron_acc.transpose().tolist()

    with open(os.path.join(OUTPUT_PATH, 'r_pron_acc.csv'), 'w') as f:
        f.write('resp;attacl;attcopy;attprectxt;attctxt;attcpre;attcpos\n')
        for row in pron_acc:
            f.write(';'.join(map(lambda x: str(x), row)))
            f.write('\n')

    # Text accuracy
    resp = np.arange(1, len(attcopy_text_acc) + 1)
    pron_acc = np.concatenate(
        [[resp], [attacl_text_acc], [attcopy_text_acc], [attprectxt_ref_acc], [attctxt_text_acc], [attcpre_text_acc], [attcpos_text_acc]])
    pron_acc = pron_acc.transpose().tolist()

    with open(os.path.join(OUTPUT_PATH, 'r_text_acc.csv'), 'w') as f:
        f.write('resp;attacl;attcopy;attprectxt;attctxt;attcpre;attcpos\n')
        for row in pron_acc:
            f.write(';'.join(map(lambda x: str(x), row)))
            f.write('\n')

    # String edit distance
    resp = np.arange(1, len(y_real) + 1)
    r_distances = np.concatenate(
        [[resp], [attacl_distances], [attcopy_distances], [attprectxt_distances],  [attctxt_distances], [attcpre_distances],
         [attcpos_distances]])
    r_distances = r_distances.transpose().tolist()

    with open(os.path.join(OUTPUT_PATH, 'r_distances.csv'), 'w') as f:
        f.write('resp;attacl;attcopy;attprectxt;attctxt;attcpre;attcpos\n')
        for row in r_distances:
            f.write(';'.join(map(lambda x: str(x), row)))
            f.write('\n')


if __name__ == '__main__':
    run()
