__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 12/12/2017
Description:
    Automatic evaluation scripts to obtain accuracy, pronoun measures, string edit distance and BLEU scores.
    Moreover, it saves .csv files to test statistica significance in R.

    PYTHON VERSION :2.7

    DEPENDENCIES:
        NLTK:           http://www.nltk.org/
        SciKit learn:   http://scikit-learn.org/stable/

    UPDATE CONSTANT PATHS:
        ORIGINAL
        ONLYNAMES
        FERREIRA
        CATT
        HIERATT

        MULTIBLEU
"""

from nltk.metrics.distance import edit_distance
from sklearn.metrics import classification_report

import numpy as np
import cPickle as p
import os

# ORIGINAL
ORIGINAL = 'data/test/data.cPickle'
ORIGINAL_INFO = 'data/test/info.txt'
# ONLY NAMES RESULTS PATH
ONLYNAMES = 'eval/onlynames.cPickle'
# FERREIRA RESULTS PATH
FERREIRA = 'eval/ferreira.cPickle'
# NEURAL-SEQ2SEQ RESULTS PATH
SEQ2SEQ = 'eval/seq2seq/results/test_best_1_300_512_3_False_5/0'
# NEURAL-CATT RESULTS PATH
CATT = 'eval/att/results/test_best_1_300_512_512_3_False_5/0'
# NEURAL-HIERATT RESULTS PATH
HIERATT = 'eval/hier/results/test_best_1_300_512_512_2_False_1/0'

MULTIBLEU = 'eval/multi-bleu.perl'

def load_models():
    original = p.load(open(ORIGINAL))

    with open(ORIGINAL_INFO) as f:
        original_info = f.read().split('\n')
        original_info = map(lambda x: x.split(), original_info)

    # ONLY NAMES RESULTS AND GOLD-STANDARDS
    only = p.load(open(ONLYNAMES))

    # FERREIRA ET AL., 2016 RESULTS
    ferreira = p.load(open(FERREIRA))

    # NEURAL SEQ2SEQ RESULTS
    with open(SEQ2SEQ) as f:
        y_seq2seq = f.read().decode('utf-8').lower().split('\n')

    # NEURAL CATT RESULTS
    with open(CATT) as f:
        y_catt = f.read().decode('utf-8').lower().split('\n')

    # NEURAL HIERATT RESULTS
    with open(HIERATT) as f:
        y_hieratt = f.read().decode('utf-8').lower().split('\n')

    return original, original_info, only, ferreira, y_seq2seq, y_catt, y_hieratt

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
            wrong.append({'real': real, 'pred':pred})
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

    print('ACCURACY: ', str(round(num/dem, 4)))
    print('DISTANCE: ', str(round(np.mean(edit_distances), 4)))
    print('PRONOUN ACCURACY: ', str(round(pronoun_num/pronoun_dem, 4)))
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
        domain = info[i][1]
        if domain not in domains:
            domains[domain] = {'num':0.0, 'dem':0.00000001, 'pronoun_num':0.0, 'pronoun_dem':0.00000001, 'distance':[]}

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
        print('ACCURACY: ', str(round(domains[domain]['num']/domains[domain]['dem'], 4)))
        print('DISTANCE: ', str(round(np.mean(domains[domain]['distance']), 4)))
        print('PRONOUN ACCURACY: ', str(round(domains[domain]['pronoun_num']/domains[domain]['pronoun_dem'], 4)))

def evaluate_text(data, y_pred):
    originals = []
    templates = []
    text_acc = []

    for i, reference in enumerate(data):
        reference['pred'] = y_pred[i]

    text_ids = sorted(list(set(map(lambda x: x['text_id'], data))))
    for text_id in text_ids:
        references = filter(lambda x: x['text_id'] == text_id, data)
        references = sorted(references, key=lambda x: x['general_pos'])

        text = references[0]['text'].lower()
        template = references[0]['pre_context'] + ' ' + references[0]['entity'] + ' ' + references[0]['pos_context']

        for reference in references:
            entity = reference['entity'] + ' '

            refex = '~'.join(reference['pred'].replace('eos', '').strip().split()) + ' '
            template = template.replace(entity, refex, 1)

        originals.append(text)
        templates.append(template.replace('_', ' ').replace('~', ' ').replace('eos', '').strip())

    # Original accuracy
    num = len(filter(lambda x: x[0].lower().replace('@', '')==x[1].lower().replace('@', ''), zip(originals, templates)))
    dem = len(originals)
    for original, template in zip(originals, templates):
        if original.lower().replace('@', '')==template.lower().replace('@', ''):
            num += 1
            text_acc.append(1)
        else:
            text_acc.append(0)
        dem+= 1


    with open('reference', 'w') as f:
        f.write('\n'.join(originals).lower().replace('@', '').encode('utf-8'))

    with open('output', 'w') as f:
        f.write('\n'.join(templates).lower().encode('utf-8'))

    os.system('perl ' + MULTIBLEU + ' reference < output')
    os.remove('reference')
    os.remove('output')

    with open('eval/stats/refs.txt', 'w') as f:
        f.write('\n'.join(originals).encode('utf-8'))

    return originals, templates, num, dem, text_acc

def model_report(model_name, original, y_real, y_pred):
    print model_name
    wrong, edit_distances, pron_acc = evaluate_references(y_real, y_pred)
    print '\n'
    originals, templates, num, dem, text_acc = evaluate_text(original, y_pred)
    print('TEXT ACCURACY: ', str(round(float(num)/dem, 4)), str(num), str(dem))
    print 10 * '-'
    return originals, templates, edit_distances, pron_acc, text_acc

def run():
    original, original_info, only, ferreira, y_seq2seq, y_catt, y_hieratt = load_models()

    y_real = map(lambda x: x['refex'].lower(), original)
    y_only = map(lambda x: x['y_pred'], only)
    # RESULTS ARE SAVE IN A DIFFERENT ORDER THAN OTHERS
    # y_real_ferreira = map(lambda x: x['refex'].lower(), ferreira)
    _ferreira = []
    for inst in original:
        reference = filter(lambda x: x['text_id'] == inst['text_id'] and
                         x['sentence'] == inst['sentence'] and
                         x['pos'] == inst['pos'] and
                         x['refex'] == inst['refex'], ferreira)[0]
        _ferreira.append(reference)
    ferreira = _ferreira
    y_ferreira = map(lambda x: x['realization'].lower(), ferreira)


    # ONLY - NAMES ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY
    originals, templates, only_distances, only_pron_acc, only_text_acc = model_report('ONLY NAMES', original, y_real, y_only)
    with open('eval/stats/only.txt', 'w') as f:
        f.write('\n'.join(templates).encode('utf-8'))

    only_ref_acc = []
    for real, pred in zip(y_real, y_only):
        if real.replace('eos', '').strip() == pred.replace('eos', '').strip():
            only_ref_acc.append(1)
        else:
            only_ref_acc.append(0)

    # FERREIRA ET AL., 2016 - ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY
    originals, templates, ferreira_distances, ferreira_pron_acc, ferreira_text_acc = model_report('FERREIRA ET AL. 2016', original, y_real, y_ferreira)
    with open('eval/stats/ferreira.txt', 'w') as f:
        f.write('\n'.join(templates).encode('utf-8'))

    ferreira_ref_acc = []
    for real, pred in zip(y_real, y_ferreira):
        if real.replace('eos', '').strip() == pred.replace('eos', '').strip():
            ferreira_ref_acc.append(1)
        else:
            ferreira_ref_acc.append(0)

    # SEQ2SEQ - ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY
    originals, templates, seq2seq_distances, seq2seq_pron_acc, seq2seq_text_acc = model_report('NEURAL SEQ2SEQ', original, y_real, y_seq2seq)
    with open('eval/stats/seq2seq.txt', 'w') as f:
        f.write('\n'.join(templates).encode('utf-8'))

    seq2seq_ref_acc = []
    for real, pred in zip(y_real, y_seq2seq):
        if real.replace('eos', '').strip() == pred.replace('eos', '').strip():
            seq2seq_ref_acc.append(1)
        else:
            seq2seq_ref_acc.append(0)

    # CATT - ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY
    originals, templates, catt_distances, catt_pron_acc, catt_text_acc = model_report('NEURAL CATT', original, y_real, y_catt)
    with open('eval/stats/catt.txt', 'w') as f:
        f.write('\n'.join(templates).encode('utf-8'))

    catt_ref_acc = []
    for real, pred in zip(y_real, y_catt):
        if real.replace('eos', '').strip() == pred.replace('eos', '').strip():
            catt_ref_acc.append(1)
        else:
            catt_ref_acc.append(0)

    # HIER - ACCURACY, STRING EDIT DISTANCE AND PRONOUN ACCURACY
    originals, templates, hieratt_distances, hier_pron_acc, hier_text_acc = model_report('NEURAL HIERATT', original, y_real, y_hieratt)
    with open('eval/stats/hieratt.txt', 'w') as f:
        f.write('\n'.join(templates).encode('utf-8'))

    hier_ref_acc = []
    for real, pred in zip(y_real, y_hieratt):
        if real.replace('eos', '').strip() == pred.replace('eos', '').strip():
            hier_ref_acc.append(1)
        else:
            hier_ref_acc.append(0)

    # Save files to perform statistical tests in R
    # Reference accuracy file
    resp = np.arange(1, len(y_real)+1)
    ref_acc = np.concatenate([[resp], [only_ref_acc], [ferreira_ref_acc], [seq2seq_ref_acc], [catt_ref_acc], [hier_ref_acc]])
    ref_acc = ref_acc.transpose().tolist()

    with open('eval/stats/r_ref_acc.csv', 'w') as f:
        f.write('resp;only;ferreira;seq2seq;catt;hieratt\n')
        for row in ref_acc:
            f.write(';'.join(map(lambda x: str(x), row)))
            f.write('\n')

    # Pronoun accuracy
    resp = np.arange(1, len(only_pron_acc)+1)
    pron_acc = np.concatenate([[resp], [only_pron_acc], [ferreira_pron_acc], [seq2seq_pron_acc], [catt_pron_acc], [hier_pron_acc]])
    pron_acc = pron_acc.transpose().tolist()

    with open('eval/stats/r_pron_acc.csv', 'w') as f:
        f.write('resp;only;ferreira;seq2seq;catt;hieratt\n')
        for row in pron_acc:
            f.write(';'.join(map(lambda x: str(x), row)))
            f.write('\n')

    # Text accuracy
    resp = np.arange(1, len(only_text_acc)+1)
    pron_acc = np.concatenate([[resp], [only_text_acc], [ferreira_text_acc], [seq2seq_text_acc], [catt_text_acc], [hier_text_acc]])
    pron_acc = pron_acc.transpose().tolist()

    with open('eval/stats/r_text_acc.csv', 'w') as f:
        f.write('resp;only;ferreira;seq2seq;catt;hieratt\n')
        for row in pron_acc:
            f.write(';'.join(map(lambda x: str(x), row)))
            f.write('\n')

    # String edit distance
    resp = np.arange(1, len(y_real)+1)
    r_distances = np.concatenate([[resp], [only_distances], [ferreira_distances], [seq2seq_distances], [catt_distances], [hieratt_distances]])
    r_distances = r_distances.transpose().tolist()

    with open('eval/stats/r_distances.csv', 'w') as f:
        f.write('resp;only;ferreira;seq2seq;catt;hieratt\n')
        for row in r_distances:
            f.write(';'.join(map(lambda x: str(x), row)))
            f.write('\n')

if __name__ == '__main__':
    run()