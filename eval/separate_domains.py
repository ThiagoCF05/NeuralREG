import codecs
import json

"""
Date: 29/03/2020
Description:
    Scripts used to separate unseen and seen domains

    PYTHON VERSION :3
 
"""

__author__ = ''

import os

DATA_PATH = '../data/v1.5/'
EVAL_PATH = 'data/'
OUTPUT_PATH = 'stats/beta/v1.5/'
GOLD_PATH = '../humaneval/trials/beta/'

# ORIGINAL
ORIGINAL = os.path.join(GOLD_PATH, 'gold-ids.json')
# TRAIN
TRAIN = os.path.join(DATA_PATH, 'train.json')

# ONLY NAMES RESULTS PATH
ONLYNAMES = EVAL_PATH + 'onlynames/results/onlynames.json'
ONLYNAMES_OUT = EVAL_PATH + 'onlynames/results/onlynames'
# ATTENTION ACL NAMES RESULTS PATH
ATTENTION_ACL = EVAL_PATH + 'attention_acl/results/test_1'
# ATTENTION COPY NAMES RESULTS FOLDER
ATTENTION_COPY = EVAL_PATH + 'attention/results/test_1'
# PROFILEREG
PROFILEREG = EVAL_PATH + 'profilereg/results/preds.txt'
PROFILEREG_OUT = EVAL_PATH + 'profilereg/results/preds'


def load_models():
    train = json.load(open(TRAIN, encoding='utf-8'))

    original = json.load(open(ORIGINAL, encoding='utf-8'))
    # y_original = []
    # for i, row in enumerate(original):
    #     refex = [w.lower() for w in row['refex']]
    #     refex = ' '.join(refex).strip()
    #     y_original.append(refex)

    # ONLY NAMES RESULTS
    only = json.load(open(ONLYNAMES, encoding='utf-8'))
    y_only = list(map(lambda x: x['y_pred'].lower().strip(), only))

    # ATTENTION_ACL RESULTS
    with open(ATTENTION_ACL, encoding='utf-8') as f:
        y_attacl = f.read().lower().split('\n')

    # ATTENTION COPY RESULTS
    with open(ATTENTION_COPY, encoding='utf-8') as f:
        y_attcopy = f.read().lower().split('\n')

    # PROFILEREG RESULTS
    with open(PROFILEREG, encoding='utf-8') as f:
        y_profilereg = f.read().lower().split('\n')

    return original, train, y_only, y_attacl, y_attcopy, y_profilereg


def generate_domains(original, train, y_only, y_attacl, y_attcopy, y_profilereg):
    train_domains = set([e['category'] for e in train])
    original_domains = set([e['category'] for e in original])
    domains = list(set(original_domains) - set(train_domains))

    # ATTENTION_ACL
    seen, unseen, test_seen, test_unseen = separate_domains(original, y_attacl, domains)

    json.dump(test_seen, open(os.path.join(DATA_PATH, 'test_seen.json'), 'w'))
    json.dump(test_unseen, open(os.path.join(DATA_PATH, 'test_unseen.json'), 'w'))

    with codecs.open(ATTENTION_ACL + '_seen', 'w', encoding='utf8') as f:
        f.write('\n'.join(seen).lower())
    with codecs.open(ATTENTION_ACL + '_unseen', 'w', encoding='utf8') as f:
        f.write('\n'.join(unseen).lower())

    # ATTENTION COPY
    seen, unseen, _, _ = separate_domains(original, y_attcopy, domains)
    with codecs.open(ATTENTION_COPY + '_seen', 'w', encoding='utf8') as f:
        f.write('\n'.join(seen).lower())
    with codecs.open(ATTENTION_COPY + '_unseen', 'w', encoding='utf8') as f:
        f.write('\n'.join(unseen).lower())

    # ONLYNAMES
    seen, unseen, _, _ = separate_domains(original, y_only, domains)
    json.dump(seen, open(os.path.join(ONLYNAMES_OUT + '_seen.json'), 'w'))
    json.dump(unseen, open(os.path.join(ONLYNAMES_OUT + '_unseen.json'), 'w'))

    # PROFILEREG
    seen, unseen, _, _ = separate_domains(original, y_profilereg, domains)
    with codecs.open(PROFILEREG_OUT + '_seen.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(seen).lower())
    with codecs.open(PROFILEREG_OUT + '_unseen.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(unseen).lower())


def separate_domains(original, model, domains):
    seen, unseen = [], []
    test_seen, test_unseen = [], []
    for i, reference in enumerate(original):
        if reference['category'] in domains:
            unseen.append(model[i])
            test_unseen.append(reference)
        else:
            seen.append(model[i])
            test_seen.append(reference)
    return seen, unseen, test_seen, test_unseen


if __name__ == '__main__':
    original, train, y_only, y_attacl, y_attcopy, y_profilereg = load_models()

    generate_domains(original, train, y_only, y_attacl, y_attcopy, y_profilereg)
