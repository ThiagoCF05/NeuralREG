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
EVAL_PATH = 'coling/'
OUTPUT_PATH = 'stats/coling/v1.5/ablation/test/'
GOLD_PATH = '../humaneval/trials/coling/'

# ORIGINAL
ORIGINAL = os.path.join(GOLD_PATH, 'gold-ids.json')
# TRAIN
TRAIN = os.path.join(DATA_PATH, 'train.json')

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


def load_models():
    original = json.load(open(ORIGINAL, encoding='utf-8'))
    train = json.load(open(TRAIN, encoding='utf-8'))

    # ATTENTION_ACL RESULTS
    with open(ATTENTION_ACL, encoding='utf-8') as f:
        y_attacl = f.read().lower().split('\n')

    # ATTENTION COPY RESULTS
    with open(ATTENTION_COPY, encoding='utf-8') as f:
        y_attcopy = f.read().lower().split('\n')

    # NEURAL ATTENTION_PRECONTEXT RESULTS
    with open(ATTENTION_PRECONTEXT, encoding='utf-8') as f:
        y_attprectxt = f.read().lower().split('\n')

    # NEURAL ATTENTION_COPY_CONTEXT RESULTS
    with open(ATTENTION_COPY_CONTEXT, encoding='utf-8') as f:
        y_attctxt = f.read().lower().split('\n')

    # NEURAL ATTENTION_COPY_PRECONTEXT RESULTS
    with open(ATTENTION_COPY_PRECONTEXT, encoding='utf-8') as f:
        y_attcpre = f.read().lower().split('\n')

    # NEURAL ATTENTION_COPY_POSCONTEXT RESULTS
    with open(ATTENTION_COPY_POSCONTEXT, encoding='utf-8') as f:
        y_attcpos = f.read().lower().split('\n')

    return original, train, y_attacl, y_attcopy, y_attprectxt, y_attctxt, y_attcpre, y_attcpos


def generate_domains(original, train, y_attacl, y_attcopy, y_attprectxt, y_attctxt, y_attcpre, y_attcpos):
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

    # ATTENTION PRE CONTEXT
    seen, unseen, _, _ = separate_domains(original, y_attprectxt, domains)
    with codecs.open(ATTENTION_PRECONTEXT + '_seen', 'w', encoding='utf8') as f:
        f.write('\n'.join(seen).lower())
    with codecs.open(ATTENTION_PRECONTEXT + '_unseen', 'w', encoding='utf8') as f:
        f.write('\n'.join(unseen).lower())

    # ATTENTION_COPY_CONTEXT
    seen, unseen, _, _ = separate_domains(original, y_attctxt, domains)
    with codecs.open(ATTENTION_COPY_CONTEXT + '_seen', 'w', encoding='utf8') as f:
        f.write('\n'.join(seen).lower())
    with codecs.open(ATTENTION_COPY_CONTEXT + '_unseen', 'w', encoding='utf8') as f:
        f.write('\n'.join(unseen).lower())

    # ATTENTION COPY PRE CONTEXT
    seen, unseen, _, _ = separate_domains(original, y_attcpre, domains)
    with codecs.open(ATTENTION_COPY_PRECONTEXT + '_seen', 'w', encoding='utf8') as f:
        f.write('\n'.join(seen).lower())
    with codecs.open(ATTENTION_COPY_PRECONTEXT + '_unseen', 'w', encoding='utf8') as f:
        f.write('\n'.join(unseen).lower())

    # ATTENTION COPY POS CONTEXT
    seen, unseen, _, _ = separate_domains(original, y_attcpos, domains)
    with codecs.open(ATTENTION_COPY_POSCONTEXT + '_seen', 'w', encoding='utf8') as f:
        f.write('\n'.join(seen).lower())
    with codecs.open(ATTENTION_COPY_POSCONTEXT + '_unseen', 'w', encoding='utf8') as f:
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
    original, train, y_attacl, y_attcopy, y_attprectxt, y_attctxt, y_attcpre, y_attcpos = load_models()

    generate_domains(original, train, y_attacl, y_attcopy, y_attprectxt, y_attctxt, y_attcpre, y_attcpos)
