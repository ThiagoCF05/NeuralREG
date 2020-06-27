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
TRIALS_PATH = 'trials/beta/'
EVAL_PATH = '../eval/stats/beta/v1.5/'
OUTPUT_PATH = '../eval/stats/beta/v1.5/'

# ORIGINAL
ORIGINAL = os.path.join(DATA_PATH, 'test.json')

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

# TEST INFO
TEST_INFO = DATA_PATH + 'test_info.json'
# GOLD - Source info
GOLD_INFO = TRIALS_PATH + 'gold.json'

def load_models():
    original = json.load(open(ORIGINAL, encoding='utf-8'))
    test_info = json.load(open(TEST_INFO, encoding='utf-8'))
    gold_info = json.load(open(GOLD_INFO, encoding='utf-8'))
    for i, row in enumerate(original):
        original[i]['eid'] = test_info[i]['eid']
        original[i]['lid'] = test_info[i]['lid']
        original[i]['category'] = test_info[i]['category']
        original[i]['text'] = ' '.join(test_info[i]['targets'][0]['output'])

        source = [w for w in gold_info if w['eid'] == original[i]['eid']]
        original[i]['size'] = source[0]['size']
        original[i]['source'] = ' '.join(source[0]['source'])

    del test_info, gold_info

    y_original = []
    for i, row in enumerate(original):
        refex = ' '.join(row['refex']).lower().strip()
        y_original.append(refex)

    # REFS
    with open(REFS, encoding='utf-8') as f:
        y_refs = f.read().lower().split('\n')

    with open(ONLYNAMES, encoding='utf-8') as f:
        y_only = f.read().lower().split('\n')

    # ATTENTION ACL RESULTS
    with open(ATTENTION_ACL, encoding='utf-8') as f:
        y_attacl = f.read().lower().split('\n')

    # ATTENTION COPY RESULTS
    with open(ATTENTION_COPY, encoding='utf-8') as f:
        y_attcopy = f.read().lower().split('\n')

    # PROFILEREG RESULTS
    with open(PROFILEREG, encoding='utf-8') as f:
        y_profilereg = f.read().lower().split('\n')

    return original, y_original, y_only, y_attacl, y_attcopy, y_profilereg, y_refs


def save_trials(data, refs, only, attacl, attcopy, profilereg):
    if not os.path.exists('trials/beta/'):
        os.mkdir('trials/beta/')

    text_ids = [w['eid'] for w in data]
    text_ids = sorted(list(set(text_ids)))

    trials, samples = [], []
    for i, text_id in enumerate(text_ids):
        eid = text_id
        original = [w for w in data if w['eid'] == eid]

        trial = {
            'eid': original[0]['eid'],
            'lid': original[0]['lid'],
            'category': original[0]['category'],
            'size': original[0]['size'],
            'source': original[0]['source'],
            'text': original[0]['text'],
            'original': refs[i],
            'only': only[i],
            'attacl': attacl[i],
            'attcopy': attcopy[i],
            'profilereg': profilereg[i]
        }
        trials.append(trial)

        if (only[i] != attcopy[i]) and (attacl[i] != attcopy[i]) and (profilereg[i] != attcopy[i]) and \
                (only[i] != refs[i]) and (attacl[i] != refs[i]) and (profilereg[i] != refs[i]):
            if trial not in samples:
                samples.append(trial)

    json.dump(trials, open(os.path.join('trials/beta/', 'samples_paper.json'), 'w'))
    with open(os.path.join('trials/beta', 'samples_paper.csv'), 'w') as f:
        f.write('eid\tlid\tsize\tcategory\toriginal\tonly\tattacl\tattcopy\tprofilereg\n')
        for e in samples:
            f.write(e['eid'] + '\t' + e['lid'] + '\t' + e['size'] + '\t' + e['category'] + '\t' + e['original']
                    + '\t' + e['only'] + '\t' + e['attacl'] + '\t' + e['attcopy'] + '\t' + e['profilereg'] + '\n')

    return trials


if __name__ == '__main__':
    original, y_original, y_only, y_attacl, y_attcopy, y_profilereg, y_refs = load_models()

    if not os.path.exists('trials/beta/'):
        os.mkdir('trials/beta')

    trials = save_trials(original, y_refs, y_only, y_attacl, y_attcopy, y_profilereg)

    json.dump(trials, open(os.path.join('trials/beta/', 'trials_b.json'), 'w'))
