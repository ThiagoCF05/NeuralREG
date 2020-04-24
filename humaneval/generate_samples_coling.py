__author__ = 'thiagocastroferreira'

import json
import os

from random import shuffle

if __name__ == '__main__':

    with open('trials/coling/sample-ids.txt', encoding='utf-8') as f:
        ids = f.read().split('\n')

    entries = json.load(open('trials/coling/gold-ids.json', encoding='utf-8'))
    gold = json.load(open('trials/coling/gold.json', encoding='utf-8'))
    texts = json.load(open('trials/coling/trials.json', encoding='utf-8'))

    entry_ids, samples = [], []

    samples_ids = [eid for eid in ids]

    # for sid, sample in enumerate(samples_ids):
    #     eid = sample.replace('Id', '').strip()

    # triples = {i: t for i, t in enumerate(entries) if t['eid'] in samples_ids}
    triples = []
    for i, t in enumerate(entries):
        if t['eid'] in samples_ids:
            lex = [lex for lex in gold if lex['eid'] == t['eid']][0]
            triples.append({'row': i,
                            'eid': t['eid'],
                            'lid': t['lid'],
                            'source': lex['source'],
                            'entity': t['entity'],
                            'size': lex['size']})
            print(t['eid'], t['lid'], lex['size'])

    tids, eids = [], []
    for t in triples:
        lexids = [lex['row'] for lex in triples if lex['eid'] == t['eid']]
        shuffle(lexids)
        row = lexids[0]

        if t['eid'] not in eids:
            eids.append(t['eid'])
            tids.append({t['eid']: row})
            # print(t['eid'], ';', row)

            text = texts[row]
            text['eid'] = t['eid']
            text['lid'] = t['lid']
            text['row'] = row
            text['size'] = t['size']
            text['source'] = ' '.join(t['source']).strip()

            samples.append(text)

    sample_texts = sorted(samples, key=lambda x: float(x['eid'].replace('Id', '')))
    json.dump(sample_texts, open(os.path.join('trials/coling/', 'samples.json'), 'w'))
