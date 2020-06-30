__author__ = 'thiagocastroferreira'

import json
import os

from random import shuffle

if __name__ == '__main__':

    with open('trials/beta/sample-ids.txt', encoding='utf-8') as f:
        ids = f.read().split('\n')

    entries = json.load(open(os.path.join('trials/beta', 'trials_b.json'), encoding='utf-8'))

    source, texts = [], []

    for i in ids:
        e = [w for w in entries if w['eid'] == i][0]

    # for e in entries:
        texts.append(
            "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(e['eid'], e['lid'], e['category'], e['size'], 'only', e['only'],
                                                       e['source']).replace('<TRIPLE> ', '').replace('</TRIPLE>',
                                                                                                    '.'))
        texts.append(
            "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(e['eid'], e['lid'], e['category'], e['size'], 'attacl', e['attacl'],
                                                       e['source']).replace('<TRIPLE> ', '').replace('</TRIPLE>',
                                                                                                    '.'))
        texts.append(
            "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(e['eid'], e['lid'], e['category'], e['size'], 'attcopy', e['attcopy'],
                                                       e['source']).replace('<TRIPLE> ', '').replace('</TRIPLE>',
                                                                                                    '.'))
        texts.append("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(e['eid'], e['lid'], e['category'], e['size'], 'profilereg',
                                                        e['profilereg'], e['source']).replace('<TRIPLE> ', '').replace('</TRIPLE>',
                                                                                            '.'))

    with open(os.path.join('trials/beta', 'samples_b_ordered.csv'), 'w') as f:
        f.write('eid\tlid\tcategory\tsize\tmodel\ttext\tsource\n')
        for e in texts:
            f.write(e + '\n')

    shuffle(texts)

    with open(os.path.join('trials/beta', 'samples_b.csv'), 'w') as f:
        f.write('eid\tlid\tcategory\tsize\tmodel\ttext\tsource\n')
        for e in texts:
            f.write(e + '\n')
