__author__ = 'thiagocastroferreira'

import json
import os

from random import shuffle

if __name__ == '__main__':

    with open('trials/coling/sample-ids.txt', encoding='utf-8') as f:
        ids = f.read().split('\n')

    entries = json.load(open(os.path.join('trials/coling', 'samples.json'), encoding='utf-8'))
    source, texts = [], []

    for e in entries:
        texts.append("{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(e['eid'], e['lid'], e['row'], e['size'], e['only'],
                                                           e['source'].replace('<TRIPLE> ', '').replace('</TRIPLE>',
                                                                                                        '.')))
        texts.append("{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(e['eid'], e['lid'], e['row'], e['size'], e['attacl'],
                                                           e['source'].replace('<TRIPLE> ', '').replace('</TRIPLE>',
                                                                                                        '.')))
        texts.append("{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(e['eid'], e['lid'], e['row'], e['size'], e['attcopy'],
                                                           e['source'].replace('<TRIPLE> ', '').replace('</TRIPLE>',
                                                                                                        '.')))
        texts.append("{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(e['eid'], e['lid'], e['row'], e['size'], e['profilereg'],
                                                           e['source'].replace('<TRIPLE> ', '').replace('</TRIPLE>',
                                                                                                        '.')))

    shuffle(texts)

    with open(os.path.join('trials/coling', 'samples_c.csv'), 'w') as f:
        f.write('original;only;attacl,attcopy,profilereg\n')
        for e in entries:
            f.write(e['original'] + ';' + e['only'] + ';' + e['attacl'] + ';' + e['attcopy'] + ';' + e['profilereg'])
            f.write('\n')

    with open(os.path.join('trials/coling', 'samples.csv'), 'w') as f:
        f.write('eid\tlid\trow\tsize\ttext\tsource\n')
        for e in texts:
            f.write(e + '\n')
