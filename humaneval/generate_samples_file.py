__author__ = 'thiagocastroferreira'

import json
import os

from random import shuffle

if __name__ == '__main__':

    with open('trials/beta/sample-ids.txt', encoding='utf-8') as f:
        ids = f.read().split('\n')

    entries = json.load(open(os.path.join('trials/beta', 'samples.json'), encoding='utf-8'))
    source, texts = [], []

    for e in entries:
        texts.append("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(e['eid'], e['lid'], e['row'], e['size'], 'only', e['only'],
                                                           e['source'].replace('<TRIPLE> ', '').replace('</TRIPLE>',
                                                                                                        '.')))
        texts.append("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(e['eid'], e['lid'], e['row'], e['size'], 'attacl', e['attacl'],
                                                           e['source'].replace('<TRIPLE> ', '').replace('</TRIPLE>',
                                                                                                        '.')))
        texts.append("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(e['eid'], e['lid'], e['row'], e['size'], 'attcopy', e['attcopy'],
                                                           e['source'].replace('<TRIPLE> ', '').replace('</TRIPLE>',
                                                                                                        '.')))
        texts.append("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(e['eid'], e['lid'], e['row'], e['size'], 'profilereg', e['profilereg'],
                                                           e['source'].replace('<TRIPLE> ', '').replace('</TRIPLE>',
                                                                                                        '.')))

    shuffle(texts)

    with open(os.path.join('trials/beta', 'samples.csv'), 'w') as f:
        f.write('eid\tlid\trow\tsize\tmodel\ttext\tsource\n')
        for e in texts:
            f.write(e + '\n')
