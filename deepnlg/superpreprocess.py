__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 25/03/2019
Description:
    Preprocessing super class
"""

import sys

sys.path.append('./')

import json
import os


class Preprocess:
    def __init__(self, data_path, write_path):
        self.write_path = write_path
        self.data_path = data_path

    def run(self, traindata, devdata, testdata):
        if not os.path.exists(self.write_path):
            os.mkdir(self.write_path)
        path = os.path.join(self.write_path, 'transformer')
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(self.write_path, 'transformer', 'model')
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(self.write_path, 'rnn')
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(self.write_path, 'rnn', 'model')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(self.write_path, 'data')
        self.save(data=traindata, path=path, fname='train')
        self.save(data=devdata, path=path, fname='dev')
        self.save(data=testdata, path=path, fname='test')

        json.dump(traindata, open(os.path.join(path, 'train.json'), 'w'))
        json.dump(devdata, open(os.path.join(path, 'dev.json'), 'w'))
        json.dump(testdata, open(os.path.join(path, 'test.json'), 'w'))

        self.stats(traindata, path=os.path.join(path, 'train.stats'))
        self.stats(devdata, path=os.path.join(path, 'dev.stats'))
        self.stats(testdata, path=os.path.join(path, 'test.stats'))

    def stats(self, data, path):
        srcsize, srcvocab = [], []
        trgtsize, trgtvocab = [], []

        for entry in data:
            source = entry['source']
            srcsize.append(len(source))
            srcvocab.extend(source)
            for target in entry['targets']:
                trgtsize.append(len(target['output']))
                trgtvocab.extend(target['output'])

        srcvocabsize = len(set(srcvocab))
        trgtvocabsize = len(set(trgtvocab))

        with open(path, 'w') as f:
            f.write('Data size: ' + str(len(data)))
            f.write('\n')
            f.write('Source: ')
            f.write('\n')
            f.write('Avg. Size: ' + str(round(sum(srcsize) / len(srcsize), 2)))
            f.write('\n')
            f.write('Max. Size: ' + str(max(srcsize)))
            f.write('\n')
            f.write('Min. Size: ' + str(min(srcsize)))
            f.write('\n')
            f.write('Vocab Size: ' + str(srcvocabsize))
            f.write('\n')
            f.write('Target: ')
            f.write('\n')
            f.write('Avg. Size: ' + str(round(sum(trgtsize) / len(trgtsize), 2)))
            f.write('\n')
            f.write('Max. Size: ' + str(max(trgtsize)))
            f.write('\n')
            f.write('Min. Size: ' + str(min(trgtsize)))
            f.write('\n')
            f.write('Vocab Size: ' + str(trgtvocabsize))

    def save(self, data, path, fname):
        nfiles = max([len(entry['targets']) for entry in data])

        if not os.path.exists(path):
            os.mkdir(path)

        fsrc = open(os.path.join(path, fname) + '.src', 'w')
        ftrgt = open(os.path.join(path, fname) + '.trg', 'w')

        feval = open(os.path.join(path, fname) + '.eval', 'w')
        finfo = open(os.path.join(path, fname) + '.info', 'w')

        ref_path = os.path.join(path, 'references')
        if not os.path.exists(ref_path):
            os.mkdir(ref_path)
        frefs = [open(os.path.join(ref_path, fname + '.trg' + str(i + 1)), 'w') for i in range(nfiles)]

        for entry in data:
            src = entry['source']
            feval.write(' '.join(src))
            feval.write('\n')

            eid, category, size = str(entry['eid']), str(entry['category']), str(entry['size'])
            finfo.write(','.join([eid, category, size]))
            finfo.write('\n')

            for trgt in entry['targets']:
                fsrc.write(' '.join(src))
                fsrc.write('\n')

                ftrgt.write(' '.join(trgt['output']))
                ftrgt.write('\n')

            targets = entry['targets']
            for i in range(nfiles):
                if i < len(targets):
                    if 'text' in targets[i]:
                        target = ' '.join(targets[i]['text'].split())
                    else:
                        target = ' '.join(targets[i]['output'])
                    frefs[i].write(target)
                frefs[i].write('\n')

        fsrc.close()
        ftrgt.close()
        feval.close()
        finfo.close()
        for fref in frefs:
            fref.close()
