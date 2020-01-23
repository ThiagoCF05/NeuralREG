__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 28/02/2019
Description:
    This script aims to extract the gold-standard structured triple sets for the Text Structuring step.

    ARGS:
        [1] Path to the folder where WebNLG corpus is available (versions/v1.5/en)
        [2] Path to the folder where the data will be saved (Folder will be created in case it does not exist)

    EXAMPLE:
        python3 preprocess.py ../webnlg/v1.5/deepnlg/
"""

import sys

sys.path.append('./')
sys.path.append('../')

import os
from deepnlg import load
from deepnlg import parsing
from deepnlg import utils

from deepnlg.superpreprocess import Preprocess


class Structing(Preprocess):
    def __init__(self, data_path, write_path):
        super().__init__(data_path=data_path, write_path=write_path)

        self.traindata, self.vocab = self.load(os.path.join(data_path, 'train'))
        self.devdata, _ = self.load(os.path.join(data_path, 'dev'))
        self.testdata, _ = self.load(os.path.join(data_path, 'test'))

    def __call__(self):
        self.run(traindata=self.traindata, devdata=self.devdata, testdata=self.testdata)

    def load(self, path):
        flat = lambda struct: [w for w in struct if w not in ['<SNT>', '</SNT>']]

        entryset = parsing.run_parser(path)

        data, size = [], 0
        invocab, outvocab = [], []
        for entry in entryset:
            entitymap = {b: a for a, b in entry.entitymap_to_dict().items()}

            if len(entry.modifiedtripleset) > 1:
                visited = []
                for lex in entry.lexEntries:
                    # process ordered tripleset
                    source, delex_source, _ = load.snt_source(lex.orderedtripleset, entitymap, {})
                    source, delex_source = flat(source), flat(delex_source)

                    if source not in visited and ' '.join(source).strip() != '':
                        visited.append(source)
                        invocab.extend(source)

                        targets = []
                        for lex2 in entry.lexEntries:
                            _, text, _ = load.snt_source(lex2.orderedtripleset, entitymap, {})
                            flatten = flat(text)
                            if delex_source == flatten:
                                trgt_preds = []
                                for snt in utils.split_struct(text):
                                    trgt_preds.append('<SNT>')
                                    trgt_preds.extend([t[1] for t in snt])
                                    trgt_preds.append('</SNT>')
                                target = {'lid': lex2.lid, 'comment': lex2.comment, 'output': trgt_preds}
                                targets.append(target)
                                outvocab.extend(trgt_preds)

                        data.append({
                            'eid': entry.eid,
                            'category': entry.category,
                            'size': entry.size,
                            'source': source,
                            'targets': targets})
                        size += len(targets)

        invocab.append('unk')
        outvocab.append('unk')

        invocab = list(set(invocab))
        outvocab = list(set(outvocab))
        vocab = {'input': invocab, 'output': outvocab}

        print('Path:', path, 'Size: ', size)
        return data, vocab

    def load_simple(self, path):
        flat = lambda struct: [w for w in struct if w not in ['<SNT>', '</SNT>']]

        entryset = parsing.run_parser(path)

        data, size = [], 0
        invocab, outvocab = [], []
        for entry in entryset:
            entitymap = {b: a for a, b in entry.entitymap_to_dict().items()}

            if len(entry.modifiedtripleset) > 1:
                visited = []
                for lex in entry.lexEntries:
                    # process ordered tripleset
                    source, delex_source, _ = load.snt_source(lex.orderedtripleset, entitymap, {})
                    source, delex_source = flat(source), flat(delex_source)

                    if source not in visited and ' '.join(source).strip() != '':
                        visited.append(source)
                        invocab.extend(source)

                        targets = []
                        for lex2 in entry.lexEntries:
                            _, text, _ = load.snt_source(lex2.orderedtripleset, entitymap, {})
                            flatten = flat(text)
                            if delex_source == flatten:
                                trgt_preds = []
                                for snt in utils.split_struct(text):
                                    trgt_preds.append('<SNT>')
                                    trgt_preds.extend(['<TRIPLE>' for _ in snt])
                                    trgt_preds.append('</SNT>')

                                target = {'lid': lex2.lid, 'comment': lex2.comment, 'output': trgt_preds}
                                targets.append(target)
                                outvocab.extend(trgt_preds)

                        data.append({
                            'eid': entry.eid,
                            'category': entry.category,
                            'size': entry.size,
                            'source': source,
                            'targets': targets})
                        size += len(targets)

        invocab.append('unk')
        outvocab.append('unk')

        invocab = list(set(invocab))
        outvocab = list(set(outvocab))
        vocab = {'input': invocab, 'output': outvocab}

        print('Path:', path, 'Size: ', size)
        return data, vocab


if __name__ == '__main__':
    # data_path = '../webnlg/v1.5/en'
    # path='/roaming/tcastrof/NeuralREG/deepnlg'

    data_path = sys.argv[1]
    write_path = sys.argv[2]
    struct = Structing(data_path=data_path, write_path=write_path)
    struct()
