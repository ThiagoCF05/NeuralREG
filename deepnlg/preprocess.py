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

import copy
import json
import os
import re
import sys
from deepnlg import parsing as parser
from deepnlg.lexicalization.preprocess import TemplateExtraction
from stanfordcorenlp import StanfordCoreNLP

sys.path.append('./')
sys.path.append('../')


class REGPrec:
    def __init__(self, data_path, write_path, stanford_path):
        self.data_path = data_path
        self.write_path = write_path

        self.temp_extractor = TemplateExtraction(stanford_path)
        self.corenlp = StanfordCoreNLP(stanford_path)

        self.traindata, self.vocab = self.process(entry_path=os.path.join(data_path, 'train'))
        self.devdata, _ = self.process(entry_path=os.path.join(data_path, 'dev'))
        self.testdata, _ = self.process(entry_path=os.path.join(data_path, 'test'))

        self.corenlp.close()
        self.temp_extractor.close()

        json.dump(self.traindata, open(os.path.join(write_path, 'train.json'), 'w'))
        json.dump(self.vocab, open(os.path.join(write_path, 'vocab.json'), 'w'))
        json.dump(self.devdata, open(os.path.join(write_path, 'dev.json'), 'w'))
        json.dump(self.testdata, open(os.path.join(write_path, 'test.json'), 'w'))

    def tokenize(self, text):
        props = {'annotators': 'tokenize,ssplit', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
        text = text.replace('@', ' ')
        tokens = []
        # tokenizing text
        try:
            out = self.corenlp.annotate(text.strip(), properties=props)
            out = json.loads(out)

            for snt in out['sentences']:
                sentence = list(map(lambda w: w['originalText'], snt['tokens']))
                tokens.extend(sentence)
        except:
            print('Parsing error...')

        return tokens

    def process(self, entry_path):
        entryset = parser.run_parser(entry_path)

        data, size = [], 0
        invocab, outvocab = [], []

        for i, entry in enumerate(entryset):
            progress = round(float(i) / len(entryset), 2)
            print('Progress: {0}'.format(progress), end='   \r')
            # process source
            entitymap = entry.entitymap_to_dict()

            for lex in entry.lexEntries:
                # process ordered tripleset
                template = self.temp_extractor.extract(lex.template)[0]
                template = ' '.join(template).split()

                refcount = {}
                for reference in lex.references:
                    tag = reference.tag
                    if tag not in refcount:
                        refcount[tag] = 0
                    refcount[tag] += 1

                    entity = '_'.join(reference.entity.split())
                    if entity != '':
                        refex = self.tokenize(reference.refex)

                        isDigit = entity.replace('.', '').strip().isdigit()
                        regex = '([0-9]{4})-([0-9]{2})-([0-9]{2})'
                        isDate = len(re.findall(regex, entity)) > 0
                        if entity[0] not in ['\'', '\"'] and not isDigit and not isDate:
                            context, pos = [], 0
                            for i, w in enumerate(template):
                                if w.strip() == tag.strip():
                                    pos += 1
                                    if pos == refcount[tag]:
                                        pre_context = copy.copy(context)
                                        pos_context = []
                                        for j in range(i + 1, len(template)):
                                            if template[j].strip() not in entitymap:
                                                pos_context.append(template[j].lower())
                                            else:
                                                pos_context.append('_'.join(entitymap[template[j]].split()))

                                        data.append({
                                            'entity': entity,
                                            'category': entry.category,
                                            'pre_context': pre_context,
                                            'pos_context': pos_context,
                                            'refex': refex
                                        })
                                        size += 1
                                        invocab.extend(pre_context)
                                        invocab.extend(pos_context)
                                        invocab.append(entity)
                                        outvocab.extend(refex)
                                else:
                                    if w.strip() not in entitymap:
                                        context.append(w.lower())
                                    else:
                                        context.append('_'.join(entitymap[w].split()))

        invocab.append('unk')
        outvocab.append('unk')
        invocab.append('eos')
        outvocab.append('eos')

        invocab = list(set(invocab))
        outvocab = list(set(outvocab))
        vocab = {'input': invocab, 'output': outvocab}

        print('Path:', entry_path, 'Size: ', size)
        return data, vocab


if __name__ == '__main__':
    # write_path='/roaming/tcastrof/emnlp2019/reg'
    # data_path = '/home/tcastrof/Experiments/versions/v1.5/en'
    # stanford_path = r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'

    data_path = sys.argv[1]
    write_path = sys.argv[2]
    stanford_path = sys.argv[3]
    s = REGPrec(data_path=data_path, write_path=write_path, stanford_path=stanford_path)
