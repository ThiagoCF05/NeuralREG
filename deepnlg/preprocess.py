__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 28/02/2019
Description:
    This script aims to extract the gold-standard structured triple sets for the Text Structuring step.

    ARGS:
        [1] Path to the folder where WebNLG corpus is available (versions/v1.5/en)
        [2] Path to the folder where the data will be saved (Folder will be created in case it does not exist)
        [3] Path to the StanfordCoreNLP software (https://stanfordnlp.github.io/CoreNLP/)

    EXAMPLE:
        python3 preprocess.py ../webnlg/v1.5/en/
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
re_date = '([0-9]{4})-([0-9]{2})-([0-9]{2})'


class Entry:
    def __init__(self, category, eid, size, originaltripleset, modifiedtripleset, entitymap, lexEntries):
        self.category = category
        self.eid = eid
        self.size = size
        self.originaltripleset = originaltripleset
        self.modifiedtripleset = modifiedtripleset
        self.lexEntries = lexEntries
        self.entitymap = entitymap

    def entitymap_to_dict(self):
        return dict(map(lambda tagentity: tagentity.to_tuple(), self.entitymap))


class Triple:
    def __init__(self, subject, predicate, object):
        self.subject = subject
        self.predicate = predicate
        self.object = object


class Lex:
    def __init__(self, text, template, references=[], input_vocab=set(), output_vocab=set(), character_vocab=set()):
        self.text = text
        self.template = template
        self.tree = ''
        self.references = references
        self.ref_input_vocab = input_vocab
        self.ref_output_vocab = output_vocab
        self.ref_character_vocab = character_vocab


class TagEntity:
    def __init__(self, tag, entity):
        self.tag = tag
        self.entity = entity

    def to_tuple(self):
        return self.tag, self.entity


class Reference:
    def __init__(self, entity, tag, text, pre_context, pos_context, refex, number, reftype, size, syntax, general_pos,
                 sentence, pos, category, filename):
        self.entity = entity
        self.tag = tag
        self.text = text
        self.pre_context = pre_context
        self.pos_context = pos_context
        self.refex = refex
        self.number = number
        self.reftype = reftype
        self.size = size
        self.syntax = syntax
        self.general_pos = general_pos
        self.sentence = sentence
        self.pos = pos
        self.category = category
        self.filename = filename


class Entry:
    def __init__(self, eid, entity, category, pre_context, pos_context, size, refex, path, fname):
        self.eid = eid
        self.entity = entity
        self.category = category
        self.pre_context = pre_context
        self.pos_context = pos_context
        self.size = size
        self.refex = refex
        self.path = path
        self.fname = fname


class REGPrec:
    def __init__(self, data_path, write_path, stanford_path):
        self.data_path = data_path
        self.write_path = write_path

        self.temp_extractor = TemplateExtraction(stanford_path)
        self.corenlp = StanfordCoreNLP(stanford_path)

        # self.traindata, self.vocab = self.process(entry_path=os.path.join(data_path, 'train'))
        # self.devdata, _ = self.process(entry_path=os.path.join(data_path, 'dev'))
        self.testdata, _ = self.process(entry_path=os.path.join(data_path, 'test'))

        self.corenlp.close()
        self.temp_extractor.close()

        # json.dump(self.traindata, open(os.path.join(write_path, 'train.json'), 'w'))
        # json.dump(self.vocab, open(os.path.join(write_path, 'vocab.json'), 'w'))
        # json.dump(self.devdata, open(os.path.join(write_path, 'dev.json'), 'w'))
        json.dump(self.testdata, open(os.path.join(write_path, 'test.json'), 'w'))

    def process(self, entry_path):
        entryset = parser.run_parser(entry_path)

        data, size = [], 0
        invocab, outvocab = [], []

        for i, entry in enumerate(entryset):
            progress = round(float(i) / len(entryset), 2)
            print('Progress: {0}'.format(progress), end='   \r')
            # process source
            entity_map = entry.entitymap_to_dict()

            for lex in entry.lexEntries:
                text = self.tokenize(lex.text)
                text = ' '.join(text).split()

                template = self.tokenize(lex.template)
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
                        isDate = len(re.findall(re_date, entity)) > 0
                        if entity[0] not in ['\'', '\"'] and not isDigit and not isDate:
                            context, pos = [], 0
                            for i, w in enumerate(template):
                                if w.strip() == tag.strip():
                                    pos += 1
                                    if pos == refcount[tag]:
                                        pre_context = copy.copy(context)
                                        pos_context = []
                                        for j in range(i + 1, len(template)):
                                            if template[j].strip() not in entity_map:
                                                pos_context.append(template[j].lower())
                                            else:
                                                pos_context.append('_'.join(entity_map[template[j]].split()))

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
                                    if w.strip() not in entity_map:
                                        context.append(w.lower())
                                    else:
                                        context.append('_'.join(entity_map[w].split()))

        invocab.append('unk')
        outvocab.append('unk')
        invocab.append('eos')
        outvocab.append('eos')

        invocab = list(set(invocab))
        outvocab = list(set(outvocab))
        vocab = {'input': invocab, 'output': outvocab}

        print('Path:', entry_path, 'Size: ', size)
        return data, vocab

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
            print('Parsing error (tokenize).')

        return tokens


if __name__ == '__main__':
    # data_path = '../webnlg/data/v1.5/en'
    # write_path = '../data/v1.5'
    # stanford_path = r'/home/stanford/stanford-corenlp-full-2018-10-05'

    data_path = sys.argv[1]
    write_path = sys.argv[2]
    stanford_path = sys.argv[3]
    s = REGPrec(data_path=data_path, write_path=write_path, stanford_path=stanford_path)
