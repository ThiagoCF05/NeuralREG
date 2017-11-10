__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 24/07/2017
Description:
    Scripts based on the manual annotation of the corpus.
"""

import copy
import os
import random
import re
import sys
sys.path.append('../')
sys.path.append('/home/tcastrof/workspace/stanford_corenlp_pywrapper')

from stanford_corenlp_pywrapper import CoreNLP

from db.model import *

class Preprocessing(object):
    def __init__(self, ftrain, fdev):
        self.proc = CoreNLP('ssplit')
        self.ftrain = ftrain
        self.fdev = fdev

        self.trainset()

    def trainset(self):
        input_vocab, output_vocab = set(), set()
        train, dev = [], []
        train_info, dev_info = [], []

        dirs = filter(lambda x: x != '.DS_Store', os.listdir(self.ftrain))
        for path in dirs:
            dirs2 = filter(lambda x: x != '.DS_Store', os.listdir(os.path.join(self.ftrain, path)))
            for fname in dirs2:
                f = open(os.path.join(self.ftrain, path, fname))
                doc = f.read().decode('utf-8')
                f.close()

                doc = doc.split((50*'*')+'\n')

                print('Doc size: ', len(doc))

                data, in_vocab, out_vocab = self.annotation_parse(doc)

                input_vocab = input_vocab.union(in_vocab)
                output_vocab = output_vocab.union(out_vocab)

                train_size = int(0.9 * len(data))

                random.shuffle(data)
                train.extend(data[:train_size])
                dev.extend(data[train_size:])

                info = len(train) * [path + ' ' + fname]
                train_info.extend(info)

                info = len(dev) * [path + ' ' + fname]
                dev_info.extend(info)

        self.write('data/train', train, train_info)
        self.write('data/dev', dev, dev_info)

        with open('data/input_vocab.txt', 'w') as f:
            f.write(('\n'.join(list(input_vocab))).encode("utf-8"))

        with open('data/output_vocab.txt', 'w') as f:
            f.write(('\n'.join(list(output_vocab))).encode("utf-8"))

    def write(self, fname, instances, info):
        pre_context = '\n'.join(map(lambda x: x['pre_context'], instances)).encode('utf-8')
        with open(os.path.join(fname, 'pre_context.txt'), 'w') as f:
            f.write(pre_context)
        pos_context = '\n'.join(map(lambda x: x['pos_context'], instances)).encode('utf-8')
        with open(os.path.join(fname, 'pos_context.txt'), 'w') as f:
            f.write(pos_context)
        entity = '\n'.join(map(lambda x: x['entity'], instances)).encode('utf-8')
        with open(os.path.join(fname, 'entity.txt'), 'w') as f:
            f.write(entity)
        refex = '\n'.join(map(lambda x: x['refex'], instances)).encode('utf-8')
        with open(os.path.join(fname, 'refex.txt'), 'w') as f:
            f.write(refex)
        size = '\n'.join(map(lambda x: str(x['size']), instances))
        with open(os.path.join(fname, 'size.txt'), 'w') as f:
            f.write(size)
        info = '\n'.join(info).encode('utf-8')
        with open(os.path.join(fname, 'info.txt'), 'w') as f:
            f.write(info)

    def check_entity(self, entity):
        f = Entity.objects(name=entity.strip(), type='wiki')
        if f.count() > 0:
            return '_'.join(entity.replace('\"', '').replace('\'', '').lower().split())
        else:
            return ''

    def normalize_entities(self, entity_map):
        for tag, entity in entity_map.iteritems():
            entity_map[tag] = '_'.join(entity_map[tag].replace('\"', '').replace('\'', '').lower().split())
        return entity_map

    def _get_refexes(self, text, template, entity_map):
        '''
        Extract referring expressions for each reference overlapping text and template
        :param text: original text
        :param template: template (delexicalized text)
        :return:
        '''
        def process_template(template):
            stemplate = template.split()

            tag = ''
            pre_tag, pos_tag, i = [], [], 0
            for token in stemplate:
                i += 1
                if token.split('-')[0] in ['AGENT', 'PATIENT', 'BRIDGE']:
                    tag = token
                    for pos_token in stemplate[i:]:
                        if pos_token.split('-')[0] in ['AGENT', 'PATIENT', 'BRIDGE']:
                            break
                        else:
                            pos_tag.append(pos_token)
                    break
                else:
                    pre_tag.append(token)
            return pre_tag, tag, pos_tag

        def process_context(context, entity_map):
            scontext = context.split()
            pre_context, pos_context, i = [], [], 0
            for token in scontext:
                i += 1
                if token.split('-')[0] in ['AGENT', 'PATIENT', 'BRIDGE']:
                    pos_context = scontext[i:]
                    break
                else:
                    pre_context.append(token)

            pre_context = ' '.join(['EOS'] + pre_context)
            pos_context = ' '.join(pos_context + ['EOS'])
            for tag, entity in entity_map.iteritems():
                pre_context = pre_context.replace(tag, entity_map[tag])
                pos_context = pos_context.replace(tag, entity_map[tag])

            return pre_context.lower(), pos_context.lower()

        text = 'BEGIN BEGIN BEGIN ' + text
        context = copy.copy(template)
        template = 'BEGIN BEGIN BEGIN ' + template

        data, input_vocab, output_vocab = [], set(), set()

        isOver = False
        while not isOver:
            pre_tag, tag, pos_tag = process_template(template)
            pre_context, pos_context = process_context(context, entity_map)

            if tag == '':
                isOver = True
            else:
                regex = re.escape(' '.join(pre_tag[-3:]).strip()) + ' (.+?) ' + re.escape(' '.join(pos_tag[:3]).strip())
                f = re.findall(regex, text)

                if len(f) > 0:
                    refex = f[0].lower()
                    template = template.replace(tag, refex, 1)

                    # Do not include numbers
                    normalized = self.check_entity(entity_map[tag])
                    if normalized != '':
                        refex = ['eos'] + refex.split() + ['eos']
                        row = {
                            'pre_context':pre_context.replace('@', ''),
                            'pos_context':pos_context.replace('@', ''),
                            'entity':normalized,
                            'refex':' '.join(refex),
                            'size':len(entity_map.keys())
                        }
                        data.append(row)
                        output_vocab = output_vocab.union(set(refex))
                        input_vocab = input_vocab.union(set(pre_context.split()))
                        input_vocab = input_vocab.union(set(pos_context.split()))
                        input_vocab = input_vocab.union(set([normalized]))

                    context = context.replace(tag, entity_map[tag], 1)
                else:
                    template = template.replace(tag, ' ', 1)
                    context = context.replace(tag, entity_map[tag], 1)

        return data, input_vocab, output_vocab

    def annotation_parse(self, doc):
        data = []
        input_vocab, output_vocab = set(), set()
        for entry in doc:
            entry = entry.split('\n\n')

            try:
                _, entryId, size, semcategory = entry[0].replace('\n', '').split()

                entity_map = dict(map(lambda entity: entity.split(' | '), entry[2].replace('\nENTITY MAP\n', '').split('\n')))
                # entity_map = self.normalize_entities(entity_map)

                lexEntries = entry[3].replace('\nLEX\n', '').split('\n-')[:-1]

                for lex in lexEntries:
                    if lex[0] == '\n':
                        lex = lex[1:]
                    lex = lex.split('\n')

                    text = lex[1].replace('TEXT: ', '').strip()
                    template = lex[2].replace('TEMPLATE: ', '')
                    correct = lex[3].replace('CORRECT: ', '').strip()
                    comment = lex[4].replace('COMMENT: ', '').strip()

                    if comment in ['g', 'good']:
                        print('{}\r'.format(template))

                        text, template = self.stanford_parse(text, template)
                        references, in_vocab, out_vocab = self._get_refexes(text, template, entity_map)
                        data.extend(references)
                        input_vocab = input_vocab.union(in_vocab)
                        output_vocab = output_vocab.union(out_vocab)
                    elif correct != '' and comment != 'wrong':
                        if correct.strip() == 'CORRECT:':
                            correct = template
                        print('{}\r'.format(correct))
                        text, template = self.stanford_parse(text, correct)
                        references, in_vocab, out_vocab = self._get_refexes(text, template, entity_map)
                        data.extend(references)
                        input_vocab = input_vocab.union(in_vocab)
                        output_vocab = output_vocab.union(out_vocab)
            except:
                print('ERROR')

        return data, input_vocab, output_vocab


    def stanford_parse(self, text, template):
        '''
        Obtain information of references and their referring expressions
        :param text:
        :param template:
        :param entities:
        :return:
        '''
        out = self.proc.parse_doc(text)
        text = []
        for i, snt in enumerate(out['sentences']):
            text.extend(snt['tokens'])
        text = ' '.join(text).replace('-LRB-', '(').replace('-RRB-', ')').strip()

        out = self.proc.parse_doc(template)
        temp = []
        for i, snt in enumerate(out['sentences']):
            temp.extend(snt['tokens'])
        template = ' '.join(temp).replace('-LRB-', '(').replace('-RRB-', ')').strip()

        return text, template

if __name__ == '__main__':
    ftrain = 'annotation/train'
    fdev = 'annotation/dev'
    Preprocessing(ftrain, fdev)