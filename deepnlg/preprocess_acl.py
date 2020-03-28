__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 28/02/2019
Description:
    This script aims to extract the gold-standard structured triple sets for the Text Structuring step.

    ARGS:
        [1] Path to the folder where ACL format from WebNLG corpus is available (versions/v1.0/acl_format)
        [2] Path to the folder where the data will be saved (Folder will be created in case it does not exist)
        [3] Path to the StanfordCoreNLP software (https://stanfordnlp.github.io/CoreNLP/)

    EXAMPLE:
        python3 preprocess_acl.py ../data/v1.0/acl_format/
"""

import copy
import json
import os
import re
import sys
import pickle
from deepnlg.lexicalization.preprocess import TemplateExtraction
from stanfordcorenlp import StanfordCoreNLP

sys.path.append('./')
sys.path.append('../')
re_date = '([0-9]{4})-([0-9]{2})-([0-9]{2})'


def load(pre_context_file, pos_context_file, entity_file, refex_file, size_file, info_file, data_file,
         original_file=''):
    original_data = []
    with open(pre_context_file) as f:
        pre_context = map(lambda x: x.split(), f.read().split('\n'))

    with open(pos_context_file) as f:
        pos_context = map(lambda x: x.split(), f.read().split('\n'))

    with open(entity_file) as f:
        entity = f.read().split('\n')

    with open(refex_file) as f:
        refex = map(lambda x: x.split(), f.read().split('\n'))

    with open(size_file) as f:
        size = f.read().split('\n')

    with open(info_file) as f:
        info = f.read().split('\n')

    if original_file != '':
        original_data = json.load(open(original_file, encoding='utf-8'))

    with open(data_file, 'rb') as file:
        data = pickle.load(file)

    data = extend_data(data, info, original_data)

    return {
        'pre_context': list(pre_context),
        'pos_context': list(pos_context),
        'entity': list(entity),
        'refex': list(refex),
        'size': list(size),
        'data': list(data),
        'original': list(original_data)
    }


def extend_data(data_set, data_info, original_data):
    # Get ordered entities
    text_ids = sorted(list(set(map(lambda x: x['text_id'], data_set))))
    new_set = []
    for i, text_id in enumerate(text_ids):
        references = filter(lambda x: x['text_id'] == text_id, data_set)
        references = sorted(references, key=lambda x: x['text_id'])

        for r, reference in enumerate(references):
            idx_info = len(reference.keys()) * (r + 1)
            reference['eid'] = 'Id' + str(text_id)
            _, xml_file = data_info[idx_info].split(' ')
            reference['info_category'] = xml_file.replace('.xml', '')
            reference['category'] = ''

            # find reference in train
            if len(original_data) > 0:
                original_entries = list(filter(lambda o: o['entity'] == reference['entity'] and
                                                         o['syntax'] == reference['syntax'] and
                                                         o['size'] == reference['size'] and
                                                         (o['text'] == reference['text'] or
                                                          o['pre_context'] == reference['pre_context'] or
                                                          o['pos_context'] == reference['pos_context'] or
                                                          o['refex'] == reference['refex'] or
                                                          o['text_status'] == reference['text_status'] or
                                                          o['sentence_status'] == reference['sentence_status']),
                                               original_data))
                if len(original_entries) == 0:
                    original_entries = list(filter(lambda o: o['entity'] == reference['entity'], original_data))
                    if len(original_entries) == 0:
                        print('Category not found for Entity :', reference['entity'])

                if len(original_entries) > 0:
                    categories = list(set([e['category'] for e in original_entries]))
                    if len(categories) > 1:
                        print('More than one category to entity: ', reference['entity'])
                    reference['category'] = categories[0]

            new_set.append(reference)

    return new_set


def load_data(entry_path, original_path):
    foriginal = ''
    fprecontext = os.path.join(entry_path, 'pre_context.txt')
    fposcontext = os.path.join(entry_path, 'pos_context.txt')
    fentity = os.path.join(entry_path, 'entity.txt')
    frefex = os.path.join(entry_path, 'refex.txt')
    fsize = os.path.join(entry_path, 'size.txt')
    finfo = os.path.join(entry_path, 'info.txt')
    fdata = os.path.join(entry_path, 'data.cPickle')
    if original_path != '':
        foriginal = os.path.join(original_path, 'data.json')
    data_set = load(fprecontext, fposcontext, fentity, frefex, fsize, finfo, fdata, foriginal)
    return data_set


class REGPrecACL:
    def __init__(self, data_path, write_path, stanford_path):
        self.data_path = data_path
        self.write_path = write_path

        self.temp_extractor = TemplateExtraction(stanford_path)
        self.corenlp = StanfordCoreNLP(stanford_path)

        self.traindata, self.vocab = self.process(entry_path=os.path.join(data_path, 'train'),
                                                  original_path=os.path.join(data_path, 'dev', 'reference'))
        self.devdata, _ = self.process(entry_path=os.path.join(data_path, 'dev'),
                                       original_path=os.path.join(data_path, 'dev', 'reference'))
        self.testdata, _ = self.process(entry_path=os.path.join(data_path, 'test'),
                                        original_path=os.path.join(data_path, 'dev', 'reference'))

        self.corenlp.close()
        self.temp_extractor.close()

        json.dump(self.traindata, open(os.path.join(write_path, 'train.json'), 'w'))
        json.dump(self.vocab, open(os.path.join(write_path, 'vocab.json'), 'w'))
        json.dump(self.devdata, open(os.path.join(write_path, 'dev.json'), 'w'))
        json.dump(self.testdata, open(os.path.join(write_path, 'test.json'), 'w'))

    def process(self, entry_path, original_path=''):
        data_set = load_data(entry_path, original_path)
        entry_set = data_set['data']
        data, size = [], 0
        in_vocab, out_vocab = [], []

        for i, reference in enumerate(entry_set):
            progress = round(float(i) / len(entry_set), 2)
            print('Progress: {0}'.format(progress), end='   \r')

            # process source
            text = str(reference['text']).split(' ')
            template = reference['pre_context'] + reference['entity'] + reference['pos_context']
            template = template.split(' ')

            pre = re.split(' ', reference['pre_context'].replace('eos', '').strip())
            pre_context = [elem for elem in pre if elem != '']

            pos = re.split(' ', reference['pos_context'].replace('eos', '').strip())
            pos_context = [elem for elem in pos if elem != '']

            entity = '_'.join(reference['entity'].replace('\"', '').replace('\'', '').lower().strip().split())
            if entity != '':
                _refex = re.split(' ', reference['refex'].replace('eos', '').strip())
                refex = [elem for elem in _refex if elem != '']

                isDigit = entity.replace('.', '').strip().isdigit()
                isDate = len(re.findall(re_date, entity)) > 0
                if entity[0] not in ['\'', '\"'] and not isDigit and not isDate and not '':
                    data.append({
                        'entity': entity,
                        'category': reference['category'],
                        'pre_context': pre_context,
                        'pos_context': pos_context,
                        'refex': refex
                    })
                    size += 1
                    in_vocab.extend(pre_context)
                    in_vocab.extend(pos_context)
                    in_vocab.append(entity)
                    out_vocab.extend(refex)

        in_vocab.append('unk')
        out_vocab.append('unk')
        in_vocab.append('eos')
        out_vocab.append('eos')

        in_vocab = list(set(in_vocab))
        out_vocab = list(set(out_vocab))
        vocab = {'input': in_vocab, 'output': out_vocab}

        print('Path:', entry_path, 'Size: ', size)
        return data, vocab


if __name__ == '__main__':
    # data_path = '../data/v1.0/acl_format'
    # write_path = '../data/v1.0'
    # stanford_path = r'../stanford/stanford-corenlp-full-2018-10-05'

    data_path = sys.argv[1]
    write_path = sys.argv[2]
    stanford_path = sys.argv[3]
    s = REGPrecACL(data_path=data_path, write_path=write_path, stanford_path=stanford_path)
