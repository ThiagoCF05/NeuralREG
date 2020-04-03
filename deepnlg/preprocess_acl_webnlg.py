__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 28/02/2019
Description:
    This script aims to extract the gold-standard structured triple sets for the Text Structuring step.

    ARGS:
        [1] Path to the folder where WebNLG corpus is available (versions/v1.0/en)
        [2] Path to the folder where the data will be saved (Folder will be created in case it does not exist)
        [3] Path to the StanfordCoreNLP software (https://stanfordnlp.github.io/CoreNLP/)

    EXAMPLE:
        python3 preprocess_acl_webnlg.py ../webnlg/v1.0/en/
"""

import copy
import json
import os
import re
import sys
import pickle
import xml.etree.ElementTree as ET
from deepnlg import parsing as parser
from deepnlg.lexicalization.preprocess import TemplateExtraction
from stanfordcorenlp import StanfordCoreNLP

sys.path.append('./')
sys.path.append('../')
re_date = '([0-9]{4})-([0-9]{2})-([0-9]{2})'
re_tag = re.compile(r"((AGENT|PATIENT|BRIDGE)\-\d{1})")


def extract_entity_type(entity):
    aux = entity.split('^^')
    if len(aux) > 1:
        return aux[-1]

    aux = entity.split('@')
    if len(aux) > 1:
        return aux[-1]

    return 'wiki'


def classify(references):
    references = sorted(references, key=lambda x: (x['entity'], x['sentence'], x['pos']))

    sentence_statuses = {}
    for i, reference in enumerate(references):
        # text status
        if i == 0 or (reference['entity'] != references[i - 1]['entity']):
            reference['text_status'] = 'new'
        else:
            reference['text_status'] = 'given'

        if reference['sentence'] not in sentence_statuses:
            sentence_statuses[reference['sentence']] = []

        # sentence status
        if reference['entity'] not in sentence_statuses[reference['sentence']]:
            reference['sentence_status'] = 'new'
        else:
            reference['sentence_status'] = 'given'

        sentence_statuses[reference['sentence']].append(reference['entity'])

        # referential form
        reg = reference['refex'].replace('eos', '').strip()
        reference['reftype'] = 'name'
        if reg.lower().strip() in ['he', 'his', 'him', 'she', 'hers', 'her', 'it', 'its', 'we', 'our', 'ours',
                                   'they', 'theirs', 'them']:
            reference['reftype'] = 'pronoun'
        elif reg.lower().strip().split()[0] in ['the', 'a', 'an']:
            reference['reftype'] = 'description'
        elif reg.lower().strip().split()[0] in ['this', 'these', 'that', 'those']:
            reference['reftype'] = 'demonstrative'

    return references


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

        self.traindata, self.vocab = self.process(entry_path=os.path.join(data_path, 'train'))
        self.testdata, _ = self.process(entry_path=os.path.join(data_path, 'test'))

        self.corenlp.close()
        self.temp_extractor.close()

        json.dump(self.traindata, open(os.path.join(write_path, 'train.json'), 'w'))
        json.dump(self.vocab, open(os.path.join(write_path, 'vocab.json'), 'w'))
        json.dump(self.testdata, open(os.path.join(write_path, 'test.json'), 'w'))

    def process(self, entry_path):
        data, in_vocab, out_vocab = self.run_parser(entry_path)

        in_vocab.add('unk')
        out_vocab.add('unk')
        in_vocab.add('eos')
        out_vocab.add('eos')

        in_vocab = list(set(in_vocab))
        out_vocab = list(set(out_vocab))
        vocab = {'input': in_vocab, 'output': out_vocab}

        print('Path:', entry_path, 'Size: ', len(data))
        return data, vocab

    def run_parser(self, set_path):
        entryset, input_vocab, output_vocab = [], set(), set()
        dirtriples = filter(lambda item: not str(item).startswith('.'), os.listdir(set_path))
        for dirtriple in dirtriples:
            fcategories = filter(lambda item: not str(item).startswith('.'),
                                 os.listdir(os.path.join(set_path, dirtriple)))
            for fcategory in fcategories:
                references, ref_in_vocab, ref_out_vocab = self.parse(os.path.join(set_path, dirtriple, fcategory))
                entryset.extend(references)
                input_vocab = input_vocab.union(ref_in_vocab)
                output_vocab = output_vocab.union(ref_out_vocab)

        return entryset, input_vocab, output_vocab

    def parse(self, in_file):
        tree = ET.parse(in_file)
        root = tree.getroot()

        data = []
        input_vocab, output_vocab = set(), set()

        entries = root.find('entries')

        for entry in entries:
            eid = entry.attrib['eid']
            size = entry.attrib['size']
            category = entry.attrib['category']

            # get entity map
            entitymap_xml = entry.find('entitymap')
            entity_map = {}
            for entitytag in entitymap_xml:
                tag, entity = entitytag.text.split(' | ')
                entity_map[tag] = entity

            # Reading original triples to extract the entities type
            types = []
            originaltripleset = entry.find('originaltripleset')
            for otriple in originaltripleset:
                e1, pred, e2 = otriple.text.split(' | ')
                entity1_type = extract_entity_type(e1.strip())
                entity2_type = extract_entity_type(e2.strip())
                types.append({'e1_type': entity1_type, 'e2_type': entity2_type})

            entity_type = {}
            modifiedtripleset = entry.find('modifiedtripleset')
            for i, mtriple in enumerate(modifiedtripleset):
                e1, pred, e2 = mtriple.text.split(' | ')
                entity_type[e1.replace('\'', '')] = types[i]['e1_type']
                entity_type[e2.replace('\'', '')] = types[i]['e2_type']

            lexList = []
            lexEntries = entry.findall('lex')
            for lex in lexEntries:
                try:
                    text = lex.find('text').text
                    template = lex.find('template').text

                    if template:
                        # print('{}\r'.format(template))

                        text = self.tokenize(text)
                        text = ' '.join(text).strip()

                        template = self.tokenize(template)
                        template = ' '.join(template).strip()

                        references, in_vocab, out_vocab = self.get_references(text, template,
                                                                              entity_map, entity_type,
                                                                              category, in_file)

                        data.extend(references)
                        input_vocab = input_vocab.union(in_vocab)
                        output_vocab = output_vocab.union(out_vocab)
                except:
                    print("Parsing error: ", sys.exc_info()[0])

        return data, input_vocab, output_vocab

    def get_references(self, text, template, entity_map, entity_type, category, filename):
        context = copy.copy(template)
        references, input_vocab, output_vocab = [], set(), set()

        isOver = False
        while not isOver:
            pre_tag, tag, pos_tag = self.process_template(template)
            pre_context, pos_context = self.process_context(context, entity_map)

            if tag == '':
                isOver = True
            else:
                # Look for reference from 5-gram to 2-gram
                i, f = 5, []
                try:
                    while i > 1:
                        begin = ' '.join(i * ['BEGIN'])
                        text = begin + ' ' + text
                        template = begin + ' ' + template
                        pre_tag, tag, pos_tag = self.process_template(template)

                        regex = re.escape(' '.join(pre_tag[-i:]).strip()) + ' (.+?) ' + re.escape(
                            ' '.join(pos_tag[:i]).strip())
                        f = re.findall(regex, text)

                        template = template.replace('BEGIN', '').strip()
                        text = text.replace('BEGIN', '').strip()
                        i -= 1

                        if len(f) == 1:
                            break

                    if len(f) > 0:
                        # DO NOT LOWER CASE HERE!!!!!!
                        template = template.replace(tag, f[0], 1)
                        refex = self.tokenize(f[0])
                        refex = ' '.join(refex).strip()

                        # Do not include literals
                        entity = entity_map[tag]

                        if entity_type[entity] == 'wiki':
                            refex = refex.replace('eos', '').split()

                            normalized = '_'.join(entity.replace('\"', '').replace('\'', '').lower().split())
                            # aux = context.replace(tag, 'ENTITY', 1)
                            # reference_info = self.process_reference_info(aux, 'ENTITY')

                            if normalized != '':
                                isDigit = normalized.replace('.', '').strip().isdigit()
                                isDate = len(re.findall(re_date, normalized)) > 0
                                if normalized[0] not in ['\'', '\"'] and not isDigit and not isDate:
                                    mapped_reference = {
                                        'entity': normalized,
                                        'category': category,
                                        'pre_context': pre_context.replace('@', '').replace('eos', '').split(),
                                        'pos_context': pos_context.replace('@', '').replace('eos', '').split(),
                                        'refex': refex
                                        # 'size': len(entity_map.keys()),
                                        # 'syntax': reference_info['syntax'],
                                        # 'general_pos': reference_info['general_pos'],
                                        # 'sentence': reference_info['sentence'],
                                        # 'pos': reference_info['pos'],
                                        # 'text': text,
                                        # 'filename': filename,
                                    }
                                    references.append(mapped_reference)

                                    output_vocab = output_vocab.union(set(refex))
                                    input_vocab = input_vocab.union(set(pre_context.split()))
                                    input_vocab = input_vocab.union(set(pos_context.split()))
                                    input_vocab = input_vocab.union(set([normalized]))

                                    context = context.replace(tag, normalized, 1)
                                else:
                                    print('Not normalized entity: ', normalized)
                        else:
                            context = context.replace(tag, '_'.join(
                                entity_map[tag].replace('\"', '').replace('\'', '').lower().split()), 1)
                    else:
                        template = template.replace(tag, ' ', 1)
                        context = context.replace(tag, '_'.join(
                            entity_map[tag].replace('\"', '').replace('\'', '').lower().split()), 1)
                except:
                    print("Parsing error (Processing template): ", sys.exc_info()[0])

        # references = classify(references)

        return references, input_vocab, output_vocab

    def process_template(self, text):
        template = text.split()

        tag = ''
        pre_tag, pos_tag, i = [], [], 0
        try:
            for token in template:
                i += 1
                token_tag = re.search(re_tag, token)

                if token_tag:
                    if token != token_tag.group():
                        print('Token error: ', token)

                    tag = token_tag.group()
                    for pos_token in template[i:]:
                        pos_token_tag = re.search(re_tag, pos_token)

                        if pos_token_tag:
                            break
                        else:
                            pos_tag.append(pos_token)
                    break
                else:
                    pre_tag.append(token)
        except:
            print("Parsing error (processing template): ", sys.exc_info()[0])

        return pre_tag, tag, pos_tag

    def process_context(self, text, entity_map):
        context = text.split()
        pre_context, pos_context, i = [], [], 0
        try:
            for token in context:
                i += 1

                token_tag = re.search(re_tag, token)
                if token_tag:
                    pos_context = context[i:]
                    break
                else:
                    pre_context.append(token)

            pre_context = ' '.join(['eos'] + pre_context)
            pos_context = ' '.join(pos_context + ['eos'])
            for tag in entity_map:
                pre_context = pre_context.replace(tag, '_'.join(
                    entity_map[tag].replace('\"', '').replace('\'', '').lower().split()))
                pos_context = pos_context.replace(tag, '_'.join(
                    entity_map[tag].replace('\"', '').replace('\'', '').lower().split()))
        except:
            print("Parsing error (processing context): ", sys.exc_info()[0])

        return pre_context.lower(), pos_context.lower()

    def process_reference_info(self, template, tag):
        props = {'annotators': 'tokenize,ssplit,pos,depparse', 'pipelineLanguage': 'en', 'outputFormat': 'json'}

        out = self.corenlp.annotate(template.strip(), properties=props)
        out = json.loads(out)

        reference = {'syntax': '', 'sentence': -1, 'pos': -1, 'general_pos': -1, 'tag': tag}
        general_pos = 0
        for i, snt in enumerate(out['sentences']):
            for token in snt['enhancedDependencies']:
                # get syntax
                if token['dependentGloss'] == tag:
                    reference = {'syntax': '', 'sentence': i, 'pos': int(token['dependent']),
                                 'general_pos': general_pos + int(token['dependent']), 'tag': tag}
                    if 'nsubj' in token['dep'] or 'nsubjpass' in token['dep']:
                        reference['syntax'] = 'np-subj'
                    elif 'nmod:poss' in token['dep'] or 'compound' in token['dep']:
                        reference['syntax'] = 'subj-det'
                    else:
                        reference['syntax'] = 'np-obj'
                    break
            general_pos += len(snt['tokens'])
        return reference

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
            print('Parsing error (tokenize):', sys.exc_info()[0])

        return tokens


if __name__ == '__main__':
    # data_path = '/webnlg/data/v1.0/en'
    # write_path = '/data/v1.0'
    # stanford_path = r'/stanford/stanford-corenlp-full-2018-10-05'

    data_path = sys.argv[1]
    write_path = sys.argv[2]
    stanford_path = sys.argv[3]
    s = REGPrecACL(data_path=data_path, write_path=write_path, stanford_path=stanford_path)
