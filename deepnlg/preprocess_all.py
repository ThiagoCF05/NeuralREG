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
import pickle
import xml.etree.ElementTree as ET
from deepnlg import parsing as parser
from deepnlg.lexicalization.preprocess import TemplateExtraction
from stanfordcorenlp import StanfordCoreNLP

sys.path.append('./')
sys.path.append('../')
re_date = '([0-9]{4})-([0-9]{2})-([0-9]{2})'
re_tag = re.compile(r"((AGENT|PATIENT|BRIDGE)\-\d{1})")


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
                    print(reference['entity'])
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


class REGPrec:
    def __init__(self, data_path, write_path, stanford_path, version='1.5', txt_format=False):
        self.data_path = data_path
        self.write_path = write_path

        self.temp_extractor = TemplateExtraction(stanford_path)
        self.corenlp = StanfordCoreNLP(stanford_path)

        if txt_format:
            # self.traindata, self.vocab = self.process_txt(entry_path=os.path.join(data_path, 'train'))
            self.devdata, _ = self.process_txt(entry_path=os.path.join(data_path, 'dev'),
                                               original_path=os.path.join(data_path, 'dev', 'reference'))
            # self.testdata, _ = self.process_txt(entry_path=os.path.join(data_path, 'test'))
        else:
            # self.traindata, self.vocab = self.process(entry_path=os.path.join(data_path, 'train'), version=version)
            self.devdata, _ = self.process(entry_path=os.path.join(data_path, 'dev'), version=version)
            # self.testdata, _ = self.process(entry_path=os.path.join(data_path, 'test'), version=version)
            # self.testdata, _ = self.process(entry_path=os.path.join(data_path, 'temp'), version=version)

        self.corenlp.close()
        self.temp_extractor.close()

        # json.dump(self.traindata, open(os.path.join(write_path, 'train.json'), 'w'))
        # json.dump(self.vocab, open(os.path.join(write_path, 'vocab.json'), 'w'))
        json.dump(self.devdata, open(os.path.join(write_path, 'dev.json'), 'w'))
        # json.dump(self.testdata, open(os.path.join(write_path, 'test.json'), 'w'))
        # json.dump(self.testdata, open(os.path.join(write_path, 'temp.json'), 'w'))

    def process(self, entry_path, version):
        if version == '1.5':
            data, vocab = self.process_version_1_5(entry_path)
        else:
            data, vocab = self.process_version_1_0(entry_path)
        return data, vocab

    def process_version_1_5(self, entry_path):
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


    def process_version_1_0(self, entry_path):
        data, in_vocab, out_vocab = self.run_parser(entry_path)

        in_vocab = list(set(in_vocab))
        out_vocab = list(set(out_vocab))
        vocab = {'input': in_vocab, 'output': out_vocab}

        print('Path:', entry_path, 'Size: ', len(data))
        return data, vocab

    def process_txt(self, entry_path, original_path):
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

            pre_context = str(reference['pre_context']).split(' ')
            pos_context = str(reference['pos_context']).split(' ')

            entity = '_'.join(reference['entity'].replace('\"', '').replace('\'', '').lower().split())
            if entity != '':
                refex = str(reference['refex']).split(' ')

                isDigit = entity.replace('.', '').strip().isdigit()
                isDate = len(re.findall(re_date, entity)) > 0
                if entity[0] not in ['\'', '\"'] and not isDigit and not isDate:
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
                # .append(TagEntity(tag=tag, entity=entity)

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
                        print('{}\r'.format(template))

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
                            normalized = '_'.join(entity.replace('\"', '').replace('\'', '').lower().split())
                            # aux = context.replace(tag, 'ENTITY', 1)
                            # reference_info = self.process_reference_info(aux, 'ENTITY')

                            refex = ['eos'] + refex.split() + ['eos']

                            mapped_reference = {
                                'entity': normalized,
                                'category': category,
                                'pre_context': pre_context.replace('@', '').split(),
                                'pos_context': pos_context.replace('@', '').split(),
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
        # re_tag = re.compile(r"((AGENT|PATIENT|BRIDGE)\-\d{1})")

        tag = ''
        pre_tag, pos_tag, i = [], [], 0
        try:
            for token in template:
                i += 1
                token_tag = re.search(re_tag, token)
                # re.findall(re_tag, token)

                # if len(token_tag) > 0:
                if token_tag:
                    # token.split('-')[0] in ['AGENT', 'PATIENT', 'BRIDGE']:
                    # tag = ' '.join(token_tag[0]).strip()
                    if token != token_tag.group():
                        print(token)
                    tag = token_tag.group()

                    for pos_token in template[i:]:
                        pos_token_tag = re.search(re_tag, pos_token)
                        if pos_token_tag:
                            # pos_token_tag = re.findall(re_tag, pos_token)
                            # if len(pos_token_tag) > 0:
                            # pos_token.split('-')[0] in ['AGENT', 'PATIENT', 'BRIDGE']:
                            break
                        else:
                            pos_tag.append(pos_token)
                    break
                else:
                    pre_tag.append(token)
        except:
            print("Parsing error (Processing template): ", sys.exc_info()[0])

        return pre_tag, tag, pos_tag

    def process_context(self, text, entity_map):
        context = text.split()
        pre_context, pos_context, i = [], [], 0
        try:
            for token in context:
                i += 1

                token_tag = re.search(re_tag, token)
                # if len(token_tag) > 0:
                if token_tag:
                    # token.split('-')[0] in ['AGENT', 'PATIENT', 'BRIDGE']:
                    pos_context = context[i:]
                    break
                else:
                    pre_context.append(token)

            pre_context = ' '.join(['EOS'] + pre_context)
            pos_context = ' '.join(pos_context + ['EOS'])
            for tag in entity_map:
                pre_context = pre_context.replace(tag, '_'.join(
                    entity_map[tag].replace('\"', '').replace('\'', '').lower().split()))
                pos_context = pos_context.replace(tag, '_'.join(
                    entity_map[tag].replace('\"', '').replace('\'', '').lower().split()))
        except:
            print("Parsing error (Processing template): ", sys.exc_info()[0])

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
            print('Parsing error (tokenize).')

        return tokens


if __name__ == '__main__':
    # write_path='/roaming/tcastrof/emnlp2019/reg'
    # data_path = '/home/tcastrof/Experiments/versions/v1.5/en'
    # stanford_path = r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'

    # data_path = '/home/rossana/Projects/NeuralREG/webnlg/data/v1.5/en'
    # old_format_path = '/home/rossana/Projects/NeuralREG/data/v1.0/old_format'
    # webnlg_format_path = '/home/rossana/Projects/NeuralREG/webnlg/data/v1.0/en'
    # dependencies_path = '/home/rossana/Projects/NeuralREG/webnlg/dependencies/original'

    data_path = '/home/rossana/Projects/NeuralREG/webnlg/data/v1.5/en'
    write_path = '/home/rossana/Projects/NeuralREG/data/v1.0'
    stanford_path = r'/home/rossana/Projects/stanford/stanford-corenlp-full-2018-10-05'

    # data_path = sys.argv[1]
    # write_path = sys.argv[2]
    # stanford_path = sys.argv[3]
    # s = REGPrec(data_path=data_path, write_path=write_path, stanford_path=stanford_path, version='1.0', txt_format=True)
    s = REGPrec(data_path=data_path, write_path=write_path, stanford_path=stanford_path, version='1.5')
