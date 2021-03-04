__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 24/07/2017
Description:
    REFERRING EXPRESSION COLLECTION:
    Preprocessing script in order to have a train, dev and test set as well as input, output
    (word- and character-based) vocabularies.

    Development set consist in 10% of the original training set, whereas test set is the original dev set

    For each instance, context (pre and pos), status (text and sentence), syntax, entity id, referential form,
    referring expression and domain info are extracted

    INPUT CONSTANTS:
        IN_PATH: directory for the delexicalized WebNLG
        OUT_PATH: path to write the result
"""

import argparse
import copy
import json
import os
import random
import re
import traceback
import xml.etree.ElementTree as ET

import nltk

class Preprocessing:
    def __init__(self, in_file, out_file):
        try:
            self.in_train = os.path.join(in_file, 'train')
            self.in_dev = os.path.join(in_file, 'dev')

            if not os.path.exists(out_file):
                os.mkdir(out_file)
            self.out_vocab = out_file
            self.out_train = os.path.join(out_file, 'train')
            self.out_dev = os.path.join(out_file, 'dev')
            self.out_test = os.path.join(out_file, 'test')

            self.text_id = 0
            self.trainset()
            self.testset()
        except:
            print(traceback.format_exc())


    def trainset(self):
        input_vocab, output_vocab, character_vocab = set(), set(), set()
        train, dev = [], []
        train_info, dev_info = [], []

        dirs = filter(lambda x: not str(x).startswith('.'), os.listdir(self.in_train))
        for path in dirs:
            dirs2 = filter(lambda x: not str(x).startswith('.'), os.listdir(os.path.join(self.in_train, path)))
            for fname in dirs2:
                f = open(os.path.join(self.in_train, path, fname))

                data, in_vocab, out_vocab, c_vocab = self.annotation_parse(f)

                input_vocab = input_vocab.union(in_vocab)
                output_vocab = output_vocab.union(out_vocab)
                character_vocab = character_vocab.union(c_vocab)

                text_ids = list(set(map(lambda x: x['text_id'], data)))

                train_size = int(0.9 * len(text_ids))

                random.shuffle(text_ids)
                train.extend(filter(lambda x: x['text_id'] in text_ids[:train_size], data))
                dev.extend(filter(lambda x: x['text_id'] in text_ids[train_size:], data))

                info = len(train) * [path + ' ' + fname]
                train_info.extend(info)

                info = len(dev) * [path + ' ' + fname]
                dev_info.extend(info)

        self.write(self.out_train, train, train_info)
        self.write(self.out_dev, dev, dev_info)

        with open(os.path.join(self.out_vocab, 'input_vocab.txt'), 'w') as f:
            f.write('\n'.join(list(input_vocab)))

        with open(os.path.join(self.out_vocab, 'output_vocab.txt'), 'w') as f:
            f.write('\n'.join(list(output_vocab)))

        with open(os.path.join(self.out_vocab, 'character_vocab.txt'), 'w') as f:
            f.write('\n'.join(list(character_vocab)))


    def testset(self):
        test = []
        test_info = []

        dirs = filter(lambda x: not str(x).startswith('.'), os.listdir(self.in_dev))
        for path in dirs:
            dirs2 = filter(lambda x: not str(x).startswith('.'), os.listdir(os.path.join(self.in_dev, path)))
            for fname in dirs2:
                f = open(os.path.join(self.in_dev, path, fname))

                data, in_vocab, out_vocab, c_vocab = self.annotation_parse(f)

                test.extend(data)

                info = len(data) * [path + ' ' + fname]
                test_info.extend(info)

        self.write(self.out_test, test, test_info)

    def extract_entity_type(self, entity):
        aux = entity.split('^^')
        if len(aux) > 1:
            return aux[-1]

        aux = entity.split('@')
        if len(aux) > 1:
            return aux[-1]

        return 'wiki'

    def annotation_parse(self, doc):
        '''
        Parse an annotation document and extract references from the texts
        :param doc:
        :return:
        '''
        tree = ET.parse(doc)
        root =  tree.getroot()

        data = []
        input_vocab, output_vocab, character_vocab = set(), set(), set()

        entries = root.find('entries')
        for entry in entries:
            entryId = entry.attrib['eid']
            size = entry.attrib['size']
            semcategory = entry.attrib['category']

            # get entity map
            entitymap_xml = entry.find('entitymap')
            entity_map = {}
            for inst in entitymap_xml:
                tag, entity = inst.text.split(' | ')
                entity_map[tag] = entity

            # Reading original triples to extract the entities type
            types = []
            otripleset = entry.find('originaltripleset')
            for otriple in otripleset:
                e1, pred, e2 = otriple.text.split(' | ')

                entity1_type = self.extract_entity_type(e1.strip())
                entity2_type = self.extract_entity_type(e2.strip())

                types.append({'e1_type':entity1_type, 'e2_type':entity2_type})

            # Reading modified triples to extract entities and classify them according to type
            mtripleset = entry.find('modifiedtripleset')
            entity_type = {}
            for i, mtriple in enumerate(mtripleset):
                e1, pred, e2 = mtriple.text.split(' | ')

                entity_type[e1.replace('\'', '')] = types[i]['e1_type']
                entity_type[e2.replace('\'', '')] = types[i]['e2_type']

            lexEntries = entry.findall('lex')

            for lex in lexEntries:
                try:
                    text = lex.find('text').text
                    template = lex.find('template').text

                    if template:
                        print('{}\r'.format(template))
                        text, template = self.stanford_parse(text, template)
                        references, in_vocab, out_vocab, c_vocab = self.get_refexes(text, template, entity_map, entity_type)
                        data.extend(references)
                        input_vocab = input_vocab.union(in_vocab)
                        output_vocab = output_vocab.union(out_vocab)
                        character_vocab = character_vocab.union(c_vocab)
                except:
                    print(traceback.format_exc())

        return data, input_vocab, output_vocab, character_vocab


    def stanford_parse(self, text, template):
        '''
        Tokenizing text and template
        :param text: original text
        :param template: original template
        :return: Tokenized text and template
        '''
        text = []
        for snt in nltk.sent_tokenize(text.strip()):
            text.extend(nltk.word_tokenize(snt)
        text = ' '.join(text).strip()

        # out = self.proc.parse_doc(text)
        # text = []
        # for i, snt in enumerate(out['sentences']):
        #     text.extend(snt['tokens'])
        # text = ' '.join(text).replace('-LRB-', '(').replace('-RRB-', ')').strip()

        temp = []
        for snt in nltk.sent_tokenize(template.strip()):
            temp.extend(nltk.word_tokenize(snt))
        template = ' '.join(temp).strip()

        # out = self.proc.parse_doc(template)
        # temp = []
        # for i, snt in enumerate(out['sentences']):
        #     temp.extend(snt['tokens'])
        # template = ' '.join(temp).replace('-LRB-', '(').replace('-RRB-', ')').strip()

        return text, template


    def write(self, fname, instances, info):
        if not os.path.exists(fname):
            os.mkdir(fname)

        pre_context = '\n'.join(map(lambda x: x['pre_context'], instances))
        with open(os.path.join(fname, 'pre_context.txt'), 'w') as f:
            f.write(pre_context)
        pos_context = '\n'.join(map(lambda x: x['pos_context'], instances))
        with open(os.path.join(fname, 'pos_context.txt'), 'w') as f:
            f.write(pos_context)
        entity = '\n'.join(map(lambda x: x['entity'], instances))
        with open(os.path.join(fname, 'entity.txt'), 'w') as f:
            f.write(entity)
        refex = '\n'.join(map(lambda x: x['refex'], instances))
        with open(os.path.join(fname, 'refex.txt'), 'w') as f:
            f.write(refex)
        size = '\n'.join(map(lambda x: str(x['size']), instances))
        with open(os.path.join(fname, 'size.txt'), 'w') as f:
            f.write(size)
        info = '\n'.join(info)
        with open(os.path.join(fname, 'info.txt'), 'w') as f:
            f.write(info)

        json.dump(instances, open(os.path.join(fname, 'data.json'), 'w'))


    def get_reference_info(self, template, tag):
        '''
        get info about a reference like syntactic position
        :param out: stanford corenlp result
        :param tag: tag (agent, patient or bridge)
        :param entity: wikipedia id
        :return:
        '''

        reference = {'sentence':-1, 'pos':-1, 'general_pos':-1, 'tag':tag}
        general_pos = 0
        sentences = nltk.sent_tokenize(template)
        for i, snt in sentences:
            tokens = nltk.word_tokenize(snt)
            for j, token in enumerate(tokens):
                # get syntax
                if token == tag:
                    reference = {'sentence':i, 'pos':j, 'general_pos':general_pos+j, 'tag':tag}
                    break
            general_pos += len(tokens)
        return reference


    def process_template(self, template):
        '''
        Return previous and subsequent tokens from a specific tag in a template
        :param template:
        :return:
        '''
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


    def process_context(self, context, entity_map):
        '''
        Return pre- and pos- wikified context
        :param context:
        :param entity_map:
        :return:
        '''
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
        for tag in entity_map:
            # pre_context = pre_context.replace(tag, entity_map[tag])
            # pos_context = pos_context.replace(tag, entity_map[tag])
            pre_context = pre_context.replace(tag, '_'.join(entity_map[tag].replace('\"', '').replace('\'', '').split()))
            pos_context = pos_context.replace(tag, '_'.join(entity_map[tag].replace('\"', '').replace('\'', '').split()))

        return pre_context, pos_context


    def classify(self, references):
        '''
        Classify referring expression by their status and form
        :param references:
        :return:
        '''
        references = sorted(references, key=lambda x: (x['entity'], x['sentence'], x['pos']))

        sentence_statuses = {}
        for i, reference in enumerate(references):
            # text status
            if i == 0 or (reference['entity'] != references[i-1]['entity']):
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
            if reg.lower().strip() in ['he', 'his', 'him', 'she', 'hers', 'her', 'it', 'its', 'we', 'our', 'ours', 'they', 'theirs', 'them']:
                reference['reftype'] = 'pronoun'
            elif reg.lower().strip().split()[0] in ['the', 'a', 'an']:
                reference['reftype'] = 'description'
            elif reg.lower().strip().split()[0] in ['this', 'these', 'that', 'those']:
                reference['reftype'] = 'demonstrative'

        return references


    def get_refexes(self, text, template, entity_map, entity_type):
        '''
        Extract referring expressions for each reference overlapping text and template
        :param text: original text
        :param template: template (delexicalized text)
        :return:
        '''
        context = copy.copy(template)

        data, input_vocab, output_vocab, character_vocab = [], set(), set(), set()

        isOver = False
        while not isOver:
            pre_tag, tag, pos_tag = self.process_template(template)
            pre_context, pos_context = self.process_context(context, entity_map)

            if tag == '':
                isOver = True
            else:
                # Look for reference from 5-gram to 2-gram
                i, f = 5, []
                while i > 1:
                    begin = ' '.join(i * ['BEGIN'])
                    text = begin + ' ' + text
                    template = begin + ' ' + template
                    pre_tag, tag, pos_tag = self.process_template(template)

                    regex = re.escape(' '.join(pre_tag[-i:]).strip()) + ' (.+?) ' + re.escape(' '.join(pos_tag[:i]).strip())
                    f = re.findall(regex, text)

                    template = template.replace('BEGIN', '').strip()
                    text = text.replace('BEGIN', '').strip()
                    i -= 1

                    if len(f) == 1:
                        break

                if len(f) > 0:
                    # DO NOT LOWER CASE HERE!!!!!!
                    template = template.replace(tag, f[0], 1)
                    refex = f[0]

                    # Do not include literals
                    entity = entity_map[tag]
                    if entity_type[entity] == 'wiki':
                        normalized = '_'.join(entity.replace('\"', '').replace('\'', '').split())
                        aux = context.replace(tag, 'ENTITY', 1)
                        reference = self.get_reference_info(aux, 'ENTITY')

                        character = ['eos'] + list(refex) + ['eos']
                        refex = ['eos'] + refex.split() + ['eos']
                        row = {
                            'pre_context':pre_context.replace('@', ''),
                            'pos_context':pos_context.replace('@', ''),
                            'entity':normalized,
                            'refex':' '.join(refex),
                            'size':len(entity_map.keys()),
                            'text_id':self.text_id,
                            'general_pos':reference['general_pos'],
                            'sentence':reference['sentence'],
                            'pos':reference['pos'],
                            'text':text
                        }
                        data.append(row)
                        output_vocab = output_vocab.union(set(refex))
                        character_vocab = character_vocab.union(set(character))
                        input_vocab = input_vocab.union(set(pre_context.split()))
                        input_vocab = input_vocab.union(set(pos_context.split()))
                        input_vocab = input_vocab.union(set([normalized]))

                        context = context.replace(tag, normalized, 1)
                    else:
                        context = context.replace(tag, '_'.join(entity_map[tag].replace('\"', '').replace('\'', '').split()), 1)
                else:
                    template = template.replace(tag, ' ', 1)
                    context = context.replace(tag, '_'.join(entity_map[tag].replace('\"', '').replace('\'', '').split()), 1)

        self.text_id += 1
        data = self.classify(data)
        return data, input_vocab, output_vocab, character_vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing train, dev and test sets.')
    parser.add_argument('in_path', help='directory for the delexicalized WebNLG dataset')
    parser.add_argument('out_path', help='path to write the result')

    args = parser.parse_args()
    try:
        IN_PATH = args.in_path
        OUT_PATH = args.out_path
    except:
        IN_PATH = 'webnlg/delexicalized'
        OUT_PATH='new_data/'
    Preprocessing(in_file=IN_PATH, out_file=OUT_PATH)