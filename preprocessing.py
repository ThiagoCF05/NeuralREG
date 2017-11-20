__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 24/07/2017
Description:
    Preprocessing script in order to have a train, dev and test set as well as input, output
    (word- and character-based) vocabularies.

    Development set consist in 10% of the original training set, whereas test set is the original dev set

    For each instance, context (pre and pos), status (text and sentence), syntax, entity id, referential form,
    referring expression and domain info are extracted
"""

import copy
import cPickle as p
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
        self.parser = CoreNLP('parse')
        self.ftrain = ftrain
        self.fdev = fdev

        self.run()


    def run(self):
        self.text_id = 0
        input_vocab, output_vocab, character_vocab = set(), set(), set()
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

                data, in_vocab, out_vocab, c_vocab = self.annotation_parse(doc)

                input_vocab = input_vocab.union(in_vocab)
                output_vocab = output_vocab.union(out_vocab)
                character_vocab = character_vocab.union(c_vocab)

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

        with open('data/character_vocab.txt', 'w') as f:
            f.write(('\n'.join(list(character_vocab))).encode("utf-8"))


    def write(self, fname, instances, info):
        if not os.path.exists(fname):
            os.mkdir(fname)

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

        p.dump(instances, open(os.path.join(fname, 'data.cPickle'), 'w'))


    def check_entity(self, entity):
        f = Entity.objects(name=entity.strip(), type='wiki')
        if f.count() > 0:
            return '_'.join(entity.replace('\"', '').replace('\'', '').lower().split())
        else:
            return ''


    def get_reference_info(self, template, tag):
        '''
        get info about a reference like syntactic position
        :param out: stanford corenlp result
        :param tag: tag (agent, patient or bridge)
        :param entity: wikipedia id
        :return:
        '''
        out = self.parser.parse_doc(template)['sentences']
        reference = {'syntax':'', 'sentence':-1, 'pos':-1, 'general_pos':-1, 'tag':tag}
        general_pos = 0
        for i, snt in enumerate(out):
            deps = snt['deps_cc']
            for dep in deps:
                # get syntax
                if snt['tokens'][dep[2]] == tag:
                    reference = {'syntax':'', 'sentence':i, 'pos':dep[2], 'general_pos':general_pos+dep[2], 'tag':tag}
                    if 'nsubj' in dep[0] or 'nsubjpass' in dep[0]:
                        reference['syntax'] = 'np-subj'
                    elif 'nmod:poss' in dep[0] or 'compound' in dep[0]:
                        reference['syntax'] = 'subj-det'
                    else:
                        reference['syntax'] = 'np-obj'
                    break
            general_pos += len(snt['tokens'])
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
        for tag, entity in entity_map.iteritems():
            pre_context = pre_context.replace(tag, entity_map[tag])
            pos_context = pos_context.replace(tag, entity_map[tag])

        return pre_context.lower(), pos_context.lower()


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


    def get_refexes(self, text, template, entity_map):
        '''
        Extract referring expressions for each reference overlapping text and template
        :param text: original text
        :param template: template (delexicalized text)
        :return:
        '''
        text = 'BEGIN BEGIN BEGIN ' + text
        context = copy.copy(template)
        template = 'BEGIN BEGIN BEGIN ' + template

        data, input_vocab, output_vocab, character_vocab = [], set(), set(), set()

        isOver = False
        while not isOver:
            pre_tag, tag, pos_tag = self.process_template(template)
            pre_context, pos_context = self.process_context(context, entity_map)

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
                            'syntax':reference['syntax'],
                            'text_id':self.text_id,
                            'general_pos':reference['general_pos'],
                            'sentence':reference['sentence'],
                            'pos':reference['pos']
                        }
                        data.append(row)
                        output_vocab = output_vocab.union(set(refex))
                        character_vocab = character_vocab.union(set(character))
                        input_vocab = input_vocab.union(set(pre_context.split()))
                        input_vocab = input_vocab.union(set(pos_context.split()))
                        input_vocab = input_vocab.union(set([normalized]))

                        context = context.replace(tag, normalized, 1)
                    else:
                        context = context.replace(tag, entity_map[tag], 1)
                else:
                    template = template.replace(tag, ' ', 1)
                    context = context.replace(tag, entity_map[tag], 1)

        self.text_id += 1
        data = self.classify(data)
        return data, input_vocab, output_vocab, character_vocab


    def annotation_parse(self, doc):
        '''
        Parse an annotation document and extract references from the texts
        :param doc:
        :return:
        '''
        data = []
        input_vocab, output_vocab, character_vocab = set(), set(), set()
        for entry in doc:
            entry = entry.split('\n\n')

            try:
                _, entryId, size, semcategory = entry[0].replace('\n', '').split()

                entity_map = dict(map(lambda entity: entity.split(' | '), entry[2].replace('\nENTITY MAP\n', '').split('\n')))

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
                        references, in_vocab, out_vocab, c_vocab = self.get_refexes(text, template, entity_map)
                        data.extend(references)
                        input_vocab = input_vocab.union(in_vocab)
                        output_vocab = output_vocab.union(out_vocab)
                        character_vocab = character_vocab.union(c_vocab)
                    elif correct != '' and comment != 'wrong':
                        if correct.strip() == 'CORRECT:':
                            correct = template
                        print('{}\r'.format(correct))
                        text, template = self.stanford_parse(text, correct)
                        references, in_vocab, out_vocab, c_vocab = self.get_refexes(text, template, entity_map)
                        data.extend(references)
                        input_vocab = input_vocab.union(in_vocab)
                        output_vocab = output_vocab.union(out_vocab)
                        character_vocab = character_vocab.union(c_vocab)
            except:
                print('ERROR')

        return data, input_vocab, output_vocab, character_vocab


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