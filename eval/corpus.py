__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 12/12/2017
Description:
    Script for generating information about the corpus as number of distinct sets of triples,
    texts, templates and entities.

    UPDATE CONSTANTS:
        TRAIN_FILE
        DEV_FILE

        TRAIN_REFEX_FILE
        DEV_REFEX_FILE
        TEST_REFEX_FILE
"""

import os
import xml.etree.ElementTree as ET

import cPickle as p
import nltk

from stanford_corenlp_pywrapper import CoreNLP

# PATHS FOR TRAINING AND DEVELOPMENT SETS OF DELEXICALIZED WEBNLG
TRAIN_FILE = '../annotation/final/train'
DEV_FILE = '../annotation/final/dev'

# PATH FOR REFERRING EXPRESSION COLLECTIONS
TRAIN_REFEX_FILE = '../data/v1.0/train/data.cPickle'
DEV_REFEX_FILE = '../data/v1.0/dev/data.cPickle'
TEST_REFEX_FILE = '../data/v1.0/test/data.cPickle'

def corpus_info():
    '''
    Number of sets of triples, texts and templates
    :return:
    '''
    def read(in_file):
        inputs = 0

        dirs = filter(lambda x: x != '.DS_Store', os.listdir(in_file))
        for path in dirs:
            dirs2 = filter(lambda x: x != '.DS_Store', os.listdir(os.path.join(in_file, path)))
            for fname in dirs2:
                # print os.path.join(in_file, path, fname)
                tree = ET.parse(os.path.join(in_file, path, fname))
                root = tree.getroot()

                entries = root.find('entries')
                for entry in entries:
                    lexEntries = entry.findall('lex')
                    inputs = inputs + 1
                    for lex in lexEntries:
                        try:
                            text = lex.find('text').text
                            template = lex.find('template').text

                            if template and text:
                                texts.append(text.lower())
                                templates.append(template.lower())
                        except:
                            pass
        return inputs

    texts = []
    templates = []

    inputs = read(TRAIN_FILE)
    _inputs = read(DEV_FILE)
    inputs += _inputs

    print 'Number of Representations: ', inputs

    print 'Sentences: ', len(texts)
    print 'Distinct Sentences: ', len(list(set(texts)))

    print 'Templates: ', len(templates)
    print 'Distinct Templates: ', len(list(set(templates)))

def corpus_entities_count():
    '''
    Number of entities in training, development and test sets
    :return:
    '''
    def pronoun_count(data):
        num = 0
        for reference in data:
            if reference['refex'] in ['he', 'his', 'him', 'she', 'hers', 'her', 'it', 'its', 'we', 'our', 'ours', 'they', 'theirs', 'them']:
                num += 1
        return num
    train_data = p.load(open(TRAIN_REFEX_FILE))
    dev_data = p.load(open(DEV_REFEX_FILE))
    test_data = p.load(open(TEST_REFEX_FILE))

    train_entities = set(map(lambda x: x['entity'], train_data))
    dev_entities = set(map(lambda x: x['entity'], dev_data))
    test_entities = set(map(lambda x: x['entity'], test_data))

    print 'Train entities: ', len(list(train_entities))
    print 'Dev entities: ', len(list(dev_entities))
    print 'Test entities: ', len(list(test_entities))

    dev_intersect = train_entities.intersection(dev_entities)
    print 'Train intersect dev: ', str(len(list(dev_intersect)))

    test_intersect = train_entities.intersection(test_entities)
    print 'Train intersect test: ', str(len(list(test_intersect)))

def entity_forms():
    '''
    Number of entities per referential form
    :return:
    '''
    train_data = p.load(open(TRAIN_REFEX_FILE))
    dev_data = p.load(open(DEV_REFEX_FILE))
    test_data = p.load(open(TEST_REFEX_FILE))

    train_entities = map(lambda x: x['reftype'], train_data)
    dev_entities = map(lambda x: x['reftype'], dev_data)
    test_entities = map(lambda x: x['reftype'], test_data)

    print 'TRAIN:'
    count = len(filter(lambda x: x == 'name', train_entities))
    print 'Proper Names:', count, float(count)/len(train_entities)

    count = len(filter(lambda x: x == 'pronoun', train_entities))
    print 'Pronoun:', count, float(count)/len(train_entities)

    count = len(filter(lambda x: x == 'description', train_entities))
    print 'Description:', count, float(count)/len(train_entities)

    count = len(filter(lambda x: x == 'demonstrative', train_entities))
    print 'Demonstrative:', count, float(count)/len(train_entities)

    print 'DEV:'
    count = len(filter(lambda x: x == 'name', dev_entities))
    print 'Proper Names:', count, float(count)/len(dev_entities)

    count = len(filter(lambda x: x == 'pronoun', dev_entities))
    print 'Pronoun:', count, float(count)/len(dev_entities)

    count = len(filter(lambda x: x == 'description', dev_entities))
    print 'Description:', count, float(count)/len(dev_entities)

    count = len(filter(lambda x: x == 'demonstrative', dev_entities))
    print 'Demonstrative:', count, float(count)/len(dev_entities)

    print 'TEST:'
    count = len(filter(lambda x: x == 'name', test_entities))
    print 'Proper Names:', count, float(count)/len(test_entities)

    count = len(filter(lambda x: x == 'pronoun', test_entities))
    print 'Pronoun:', count, float(count)/len(test_entities)

    count = len(filter(lambda x: x == 'description', test_entities))
    print 'Description:', count, float(count)/len(test_entities)

    count = len(filter(lambda x: x == 'demonstrative', test_entities))
    print 'Demonstrative:', count, float(count)/len(test_entities)

    total_entities = train_entities + dev_entities + test_entities
    print '\nTOTAL:'
    count = len(filter(lambda x: x == 'name', total_entities))
    print 'Proper Names:', count, float(count)/len(total_entities)

    count = len(filter(lambda x: x == 'pronoun', total_entities))
    print 'Pronoun:', count, float(count)/len(total_entities)

    count = len(filter(lambda x: x == 'description', total_entities))
    print 'Description:', count, float(count)/len(total_entities)

    count = len(filter(lambda x: x == 'demonstrative', total_entities))
    print 'Demonstrative:', count, float(count)/len(total_entities)

def entity_ner():
    '''
    Named entity types of the entities
    :return:
    '''
    def get_stats(dataset, setname):
        stats = []
        for text, refex in dataset:
            refex_tokens = refex.split()
            out = proc.parse_doc(text)

            tokens, ners = [], []
            for snt in out['sentences']:
                tokens.extend(snt['tokens'])
                ners.extend(snt['ner'])

            for i, token in enumerate(tokens):
                found = True
                if refex_tokens[0] == token:
                    for j, refex_token in enumerate(refex_tokens):
                        if refex_token != tokens[i+j]:
                            found = False
                            break

                    if found:
                        ner = ners[i]
                        stats.append(ner)
                        break

        print setname
        freq = dict(nltk.FreqDist(stats))
        total = sum(freq.values())
        for name, freq in freq.iteritems():
            print name, freq, float(freq)/total
        print 10 * '-'

    proc = CoreNLP('ner')

    train_data = p.load(open(TRAIN_REFEX_FILE))
    dev_data = p.load(open(DEV_REFEX_FILE))
    test_data = p.load(open(TEST_REFEX_FILE))

    train_refex = map(lambda x: (x['text'], x['refex'].replace('eos', '').strip()), train_data)
    dev_refex = map(lambda x: (x['text'], x['refex'].replace('eos', '').strip()), dev_data)
    test_refex = map(lambda x: (x['text'], x['refex'].replace('eos', '').strip()), test_data)

    get_stats(train_refex, 'TRAIN')
    get_stats(dev_refex, 'DEV')
    get_stats(test_refex, 'TEST')

if __name__ == '__main__':
    corpus_info()
    print '\n\n'

    corpus_entities_count()
    print '\n\n'

    entity_forms()
    print '\n\n'

    entity_ner()