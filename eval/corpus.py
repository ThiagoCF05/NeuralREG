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

# PATHS FOR TRAINING AND DEVELOPMENT SETS OF DELEXICALIZED WEBNLG
TRAIN_FILE = 'annotation/final/train'
DEV_FILE = 'annotation/final/dev'

# PATH FOR REFERRING EXPRESSION COLLECTIONS
TRAIN_REFEX_FILE = 'data/train/data.cPickle'
DEV_REFEX_FILE = 'data/dev/data.cPickle'
TEST_REFEX_FILE = 'data/test/data.cPickle'

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

if __name__ == '__main__':
    corpus_info()

    print '\n\n'

    corpus_entities_count()