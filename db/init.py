__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 02/05/2017
Description:
    This script aims to populate the database with training and dev sets of the WebNLG Challenge
"""

import sys
sys.path.append('../')
import json
import os
import xml.etree.ElementTree as ET

from db import operations as dbop
sys.path.append('/home/tcastrof/workspace/stanford_corenlp_pywrapper')
from stanford_corenlp_pywrapper import CoreNLP

class DBInit(object):
    def __init__(self):
        self.proc = CoreNLP('parse')

    def run(self, dir, typeset):
        self.typeset = typeset

        for fname in os.listdir(dir):
            if fname != '.DS_Store':
                self.proc_file(os.path.join(dir, fname))

    def extract_entity_type(self, entity):
        aux = entity.split('^^')
        if len(aux) > 1:
            return aux[-1]

        aux = entity.split('@')
        if len(aux) > 1:
            return aux[-1]

        return 'wiki'

    def extract_parse_tree(self, text):
        out = self.proc.parse_doc(text)

        parse_trees = []
        for snt in out['sentences']:
            parse_trees.append(snt['parse'])

        if len(parse_trees) > 1:
            parse = '(MULTI-SENTENCE '
            for tree in parse_trees:
                parse += tree + ' '
            parse = parse.strip() + ')'
        else:
            parse = parse_trees[0]
        return parse

    def proc_file(self, fname):
        tree = ET.parse(fname)
        root = tree.getroot()

        entries = root.find('entries')

        for _entry in entries:
            entry = dbop.save_entry(docid=_entry.attrib['eid'], size=int(_entry.attrib['size']), category=_entry.attrib['category'], set=self.typeset)

            entities_type = []

            # Reading original triples to extract the entities type
            otripleset = _entry.find('originaltripleset')
            for otriple in otripleset:
                e1, pred, e2 = otriple.text.split(' | ')

                entity1_type = self.extract_entity_type(e1.strip())
                entity2_type = self.extract_entity_type(e2.strip())

                types = {'e1_type':entity1_type, 'e2_type':entity2_type}
                entities_type.append(types)

            # Reading modified triples to extract entities and predicate
            mtripleset = _entry.find('modifiedtripleset')
            for i, mtriple in enumerate(mtripleset):
                e1, pred, e2 = mtriple.text.split(' | ')

                entity1 = dbop.save_entity(name=e1.replace('\'', '').strip(), type=entities_type[i]['e1_type'], ner='', category='', description='')

                predicate = dbop.save_predicate(pred)

                entity2 = dbop.save_entity(e2.replace('\'', '').strip(), entities_type[i]['e2_type'], ner='', category='', description='')

                triple = dbop.save_triple(entity1, predicate, entity2)

                dbop.add_triple(entry, triple)

            # process lexical entries
            lexEntries = _entry.findall('lex')
            for lexEntry in lexEntries:
                text = lexEntry.text.strip()
                parse_tree = self.extract_parse_tree(text)
                lexEntry = dbop.save_lexEntry(docid=lexEntry.attrib['lid'], comment=lexEntry.attrib['comment'], text=text, parse_tree=parse_tree)

                dbop.add_lexEntry(entry, lexEntry)

if __name__ == '__main__':
    dbop.clean()

    dbinit = DBInit()

    # TEST SET
    TEST_DIR = '../data/cyber/test'
    for dir in os.listdir(TEST_DIR):
        if dir != '.DS_Store':
            f = os.path.join(TEST_DIR, dir)
            print f
            dbinit.run(f, 'test')

    # TRAIN SET
    TRAIN_DIR = '../data/cyber/train'
    for dir in os.listdir(TRAIN_DIR):
        if dir != '.DS_Store':
            f = os.path.join(TRAIN_DIR, dir)
            print f
            dbinit.run(f, 'train')

    # DEV SET
    DEV_DIR = '../data/cyber/dev'
    for dir in os.listdir(DEV_DIR):
        if dir != '.DS_Store':
            f = os.path.join(DEV_DIR, dir)
            print f
            dbinit.run(f, 'dev')