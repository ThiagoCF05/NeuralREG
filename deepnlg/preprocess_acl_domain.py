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

import json
import os
import sys


sys.path.append('./')
sys.path.append('../')
ORIGINAL_PATH = '../data/v1.0/acl_format/'


def load_data(entry_path, original_path):
    fdataset = json.load(open(entry_path))
    foriginal = json.load(open(original_path))

    return fdataset, foriginal


class REGPrecACLDomain:
    def __init__(self, data_path, write_path):
        self.data_path = data_path
        self.write_path = write_path

        self.traindata = self.process(entry_path=os.path.join(data_path, 'train.json'), original_path=os.path.join(ORIGINAL_PATH, 'train.json'))
        self.devdata = self.process(entry_path=os.path.join(data_path, 'dev.json'), original_path=os.path.join(ORIGINAL_PATH, 'dev.json'))
        self.testdata = self.process(entry_path=os.path.join(data_path, 'test.json'), original_path=os.path.join(ORIGINAL_PATH, 'test.json'))

        json.dump(self.traindata, open(os.path.join(write_path, 'train.json'), 'w'))
        json.dump(self.devdata, open(os.path.join(write_path, 'dev.json'), 'w'))
        json.dump(self.testdata, open(os.path.join(write_path, 'test.json'), 'w'))

    def process(self, entry_path, original_path):
        entry_set, original = load_data(entry_path, original_path)
        data = []

        for i, reference in enumerate(entry_set):
            reference['category'] = original[i]['category']
            data.append(reference)

        print('Path:', entry_path)

        return data


if __name__ == '__main__':
    data_path = '../data/v1.0/'
    write_path = '../data/v1.0'

    # data_path = sys.argv[1]
    # write_path = sys.argv[2]
    s = REGPrecACLDomain(data_path=data_path, write_path=write_path)
