__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 28/02/2019
Description:
    This script aims to extract profile information from ProfileREG API.

    ARGS:
        [1] Path to the folder where ACL format from WebNLG corpus is available (versions/v1.5)
        [2] Path to the folder where the data will be saved (Folder will be created in case it does not exist)

    EXAMPLE:
        python3 preprocess_profile.py '../data/v1.5' '../eval/data/profilereg/references/'
"""

import json
import os
import sys
import pickle

sys.path.append('./')
sys.path.append('../')
profilereg_path = '../eval/data/profilereg/references/v1.0'


def load_data(train_path, dev_path, test_path):
    with open(train_path, 'rb') as file:
        train_data = pickle.load(file)

    with open(dev_path, 'rb') as file:
        dev_data = pickle.load(file)

    with open(test_path, 'rb') as file:
        test_data = pickle.load(file)

    reference_data = list(train_data) + list(dev_data) + list(test_data)

    return list(reference_data)


class REGPrecProfileREG:
    def __init__(self, data_path, write_path):
        self.data_path = data_path
        self.write_path = write_path

        self.profile_data = self.process(train_path=os.path.join(profilereg_path, 'train.pickle'),
                                         dev_path=os.path.join(profilereg_path, 'dev.pickle'),
                                         test_path=os.path.join(profilereg_path, 'test.pickle'))

        json.dump(self.profile_data, open(os.path.join(write_path, 'profile.json'), 'w'))

    def process(self, train_path, dev_path, test_path):
        profile_set = load_data(train_path, dev_path, test_path)
        data, size = [], 0

        for i, reference in enumerate(profile_set):
            entity = {'entity': reference['entity'],
                      'profile': reference['profile']}

            if entity not in data:
                data.append(entity)
                size += 1

        return data


if __name__ == '__main__':
    # data_path = '../data/v1.5'
    # write_path = '../eval/data/profilereg/references/'

    data_path = sys.argv[1]
    write_path = sys.argv[2]
    s = REGPrecProfileREG(data_path=data_path, write_path=write_path)
