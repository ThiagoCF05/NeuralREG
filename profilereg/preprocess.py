__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 30/03/2020
Description:
    This script aims to generate WebNLG v1.5 dataset for the ProfileREG API

    ARGS:
        [1] Path to the folder where ACL format from WebNLG corpus is available (versions/v1.0/acl_format)
        [2] Path to the folder where the data will be saved (Folder will be created in case it does not exist)

    EXAMPLE:
        python3 preprocess.py '../data/v1.5' '../eval/data/profilereg/references/v1.5'
"""

import json
import os
import sys
import pickle

sys.path.append('./')
sys.path.append('../')
profilereg_path = '../eval/data/profilereg/references/'


def load_data(entry_path):
    data = json.load(open(entry_path, encoding='utf-8'))

    profile = json.load(open(os.path.join(profilereg_path, 'profile.json'), encoding='utf-8'))

    return list(data), list(profile)


class REGPrecProfileREG:
    def __init__(self, data_path, write_path):
        self.data_path = data_path
        self.write_path = write_path

        self.traindata = self.process(entry_path=os.path.join(data_path, 'train.json'))
        self.devdata = self.process(entry_path=os.path.join(data_path, 'dev.json'))
        self.testdata = self.process(entry_path=os.path.join(data_path, 'test.json'))

        # json.dump(self.traindata, open(os.path.join(write_path, 'profile_train.json'), 'w'))
        # json.dump(self.devdata, open(os.path.join(write_path, 'profile_dev.json'), 'w'))
        # json.dump(self.testdata, open(os.path.join(write_path, 'profile_test.json'), 'w'))

        pickle.dump(self.traindata, open(write_path + 'webnlg_train.pickle', 'wb'))
        pickle.dump(self.devdata, open(write_path + 'webnlg_dev.pickle', 'wb'))
        pickle.dump(self.testdata, open(write_path + 'webnlg_test.pickle', 'wb'))

    def process(self, entry_path):
        entry_set, profile_set = load_data(entry_path)
        data, size = [], 0

        print('Data path: ', entry_path)
        for i, reference in enumerate(entry_set):
            reference['pre_context'] = 'eos ' + ' '.join(reference['pre_context']).lower().strip()
            reference['pos_context'] = ' '.join(reference['pos_context']).lower().strip() + ' eos'
            reference['refex'] = 'eos ' + ' '.join(reference['refex']).lower().strip() + ' eos'
            reference['entity'] = reference['entity'].lower()

            entities = list(
                filter(lambda o: o['entity'].lower().strip() == reference['entity'].lower().strip(), profile_set))
            reference['profile'] = ''
            if len(entities) > 0:
                reference['profile'] = entities[0]['profile']
            else:
                reference['profile'] = 'eos ' + ' '.join(reference['entity'].lower().split('_')).strip() + ' eos'

            data.append(reference)
            size += 1

        return data


if __name__ == '__main__':
    # data_path = '../data/v1.5'
    # write_path = '../eval/data/profilereg/references/v1.5'

    data_path = sys.argv[1]
    write_path = sys.argv[2]
    s = REGPrecProfileREG(data_path=data_path, write_path=write_path)
