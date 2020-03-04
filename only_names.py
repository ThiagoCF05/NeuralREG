__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 12/12/2017
Description:
    Only Names model

    PYTHON VERSION :2.7

    DEPENDENCIES:
        cPickle

    UPDATE CONSTANT PATHS:
        IN_FILE: path to reference collection to be realized
        OUT_FILE: path to save results
"""

import json

class OnlyNames():
    def __init__(self, in_file, out_file):
        self.references = json.load(open(in_file))
        self.out_file = out_file

        self.run()

    def run(self):
        results = []

        for i, testinst in enumerate(self.references):
            refex = testinst['refex'].replace('eos', '').strip()
            entity = testinst['entity']

            output = ' '.join(entity.split('_'))

            results.append({'y_real':refex, 'y_pred':output})

        json.dump(results, open(self.out_file, 'w'))


if __name__ == '__main__':
    IN_FILE = 'data/v1.0/old_format/test/data.json'
    OUT_FILE = 'onlynames.json'

    b = OnlyNames(in_file=IN_FILE, out_file=OUT_FILE)