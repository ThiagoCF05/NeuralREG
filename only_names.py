__author__ = ''

"""
Author: ANONYMOUS
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

import cPickle as p

class OnlyNames():
    def __init__(self, in_file, out_file):
        self.references = p.load(open(in_file))
        self.out_file = out_file

        self.run()

    def run(self):
        results = []

        for i, testinst in enumerate(self.references):
            refex = testinst['refex'].replace('eos', '').strip()
            entity = testinst['entity']

            output = ' '.join(entity.split('_'))

            results.append({'y_real':refex, 'y_pred':output})

        p.dump(results, open(self.out_file, 'w'))


if __name__ == '__main__':
    IN_FILE = 'data/test/data.cPickle'
    OUT_FILE = 'onlynames.cPickle'

    b = OnlyNames(in_file=IN_FILE, out_file=OUT_FILE)