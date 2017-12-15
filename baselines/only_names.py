
import cPickle as p

class Baseline():
    def __init__(self):
        self.references = p.load(open('../data/test/data.cPickle'))

    def only_names(self):
        fname = 'baseline_names.cPickle'
        results = []

        for i, testinst in enumerate(self.references):
            refex = testinst['refex'].replace('eos', '').strip()
            entity = testinst['entity']

            output = ' '.join(entity.split('_'))

            results.append({'y_real':refex, 'y_pred':output})

        p.dump(results, open(fname, 'w'))

b = Baseline()
b.only_names()