
import utils

class Baseline():
    def __init__(self):
        self.vocab, self.trainset, self.devset, self.testset = utils.load_data()

    def only_names(self):
        fname = 'data/results/baseline_names.txt'
        f = open(fname, 'w')
        for i, testinst in enumerate(self.devset['refex']):
            refex = ' '.join(self.devset['refex'][i]).replace('eos', '').strip()
            entity = self.devset['entity'][i]

            output = ' '.join(entity.split('_'))

            f.write(output)
            f.write('\n')
        f.close()

b = Baseline()
b.only_names()