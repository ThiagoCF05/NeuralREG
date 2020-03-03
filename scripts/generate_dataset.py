import pickle as p

# PATHS FOR TRAINING AND DEVELOPMENT SETS OF NEURAL v1.0 DATA
TRAIN_PATH = 'NeuralREG/data/v1.0/old_format/train'
DEV_PATH = 'NeuralREG/data/v1.0/old_format/dev'
TEST_PATH = 'NeuralREG/data/v1.0/old_format/dev'

# PATH FOR DATA COLLECTION
TRAIN_FILE = 'NeuralREG/data/v1.0/old_format/train/data.cPickle'
DEV_FILE = 'NeuralREG/data/v1.0/old_format/dev/data.cPickle'
TEST_FILE = 'NeuralREG/data/v1.0/old_format/test/data.cPickle'

WEBNLG_PATH = 'NeuralREG/data/v1.0/en'


def load_data():

    with open(TRAIN_FILE, 'rb') as file1:
        train_data = p.load(file1)

    with open(DEV_FILE, 'rb') as file2:
        dev_data = p.load(file2)

    with open(TEST_FILE, 'rb') as file3:
        test_data = p.load(file3)

    train_entities = map(lambda x: x['entity'], train_data)
    dev_entities = map(lambda x: x['entity'], dev_data)
    test_entities = map(lambda x: x['entity'], test_data)


if __name__ == '__main__':
    load_data()