import os
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pickle as p

# PATHS FOR TRAINING AND DEVELOPMENT SETS OF NEURAL v1.0 DATA
TRAIN_PATH = '/NeuralREG/data/v1.0/old_format/train'
DEV_PATH = '/NeuralREG/data/v1.0/old_format/dev'
TEST_PATH = '/NeuralREG/data/v1.0/old_format/dev'

# PATH FOR DATA COLLECTION
TRAIN_FILE = '/NeuralREG/data/v1.0/old_format/train/data.cPickle'
DEV_FILE = '/NeuralREG/data/v1.0/old_format/dev/data.cPickle'
TEST_FILE = '/NeuralREG/data/v1.0/old_format/test/data.cPickle'

INFO_FILENAME = 'info.txt'

WEBNLG_PATH = '/NeuralREG/webnlg/data/v1.0/en'


def run_generator():
    dev_info = os.path.join(DEV_PATH, INFO_FILENAME)

    with open(DEV_FILE, 'rb') as file:
        dev_data = p.load(file)

    with open(dev_info) as info:
        dev_info = info.read().split('\n')

    # Get ordered entities
    text_ids = sorted(list(set(map(lambda x: x['text_id'], dev_data))))

    # Concatenate the dev set
    dev_set = []
    for i, text_id in enumerate(text_ids):
        references = filter(lambda x: x['text_id'] == text_id, dev_data)
        references = sorted(references, key=lambda x: x['text_id'])

        for reference in references:
            reference['eid'] = 'Id' + str(text_id)
            path, xml_file = dev_info[i].split(' ')
            reference['path'] = path
            reference['fname'] = xml_file
            reference['category'] = xml_file.replace('.xml', '')

            dev_set.append(reference)

    generate(dev_set)


def generate(entry_set):
    entry_info = []
    for reference in entry_set:
        out_file = os.path.join(WEBNLG_PATH, 'dev', reference['path'], reference['fname'])
        in_file = os.path.join(WEBNLG_PATH, 'train', reference['path'], reference['fname'])

        tree = ET.parse(in_file)
        root = tree.getroot()

        entries = root.find('entries')

        entries_xml = list(filter(lambda entry: entry.attrib['eid'] == reference['eid'] and
                                            entry.attrib['size'] == reference['size'] and
                                            entry.attrib['category'] == reference['category'], entries))

        if len(entries_xml) >0:
            entry_info.append(entries_xml)

    # TODO
    # Find entries and generate XML on dev folder


if __name__ == '__main__':
    run_generator()
