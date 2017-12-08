
from nltk.metrics.distance import edit_distance

import xml.etree.ElementTree as ET

import numpy as np
import cPickle as p
import os

# from stanford_corenlp_pywrapper import CoreNLP

def corpus_info():
    def read(in_file):
        inputs = 0

        dirs = filter(lambda x: x != '.DS_Store', os.listdir(in_file))
        for path in dirs:
            dirs2 = filter(lambda x: x != '.DS_Store', os.listdir(os.path.join(in_file, path)))
            for fname in dirs2:
                print os.path.join(in_file, path, fname)
                tree = ET.parse(os.path.join(in_file, path, fname))
                root = tree.getroot()

                entries = root.find('entries')
                for entry in entries:
                    lexEntries = entry.findall('lex')
                    inputs = inputs + 1
                    for lex in lexEntries:
                        try:
                            text = lex.find('text').text
                            template = lex.find('template').text

                            if template and text:
                                # out = parser.parse_doc(text)
                                # tokens = []
                                # for snt in out['sentences']:
                                #     tokens.extend(snt['tokens'])
                                # originals.append(' '.join(tokens).strip().lower())
                                #
                                # out = parser.parse_doc(template)
                                # for snt in out['sentences']:
                                #     templates.append(' '.join(snt['tokens']).strip().lower())

                                texts.append(text.lower())
                                templates.append(template.lower())
                        except:
                            pass
        return inputs

    # parser = CoreNLP('ssplit')
    ftrain = 'annotation/final/train'
    fdev = 'annotation/final/dev'

    texts = []
    templates = []

    inputs = read(ftrain)
    _inputs = read(fdev)
    inputs += _inputs

    print 'Number of Representations: ', inputs

    print 'Sentences: ', len(texts)
    print 'Distinct Sentences: ', len(list(set(texts)))

    print 'Templates: ', len(templates)
    print 'Distinct Templates: ', len(list(set(templates)))

def corpus_evaluation():
    def pronoun_count(data):
        num = 0
        for reference in data:
            if reference['refex'] in ['he', 'his', 'him', 'she', 'hers', 'her', 'it', 'its', 'we', 'our', 'ours', 'they', 'theirs', 'them']:
                num += 1
        return num
    train_data = p.load(open('data/train/data.cPickle'))
    dev_data = p.load(open('data/dev/data.cPickle'))
    test_data = p.load(open('data/test/data.cPickle'))

    train_entities = set(map(lambda x: x['entity'], train_data))
    dev_entities = set(map(lambda x: x['entity'], dev_data))
    test_entities = set(map(lambda x: x['entity'], test_data))

    print 'Train entities: ', len(list(train_entities))
    print 'Dev entities: ', len(list(dev_entities))
    print 'Test entities: ', len(list(test_entities))

    dev_intersect = train_entities.intersection(dev_entities)
    print 'Train intersect dev: ', str(len(list(dev_intersect)))

    test_intersect = train_entities.intersection(test_entities)
    print 'Train intersect test: ', str(len(list(test_intersect)))

def evaluate(y_real, y_pred):
    edit_distances = []
    pronoun_num, pronoun_dem = 0.0, 0.0
    num, dem = 0.0, 0.0
    wrong = []

    for real, pred in zip(y_real, y_pred):
        real = real.replace('eos', '').strip()
        pred = pred.replace('eos', '').strip()

        edit_distances.append(edit_distance(real, pred.strip()))

        if pred.strip() == real:
            num += 1
        else:
            wrong.append({'real': real, 'pred':pred})
        dem += 1

        if real.lower() in ['he', 'his', 'him',
                    'she', 'her', 'hers',
                    'it', 'its', 'we', 'us', 'our', 'ours',
                    'they', 'them', 'their', 'theirs']:
            # print(real)
            if pred.strip() == real:
                pronoun_num += 1
            pronoun_dem += 1

    print('ACCURACY: ', str(round(num/dem, 4)))
    print('DISTANCE: ', str(round(np.mean(edit_distances), 4)))
    print('PRONOUN ACCURACY: ', str(round(pronoun_num/pronoun_dem, 4)))
    return wrong

def generate_sentence(data, y_neural, y_only, y_ferreira):
    originals, neurals, onlys, ferreiras = [], [], [], []

    for i, reference in enumerate(data):
        reference['neural'] = y_neural[i]
        reference['only'] = y_only[i]['y_pred']
        reference['ferreira'] = y_ferreira[i]['realization']

    text_ids = sorted(list(set(map(lambda x: x['text_id'], data))))
    print len(text_ids)
    for text_id in text_ids:
        references = filter(lambda x: x['text_id'] == text_id, data)
        references = sorted(references, key=lambda x: x['general_pos'])

        template = references[0]['pre_context'] + ' ' + references[0]['entity'] + ' ' + references[0]['pos_context']
        neural = template
        only = template
        ferreira = template

        for reference in references:
            entity = reference['entity']

            neural_refex = reference['neural'].replace('eos', '').strip()
            neural = neural.replace(entity, neural_refex, 1)

            only_refex = reference['only'].replace('eos', '').strip()
            only = only.replace(entity, only_refex, 1)

            ferreira_refex = reference['ferreira'].replace('eos', '').strip()
            ferreira = ferreira.replace(entity, ferreira_refex, 1)

        originals.append(references[0]['text'].replace('@', ''))
        neurals.append(neural.replace('_', ' ').replace('eos', '').strip())
        onlys.append(only.replace('_', ' ').replace('eos', '').strip())
        ferreiras.append(ferreira.replace('_', ' ').replace('eos', '').strip())

    with open('data/original', 'w') as f:
        f.write('\n'.join(originals).lower().encode('utf-8'))

    with open('data/neural', 'w') as f:
        f.write('\n'.join(neurals).lower().encode('utf-8'))

    with open('data/only', 'w') as f:
        f.write('\n'.join(onlys).lower().encode('utf-8'))

    with open('data/ferreira', 'w') as f:
        f.write('\n'.join(ferreiras).lower().encode('utf-8'))

if __name__ == '__main__':
    # y_only = p.load(open('baseline/baseline_names.cPickle'))
    # y_pred = map(lambda x: x['y_pred'], y_only)
    #
    # y_real = map(lambda x: x['y_real'].lower(), y_only)
    # print 'BASELINE: ONLY NAMES'
    # evaluate(y_real, y_pred)
    # print '\n'
    #
    # y_ferreira = p.load(open('baseline/reg/result.cPickle'))
    # y_pred = map(lambda x: x['realization'], y_ferreira)
    #
    # y_real = map(lambda x: x['refex'].lower(), y_ferreira)
    # print 'BASELINE:'
    # evaluate(y_real, y_pred)
    # print '\n'
    #
    # # ATT
    # with open('data/att/results/test_best_1_300_512_512_2_False_1/0') as f:
    #     y_pred = f.read().lower().split('\n')
    #
    # with open('data/test/refex.txt') as f:
    #     y_real = f.read().lower().split('\n')
    # print 'MODEL: att test_best_1_300_512_512_2_False_1'
    # evaluate(y_real, y_pred)
    # print '\n'
    #
    # with open('data/att/results/test_best_1_300_512_512_3_False_1/0') as f:
    #     y_pred = f.read().decode('utf-8').lower().split('\n')
    #
    # with open('data/test/refex.txt') as f:
    #     y_real = f.read().lower().split('\n')
    # print 'MODEL: att test_best_1_300_512_512_3_False_1'
    # wrong = evaluate(y_real, y_pred)
    # print '\n'
    #
    # # HIER
    # with open('data/hier/results/test_best_1_300_512_512_2_False_1/0') as f:
    #     y_pred = f.read().lower().split('\n')
    #
    # with open('data/test/refex.txt') as f:
    #     y_real = f.read().lower().split('\n')
    # print 'MODEL: hier test_best_1_300_512_512_2_False_1'
    # evaluate(y_real, y_pred)
    # print '\n'
    #
    # with open('data/hier/results/test_best_1_300_512_512_3_False_1/0') as f:
    #     y_pred = f.read().lower().split('\n')
    #
    # with open('data/test/refex.txt') as f:
    #     y_real = f.read().lower().split('\n')
    # print 'MODEL: hier test_best_1_300_512_512_3_False_1'
    # wrong = evaluate(y_real, y_pred)
    # print '\n'

    # for e in wrong:
    #     print 'REAL: ', e['real']
    #     print 'PRED: ', e['pred']
    #     print 10 * '-'
    #
    corpus_evaluation()

    corpus_info()

    # references = p.load(open('data/test/data.cPickle'))
    # generate_sentence(references, y_pred, y_only, y_ferreira)