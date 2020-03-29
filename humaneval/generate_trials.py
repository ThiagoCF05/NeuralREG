__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 02/02/2018
Description:
    Scripts used to surface realized the texts of the human evaluation in the HTML format

    PYTHON VERSION :2.7

    DEPENDENCIES:
        Numpy
"""

__author__ = ''

import copy
import pickle as p
import numpy as np
import os
import random

from tabulate import tabulate

# ORIGINAL
ORIGINAL = '../data/test/data.cPickle'
ORIGINAL_INFO = '../data/test/info.txt'
# ONLY NAMES RESULTS PATH
ONLYNAMES = '../eval/onlynames.cPickle'
# FERREIRA RESULTS PATH
FERREIRA = '../eval/ferreira.cPickle'
# NEURAL-SEQ2SEQ RESULTS PATH
SEQ2SEQ = '../eval/seq2seq/results/test_best_1_300_512_3_False_5/0'
# NEURAL-CATT RESULTS PATH
CATT = '../eval/att/results/test_best_1_300_512_512_3_False_5/0'
# NEURAL-HIERATT RESULTS PATH
HIERATT = '../eval/hier/results/test_best_1_300_512_512_2_False_1/0'

def load_models():
    original = p.load(open(ORIGINAL))

    with open(ORIGINAL_INFO) as f:
        original_info = f.read().split('\n')
        original_info = map(lambda x: x.split(), original_info)

    # ONLY NAMES RESULTS AND GOLD-STANDARDS
    only = p.load(open(ONLYNAMES))
    y_only = map(lambda x: x['y_pred'], only)

    # FERREIRA ET AL., 2016 RESULTS
    ferreira = p.load(open(FERREIRA))
    _ferreira = []
    for inst in original:
        reference = filter(lambda x: x['text_id'] == inst['text_id'] and
                                     x['sentence'] == inst['sentence'] and
                                     x['pos'] == inst['pos'] and
                                     x['refex'] == inst['refex'], ferreira)[0]
        _ferreira.append(reference)
    ferreira = _ferreira
    y_ferreira = map(lambda x: x['realization'].lower(), ferreira)

    # NEURAL SEQ2SEQ RESULTS
    with open(SEQ2SEQ) as f:
        y_seq2seq = f.read().decode('utf-8').lower().split('\n')

    # NEURAL CATT RESULTS
    with open(CATT) as f:
        y_catt = f.read().decode('utf-8').lower().split('\n')

    # NEURAL HIERATT RESULTS
    with open(HIERATT) as f:
        y_hieratt = f.read().decode('utf-8').lower().split('\n')

    return original, original_info, y_only, y_ferreira, y_seq2seq, y_catt, y_hieratt

def generate_texts(data, y_pred, fname):
    templates, templates_html = [], []

    for i, reference in enumerate(data):
        reference['pred'] = y_pred[i].lower().strip()

    text_ids = sorted(list(set(map(lambda x: x['text_id'], data))))
    for text_id in text_ids:
        references = filter(lambda x: x['text_id'] == text_id, data)
        references = sorted(references, key=lambda x: x['general_pos'])

        # text = references[0]['text'].lower()
        template = references[0]['pre_context'] + ' ' + references[0]['entity'] + ' ' + references[0]['pos_context']
        template_html = references[0]['pre_context'] + ' ' + references[0]['entity'] + ' ' + references[0]['pos_context']

        for reference in references:
            entity = reference['entity'] + ' '

            refex = '~'.join(reference['pred'].replace('eos', '').strip().split())
            html = '<span style=\"background-color: #FFFF00\">' + refex + '</span> '
            template = template.replace(entity, refex + ' ', 1)
            template_html = template_html.replace(entity, html, 1)

        templates.append(template.replace('_', ' ').replace('~', ' ').replace('eos', '').strip())
        templates_html.append(template_html.replace('_', ' ').replace('~', ' ').replace('eos', '').strip())

    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write('\n'.join(templates).lower().encode('utf-8'))
    return templates, templates_html

def generate_original(data, fname):
    templates, templates_html = [], []

    text_ids = sorted(list(set(map(lambda x: x['text_id'], data))))
    for text_id in text_ids:
        references = filter(lambda x: x['text_id'] == text_id, data)
        references = sorted(references, key=lambda x: x['general_pos'])

        template = references[0]['text'].lower()
        template_html = references[0]['text'].lower()

        for reference in references:
            entity = reference['refex'].lower().replace('eos', '').strip() + ' '

            refex = '~'.join(reference['refex'].lower().replace('eos', '').strip().split())
            html = '<span style=\"background-color: #FFFF00\">' + refex + '</span> '

            template_html = template_html.replace(entity, html, 1)

        templates.append(template.replace('@', '').replace('eos', '').strip())
        templates_html.append(template_html.replace('_', ' ').replace('~', ' ').replace('@', '').replace('eos', '').strip())

    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write('\n'.join(templates).lower().encode('utf-8'))
    return templates, templates_html

def compute_difference(text_sizes, originals, only, ferreira, seq2seq, catt, hieratt):
    text_diff = {}

    acc_only, acc_ferreira, acc_seq, acc_catt, acc_hier = 0, 0, 0, 0, 0
    for text, row in enumerate(text_sizes.items()):
        text_id, size = row
        if size not in text_diff:
            text_diff[size] = []
        diff = 0

        if originals[text] != only[text]:
            diff += 1
        else:
            acc_only += 1
        if originals[text] != ferreira[text]:
            diff += 1
        else:
            acc_ferreira += 1
        if originals[text] != seq2seq[text]:
            diff += 1
        else:
            acc_seq += 1
        if originals[text] != catt[text]:
            diff += 1
        else:
            acc_catt += 1
        if originals[text] != hieratt[text]:
            diff += 1
        else:
            acc_hier += 1

        if only[text] != ferreira[text]:
            diff += 1
        if only[text] != seq2seq[text]:
            diff += 1
        if only[text] != catt[text]:
            diff += 1
        if only[text] != hieratt[text]:
            diff += 1

        if ferreira[text] != seq2seq[text]:
            diff += 1
        if ferreira[text] != catt[text]:
            diff += 1
        if ferreira[text] != hieratt[text]:
            diff += 1

        if seq2seq[text] != catt[text]:
            diff += 1
        if seq2seq[text] != hieratt[text]:
            diff += 1

        if catt[text] != hieratt[text]:
            diff += 1

        text_diff[size].append((text, diff, text_id))
    print 'Only: ', str(float(acc_only) / len(text_sizes)), str(acc_only), str(len(text_sizes))
    print 'Ferreira: ', str(float(acc_ferreira) / len(text_sizes)), str(acc_ferreira), str(len(text_sizes))
    print 'Seq2seq: ', str(float(acc_seq) / len(text_sizes)), str(acc_seq), str(len(text_sizes))
    print 'CAtt: ', str(float(acc_catt) / len(text_sizes)), str(acc_catt), str(len(text_sizes))
    print 'HierAtt: ', str(float(acc_hier) / len(text_sizes)), str(acc_hier), str(len(text_sizes))
    return text_diff

def save_trials(size_text_diff, originals, only, ferreira, seq2seq, catt, hieratt):
    if not os.path.exists('trials'):
        os.mkdir('trials')

    original_trial, only_trial, ferreira_trial, seq2seq_trial, catt_trial, hieratt_trial = {}, {}, {}, {}, {}, {}
    for size in size_text_diff:
        if not os.path.exists(os.path.join('trials', str(size-1)+'triple')):
            os.mkdir(os.path.join('trials', str(size-1)+'triple'))

        random.shuffle(size_text_diff[size])
        text_diff = sorted(size_text_diff[size], key=lambda x:x[1], reverse=True)

        templates = map(lambda x: {'size':size, 'text_pos':x[0], 'diff':x[1], 'text_id':x[2], 'text':originals[x[0]]}, text_diff)
        original_trial[size] = templates
        templates = map(lambda x: ' \t'.join([str(x[0]), str(x[2]), str(x[1]), originals[x[0]]]), text_diff)
        with open(os.path.join('trials', str(size-1)+'triple', 'original.txt'), 'w') as f:
            f.write('\t'.join(['position', 'text_id', 'difference', 'html']) + '\n')
            f.write('\n'.join(templates).lower().encode('utf-8'))

        templates = map(lambda x: {'size':size, 'text_pos':x[0], 'diff':x[1], 'text_id':x[2], 'text':only[x[0]]}, text_diff)
        only_trial[size] = templates
        templates = map(lambda x: ' \t'.join([str(x[0]), str(x[2]), str(x[1]), only[x[0]]]), text_diff)
        with open(os.path.join('trials', str(size-1)+'triple', 'only.txt'), 'w') as f:
            f.write('\t'.join(['position', 'text_id', 'difference', 'html']) + '\n')
            f.write('\n'.join(templates).lower().encode('utf-8'))

        templates = map(lambda x: {'size':size, 'text_pos':x[0], 'diff':x[1], 'text_id':x[2], 'text':ferreira[x[0]]}, text_diff)
        ferreira_trial[size] = templates
        templates = map(lambda x: ' \t'.join([str(x[0]), str(x[2]), str(x[1]), ferreira[x[0]]]), text_diff)
        with open(os.path.join('trials', str(size-1)+'triple', 'ferreira.txt'), 'w') as f:
            f.write('\t'.join(['position', 'text_id', 'difference', 'html']) + '\n')
            f.write('\n'.join(templates).lower().encode('utf-8'))

        templates = map(lambda x: {'size':size, 'text_pos':x[0], 'diff':x[1], 'text_id':x[2], 'text':seq2seq[x[0]]}, text_diff)
        seq2seq_trial[size] = templates
        templates = map(lambda x: ' \t'.join([str(x[0]), str(x[2]), str(x[1]), seq2seq[x[0]]]), text_diff)
        with open(os.path.join('trials', str(size-1)+'triple', 'seq2seq.txt'), 'w') as f:
            f.write('\t'.join(['position', 'text_id', 'difference', 'html']) + '\n')
            f.write('\n'.join(templates).lower().encode('utf-8'))

        templates = map(lambda x: {'size':size, 'text_pos':x[0], 'diff':x[1], 'text_id':x[2], 'text':catt[x[0]]}, text_diff)
        catt_trial[size] = templates
        templates = map(lambda x: ' \t'.join([str(x[0]), str(x[2]), str(x[1]), catt[x[0]]]), text_diff)
        with open(os.path.join('trials', str(size-1)+'triple', 'catt.txt'), 'w') as f:
            f.write('\t'.join(['position', 'text_id', 'difference', 'html']) + '\n')
            f.write('\n'.join(templates).lower().encode('utf-8'))

        templates = map(lambda x: {'size':size, 'text_pos':x[0], 'diff':x[1], 'text_id':x[2], 'text':hieratt[x[0]]}, text_diff)
        hieratt_trial[size] = templates
        templates = map(lambda x: ' \t'.join([str(x[0]), str(x[2]), str(x[1]), hieratt[x[0]]]), text_diff)
        with open(os.path.join('trials', str(size-1)+'triple', 'hieratt.txt'), 'w') as f:
            f.write('\t'.join(['position', 'text_id', 'difference', 'html']) + '\n')
            f.write('\n'.join(templates).lower().encode('utf-8'))

    trials = {
        'original': original_trial,
        'only': only_trial,
        'ferreira': ferreira_trial,
        'seq2seq': seq2seq_trial,
        'catt': catt_trial,
        'hieratt': hieratt_trial
    }
    return trials

def generate_htmls(size_text_diff, trials):
    with open('htmls/layout.html') as f:
        layout = f.read().decode('utf-8')

    trial_info = []
    text_trials = []
    sizes = range(3, 9)
    models_med = ['original', 'only', 'ferreira', 'seq2seq', 'catt', 'hieratt']
    models_cri = ['original', 'only', 'ferreira', 'seq2seq', 'catt', 'hieratt']
    for size in sizes:
        text_diff = size_text_diff[size]
        random.shuffle(text_diff)
        text_diff = sorted(size_text_diff[size], key=lambda x:x[1], reverse=True)

        # choose 2 triple in which all text version are different (critical cases)
        critical_pos = map(lambda x: x[0], text_diff[:2])
        for position in critical_pos:
            info = {'position':position, 'size': str(size-1), 'difficult':'critical'}
            text_trial = {'position':position}

            for i, model_cri in enumerate(models_cri):
                trial = filter(lambda x: x['text_pos'] == position, trials[model_cri][size])[0]

                l = 'list' + str(i+1)
                if not os.path.exists(os.path.join('htmls', l)):
                    os.mkdir(os.path.join('htmls', l))

                html = copy.copy(layout)
                old = '<p class=\"lead\" id=\"text_article\"></p>'
                new = '<p class=\"lead\" id=\"text_article\">' + trial['text'] + '</p>'
                html = html.replace(old, new)

                old = '<input type=\"hidden\" name=\"url\" value=\"\" />'
                new = '<input type=\"hidden\" name=\"url\" value=\"' + str(trial['text_id']) + '.php\" />'
                html = html.replace(old, new)

                info['text_id'] = str(trial['text_id'])
                info[l] = model_cri

                text = trial['text'].replace('<span style=\"background-color: #FFFF00\">', '').replace('</span>', '')
                text_trial['text_id'] = str(trial['text_id'])
                text_trial[model_cri] = text

                fname = str(trial['text_id']) + '.html'
                with open(os.path.join('htmls', l, fname), 'w') as f:
                    f.write(html.encode('utf-8'))
            trial_info.append(info)
            text_trials.append(text_trial)
            models_cri.insert(0, models_cri.pop())

        # choose 2 triple in which text versions are different according to the difference median
        median = np.median(map(lambda x: x[1], text_diff))
        ftext_diff = filter(lambda x: x[1] == median, text_diff)
        while len(ftext_diff) < 2:
            median += 1
            ftext_diff = filter(lambda x: x[1] == median, text_diff)
        random.shuffle(ftext_diff)
        median_pos = map(lambda x: x[0], ftext_diff[:2])
        for position in median_pos:
            info = {'position':position, 'size': str(size-1), 'difficult':'median'}
            text_trial = {'position':position}
            for i, model_med in enumerate(models_med):
                trial = filter(lambda x: x['text_pos'] == position, trials[model_med][size])[0]

                l = 'list' + str(i+1)
                if not os.path.exists(os.path.join('htmls', l)):
                    os.mkdir(os.path.join('htmls', l))

                html = copy.copy(layout)
                old = '<p class=\"lead\" id=\"text_article\"></p>'
                new = '<p class=\"lead\" id=\"text_article\">' + trial['text'] + '</p>'
                html = html.replace(old, new)

                old = '<input type=\"hidden\" name=\"url\" value=\"\" />'
                new = '<input type=\"hidden\" name=\"url\" value=\"' + str(trial['text_id']) + '.php\" />'
                html = html.replace(old, new)

                info['text_id'] = str(trial['text_id'])
                info[l] = model_med

                text = trial['text'].replace('<span style=\"background-color: #FFFF00\">', '').replace('</span>', '')
                text_trial['text_id'] = str(trial['text_id'])
                text_trial[model_med] = text

                fname = str(trial['text_id']) + '.html'
                with open(os.path.join('htmls', l, fname), 'w') as f:
                    f.write(html.encode('utf-8'))
            trial_info.append(info)
            text_trials.append(text_trial)
            models_med.insert(0, models_med.pop())

    path = os.path.join('htmls', 'text_trials')
    if not os.path.exists(path):
        os.mkdir(path)

    # save a report of the trials per list
    trial_info = sorted(trial_info, key=lambda x: int(x['text_id']))
    with open(os.path.join(path, 'trial_info'), 'w') as f:
        content = map(lambda x: [x['text_id'],x['size'],x['difficult'],x['list1'],x['list2'],x['list3'],x['list4'],x['list5'],x['list6']], trial_info)
        headers = ['text_id','size','difficult','list1','list2','list3','list4','list5','list6']
        content = tabulate(content, headers=headers)
        f.write(content)

    # save the texts used in the experiment
    for text_trial in text_trials:
        with open(os.path.join(path, text_trial['text_id']), 'w') as f:
            f.write('\n'.join(['original', text_trial['original'].encode('utf-8')]))
            f.write('\n\n')
            f.write('\n'.join(['only', text_trial['only'].encode('utf-8')]))
            f.write('\n\n')
            f.write('\n'.join(['ferreira', text_trial['ferreira'].encode('utf-8')]))
            f.write('\n\n')
            f.write('\n'.join(['seq2seq', text_trial['seq2seq'].encode('utf-8')]))
            f.write('\n\n')
            f.write('\n'.join(['catt', text_trial['catt'].encode('utf-8')]))
            f.write('\n\n')
            f.write('\n'.join(['hieratt', text_trial['hieratt'].encode('utf-8')]))

if __name__ == '__main__':
    # original, original_info, y_only, y_ferreira, y_seq2seq, y_catt, y_hieratt = load_models()
    # y_original = map(lambda x: x['refex'], original)
    #
    # if not os.path.exists('texts/'):
    #     os.mkdir('texts')
    #
    # # get different versions of the texts per model
    # originals, originals_html = generate_original(original, 'texts/original.txt')
    # only, only_html = generate_texts(original, y_only, 'texts/only.txt')
    # ferreira, ferreira_html = generate_texts(original, y_ferreira, 'texts/ferreira.txt')
    # seq2seq, seq2seq_html = generate_texts(original, y_seq2seq, 'texts/seq2seq.txt')
    # catt, catt_html = generate_texts(original, y_catt, 'texts/catt.txt')
    # hieratt, hieratt_html = generate_texts(original, y_hieratt, 'texts/hieratt.txt')
    #
    # # generate texts highlighting references and sorting texts by differences in their model versions
    # text_size = dict(map(lambda x: (x['text_id'], x['size']), original))
    # size_text_diff = compute_difference(text_size, originals, only, ferreira, seq2seq, catt, hieratt)
    #
    # trials = save_trials(size_text_diff, originals_html, only_html, ferreira_html, seq2seq_html, catt_html, hieratt_html)
    #
    # # generate trials for experiment
    # generate_htmls(size_text_diff, trials)



