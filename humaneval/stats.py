__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 02/02/2018
Description:
    Human evaluation scripts to obtain results depicted at the paper.

    PYTHON VERSION :2.7

    DEPENDENCIES:
        NLTK:           http://www.nltk.org/
"""

import json
import numpy as np
import os
import random

from nltk.translate.bleu_score import sentence_bleu
from scipy.stats import spearmanr
from tabulate import tabulate

def process_db():
    with open('htmls/official/text_trials/trial_info') as f:
        trials = f.read()

    trials = trials.split('\n')[2:]

    trials = map(lambda trial: trial.split(), trials)
    trials_ = []
    for trial in trials:
        for i, model in enumerate(trial[3:]):
            trial_ = {'id':trial[0], 'size':trial[1], 'model':model, 'difficult':trial[2], 'list':i+1, 'evaluations':[]}

            trials_.append(trial_)

    evaluations = json.load(open('experiment_results.json'))['RECORDS']
    participant_info, participant_info_ = json.load(open('participants_info.json'))['RECORDS'], []
    lists = [1,2,3,4,5,6]
    for l in lists:
        list_trials = filter(lambda x: x['list'] == l, trials_)
        assert len(list_trials) == 24

        list_evaluations = filter(lambda x: x['list_id'] == str(l), evaluations)

        # restrict to 10 participants per list
        participants = list(set(map(lambda x: x['participant_id'], list_evaluations)))[:10]
        contexts = list(set(map(lambda x: x['url'], list_evaluations)))
        assert len(contexts) == 24

        for context in contexts:
            context_trials = filter(lambda x: x['id'] in context, list_trials)
            assert len(context_trials) == 1

            context_evaluations = filter(lambda x: x['url'] == context, evaluations)

            for participant in participants:
                participant_evaluation = filter(lambda x: x['participant_id'] == participant, context_evaluations)[0]
                context_trials[0]['evaluations'].append(participant_evaluation)

        for participant in participants:
            participant_info_.append(filter(lambda x: x['id'] == participant, participant_info)[0])
    return trials_, participant_info_

def evaluate(evaluations, model):
    fluency = map(lambda x: float(x['fluency']), evaluations)
    grammar = map(lambda x: float(x['grammar']), evaluations)
    clarity = map(lambda x: float(x['clarity']), evaluations)

    print model
    print 'FLUENCY:', str(round(np.mean(fluency),2)), str(round(np.std(fluency),2))
    print 'GRAMMATICALITY:', str(round(np.mean(grammar),2)), str(round(np.std(grammar),2))
    print 'CLARITY:', str(round(np.mean(clarity),2)), str(round(np.std(clarity),2))
    print 10 * '-'

    return fluency, grammar, clarity

def evaluate_by_difficult(trials, model):
    difficulties = list(set(map(lambda x: x['difficult'], trials)))

    for difficult in difficulties:
        difficult_trials = filter(lambda x: x['difficult'] == difficult, trials)
        evaluations = []
        for trial in difficult_trials:
            evaluations.extend(trial['evaluations'])

        fluency = map(lambda x: float(x['fluency']), evaluations)
        grammar = map(lambda x: float(x['grammar']), evaluations)
        clarity = map(lambda x: float(x['clarity']), evaluations)

        print difficult.upper(), model
        print 'FLUENCY:', str(round(np.mean(fluency),2)), str(round(np.std(fluency),2))
        print 'GRAMMATICALITY:', str(round(np.mean(grammar),2)), str(round(np.std(grammar),2))
        print 'CLARITY:', str(round(np.mean(clarity),2)), str(round(np.std(clarity),2))
        print 10 * '-'

def demographics(participants):
    print 'PARTICIPANT DEMOGRAPHICS: '

    avg_age = round(np.mean(map(lambda x: float(x['age']), participants)) , 2)
    print 'AGE: ', str(avg_age)

    females = len(filter(lambda x: x['gender'] == 'F', participants))
    print 'FEMALES: ', str(females), str(len(participants)), str(float(females)/len(participants))

    print 'ENGLISH PROFICIENCY:'
    basic = len(filter(lambda x: x['english_proficiency_level'] == 'basic', participants))
    print 'BASIC: ', str(basic), str(len(participants)), str(float(basic)/len(participants))
    fluent = len(filter(lambda x: x['english_proficiency_level'] == 'fluent', participants))
    print 'FLUENT: ', str(fluent), str(len(participants)), str(float(fluent)/len(participants))
    native = len(filter(lambda x: x['english_proficiency_level'] == 'native', participants))
    print 'NATIVE: ', str(native), str(len(participants)), str(float(native)/len(participants))

def evaluate_by_size(trials, model):
    sizes = sorted(list(set(map(lambda x: x['size'], trials))))

    for size in sizes:
        size_trials = filter(lambda x: x['size'] == size, trials)
        evaluations = []
        for trial in size_trials:
            evaluations.extend(trial['evaluations'])

        fluency = map(lambda x: float(x['fluency']), evaluations)
        grammar = map(lambda x: float(x['grammar']), evaluations)
        clarity = map(lambda x: float(x['clarity']), evaluations)

        print 'SIZE ' + str(size) + ': ', model
        print 'FLUENCY:', str(round(np.mean(fluency),2)), str(round(np.std(fluency),2))
        print 'GRAMMATICALITY:', str(round(np.mean(grammar),2)), str(round(np.std(grammar),2))
        print 'CLARITY:', str(round(np.mean(clarity),2)), str(round(np.std(clarity),2))
        print 10 * '-'


def sql():
    pre_sql = 'INSERT experiment5_contexts (list_id, url, next_url) VALUES ('
    sqls = []

    path = os.path.join('htmls', 'official')
    lists = ['list1', 'list2', 'list3', 'list4', 'list5', 'list6']
    for l in lists:
        path_ = os.path.join(path, l)
        fnames = os.listdir(path_)
        urls = []

        for fname in fnames:
            # os.rename(os.path.join(path_, fname), os.path.join(path_, fname.replace('.html', '.php')))
            with open(os.path.join(path_, fname)) as f:
                html = f.read()

            if '.DS_Store' not in fname:
                urls.append(os.path.join(l, fname.replace('.html', '.php')))

        list_id = l.replace('list', '')
        random.shuffle(urls)

        for i, url in enumerate(urls):
            if i == len(urls)-1:
                next_url = 'finish.php'
            else:
                next_url = urls[i+1]

            sql = pre_sql + list_id + ',\'' + url + '\',\'' + next_url + '\');'
            sqls.append(sql)

    for sql in sqls:
        print sql

def print_report(trials, participants):
    table = []

    original = filter(lambda x: x['model'] == 'original', trials)
    evaluations = []
    for inst in original:
        evaluations.extend(inst['evaluations'])
    original_fluency, original_grammar, original_clarity = evaluate(evaluations, 'ORIGINAL')
    table.append(['ORIGINAL', round(np.mean(original_fluency),4), round(np.mean(original_grammar),4), round(np.mean(original_clarity),4)])
    evaluate_by_difficult(original, 'ORIGINAL')

    print '\n'
    evaluate_by_size(original, 'ORIGINAL')
    print 20 * '-' + '\n'

    only = filter(lambda x: x['model'] == 'only', trials)
    evaluations = []
    for inst in only:
        evaluations.extend(inst['evaluations'])
    only_fluency, only_grammar, only_clarity = evaluate(evaluations, 'ONLY')
    table.append(['ONLY', round(np.mean(only_fluency),4), round(np.mean(only_grammar),4), round(np.mean(only_clarity),4)])
    evaluate_by_difficult(only, 'ONLY')
    print '\n'
    evaluate_by_size(only, 'only')
    print 20 * '-' + '\n'

    ferreira = filter(lambda x: x['model'] == 'ferreira', trials)
    evaluations = []
    for inst in ferreira:
        evaluations.extend(inst['evaluations'])
    ferreira_fluency, ferreira_grammar, ferreira_clarity = evaluate(evaluations, 'FERREIRA')
    table.append(['FERREIRA', round(np.mean(ferreira_fluency),4), round(np.mean(ferreira_grammar),4), round(np.mean(ferreira_clarity),4)])
    evaluate_by_difficult(ferreira, 'FERREIRA')
    print '\n'
    evaluate_by_size(ferreira, 'FERREIRA')
    print 20 * '-' + '\n'

    seq2seq = filter(lambda x: x['model'] == 'seq2seq', trials)
    evaluations = []
    for inst in seq2seq:
        evaluations.extend(inst['evaluations'])
    seq2seq_fluency, seq2seq_grammar, seq2seq_clarity = evaluate(evaluations, 'SEQ2SEQ')
    table.append(['SEQ2SEQ', round(np.mean(seq2seq_fluency),4), round(np.mean(seq2seq_grammar),4), round(np.mean(seq2seq_clarity),4)])
    evaluate_by_difficult(seq2seq, 'SEQ2SEQ')
    print '\n'
    evaluate_by_size(seq2seq, 'SEQ2SEQ')
    print 20 * '-' + '\n'

    catt = filter(lambda x: x['model'] == 'catt', trials)
    evaluations = []
    for inst in catt:
        evaluations.extend(inst['evaluations'])
    catt_fluency, catt_grammar, catt_clarity = evaluate(evaluations, 'CATT')
    table.append(['CATT', round(np.mean(catt_fluency),4), round(np.mean(catt_grammar),4), round(np.mean(catt_clarity),4)])
    evaluate_by_difficult(catt, 'CATT')
    print '\n'
    evaluate_by_size(catt, 'CATT')
    print 20 * '-' + '\n'

    hieratt = filter(lambda x: x['model'] == 'hieratt', trials)
    evaluations = []
    for inst in hieratt:
        evaluations.extend(inst['evaluations'])
    hier_fluency, hier_grammar, hier_clarity = evaluate(evaluations, 'HIERATT')
    table.append(['HIERATT', round(np.mean(hier_fluency),4), round(np.mean(hier_grammar),4), round(np.mean(hier_clarity),4)])
    evaluate_by_difficult(hieratt, 'HIERATT')
    print '\n'
    evaluate_by_size(hieratt, 'HIERATT')
    print 20 * '-' + '\n'

    print '\n\n'
    demographics(participants)
    print '\n'
    print tabulate(table, headers=['model', 'fluency', 'grammaticality', 'clarity'])

    header = "resp;original_fluency;original_grammar;original_clarity;only_fluency;only_grammar;only_clarity;ferreira_fluency;ferreira_grammar;ferreira_clarity;seq2seq_fluency;seq2seq_grammar;seq2seq_clarity;catt_fluency;catt_grammar;catt_clarity;hier_fluency;hier_grammar;hier_clarity"
    with open('official_results.csv', 'w') as f:
        f.write(header + '\n')

        for i, row in enumerate(original_fluency):
            l = [
                i+1,
                original_fluency[i], original_grammar[i], original_clarity[i],
                only_fluency[i], only_grammar[i], only_clarity[i],
                ferreira_fluency[i], ferreira_grammar[i], ferreira_clarity[i],
                seq2seq_fluency[i], seq2seq_grammar[i], seq2seq_clarity[i],
                catt_fluency[i], catt_grammar[i], catt_clarity[i],
                hier_fluency[i], hier_grammar[i], hier_clarity[i]
            ]
            l = map(lambda x: str(x), l)
            f.write(';'.join(l) + '\n')

def correlation(trials):

    # get trials from the trials/ directory
    for fsize in os.listdir('trials'):
        if fsize != '.DS_Store':
            for fname in os.listdir(os.path.join('trials', fsize)):
                with open(os.path.join('trials', fsize, fname)) as f:
                    texts = f.read().split('\n')[1:]

                    for text in texts:
                        text = text.split('\t')
                        _id, text = text[1].strip(), text[3]
                        text = text.replace('<span style=\"background-color: #ffff00\">', '').replace('</span>', '')

                        for i, trial in enumerate(trials):
                            if trial['id'] == _id and trial['model'] in fname:
                                trials[i]['text'] = text

    # compute rankings
    scores = []
    trial_ids = list(set(map(lambda x: x['id'], trials)))
    for trial_id in trial_ids:
        f = filter(lambda x: x['id'] == trial_id, trials)
        original = filter(lambda x: x['model'] == 'original', f)[0]

        only = filter(lambda x: x['model'] == 'only', f)[0]

        bleu = sentence_bleu([original['text'].split()], only['text'].split())
        fluency = np.mean(map(lambda x: float(x['fluency']), only['evaluations']))
        grammar = np.mean(map(lambda x: float(x['grammar']), only['evaluations']))
        clarity = np.mean(map(lambda x: float(x['clarity']), only['evaluations']))
        scores.append([bleu, fluency, grammar, clarity])

        # FERREIRA
        ferreira = filter(lambda x: x['model'] == 'ferreira', f)[0]
        bleu = sentence_bleu([original['text'].split()], ferreira['text'].split())
        fluency = np.mean(map(lambda x: float(x['fluency']), ferreira['evaluations']))
        grammar = np.mean(map(lambda x: float(x['grammar']), ferreira['evaluations']))
        clarity = np.mean(map(lambda x: float(x['clarity']), ferreira['evaluations']))
        scores.append([bleu, fluency, grammar, clarity])

        # Seq2seq
        seq2seq = filter(lambda x: x['model'] == 'seq2seq', f)[0]
        bleu = sentence_bleu([original['text'].split()], seq2seq['text'].split())
        fluency = np.mean(map(lambda x: float(x['fluency']), seq2seq['evaluations']))
        grammar = np.mean(map(lambda x: float(x['grammar']), seq2seq['evaluations']))
        clarity = np.mean(map(lambda x: float(x['clarity']), seq2seq['evaluations']))
        scores.append([bleu, fluency, grammar, clarity])

        # CAtt
        catt = filter(lambda x: x['model'] == 'catt', f)[0]
        bleu = sentence_bleu([original['text'].split()], catt['text'].split())
        fluency = np.mean(map(lambda x: float(x['fluency']), catt['evaluations']))
        grammar = np.mean(map(lambda x: float(x['grammar']), catt['evaluations']))
        clarity = np.mean(map(lambda x: float(x['clarity']), catt['evaluations']))
        scores.append([bleu, fluency, grammar, clarity])

        # HierAtt
        hieratt = filter(lambda x: x['model'] == 'hieratt', f)[0]
        bleu = sentence_bleu([original['text'].split()], hieratt['text'].split())
        fluency = np.mean(map(lambda x: float(x['fluency']), hieratt['evaluations']))
        grammar = np.mean(map(lambda x: float(x['grammar']), hieratt['evaluations']))
        clarity = np.mean(map(lambda x: float(x['clarity']), hieratt['evaluations']))
        scores.append([bleu, fluency, grammar, clarity])

    # compute correlations
    rho, pval = spearmanr(scores)

    print np.round(rho, 2)

    print '\n\n'
    print np.round(pval, 4)
    print '\n\n'

if __name__ == '__main__':
    trials, participants = process_db()

    print_report(trials, participants)
    correlation(trials)

