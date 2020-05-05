__author__ = 'thiagocastroferreira'

import json
from csv import DictReader

"""
Author: Thiago Castro Ferreira
Date: 02/02/2018
Description:
    Human evaluation scripts to obtain results depicted at the paper.

    PYTHON VERSION :2.7

    DEPENDENCIES:
        NLTK:           http://www.nltk.org/
"""

import numpy as np

from tabulate import tabulate

unseen_domains = ['Artist', 'Politician', 'CelestialBody', 'Athlete', 'MeanOfTransportation']

def process_db():
    trials = []
    entries = json.load(open('../data/v1.5/test_info.json'))
    with open('trials/beta/trials_results.csv') as f:
        csv_file = DictReader(f, delimiter="\t")
        for trial in csv_file:
            category = list(filter(lambda o: o['eid'] == trial['eid'] and
                                             o['lid'] == trial['lid'] and
                                             o['size'] == trial['size'],
                                   entries))
            trial_ = {'id': trial['eid'], 'size': str(trial['size']), 'model': trial['model'],
                      'category': category[0]['category'], 'acceptability': str(trial['acceptability']),
                      'lexicogrammar': str(trial['lexicogrammar']), 'adequacy': str(trial['adequacy'])}

            trials.append(trial_)

    return trials


def evaluate(evaluations, model):
    adequacy = [float(x['adequacy']) for x in evaluations]
    grammar = [float(x['lexicogrammar']) for x in evaluations]
    fluency = [float(x['acceptability']) for x in evaluations]

    print(model)
    print('ADEQUACY:', str(round(np.mean(adequacy), 2)), str(round(np.std(adequacy), 2)))
    print('LEXICOGRAMMAR:', str(round(np.mean(grammar), 2)), str(round(np.std(grammar), 2)))
    print('ACCEPTABILITY:', str(round(np.mean(fluency), 2)), str(round(np.std(fluency), 2)))

    print(10 * '-')

    return fluency, grammar, adequacy


def evaluate_by_seen_domains(evaluations, model):
    adequacy = [float(x['adequacy']) for x in evaluations if x['category'] not in unseen_domains]
    grammar = [float(x['lexicogrammar']) for x in evaluations if x['category'] not in unseen_domains]
    fluency = [float(x['acceptability']) for x in evaluations if x['category'] not in unseen_domains]

    print(model)
    print('ADEQUACY:', str(round(np.mean(adequacy), 2)), str(round(np.std(adequacy), 2)))
    print('LEXICOGRAMMAR:', str(round(np.mean(grammar), 2)), str(round(np.std(grammar), 2)))
    print('ACCEPTABILITY:', str(round(np.mean(fluency), 2)), str(round(np.std(fluency), 2)))

    print(10 * '-')

    return fluency, grammar, adequacy


def evaluate_by_unseen_domains(evaluations, model):
    adequacy = [float(x['adequacy']) for x in evaluations if x['category'] in unseen_domains]
    grammar = [float(x['lexicogrammar']) for x in evaluations if x['category'] in unseen_domains]
    fluency = [float(x['acceptability']) for x in evaluations if x['category'] in unseen_domains]

    print(model)
    print('ADEQUACY:', str(round(np.mean(adequacy), 2)), str(round(np.std(adequacy), 2)))
    print('LEXICOGRAMMAR:', str(round(np.mean(grammar), 2)), str(round(np.std(grammar), 2)))
    print('ACCEPTABILITY:', str(round(np.mean(fluency), 2)), str(round(np.std(fluency), 2)))

    print(10 * '-')

    return fluency, grammar, adequacy


def evaluate_by_size(trials, model):
    sizes = sorted(list(set(map(lambda x: x['size'], trials))))

    for size in sizes:
        size_trials = list(filter(lambda x: x['size'] == size, trials))

        adequacy = [float(x['adequacy']) for x in size_trials]
        grammar = [float(x['lexicogrammar']) for x in size_trials]
        fluency = [float(x['acceptability']) for x in size_trials]

        print('SIZE ' + str(size) + ': ', model)
        print('ACCEPTABILITY:', str(round(np.mean(fluency), 2)), str(round(np.std(fluency), 2)))
        print('LEXICOGRAMMAR:', str(round(np.mean(grammar), 2)), str(round(np.std(grammar), 2)))
        print('ADEQUACY:', str(round(np.mean(adequacy), 2)), str(round(np.std(adequacy), 2)))
        print(10 * '-')


def print_report(trials):
    table, table_seen, table_unseen = [], [], []

    # ONLYNAMES
    only = list(filter(lambda x: x['model'] == 'only', trials))
    only_fluency, only_grammar, only_adequacy = evaluate(only, 'ONLY')
    table.append(
        ['ONLY', round(np.mean(only_fluency), 4), round(np.mean(only_grammar), 4), round(np.mean(only_adequacy), 4)])

    only_seen = list(filter(lambda x: x['model'] == 'only' and x['category'] not in unseen_domains, trials))
    only_seen_fluency, only_seen_grammar, only_seen_adequacy = evaluate(only_seen, 'ONLY')
    table_seen.append(
        ['ONLY', round(np.mean(only_seen_fluency), 4), round(np.mean(only_seen_grammar), 4), round(np.mean(only_seen_adequacy), 4)])

    only_unseen = list(filter(lambda x: x['model'] == 'only' and x['category'] in unseen_domains, trials))
    only_unseen_fluency, only_unseen_grammar, only_unseen_adequacy = evaluate(only_unseen, 'ONLY')
    table_unseen.append(
        ['ONLY', round(np.mean(only_unseen_fluency), 4), round(np.mean(only_unseen_grammar), 4),
         round(np.mean(only_unseen_adequacy), 4)])

    evaluate_by_size(only, 'only')
    print(20 * '-' + '\n')

    # ATTENTION ACL
    attacl = list(filter(lambda x: x['model'] == 'attacl', trials))
    attacl_fluency, attacl_grammar, attacl_adequacy = evaluate(attacl, 'Attention ACL')
    table.append(['Attention ACL', round(np.mean(attacl_fluency), 4), round(np.mean(attacl_grammar), 4),
                  round(np.mean(attacl_adequacy), 4)])

    attacl_seen = list(filter(lambda x: x['model'] == 'attacl' and x['category'] not in unseen_domains, trials))
    attacl_seen_fluency, attacl_seen_grammar, attacl_seen_adequacy = evaluate(attacl_seen, 'Attention ACL')
    table_seen.append(
        ['Attention ACL', round(np.mean(attacl_seen_fluency), 4), round(np.mean(attacl_seen_grammar), 4),
         round(np.mean(attacl_seen_adequacy), 4)])

    attacl_unseen = list(filter(lambda x: x['model'] == 'attacl' and x['category'] not in unseen_domains, trials))
    attacl_unseen_fluency, attacl_unseen_grammar, attacl_unseen_adequacy = evaluate(attacl_unseen, 'Attention ACL')
    table_unseen.append(
        ['Attention ACL', round(np.mean(attacl_unseen_fluency), 4), round(np.mean(attacl_unseen_grammar), 4),
         round(np.mean(attacl_unseen_adequacy), 4)])

    evaluate_by_size(attacl, 'Attention ACL')
    print(20 * '-' + '\n')

    # ATTENTION COPY
    attcopy = list(filter(lambda x: x['model'] == 'attcopy', trials))
    attcopy_fluency, attcopy_grammar, attcopy_adequacy = evaluate(attcopy, 'Attention Copy')
    table.append(['Attention Copy', round(np.mean(attcopy_fluency), 4), round(np.mean(attcopy_grammar), 4),
                  round(np.mean(attcopy_adequacy), 4)])

    attcopy_seen = list(filter(lambda x: x['model'] == 'attcopy' and x['category'] not in unseen_domains, trials))
    attcopy_seen_fluency, attcopy_seen_grammar, attcopy_seen_adequacy = evaluate(attcopy_seen, 'Attention Copy')
    table_seen.append(['Attention Copy', round(np.mean(attcopy_seen_fluency), 4), round(np.mean(attcopy_seen_grammar), 4),
                  round(np.mean(attcopy_seen_adequacy), 4)])

    attcopy_unseen = list(filter(lambda x: x['model'] == 'attcopy' and x['category'] in unseen_domains, trials))
    attcopy_unseen_fluency, attcopy_unseen_grammar, attcopy_unseen_adequacy = evaluate(attcopy_unseen, 'Attention Copy')
    table_unseen.append(
        ['Attention Copy', round(np.mean(attcopy_unseen_fluency), 4), round(np.mean(attcopy_unseen_grammar), 4),
         round(np.mean(attcopy_unseen_adequacy), 4)])

    evaluate_by_size(attcopy, 'Attention Copy')
    print(20 * '-' + '\n')

    # PROFILEREG
    profilereg = list(filter(lambda x: x['model'] == 'profilereg', trials))
    profilereg_fluency, profilereg_grammar, profilereg_adequacy = evaluate(profilereg, 'ProfileREG')
    table.append(['ProfileREG', round(np.mean(profilereg_fluency), 4), round(np.mean(profilereg_grammar), 4),
                  round(np.mean(profilereg_adequacy), 4)])

    profilereg_seen = list(filter(lambda x: x['model'] == 'profilereg' and x['category'] not in unseen_domains, trials))
    profilereg_seen_fluency, profilereg_seen_grammar, profilereg_seen_adequacy = evaluate(profilereg_seen, 'ProfileREG')
    table_seen.append(['ProfileREG', round(np.mean(profilereg_seen_fluency), 4), round(np.mean(profilereg_seen_grammar), 4),
                  round(np.mean(profilereg_seen_adequacy), 4)])

    profilereg_unseen = list(filter(lambda x: x['model'] == 'profilereg' and x['category'] in unseen_domains, trials))
    profilereg_unseen_fluency, profilereg_unseen_grammar, profilereg_unseen_adequacy = evaluate(profilereg_unseen, 'ProfileREG')
    table_unseen.append(
        ['ProfileREG', round(np.mean(profilereg_unseen_fluency), 4), round(np.mean(profilereg_unseen_grammar), 4),
         round(np.mean(profilereg_unseen_adequacy), 4)])

    evaluate_by_size(profilereg, 'ProfileREG')
    print(20 * '-' + '\n')

    print('ALL DOMAINS')
    print(tabulate(table, headers=['model', 'acceptability', 'lexicogrammar', 'adequacy']))
    print('\n')
    print('SEEN DOMAINS')
    print(tabulate(table_seen, headers=['model', 'acceptability', 'lexicogrammar', 'adequacy']))
    print('\n')
    print('UNSEEN DOMAINS')
    print(tabulate(table_unseen, headers=['model', 'acceptability', 'lexicogrammar', 'adequacy']))

    header = "resp;only_fluency;only_grammar;only_adequacy;attacl_fluency;attacl_grammar;attacl_adequacy;attcopy_fluency;attcopy_grammar;attcopy_adequacy;profilereg_fluency;profilereg_grammar;profilereg_adequacy"
    with open('trials/beta/official_results.csv', 'w') as f:
        f.write(header + '\n')

        for i, row in enumerate(only_fluency):
            l = [
                i + 1,
                only_fluency[i], only_grammar[i], only_adequacy[i],
                attacl_fluency[i], attacl_grammar[i], attacl_adequacy[i],
                attcopy_fluency[i], attcopy_grammar[i], attcopy_adequacy[i],
                profilereg_fluency[i], profilereg_grammar[i], profilereg_adequacy[i]
            ]
            l = map(lambda x: str(x), l)
            f.write(';'.join(l) + '\n')

    with open('trials/beta/official_results_seen.csv', 'w') as f:
        f.write(header + '\n')

        for i, row in enumerate(only_seen_fluency):
            l = [
                i + 1,
                only_seen_fluency[i], only_seen_grammar[i], only_seen_adequacy[i],
                attacl_seen_fluency[i], attacl_seen_grammar[i], attacl_seen_adequacy[i],
                attcopy_seen_fluency[i], attcopy_seen_grammar[i], attcopy_seen_adequacy[i],
                profilereg_seen_fluency[i], profilereg_seen_grammar[i], profilereg_seen_adequacy[i]
            ]
            l = map(lambda x: str(x), l)
            f.write(';'.join(l) + '\n')

    with open('trials/beta/official_results_unseen.csv', 'w') as f:
        f.write(header + '\n')

        for i, row in enumerate(only_unseen_fluency):
            l = [
                i + 1,
                only_unseen_fluency[i], only_unseen_grammar[i], only_unseen_adequacy[i],
                attacl_unseen_fluency[i], attacl_unseen_grammar[i], attacl_unseen_adequacy[i],
                attcopy_unseen_fluency[i], attcopy_unseen_grammar[i], attcopy_unseen_adequacy[i],
                profilereg_unseen_fluency[i], profilereg_unseen_grammar[i], profilereg_unseen_adequacy[i]
            ]
            l = map(lambda x: str(x), l)
            f.write(';'.join(l) + '\n')

if __name__ == '__main__':
    trials = process_db()

    print_report(trials)
