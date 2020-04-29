__author__ = 'thiagocastroferreira'

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


def process_db():
    trials = []
    with open('trials/beta/trials_results.csv') as f:
        csv_file = DictReader(f, delimiter="\t")
        for trial in csv_file:
            trial_ = {'id': trial['eid'], 'size': str(trial['size']), 'model': trial['model'],
                      'acceptability': str(trial['acceptability']),
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
    table = []

    only = list(filter(lambda x: x['model'] == 'only', trials))
    only_fluency, only_grammar, only_adequacy = evaluate(only, 'ONLY')
    table.append(
        ['ONLY', round(np.mean(only_fluency), 4), round(np.mean(only_grammar), 4), round(np.mean(only_adequacy), 4)])
    evaluate_by_size(only, 'only')
    print(20 * '-' + '\n')

    attacl = list(filter(lambda x: x['model'] == 'attacl', trials))
    attacl_fluency, attacl_grammar, attacl_adequacy = evaluate(attacl, 'Attention ACL')
    table.append(['Attention ACL', round(np.mean(attacl_fluency), 4), round(np.mean(attacl_grammar), 4),
                  round(np.mean(attacl_adequacy), 4)])
    evaluate_by_size(attacl, 'Attention ACL')
    print(20 * '-' + '\n')

    attcopy = list(filter(lambda x: x['model'] == 'attcopy', trials))
    attcopy_fluency, attcopy_grammar, attcopy_adequacy = evaluate(attcopy, 'Attention Copy')
    table.append(['Attention Copy', round(np.mean(attcopy_fluency), 4), round(np.mean(attcopy_grammar), 4),
                  round(np.mean(attcopy_adequacy), 4)])
    evaluate_by_size(attcopy, 'Attention Copy')
    print(20 * '-' + '\n')

    profilereg = list(filter(lambda x: x['model'] == 'profilereg', trials))
    profilereg_fluency, profilereg_grammar, profilereg_adequacy = evaluate(profilereg, 'ProfileREG')
    table.append(['ProfileREG', round(np.mean(profilereg_fluency), 4), round(np.mean(profilereg_grammar), 4),
                  round(np.mean(profilereg_adequacy), 4)])
    evaluate_by_size(profilereg, 'ProfileREG')
    print(20 * '-' + '\n')

    print(tabulate(table, headers=['model', 'acceptability', 'lexicogrammar', 'adequacy']))

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


if __name__ == '__main__':
    trials = process_db()

    print_report(trials)
