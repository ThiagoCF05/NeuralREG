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
import scipy.stats
from csv import DictReader
from tabulate import tabulate
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon

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


def evaluate_by_size(trials, model):
    sizes = sorted(list(set(map(lambda x: x['size'], trials))))

    for size in sizes:
        size_trials = list(filter(lambda x: x['size'] == size, trials))

        adequacy = [float(x['adequacy']) for x in size_trials]
        grammar = [float(x['lexicogrammar']) for x in size_trials]
        fluency = [float(x['acceptability']) for x in size_trials]

        print('SIZE ' + str(size) + ': ', model)
        print('ACCEPTABILITY/FLUENCY:', str(round(np.mean(fluency), 2)), str(round(np.std(fluency), 2)))
        print('LEXICOGRAMMAR:', str(round(np.mean(grammar), 2)), str(round(np.std(grammar), 2)))
        print('ADEQUACY:', str(round(np.mean(adequacy), 2)), str(round(np.std(adequacy), 2)))
        print(10 * '-')


def evaluate_confidence_interval(trials, confidence=0.95):
    a = 1.0 * np.array(trials)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)

    return m, h, m + h


def evaluate_std_deviation(trials):
    return round(np.std(trials), 4)


def evaluate_trials(fluency, grammar, adequacy, model):
    mean_fluency, ci_lower_fluency, ci_upper_fluency = evaluate_confidence_interval(fluency)
    std_fluency = evaluate_std_deviation(fluency)

    mean_grammar, ci_lower_grammar, ci_upper_grammar = evaluate_confidence_interval(grammar)
    std_grammar = evaluate_std_deviation(grammar)
    mean_adequacy, ci_lower_adequacy, ci_upper_adequacy = evaluate_confidence_interval(adequacy)
    std_adequacy = evaluate_std_deviation(adequacy)
    table = [model, mean_fluency, ci_lower_fluency, ci_upper_fluency, std_fluency, mean_grammar,
             ci_lower_grammar, ci_upper_grammar, std_grammar, mean_adequacy, ci_lower_adequacy,
             ci_upper_adequacy, std_adequacy]

    return table


def apply_wilcoxon_test(test_type, data1, data2, model1, model2):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html

    w, p = wilcoxon(data1, data2)
    print('%s between %s and %s: %.3f, p=%.3f' % (test_type, model1, model2, w, p))


def print_report(trials):
    table, table_seen, table_unseen = [], [], []

    # ONLYNAMES
    only = list(filter(lambda x: x['model'] == 'only', trials))
    only_fluency, only_grammar, only_adequacy = evaluate(only, 'ONLY')
    table.append(evaluate_trials(only_fluency, only_grammar, only_adequacy, 'ONLY'))

    only_seen = list(filter(lambda x: x['model'] == 'only' and x['category'] not in unseen_domains, trials))
    only_seen_fluency, only_seen_grammar, only_seen_adequacy = evaluate(only_seen, 'ONLY')
    table_seen.append(evaluate_trials(only_seen_fluency, only_seen_grammar, only_seen_adequacy, 'ONLY'))

    only_unseen = list(filter(lambda x: x['model'] == 'only' and x['category'] in unseen_domains, trials))
    only_unseen_fluency, only_unseen_grammar, only_unseen_adequacy = evaluate(only_unseen, 'ONLY')
    table_unseen.append(evaluate_trials(only_unseen_fluency, only_unseen_grammar, only_unseen_adequacy, 'ONLY'))

    evaluate_by_size(only, 'only')
    print(20 * '-' + '\n')

    # ATTENTION ACL
    attacl = list(filter(lambda x: x['model'] == 'attacl', trials))
    attacl_fluency, attacl_grammar, attacl_adequacy = evaluate(attacl, 'Attention ACL')
    table.append(evaluate_trials(attacl_fluency, attacl_grammar, attacl_adequacy, 'Attention ACL'))

    attacl_seen = list(filter(lambda x: x['model'] == 'attacl' and x['category'] not in unseen_domains, trials))
    attacl_seen_fluency, attacl_seen_grammar, attacl_seen_adequacy = evaluate(attacl_seen, 'Attention ACL')
    table_seen.append(evaluate_trials(attacl_seen_fluency, attacl_seen_grammar, attacl_seen_adequacy, 'Attention ACL'))

    attacl_unseen = list(filter(lambda x: x['model'] == 'attacl' and x['category'] in unseen_domains, trials))
    attacl_unseen_fluency, attacl_unseen_grammar, attacl_unseen_adequacy = evaluate(attacl_unseen, 'Attention ACL')
    table_unseen.append(
        evaluate_trials(attacl_unseen_fluency, attacl_unseen_grammar, attacl_unseen_adequacy, 'Attention ACL'))

    evaluate_by_size(attacl, 'Attention ACL')
    print(20 * '-' + '\n')

    # ATTENTION COPY
    attcopy = list(filter(lambda x: x['model'] == 'attcopy', trials))
    attcopy_fluency, attcopy_grammar, attcopy_adequacy = evaluate(attcopy, 'Attention Copy')
    table.append(evaluate_trials(attcopy_fluency, attcopy_grammar, attcopy_adequacy, 'Attention Copy'))

    attcopy_seen = list(filter(lambda x: x['model'] == 'attcopy' and x['category'] not in unseen_domains, trials))
    attcopy_seen_fluency, attcopy_seen_grammar, attcopy_seen_adequacy = evaluate(attcopy_seen, 'Attention Copy')
    table_seen.append(
        evaluate_trials(attcopy_seen_fluency, attcopy_seen_grammar, attcopy_seen_adequacy, 'Attention Copy'))

    attcopy_unseen = list(filter(lambda x: x['model'] == 'attcopy' and x['category'] in unseen_domains, trials))
    attcopy_unseen_fluency, attcopy_unseen_grammar, attcopy_unseen_adequacy = evaluate(attcopy_unseen, 'Attention Copy')
    table_unseen.append(
        evaluate_trials(attcopy_unseen_fluency, attcopy_unseen_grammar, attcopy_unseen_adequacy, 'Attention Copy'))

    evaluate_by_size(attcopy, 'Attention Copy')
    print(20 * '-' + '\n')

    # PROFILEREG
    profilereg = list(filter(lambda x: x['model'] == 'profilereg', trials))
    profilereg_fluency, profilereg_grammar, profilereg_adequacy = evaluate(profilereg, 'ProfileREG')
    table.append(evaluate_trials(profilereg_fluency, profilereg_grammar, profilereg_adequacy, 'ProfileREG'))

    profilereg_seen = list(filter(lambda x: x['model'] == 'profilereg' and x['category'] not in unseen_domains, trials))
    profilereg_seen_fluency, profilereg_seen_grammar, profilereg_seen_adequacy = evaluate(profilereg_seen, 'ProfileREG')
    table_seen.append(
        evaluate_trials(profilereg_seen_fluency, profilereg_seen_grammar, profilereg_seen_adequacy, 'ProfileREG'))

    profilereg_unseen = list(filter(lambda x: x['model'] == 'profilereg' and x['category'] in unseen_domains, trials))
    profilereg_unseen_fluency, profilereg_unseen_grammar, profilereg_unseen_adequacy = evaluate(profilereg_unseen,
                                                                                                'ProfileREG')
    table_unseen.append(
        evaluate_trials(profilereg_unseen_fluency, profilereg_unseen_grammar, profilereg_unseen_adequacy,
                        'ProfileREG'))

    evaluate_by_size(profilereg, 'ProfileREG')
    print(20 * '-' + '\n')

    print('ALL DOMAINS')
    model_header = ['model', 'acceptability (mean)', 'acceptability (lower ci)', 'acceptability (upper ci)',
                    'acceptability (std)', 'lexicogrammar (mean)', 'lexicogrammar (lower ci)',
                    'lexicogrammar (upper ci)', 'lexicogrammar (std)', 'adequacy (mean)', 'adequacy (lower ci)',
                    'adequacy (upper ci)', 'adequacy (std)']
    print(tabulate(table, headers=model_header))
    print('\n')
    print('SEEN DOMAINS')
    print(tabulate(table_seen, headers=model_header))
    print('\n')
    print('UNSEEN DOMAINS')
    print(tabulate(table_unseen, headers=model_header))
    print(20 * '-' + '\n')

    print('WILCOXON TEST - ALL DOMAINS')
    apply_wilcoxon_test('Fluency', only_fluency, attcopy_fluency, 'ONLY', 'ATTCOPY')
    apply_wilcoxon_test('Lexicogrammar', only_grammar, attcopy_grammar, 'ONLY', 'ATTCOPY')
    apply_wilcoxon_test('Adequacy', only_adequacy, attcopy_adequacy, 'ONLY', 'ATTCOPY')
    print('\n')
    apply_wilcoxon_test('Fluency', only_fluency, attcopy_fluency, 'ONLY', 'ATTCOPY')
    apply_wilcoxon_test('Lexicogrammar', only_grammar, attcopy_grammar, 'ONLY', 'ATTCOPY')
    apply_wilcoxon_test('Adequacy', only_adequacy, attcopy_adequacy, 'ONLY', 'ATTCOPY')
    print('\n')
    print('WILCOXON TEST - SEEN DOMAINS')
    apply_wilcoxon_test('Fluency', only_seen_fluency, attcopy_seen_fluency, 'ONLY', 'ATTCOPY')
    apply_wilcoxon_test('Lexicogrammar', only_seen_grammar, attcopy_seen_grammar, 'ONLY', 'ATTCOPY')
    apply_wilcoxon_test('Adequacy', only_seen_adequacy, attcopy_seen_adequacy, 'ONLY', 'ATTCOPY')
    print('\n')
    # print('WILCOXON TEST - UNSEEN DOMAINS')
    # apply_wilcoxon_test('Fluency', only_unseen_fluency, attcopy_unseen_fluency, 'ONLY', 'ATTCOPY')
    # apply_wilcoxon_test('Lexicogrammar', only_unseen_grammar, attcopy_unseen_grammar, 'ONLY', 'ATTCOPY')
    # apply_wilcoxon_test('Adequacy', only_unseen_adequacy, attcopy_unseen_adequacy, 'ONLY', 'ATTCOPY')
    print(20 * '-' + '\n')

    header = "resp;only_fluency;only_grammar;only_adequacy;attacl_fluency;attacl_grammar;attacl_adequacy" \
             ";attcopy_fluency;attcopy_grammar;attcopy_adequacy;profilereg_fluency;profilereg_grammar" \
             ";profilereg_adequacy "
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
