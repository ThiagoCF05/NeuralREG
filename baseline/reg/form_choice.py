__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 02/06/2017
Description:
    Choice of referential form model
"""

import cPickle as p
import operator
from random import shuffle

DISTRIBUTIONS = p.load(open('reg/pronoun_data/form_distributions.cPickle'))

# text-new -> name / text-old -> pronoun
def rule_form_choice(text_status):
    if text_status == 'new':
        return 'name'
    else:
        return 'pronoun'

def regular_bayes(references, distributions=DISTRIBUTIONS):
    for reference in references:
        X = (reference['syntax'], reference['text_status'], reference['sentence_status'])

        form = sorted(distributions[X].items(), key=operator.itemgetter(1), reverse=True)[0][0]
        reference['form'] = form
    return references

def variation_bayes(references):
    '''
    Apply variation in the choice of referential with distributions from ACL 2016 model
    :param references:
    :return:
    '''
    distributions = p.load(open('reg/pronoun_data/form_distributions.cPickle'))
    def group():
        g = {}
        for reference in references:
            X = (reference['syntax'], reference['text_status'], reference['sentence_status'])
            if X not in g:
                g[X] = {'distribution': distributions[X], 'references':[]}
            g[X]['references'].append(reference)
        return g

    def choose_form(_references, distribution):
        size = len(_references)
        for form in distribution:
            distribution[form] = size * distribution[form]

        # print distribution
        shuffle(_references)
        for reference in _references:
            form = sorted(distribution.items(), key=operator.itemgetter(1), reverse=True)[0][0]
            reference['form'] = form

            distribution[form] -= 1
        return _references

    groups = group()
    references = []
    for g in groups:
        _references = choose_form(groups[g]['references'], groups[g]['distribution'])
        references.extend(_references)
    return references