__author__ = ''

"""
Author: ANONYMOUS
Date: 07/06/2017
Description:
    FERREIRA ET AL., 2016
    Main code for Referring expression generation

    PYTHON VERSION :2.7

    DEPENDENCIES:
        cPickle
        stanford_corenlp_pywrapper: https://github.com/brendano/stanford_corenlp_pywrapper

    UPDATE CONSTANT PATHS:
        MODEL_PATH: path to trained model
        IN_FILE: path to reference collection to be realized
        OUT_FILE: path to save results
"""

import cPickle as p
import sys
sys.path.append('../')
sys.path.append('~/workspace/stanford_corenlp_pywrapper')
from stanford_corenlp_pywrapper import CoreNLP

import form_choice
import re

class REG(object):
    def __init__(self, freferences, fmodel, fout):
        self.references = p.load(open(freferences))
        self.model = p.load(open(fmodel))
        self.fout = fout

        self.proc = CoreNLP('parse')

    def _realize_date(self, date):
        year, month, day = date.replace('\'', '').replace('\"', '').split('-')

        if day[-1] == '1':
            day = day + 'st'
        elif day[-1] == '2':
            day = day + 'nd'
        elif day[-1] == '3':
            day = day + 'rd'
        else:
            day = day + 'th'

        month = int(month)
        if month == 1:
            month = 'january'
        elif month == 2:
            month = 'february'
        elif month == 3:
            month = 'march'
        elif month == 4:
            month = 'april'
        elif month == 5:
            month = 'may'
        elif month == 6:
            month = 'june'
        elif month == 7:
            month = 'july'
        elif month == 8:
            month = 'august'
        elif month == 9:
            month = 'september'
        elif month == 10:
            month = 'october'
        elif month == 11:
            month = 'november'
        elif month == 12:
            month = 'december'
        else:
            month = str(month)

        return ' '.join([month, day, year])

    def _realize_description(self, prev_references, reference, data):
        '''
        Generating a description/demonstrative according to reg_train.py script
        :param prev_references:
        :param reference:
        :param data:
        :return:
        '''
        syntax = reference['syntax']
        text_status = reference['text_status']
        sentence_status = reference['sentence_status']
        entity = reference['entity']

        descriptions = data[(syntax, text_status, sentence_status, entity)]
        if len(descriptions) == 0:
            name = ' '.join(entity.replace('\'', '').replace('\"', '').split('_'))
            return name
        else:
            description = descriptions[0][0]

            # Check for a competitor
            isCompetitor = False
            for prev_reference in prev_references:
                if prev_reference['entity'] != entity and prev_reference['realization'] == description:
                    isCompetitor = True
                    break

            # If it is a competitor, return the name of the entity
            if not isCompetitor:
                return description
            else:
                name = ' '.join(entity.replace('\'', '').replace('\"', '').split('_'))
                return name

    def _realize_name(self, reference, data):
        '''
        Generate based on result of reg_train.py script
        :param reference:
        :param data:
        :return:
        '''
        syntax = reference['syntax']
        text_status = reference['text_status']
        sentence_status = reference['sentence_status']
        entity = reference['entity']

        try:
            names = data[(syntax, text_status, sentence_status, entity)]
            if len(names) > 0:
                name = names[0][0]
            else:
                name = ' '.join(entity.replace('\'', '').replace('\"', '').split('_'))
        except Exception as e:
            print(e.message)
            name = ' '.join(entity.replace('\'', '').replace('\"', '').split('_'))

        return name

    def _realize_pronoun(self, prev_references, reference, data):
        entity = reference['entity']
        syntax = reference['syntax']

        pronouns = data[entity]
        if len(pronouns) == 0:
            if syntax == 'subj-det':
                pronoun = 'its'
            else:
                pronoun = 'it'
        else:
            pronoun = pronouns[0][0]

            if pronoun == 'he':
                if syntax == 'np-obj':
                    pronoun = 'him'
                elif syntax == 'subj-det':
                    pronoun = 'his'
            elif pronoun == 'she':
                if syntax != 'np-subj':
                    pronoun = 'her'
            elif pronoun == 'it':
                if syntax == 'subj-det':
                    pronoun = 'its'
                else:
                    pronoun = 'it'
            elif pronoun == 'they':
                if syntax == 'np-obj':
                    pronoun = 'them'
                elif syntax == 'subj-det':
                    pronoun = 'their'

        # Check for a competitor
        competitors = {
            'he':'he', 'his':'he', 'him':'he',
            'she':'she', 'her':'she', 'hers':'she',
            'it':'it', 'its':'it',
            'we':'we', 'our':'we', 'ours':'we', 'us':'we',
            'they':'they', 'their':'they', 'theirs':'they', 'them':'they'
        }
        isCompetitor = False
        for prev_reference in prev_references[::-1]:
            if prev_reference['entity'] != entity:
                distractor_pronouns = data[prev_reference['entity']]
                if len(distractor_pronouns) == 0:
                    distractor_pronouns = ['it']
                if competitors[pronoun] in distractor_pronouns:
                    isCompetitor = True
                    break
            else:
                break

        return isCompetitor, pronoun

    def _realize(self, prev_references, reference, topic=''):
        entity = reference['entity']
        regex = '([0-9]{4})-([0-9]{2})-([0-9]{2})'
        matcher = re.match(regex, entity.replace('\'', '').replace('\"', ''))
        if matcher is not None:
            return self._realize_date(entity)

        if reference['form'] == 'pronoun':
            isCompetitor, pronoun = self._realize_pronoun(prev_references, reference, self.model['pronouns'])

            # if reference['no_pronoun']:
            #     return self._realize_name(reference, self.model['names'])
            # if isCompetitor:
            #     return self._realize_description(prev_references, reference, self.model['descriptions'])
            # else:
            return pronoun
        elif reference['form'] == 'name':
            return self._realize_name(reference, self.model['names'])
        elif reference['form'] == 'description':
            return self._realize_description(prev_references, reference, self.model['descriptions'])
        elif reference['form'] == 'demonstrative':
            return self._realize_description(prev_references, reference, self.model['demonstratives'])

    def generate(self, references):

        references = form_choice.variation_bayes(references)

        references = sorted(references, key=lambda x: (x['sentence'], x['pos']))

        prev_references = []
        for reference in references:
            try:
                realization = self._realize(prev_references, reference).lower()
            except:
                realization = ' '.join(reference['entity'].replace('\'', '').replace('\"', '').split('_'))
            reference['realization'] = realization
            prev_references.append(reference)

        return references

    def run(self):
        results = []
        # group references per text
        text_ids = list(set(map(lambda x: x['text_id'], self.references)))

        for text_id in text_ids:
            text_references = filter(lambda x: x['text_id'] == text_id, self.references)

            results.extend(self.generate(text_references))

        results.sort(key=lambda x: (x['text_id'], x['general_pos']))
        p.dump(results, open(self.fout, 'w'))


if __name__ == '__main__':
    MODEL_PATH = 'reg.cPickle'
    IN_FILE = '../data/test/data.cPickle'
    OUT_FILE = '../eval/ferreira.cPickle'
    reg = REG(fmodel=MODEL_PATH, freferences=IN_FILE, fout=OUT_FILE)
    reg.run()