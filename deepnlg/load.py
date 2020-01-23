__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 28/02/2019
"""


def source(tripleset, entitymap, entities={}):
    src, delexsrc = [], []
    for triple in tripleset:
        subject = '_'.join(triple.subject.split())
        predicate = '_'.join(triple.predicate.split())
        patient = '_'.join(triple.object.split())

        src.append('<TRIPLE>')
        src.append(subject)
        src.append(predicate)
        src.append(patient)
        src.append('</TRIPLE>')

        delexsrc.append('<TRIPLE>')
        # SUBJECT
        if entitymap[triple.subject] not in entities:
            entity = 'ENTITY-' + str(len(list(entities.keys())) + 1)
            entities[entitymap[triple.subject]] = entity
        else:
            entity = entities[entitymap[triple.subject]]
        delexsrc.append(entity)

        # PREDICATE
        delexsrc.append(predicate)

        # OBJECT
        if entitymap[triple.object] not in entities:
            entity = 'ENTITY-' + str(len(list(entities.keys())) + 1)
            entities[entitymap[triple.object]] = entity
        else:
            entity = entities[entitymap[triple.object]]
        delexsrc.append(entity)
        delexsrc.append('</TRIPLE>')
    # src.append('eos')
    # delexsrc.append('eos')
    return src, delexsrc, entities


def snt_source(tripleset, entitymap, entities):
    aggregation = []
    delex_aggregation = []

    for sentence in tripleset:
        snt, delex_snt = ['<SNT>'], ['<SNT>']
        for striple in sentence:
            subject = '_'.join(striple.subject.split())
            predicate = '_'.join(striple.predicate.split())
            patient = '_'.join(striple.object.split())

            snt.append('<TRIPLE>')
            snt.append(subject)
            snt.append(predicate)
            snt.append(patient)
            snt.append('</TRIPLE>')

            delex_snt.append('<TRIPLE>')

            # SUBJECT
            if entitymap[striple.subject] not in entities:
                entity = 'ENTITY-' + str(len(list(entities.keys())) + 1)
                entities[entitymap[striple.subject]] = entity
            else:
                entity = entities[entitymap[striple.subject]]
            delex_snt.append(entity)

            # PREDICATE
            delex_snt.append(predicate)

            # OBJECT
            if entitymap[striple.object] not in entities:
                entity = 'ENTITY-' + str(len(list(entities.keys())) + 1)
                entities[entitymap[striple.object]] = entity
            else:
                entity = entities[entitymap[striple.object]]
            delex_snt.append(entity)

            delex_snt.append('</TRIPLE>')

        snt.append('</SNT>')
        aggregation.extend(snt)

        delex_snt.append('</SNT>')
        delex_aggregation.extend(delex_snt)

    # aggregation = ['bos'] + aggregation + ['eos']
    # delex_aggregation = ['bos'] + delex_aggregation + ['eos']
    return aggregation, delex_aggregation, entities
