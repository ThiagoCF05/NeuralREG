__author__='thiagocastroferreira'

import json
import os
import re

from SPARQLWrapper import SPARQLWrapper, JSON

def check_type(entity, sparql):
    query = """ASK { <http://dbpedia.org/resource/%ENTITY%> rdf:type dbo:%TYPE% }"""

    # is person
    sparql.setQuery(query.replace('%ENTITY%', entity).replace('%TYPE%', 'Person'))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if results['boolean']:
        return 'Person'

    # is Organisation
    sparql.setQuery(query.replace('%ENTITY%', entity).replace('%TYPE%', 'Organisation'))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if results['boolean']:
        return 'Organisation'

    # is Location
    sparql.setQuery(query.replace('%ENTITY%', entity).replace('%TYPE%', 'Location'))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if results['boolean']:
        return 'Location'
    else:
        return 'Other'

def check_gender(entity, sparql):
    query = """SELECT ?gender WHERE {
            <http://dbpedia.org/resource/%ENTITY%> foaf:gender ?gender .
            FILTER(langMatches(lang(?gender), "en")) .
            }"""

    # is person
    sparql.setQuery(query.replace('%ENTITY%', entity))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results = results['results']['bindings']

    if len(results) > 0:
        return results[0]['gender']['value']
    else:
        return 'neutral'

def run(thread_id, entities):
    print('Thread ID: ', thread_id, 'ENTITIES: ', len(entities))

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    entity_types = {}
    for i, entity in enumerate(entities):
        regex = '([0-9]{4})-([0-9]{2})-([0-9]{2})'
        date = entity.replace('\"', '').replace('\'', '').replace('.', '')
        if len(re.findall(regex, date)) > 0:
            entity_types[entity] = 'Date'
        elif len(entity.split()) > 1:
            entity_types[entity] = 'Other'
        else:
            try:
                aux = int(entity.replace('\"', '').replace('\'', '').replace('.', ''))
                entity_types[entity] = 'Number'
            except:
                try:
                    entity_types[entity] = check_type(entity, sparql)
                except:
                    entity_types[entity] = 'Other'
        progress = round(i / len(entities), 2)
        print('Thread ID: ', thread_id, 'Progress: ', progress, entity, entity_types[entity])

    return entity_types


if __name__ == '__main__':
    path = '/home/tcastrof/NeuralREG/data/v1.5'

    entities = []
    train = json.load(open(os.path.join(path, 'train.json')))
    entities += [w['entity'] for w in train]

    dev = json.load(open(os.path.join(path, 'dev.json')))
    entities += [w['entity'] for w in dev]

    test = json.load(open(os.path.join(path, 'test.json')))
    entities += [w['entity'] for w in test]

    entities = list(set(entities))

    entity_type = run(1, entities)
    json.dump(entity_type, open(os.path.join(path, 'entity_types.json'), 'w'))

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    gender = {}
    for entity in entity_type:
        print(entity)
        if entity_type[entity] == 'Person':
            gender[entity] = check_gender(entity, sparql)
        else:
            gender[entity] = 'neutral'

    json.dump(gender, open(os.path.join(path, 'gender.json'), 'w'))