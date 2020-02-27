__author__ = 'rossanacunha'

import json
import os
import re
import sys

from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError
from urllib3.exceptions import HTTPError

regex = '([0-9]{4})-([0-9]{2})-([0-9]{2})'
ONTOLOGY_URL = "http://dbpedia.org/ontology/"
DBPEDIA_RESOURCE_URL = "http://dbpedia.org/resource/"


def get_domain(entity, sparql):
    query = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT DISTINCT ?domain
            WHERE {<%ENTITY%> a ?domain.
                   ?prop a rdf:Property.
                   ?prop rdfs:domain ?domain.
            }"""
    query = query.replace('%ENTITY%', DBPEDIA_RESOURCE_URL + entity)

    sparql.setQuery(query.encode('UTF-8'))
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(60000)
    try:
        results = sparql.query().convert()
        results = results['results']['bindings']

        if len(results) > 0:
            return list(set([r['domain']['value'] for r in results]))

    except (HTTPError, EndPointInternalError):
        print('Timeout exception on get_domain.')
    except:
        print("Unexpected error on get_domain:", sys.exc_info()[0])

    return []


def get_range(entity, domain, sparql, limit=1000):
    query = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT DISTINCT ?range
            WHERE {<%ENTITY%> a <%DOMAIN%>.
                    ?prop a rdf:Property;
                    rdfs:range  ?range .
            }
            LIMIT %LIMIT%"""
    query = query.replace('%ENTITY%', DBPEDIA_RESOURCE_URL + entity)
    query = query.replace('%DOMAIN%', domain)
    query = query.replace('%LIMIT%', str(limit))

    sparql.setQuery(query.encode('UTF-8'))
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(60000)
    try:
        results = sparql.query().convert()
        results = results['results']['bindings']

        if len(results) > 0:
            return list(set([r['range']['value'] for r in results]))

    except (HTTPError, EndPointInternalError):
        print('Timeout exception on get_range.')
    except:
        print("Unexpected error on get_range:", sys.exc_info()[0])

    return []


def get_predicates(entity, sparql):
    query = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
           PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
           SELECT DISTINCT ?predicate
           WHERE { <%ENTITY%> ?predicate ?object.
           FILTER (STRSTARTS(STR(?predicate), "http://dbpedia.org/"))
           }"""
    query = query.replace('%ENTITY%', DBPEDIA_RESOURCE_URL + entity)

    sparql.setQuery(query.encode('UTF-8'))
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(60000)
    try:
        results = sparql.query().convert()
        results = results['results']['bindings']

        if len(results) > 0:
            return list(set([r['predicate']['value'] for r in results]))

    except (HTTPError, EndPointInternalError):
        print('Timeout exception on get_predicates.')
    except:
        print("Unexpected error on get_predicates:", sys.exc_info()[0])

    return []


def get_objects(entity, sparql):
    query = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT DISTINCT ?object
            WHERE {<%ENTITY%> ?property ?object.
            FILTER (STRSTARTS(STR(?object), "http://dbpedia.org/"))
            }"""
    query = query.replace('%ENTITY%', DBPEDIA_RESOURCE_URL + entity)

    sparql.setQuery(query.encode('UTF-8'))
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(60000)
    try:
        results = sparql.query().convert()
        results = results['results']['bindings']

        if len(results) > 0:
            return list(set([r['object']['value'] for r in results]))

    except (HTTPError, EndPointInternalError):
        print('Timeout exception on get_objects.')
    except:
        print("Unexpected error on get_objects:", sys.exc_info()[0])

    return []


def get_similar(entity_info, sparql, max_filter, limit=10):
    print('#SIMILAR to ENTITY: ', entity_info['entity'])

    similar = []
    main_filter = """ FILTER(%FILTER_VALUE%)"""
    query = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT DISTINCT ?entity ?domain
            WHERE { ?entity a ?domain.
            ?prop a rdf:Property;
                  rdfs:domain ?domain;
                  rdfs:range ?range .
            FILTER (!EXISTS {?entity a <%ENTITY%>  
            })
            %FILTER% 
            } LIMIT %LIMIT% """
    query_filter = ''
    entity_tokens = get_entity_tokens(entity_info['entity'])
    entity_tokens_filter = get_query_filter("""CONTAINS(LCASE(str(?entity)),'%VALUE%')""", entity_tokens[:max_filter])
    domain_filter = get_query_filter("""?domain = <%VALUE%>""", entity_info['domain'][:max_filter])

    if bool(entity_tokens_filter.strip()):
        query_filter += main_filter.replace('%FILTER_VALUE%', entity_tokens_filter) + '\n'

    if bool(domain_filter.strip()):
        query_filter += main_filter.replace('%FILTER_VALUE%', domain_filter) + '\n'

    query = query.replace('%ENTITY%', DBPEDIA_RESOURCE_URL + entity_info['entity'])
    query = query.replace('%FILTER%', query_filter)
    query = query.replace('%LIMIT%', str(limit))

    print(query)

    sparql.setQuery(query.encode('UTF-8'))
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(60000)
    try:
        results = sparql.query().convert()
        results = results['results']['bindings']

        if len(results) > 0:
            for i, result in enumerate(results):
                if result['entity']['value'] != DBPEDIA_RESOURCE_URL + entity_info['entity']:
                    similar.append(result['entity']['value'])
                    # progress = round(i/len(results), 2)
                    # print('Progress (Get similar): ', progress, result['entity']['value'])

    except (HTTPError, EndPointInternalError):
        print('Timeout exception on get_similar.')
    except:
        print("Unexpected error on get_similar:", sys.exc_info()[0])

    return list(set(similar))


def get_entity_tokens(entity):
    entity_tokens = []
    tokens = entity.replace('\"', '').replace('\'', '').replace(',', '').replace('(', '') \
        .replace(')', '').replace('-', '_').lower().split('_')

    for token in tokens:
        date = token.replace('\"', '').replace('\'', '').replace('.', '')
        if len(re.findall(regex, date)) == 0 and len(token) > 2:
            try:
                number = float(token)
            except:
                entity_tokens.append(token)
    print(entity_tokens)
    return entity_tokens


def get_query_filter(query_filter, query_values, next_line_char=" OR "):
    new_filter = ''
    if len(query_values) > 0:
        for i, value in enumerate(query_values):
            new_filter += query_filter.replace('%VALUE%', value)
            if i < len(query_values) - 1:
                new_filter += next_line_char
    return new_filter


def get_entities(entity_path):
    entities = []

    train = json.load(open(os.path.join(entity_path, 'train.json')))
    entities += [w['entity'] for w in train]
    dev = json.load(open(os.path.join(entity_path, 'dev.json')))
    entities += [w['entity'] for w in dev]
    test = json.load(open(os.path.join(entity_path, 'test.json')))
    entities += [w['entity'] for w in test]

    return list(set(entities))


def run(entities, max_filter=5, max_entities=5, max_range=100, max_intersection=2):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    print('#ENTITIES: ', len(entities))
    entity_info, entity_similar = {}, {}
    for i, entity in enumerate(entities[:max_entities]):
        print('ENTITY: ', entity)

        entity_info['entity'] = entity
        entity_info['predicate'] = get_predicates(entity, sparql)
        entity_info['object'] = get_objects(entity, sparql)
        entity_info['domain'] = get_domain(entity, sparql)

        print('DOMAIN: ', entity_info['domain'])
        entity_ranges = []
        for domain in entity_info['domain']:
            entity_ranges += get_range(entity, domain, sparql, max_range)

        entity_info['range'] = list(set(entity_ranges))
        entity_info['similar'] = get_similar(entity_info, sparql, max_filter)

        entity_similar[entity] = check_similar(entity_info, sparql, max_entities, max_intersection, max_range, )
        print('Similar: ', entity_similar[entity])

    progress = round(i / len(entities[:max_entities]), 2)
    print('Progress (entity): ', progress, entity)

    return entity_info, entity_similar


def check_similar(entity_info, sparql, max_entities, max_intersection, max_range):
    print('#SIMILAR: ', len(entity_info['similar'][:max_entities]))

    similar_entities = []
    for i, similar in enumerate(entity_info['similar'][:max_entities]):
        similar = similar.replace(DBPEDIA_RESOURCE_URL, '')

        predicates = get_predicates(similar, sparql)
        objects = get_objects(similar, sparql)
        domains = get_domain(similar, sparql)
        ranges = []
        for domain in domains:
            ranges += get_range(entity_info['entity'], domain, sparql, max_range)

        # Intersection
        similar_predicates = set(entity_info['predicate']).intersection(set(predicates))
        similar_objects = set(entity_info['object']).intersection(set(objects))
        similar_domains = set(entity_info['domain']).intersection(set(domains))
        similar_ranges = set(entity_info['range']).intersection(set(ranges))

        if len(similar_domains) >= max_intersection and len(similar_ranges) >= max_intersection \
                and (len(similar_objects) >= max_intersection or len(similar_predicates) >= max_intersection):
            similar_entities.append(similar)

        progress = round(i / len(entity_info['similar'][:max_entities]), 2)
        print('Progress (Check Similar): ', progress, similar)

    return similar_entities


if __name__ == '__main__':
    path = '/NeuralREG/data/v1.5'
    entities = get_entities(path)

    info, similar = run(entities=entities, max_filter=5, max_entities=100)
    json.dump(info, open(os.path.join(path, 'entity_similar_info.json'), 'w'), ensure_ascii=False)
    json.dump(similar, open(os.path.join(path, 'entity_similar.json'), 'w'), ensure_ascii=False)
