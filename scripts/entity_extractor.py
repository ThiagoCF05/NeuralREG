import json
import os
import sys

import rdflib
from rdflib import URIRef, RDF, Literal
from rdflib.namespace import FOAF

URI_ORGANISATION = URIRef('http://dbpedia.org/ontology/Organisation')
URI_PLACE = URIRef('http://dbpedia.org/ontology/Place')
FOAF_GENDER = FOAF["gender"]


class REGEntityExtractor:
    def __init__(self, data_path, write_path, entity_uri):
        self.EOS = "eos"
        self.data_path = data_path
        self.write_path = write_path

        self.train_data = self.extract_entities(entry_set=json.load(open(os.path.join(data_path, 'train.json'))), entity_uri=entity_uri)
        self.dev_data = self.extract_entities(entry_set=json.load(open(os.path.join(data_path, 'dev.json'))), entity_uri=entity_uri)
        self.test_data = self.extract_entities(entry_set=json.load(open(os.path.join(data_path, 'test.json'))), entity_uri=entity_uri)

        json.dump(self.train_data, open(os.path.join(write_path, 'train.ent.json'), 'w'))
        json.dump(self.dev_data, open(os.path.join(write_path, 'dev.ent.json'), 'w'))
        json.dump(self.test_data, open(os.path.join(write_path, 'test.ent.json'), 'w'))

    def get_attributes(self, graph, uri, entity):
        entity_sex = 'neutral'
        entity_type = 'other'

        if (uri, RDF.type, FOAF.Person) in graph:
            entity_type = "person"
            for sex in graph.objects(uri, FOAF_GENDER):
                entity_sex = Literal(sex).value
        elif (uri, RDF.type, URI_ORGANISATION) in graph:
            entity_type = "organisation"
        elif (uri, RDF.type, URI_PLACE) in graph:
            entity_type = "place"

        # print(uri)
        # print('Entity: {} - sex: {} - type: {}'.format(entity, entity_sex, entity_type))
        return {
            '%s' % entity: {
                'sex': entity_sex,
                'type': entity_type}
        }

    def extract_entities(self, entry_set, entity_uri):
        attributes = []

        for i, entity_set in enumerate(entry_set):
            entity = entity_set['entity']
            attributes.append(self.extract_entities_attributes(entity, entity_uri))

        return attributes

    def extract_entities_attributes(self, entity, uri):
        query_uri = rdflib.term.URIRef(uri % entity)

        graph = rdflib.Graph()
        graph.parse(query_uri)

        return self.get_attributes(graph, query_uri, entity)


if __name__ == "__main__":
    # writepath='/roaming/tcastrof/emnlp2019/reg'
    # datapath = '/home/tcastrof/Experiments/versions/v1.5/en'

    datapath = sys.argv[1]
    writepath = sys.argv[2]
    entityuri = "http://dbpedia.org/resource/%s"

    s = REGEntityExtractor(data_path=datapath, write_path=writepath, entity_uri=entityuri)
