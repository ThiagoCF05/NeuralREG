import json
import os
import sys

import rdflib
from rdflib import URIRef, RDF, Literal
from rdflib.namespace import FOAF
from urllib.parse import quote

URI_ORGANISATION = URIRef('http://dbpedia.org/ontology/Organisation')
URI_PLACE = URIRef('http://dbpedia.org/ontology/Place')
FOAF_GENDER = FOAF["gender"]


class REGEntityExtractor:
    def __init__(self, data_path, write_path):
        self.DBR = rdflib.namespace.Namespace("http://dbpedia.org/resource/")
        self.EOS = "eos"
        self.data_path = data_path
        self.write_path = write_path

        self.train_data, _ = self.extract_entities(entry_set=json.load(open(os.path.join(data_path, 'train.json'))))
        self.dev_data, _ = self.extract_entities(entry_set=json.load(open(os.path.join(data_path, 'dev.json'))))
        self.test_data, _ = self.extract_entities(entry_set=json.load(open(os.path.join(data_path, 'test.json'))))

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

        # print('Entity: {} - sex: {} - type: {}'.format(entity, entity_sex, entity_type))
        return {
            '%s' % entity: {
                'sex': entity_sex,
                'type': entity_type}
        }

    def extract_entities(self, entry_set):
        attributes, entities = [], []

        for i, entity_set in enumerate(entry_set):
            entity = entity_set['entity']
            if entity not in entities:
                entities.append(entity)
                attributes.append(self.extract_entities_attributes(entity))

        return attributes, entities

    def extract_entities_attributes(self, entity):
        uri = rdflib.term.URIRef(self.DBR[quote(entity)])
        # print(uri)
        graph = rdflib.Graph()

        try:
            graph.parse(uri)
        except:
            print('Error with entity %s' % entity)
            raise

        return self.get_attributes(graph, uri, entity)


if __name__ == "__main__":
    # datapath = '/home/tcastrof/Experiments/versions/v1.5/en'
    # writepath= '/roaming/tcastrof/emnlp2019/reg'

    datapath = sys.argv[1]
    writepath = sys.argv[2]

    s = REGEntityExtractor(data_path=datapath, write_path=writepath)
