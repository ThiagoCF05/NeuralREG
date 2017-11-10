__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 02/05/2017
Description:
    Database structure
"""

from mongoengine import *
connect('challenge')

class Predicate(Document):
    name = StringField(required=True, max_length=100)

class Entity(Document):
    name = StringField(required=True, max_length=300)
    type = StringField(required=True, max_length=100)
    ner = StringField()
    category = StringField()
    description = StringField()

class Triple(Document):
    agent = ReferenceField(Entity)
    predicate = ReferenceField(Predicate)
    patient = ReferenceField(Entity)

    meta = {'allow_inheritance': True}

class Refex(EmbeddedDocument):
    docid = StringField(required=True)
    ref_type = StringField(required=True, max_length=10)
    refex = StringField(required=True, max_length=400)
    annotation = StringField(max_length=10)

class Reference(Document):
    entity = ReferenceField(Entity)

    syntax = StringField(required=True, max_length=10)
    text_status = StringField(required=True, max_length=10)
    sentence_status = StringField(required=True, max_length=10)

    refexes = ListField(EmbeddedDocumentField(Refex))

    meta = {'allow_inheritance': True}

class Template(Document):
    docid = IntField(required=True)
    category = StringField(required=True, max_length=20)
    triples = ListField(ReferenceField(Triple))
    template = StringField(max_length=1000)
    delex_type = StringField(max_length=10)

    meta = {'allow_inheritance': True}

class Lex(Document):
    docid = StringField(required=True, max_length=10)
    comment = StringField(required=True, max_length=20)
    text = StringField(required=True, max_length=1000)
    parse_tree = StringField()

    template = StringField(max_length=1000)
    delex_type = StringField(max_length=10)
    # references = ListField(ReferenceField(Reference))
    meta = {'allow_inheritance': True}

class Entry(Document):
    docid = StringField(required=True, max_length=10)
    size = IntField(required=True)
    category = StringField(required=True, max_length=20)
    set = StringField(required=True, max_length=20)

    triples = ListField(ReferenceField(Triple))
    texts = ListField(ReferenceField(Lex))

    meta = {'allow_inheritance': True}