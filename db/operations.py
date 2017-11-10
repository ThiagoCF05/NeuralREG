__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 02/05/2017
Description:
    CRUD (Create, read, update and delete) operations in database
"""

import sys
sys.path.append('../')
from db.model import *
from mongoengine.queryset.visitor import Q

# Predicate operations
def save_predicate(name):
    predicate = Predicate(name=name)

    query = Predicate.objects(name=name)
    if query.count() == 0:
        predicate.save()
    else:
        predicate = query.get()
    return predicate

# Entity operations
def save_entity(name, type, ner, category, description):
    entity = Entity(name=name, type=type, ner=ner, category=category, description=description)

    query = Entity.objects(name=name, type=type)
    if query.count() == 0:
        entity.save()
    else:
        entity = query.get()
    return entity

def get_entity(name):
    return Entity.objects(name=name).first()

def add_description(entity, description):
    entity.modify(set__description=description)

def add_ner(entity, ner):
    entity.modify(set__ner=ner)

def add_category(entity, category):
    entity.modify(set__category=category)

# Triple operations
def save_triple(e1, pred, e2):
    triple = Triple(agent=e1, predicate=pred, patient=e2)

    query = Triple.objects(agent=e1, predicate=pred, patient=e2)
    if query.count() == 0:
        triple.save()
    else:
        triple = query.get()
    return triple

# Lexical entry operations
def save_lexEntry(docid, comment, text, parse_tree, template='', delex_type=''):
    lexEntry = Lex(docid=docid, comment=comment, text=text, template=template, parse_tree=parse_tree, delex_type=delex_type)
    response = lexEntry.save()
    if not response:
        raise NameError('LexEntry error: ', lexEntry.text)
    return lexEntry

def insert_template(lexEntry, template, delex_type='automatic'):
    response = lexEntry.modify(set__template=template)
    if not response:
        raise NameError('Insert template error: ', lexEntry.text)
    response = lexEntry.modify(set__delex_type=delex_type)
    if not response:
        raise NameError('Insert template error: ', lexEntry.text)
    return lexEntry

# Template operations
def save_template(category, triples, template, delex_type):
    size = Template.objects().count()

    template = Template(docid=size+1, category=category, triples=triples, template=template, delex_type=delex_type)
    template.save()
    return template

# Entry operations
def save_entry(docid, size, category, set):
    entry = Entry(docid=docid, size=size, category=category, set=set)

    query = Entry.objects(docid=docid, size=size, category=category, set=set)
    if query.count() == 0:
        entry.save()
    else:
        entry = query.get()
    return entry

def add_triple(entry, triple):
    query = Entry.objects(Q(id=entry.id) & Q(triples=triple))

    if query.count() == 0:
        entry.update(add_to_set__triples=[triple])

def add_lexEntry(entry, lexEntry):
    query = Entry.objects(Q(id=entry.id) & Q(texts=lexEntry))

    if query.count() == 0:
        entry.update(add_to_set__texts=[lexEntry])

# Reference operations
def save_reference(entity, syntax, text_status, sentence_status):
    if type(entity) == str or type(entity) == unicode:
        entity = Entity.objects(name=entity).first()
    reference = Reference(entity=entity, syntax=syntax, text_status=text_status, sentence_status=sentence_status)

    query = Reference.objects(entity=entity, syntax=syntax, text_status=text_status, sentence_status=sentence_status)

    if query.count() == 0:
        reference.save()
    else:
        reference = query.get()
    return reference

def add_refex(reference, reftype, refex, annotation='automatic'):
    size = len(reference.refexes)
    entry = Refex(docid=size+1, ref_type=reftype, refex=refex, annotation=annotation)

    reference.update(add_to_set__refexes=[entry])

# Referring expression
def save_refex(reftype, refex, annotation='automatic'):
    size = Refex.objects().count()
    entry = Refex(docid=size+1, ref_type=reftype, refex=refex, annotation=annotation)

    # query = Refex.objects(ref_type=reftype, refex=refex.strip(), annotation=annotation)
    # if query.count() == 0:
    #     entry.save()
    # else:
    #     entry = query.get()
    return entry

# Clean database
def clean():
    Entry.objects().delete()
    Triple.objects().delete()
    Reference.objects().delete()
    Lex.objects().delete()
    Entity.objects().delete()
    Predicate.objects().delete()

# Clean delex information
def clean_delex():
    Reference.objects().delete()
    Lex.objects().update(template='')
    Template.objects().delete()