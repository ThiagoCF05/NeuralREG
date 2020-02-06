__author__ = 'thiagocastroferreira'

import json

"""
Author: Thiago Castro Ferreira
Date: 12/12/2017
Description:
    Script for loading the referring expressions collection.

    PYTHON VERSION: 2.7

    UPDATE CONSTANTS:
        VOCAB_PATH
        TRAIN_REFEX_PATH
        DEV_REFEX_PATH
        TEST_REFEX_PATH
"""

import os

# PATH FOR VOCABULARY
VOCAB_PATH = 'data/v1.0/'

# PATH FOR REFERRING EXPRESSION COLLECTIONS
TRAIN_REFEX_PATH = 'data/v1.0/train'
DEV_REFEX_PATH = 'data/v1.0/dev'
TEST_REFEX_PATH = 'data/v1.0/test'


def load(fpre_context, fpos_context, fentity, frefex, fsize, character):
    with open(fpre_context) as f:
        pre_context = map(lambda x: x.split(), f.read().split('\n'))

    with open(fpos_context) as f:
        pos_context = map(lambda x: x.split(), f.read().split('\n'))

    with open(fentity) as f:
        entity = f.read().split('\n')

    with open(frefex) as f:
        if character:
            refex = map(lambda x: ['eos'] + list(x.replace('eos', '').strip()) + ['eos'], f.read().split('\n'))
        else:
            refex = map(lambda x: x.split(), f.read().split('\n'))

    with open(fsize) as f:
        size = f.read().split('\n')

    return {
        'pre_context': list(pre_context),
        'pos_context': list(pos_context),
        'entity': list(entity),
        'refex': list(refex),
        'size': list(size)
    }


def run(character=False):
    # VOCABULARY
    with open(os.path.join(VOCAB_PATH, 'input_vocab.txt')) as f:
        input_vocab = f.read().split('\n')

    if character:
        with open(os.path.join(VOCAB_PATH, 'character_vocab.txt')) as f:
            output_vocab = f.read().split('\n')
    else:
        with open(os.path.join(VOCAB_PATH, 'output_vocab.txt')) as f:
            output_vocab = f.read().split('\n')
    vocab = {'input': input_vocab, 'output': output_vocab}

    # ENTITY INFORMATION
    entity_types = json.load(open(os.path.join(VOCAB_PATH, 'entity_types.json')))
    entity_gender = json.load(open(os.path.join(VOCAB_PATH, 'gender.json')))

    # TRAINSET
    fprecontext = os.path.join(TRAIN_REFEX_PATH, 'pre_context.txt')
    fposcontext = os.path.join(TRAIN_REFEX_PATH, 'pos_context.txt')
    fentity = os.path.join(TRAIN_REFEX_PATH, 'entity.txt')
    frefex = os.path.join(TRAIN_REFEX_PATH, 'refex.txt')
    fsize = os.path.join(TRAIN_REFEX_PATH, 'size.txt')
    trainset = load(fprecontext, fposcontext, fentity, frefex, fsize, character)

    # DEVSET
    fprecontext = os.path.join(DEV_REFEX_PATH, 'pre_context.txt')
    fposcontext = os.path.join(DEV_REFEX_PATH, 'pos_context.txt')
    fentity = os.path.join(DEV_REFEX_PATH, 'entity.txt')
    frefex = os.path.join(DEV_REFEX_PATH, 'refex.txt')
    fsize = os.path.join(DEV_REFEX_PATH, 'size.txt')
    devset = load(fprecontext, fposcontext, fentity, frefex, fsize, character)

    # TESTSET
    fprecontext = os.path.join(TEST_REFEX_PATH, 'pre_context.txt')
    fposcontext = os.path.join(TEST_REFEX_PATH, 'pos_context.txt')
    fentity = os.path.join(TEST_REFEX_PATH, 'entity.txt')
    frefex = os.path.join(TEST_REFEX_PATH, 'refex.txt')
    fsize = os.path.join(TEST_REFEX_PATH, 'size.txt')
    testset = load(fprecontext, fposcontext, fentity, frefex, fsize, character)

    return vocab, entity_types, entity_gender, trainset, devset, testset


def run_json(path):
    vocab = json.load(open(os.path.join(path, 'vocab.json')))
    trainset = json.load(open(os.path.join(path, 'train.json')))
    devset = json.load(open(os.path.join(path, 'dev.json')))
    testset = json.load(open(os.path.join(path, 'test.json')))

    # ENTITY INFORMATION
    entity_types = json.load(open(os.path.join(path, 'entity_types.json')))
    entity_gender = json.load(open(os.path.join(path, 'gender.json')))

    return vocab, entity_types, entity_gender, trainset, devset, testset
