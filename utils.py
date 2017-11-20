

def load_data(character=False):
    # VOCABULARY
    with open('data/input_vocab.txt') as f:
        input_vocab = f.read().split('\n')

    if character:
        with open('data/character_vocab.txt') as f:
            output_vocab = f.read().split('\n')
    else:
        with open('data/output_vocab.txt') as f:
            output_vocab = f.read().split('\n')
    vocab = {'input':input_vocab, 'output':output_vocab}

    # TRAINSET
    with open('data/train/pre_context.txt') as f:
        pre_context = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/train/pos_context.txt') as f:
        pos_context = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/train/entity.txt') as f:
        entity = f.read().split('\n')

    if character:
        with open('data/train/refex.txt') as f:
            refex = map(lambda x: ['eos'] + list(x.replace('eos', '').strip()) + ['eos'], f.read().split('\n'))
    else:
        with open('data/train/refex.txt') as f:
            refex = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/train/size.txt') as f:
        size = f.read().split('\n')

    trainset = {
        'pre_context':list(pre_context),
        'pos_context':list(pos_context),
        'entity':list(entity),
        'refex':list(refex),
        'size':list(size)
    }

    # DEVSET
    with open('data/dev/pre_context.txt') as f:
        pre_context = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/dev/pos_context.txt') as f:
        pos_context = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/dev/entity.txt') as f:
        entity = f.read().split('\n')

    if character:
        with open('data/train/refex.txt') as f:
            refex = map(lambda x: ['eos'] + list(x.replace('eos', '').strip()) + ['eos'], f.read().split('\n'))
    else:
        with open('data/train/refex.txt') as f:
            refex = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/dev/size.txt') as f:
        size = f.read().split('\n')

    devset = {
        'pre_context':list(pre_context),
        'pos_context':list(pos_context),
        'entity':list(entity),
        'refex':list(refex),
        'size':list(size)
    }

    # TESTSET
    with open('data/test/pre_context.txt') as f:
        pre_context = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/test/pos_context.txt') as f:
        pos_context = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/test/entity.txt') as f:
        entity = f.read().split('\n')

    if character:
        with open('data/train/refex.txt') as f:
            refex = map(lambda x: ['eos'] + list(x.replace('eos', '').strip()) + ['eos'], f.read().split('\n'))
    else:
        with open('data/train/refex.txt') as f:
            refex = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/test/size.txt') as f:
        size = f.read().split('\n')

    testset = {
        'pre_context':list(pre_context),
        'pos_context':list(pos_context),
        'entity':list(entity),
        'refex':list(refex),
        'size':list(size)
    }

    return vocab, trainset, devset, testset