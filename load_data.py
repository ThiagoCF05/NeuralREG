
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
        'pre_context':list(pre_context),
        'pos_context':list(pos_context),
        'entity':list(entity),
        'refex':list(refex),
        'size':list(size)
    }

def run(character=False):
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
    fprecontext = 'data/train/pre_context.txt'
    fposcontext = 'data/train/pos_context.txt'
    fentity = 'data/train/entity.txt'
    frefex = 'data/train/refex.txt'
    fsize = 'data/train/size.txt'
    trainset = load(fprecontext, fposcontext, fentity, frefex, fsize, character)

    # DEVSET
    fprecontext = 'data/dev/pre_context.txt'
    fposcontext = 'data/dev/pos_context.txt'
    fentity = 'data/dev/entity.txt'
    frefex = 'data/dev/refex.txt'
    fsize = 'data/dev/size.txt'
    devset = load(fprecontext, fposcontext, fentity, frefex, fsize, character)

    # TESTSET
    fprecontext = 'data/test/pre_context.txt'
    fposcontext = 'data/test/pos_context.txt'
    fentity = 'data/test/entity.txt'
    frefex = 'data/test/refex.txt'
    fsize = 'data/test/size.txt'
    testset = load(fprecontext, fposcontext, fentity, frefex, fsize, character)

    return vocab, trainset, devset, testset