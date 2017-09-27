import dynet as dy

def load_data():
    with open('data/input_vocab.txt') as f:
        input_vocab = f.read().split('\n')

    with open('data/output_vocab.txt') as f:
        output_vocab = f.read().split('\n')
    vocab = {'input':input_vocab, 'output':output_vocab}

    with open('data/train/pre_context.txt') as f:
        pre_context = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/train/pos_context.txt') as f:
        pos_context = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/train/entity.txt') as f:
        entity = f.read().split('\n')

    with open('data/train/refex.txt') as f:
        refex = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/train/size.txt') as f:
        size = f.read().split('\n')

    _train = {
        'pre_context':list(pre_context),
        'pos_context':list(pos_context),
        'entity':list(entity),
        'refex':list(refex),
        'size':list(size)
    }

    with open('data/dev/pre_context.txt') as f:
        pre_context = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/dev/pos_context.txt') as f:
        pos_context = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/dev/entity.txt') as f:
        entity = f.read().split('\n')

    with open('data/dev/refex.txt') as f:
        refex = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/dev/size.txt') as f:
        size = f.read().split('\n')

    _dev = {
        'pre_context':list(pre_context),
        'pos_context':list(pos_context),
        'entity':list(entity),
        'refex':list(refex),
        'size':list(size)
    }

    with open('data/test/pre_context.txt') as f:
        pre_context = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/test/pos_context.txt') as f:
        pos_context = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/test/entity.txt') as f:
        entity = f.read().split('\n')

    with open('data/test/refex.txt') as f:
        refex = map(lambda x: x.split(), f.read().split('\n'))

    with open('data/test/size.txt') as f:
        size = f.read().split('\n')

    _test = {
        'pre_context':list(pre_context),
        'pos_context':list(pos_context),
        'entity':list(entity),
        'refex':list(refex),
        'size':list(size)
    }

    return vocab, _train, _dev, _test

EOS = "eos"
vocab, trainset, devset, testset = load_data()

int2input = list(vocab['input'])
input2int = {c:i for i, c in enumerate(vocab['input'])}

int2output = list(vocab['output'])
output2int = {c:i for i, c in enumerate(vocab['output'])}

INPUT_VOCAB_SIZE = len(vocab['input'])
OUTPUT_VOCAB_SIZE = len(vocab['output'])

LSTM_NUM_OF_LAYERS = 2
EMBEDDINGS_SIZE = 256
STATE_SIZE = 1024
ATTENTION_SIZE = 1024

model = dy.Model()

encpre_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
encpre_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
encpre_fwd_lstm.set_dropout(0.2)
encpre_bwd_lstm.set_dropout(0.2)

encpos_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
encpos_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
encpos_fwd_lstm.set_dropout(0.2)
encpos_bwd_lstm.set_dropout(0.2)

dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, (STATE_SIZE*4)+(EMBEDDINGS_SIZE*2), STATE_SIZE, model)
dec_lstm.set_dropout(0.2)

input_lookup = model.add_lookup_parameters((INPUT_VOCAB_SIZE, EMBEDDINGS_SIZE))
attention_w1 = model.add_parameters((ATTENTION_SIZE, STATE_SIZE*4))
attention_w2 = model.add_parameters((ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
attention_v = model.add_parameters((1, ATTENTION_SIZE))
decoder_w = model.add_parameters((OUTPUT_VOCAB_SIZE, STATE_SIZE))
decoder_b = model.add_parameters((OUTPUT_VOCAB_SIZE))
output_lookup = model.add_lookup_parameters((OUTPUT_VOCAB_SIZE, EMBEDDINGS_SIZE))

def embed_sentence(sentence):
    sentence = [EOS] + list(sentence) + [EOS]
    sentence = [input2int[c] for c in sentence]

    global input_lookup

    return [input_lookup[char] for char in sentence]


def run_lstm(init_state, input_vecs):
    s = init_state

    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode_sentence(enc_fwd_lstm, enc_bwd_lstm, sentence):
    sentence_rev = list(reversed(sentence))

    fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), sentence)
    bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    return vectors


def attend(input_mat, state, w1dt):
    global attention_w2
    global attention_v
    w2 = dy.parameter(attention_w2)
    v = dy.parameter(attention_v)

    # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
    # w1dt: (attdim x seqlen)
    # w2dt: (attdim x attdim)
    w2dt = w2*dy.concatenate(list(state.s()))
    # att_weights: (seqlen,) row vector
    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
    att_weights = dy.softmax(unnormalized)
    # context: (encoder_state)
    context = input_mat * att_weights
    return context


def decode(dec_lstm, vectors, output, entity):
    output = list(output)
    output = [output2int[c] for c in output]

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(vectors)
    w1dt = None

    last_output_embeddings = output_lookup[output2int[EOS]]
    entity_embedding = input_lookup[input2int[entity]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*4), last_output_embeddings, entity_embedding]))
    loss = []

    for word in output:
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings, entity_embedding])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)
        last_output_embeddings = output_lookup[word]
        loss.append(-dy.log(dy.pick(probs, word)))
    loss = dy.esum(loss)
    return loss


def generate(pre_context, pos_context, entity, encpre_fwd_lstm, encpre_bwd_lstm, encpos_fwd_lstm, encpos_bwd_lstm, dec_lstm):
    embedded = embed_sentence(pre_context)
    pre_encoded = encode_sentence(encpre_fwd_lstm, encpre_bwd_lstm, embedded)

    embedded = embed_sentence(pos_context)
    pos_encoded = encode_sentence(encpos_fwd_lstm, encpos_bwd_lstm, embedded)

    encoded = [dy.concatenate(list(p)) for p in zip(pre_encoded, pos_encoded)]

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None

    last_output_embeddings = output_lookup[input2int[EOS]]
    entity_embedding = input_lookup[input2int[entity]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*4), last_output_embeddings, entity_embedding]))

    out = ''
    count_EOS = 0
    for i in range(len(pre_context)*2):
        if count_EOS == 2: break
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings, entity_embedding])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_word = probs.index(max(probs))
        last_output_embeddings = output_lookup[next_word]
        if int2output[next_word] == EOS:
            count_EOS += 1
            continue

        out = out + int2output[next_word] + ' '
    return out.strip()


def get_loss(pre_context, pos_context, refex, entity, encpre_fwd_lstm, encpre_bwd_lstm, encpos_fwd_lstm, encpos_bwd_lstm, dec_lstm):
    # dy.renew_cg()
    embedded = embed_sentence(pre_context)
    pre_encoded = encode_sentence(encpre_fwd_lstm, encpre_bwd_lstm, embedded)

    embedded = embed_sentence(pos_context)
    pos_encoded = encode_sentence(encpos_fwd_lstm, encpos_bwd_lstm, embedded)

    encoded = [dy.concatenate(list(p)) for p in zip(pre_encoded, pos_encoded)]
    return decode(dec_lstm, encoded, refex, entity)


def train(model, trainset, devset):
    # trainer = dy.SimpleSGDTrainer(model)
    trainer = dy.AdadeltaTrainer(model)

    prev_acc, repeat = 0.0, 0
    for epoch in range(50):
        dy.renew_cg()
        losses = []
        closs = 0.0
        for i, traininst in enumerate(trainset['refex']):
            pre_context = trainset['pre_context'][i]
            pos_context = trainset['pos_context'][i]
            refex = trainset['refex'][i]
            entity = trainset['entity'][i]
            loss = get_loss(pre_context, pos_context, refex, entity, encpre_fwd_lstm, encpre_bwd_lstm, encpos_fwd_lstm, encpos_bwd_lstm, dec_lstm)
            losses.append(loss)

            if len(losses) == 50:
                loss = dy.esum(losses)
                closs += loss.value()
                loss.backward()
                trainer.update()
                dy.renew_cg()

                print("Epoch: {0} \t Loss: {1}".format(epoch, (closs / 50)), end='     \r')
                losses = []
                closs = 0.0

        num, dem = 0.0, 0.0
        for i, devinst in enumerate(devset['refex']):
            pre_context = devset['pre_context'][i]
            pos_context = devset['pos_context'][i]
            refex = ' '.join(devset['refex'][i]).replace('eos', '').strip()
            entity = devset['entity'][i]
            output = generate(pre_context, pos_context, entity, encpre_fwd_lstm, encpre_bwd_lstm, encpos_fwd_lstm, encpos_bwd_lstm, dec_lstm)
            output = output.replace('eos', '').strip()
            if refex == output:
                num += 1
            dem += 1

            if i < 20:
                print ("Refex: ", refex, "\t Output: ", output)
                print(10 * '-')

            if i % 100:
                dy.renew_cg()
        print("Dev: ", str(num/dem))

        if round(num/dem, 2) == prev_acc:
            repeat += 1
        if repeat == 20:
            break
        prev_acc = round(num/dem, 2)

    f = open('data/test/output.txt', 'w')
    for i, testinst in enumerate(testset['refex']):
        pre_context = testset['pre_context'][i]
        pos_context = testset['pos_context'][i]
        # refex = ' '.join(testset['refex'][i]).replace('eos', '').strip()
        entity = testset['entity'][i]

        output = generate(pre_context, pos_context, entity, encpre_fwd_lstm, encpre_bwd_lstm, encpos_fwd_lstm, encpos_bwd_lstm, dec_lstm)
        output = output.replace('eos', '').strip()

        if i % 100:
            dy.renew_cg()

        f.write(output)
        f.write('\n')
    f.close()
    model.save("data/tmp.model")

train(model, trainset, devset)