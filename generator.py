import dynet as dy
import utils

class Generator():
    def __init__(self, configs):
        self.configs = configs

        self.EOS = "eos"
        self.vocab, self.trainset, self.devset, self.testset = utils.load_data()

        self.int2input = list(self.vocab['input'])
        self.input2int = {c:i for i, c in enumerate(self.vocab['input'])}

        self.int2output = list(self.vocab['output'])
        self.output2int = {c:i for i, c in enumerate(self.vocab['output'])}

        for config in configs:
            self.init(config)
            self.train(config)


    def init(self, config):
        dy.renew_cg()

        self.INPUT_VOCAB_SIZE = len(self.vocab['input'])
        self.OUTPUT_VOCAB_SIZE = len(self.vocab['output'])

        self.LSTM_NUM_OF_LAYERS = config['LSTM_NUM_OF_LAYERS']
        self.EMBEDDINGS_SIZE = config['EMBEDDINGS_SIZE']
        self.STATE_SIZE = config['STATE_SIZE']
        self.ATTENTION_SIZE = config['ATTENTION_SIZE']
        self.DROPOUT = config['DROPOUT']

        self.model = dy.Model()

        self.encpre_fwd_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, self.model)
        self.encpre_bwd_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, self.model)
        self.encpre_fwd_lstm.set_dropout(self.DROPOUT)
        self.encpre_bwd_lstm.set_dropout(self.DROPOUT)

        self.encpos_fwd_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, self.model)
        self.encpos_bwd_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, self.model)
        self.encpos_fwd_lstm.set_dropout(self.DROPOUT)
        self.encpos_bwd_lstm.set_dropout(self.DROPOUT)

        self.dec_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, (self.STATE_SIZE*4)+(self.EMBEDDINGS_SIZE*2), self.STATE_SIZE, self.model)
        self.dec_lstm.set_dropout(self.DROPOUT)

        self.input_lookup = self.model.add_lookup_parameters((self.INPUT_VOCAB_SIZE, self.EMBEDDINGS_SIZE))
        self.attention_w1 = self.model.add_parameters((self.ATTENTION_SIZE, self.STATE_SIZE*4))
        self.attention_w2 = self.model.add_parameters((self.ATTENTION_SIZE, self.STATE_SIZE*self.LSTM_NUM_OF_LAYERS*2))
        self.attention_v = self.model.add_parameters((1, self.ATTENTION_SIZE))
        self.decoder_w = self.model.add_parameters((self.OUTPUT_VOCAB_SIZE, self.STATE_SIZE))
        self.decoder_b = self.model.add_parameters((self.OUTPUT_VOCAB_SIZE))
        self.output_lookup = self.model.add_lookup_parameters((self.OUTPUT_VOCAB_SIZE, self.EMBEDDINGS_SIZE))


    def embed_sentence(self, sentence):
        sentence = list(sentence)
        sentence = [self.input2int[c] for c in sentence]

        return [self.input_lookup[char] for char in sentence]


    def run_lstm(self, init_state, input_vecs):
        s = init_state

        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors


    def encode_sentence(self, enc_fwd_lstm, enc_bwd_lstm, sentence):
        sentence_rev = list(reversed(sentence))

        fwd_vectors = self.run_lstm(enc_fwd_lstm.initial_state(), sentence)
        bwd_vectors = self.run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

        return vectors


    def attend(self, input_mat, state, w1dt):
        w2 = dy.parameter(self.attention_w2)
        v = dy.parameter(self.attention_v)

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


    def decode(self, vectors, output, entity):
        output = list(output)
        output = [self.output2int[c] for c in output]

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)
        w1 = dy.parameter(self.attention_w1)
        input_mat = dy.concatenate_cols(vectors)
        w1dt = None

        last_output_embeddings = self.output_lookup[self.output2int[self.EOS]]
        entity_embedding = self.input_lookup[self.input2int[entity]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.STATE_SIZE*4), last_output_embeddings, entity_embedding]))
        loss = []

        for word in output:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings, entity_embedding])
            s = s.add_input(vector)
            out_vector = w * s.output() + b
            probs = dy.softmax(out_vector)
            last_output_embeddings = self.output_lookup[word]
            loss.append(-dy.log(dy.pick(probs, word)))
        loss = dy.esum(loss)
        return loss


    def generate(self, pre_context, pos_context, entity):
        embedded = self.embed_sentence(pre_context)
        pre_encoded = self.encode_sentence(self.encpre_fwd_lstm, self.encpre_bwd_lstm, embedded)

        embedded = self.embed_sentence(pos_context)
        pos_encoded = self.encode_sentence(self.encpos_fwd_lstm, self.encpos_bwd_lstm, embedded)

        encoded = [dy.concatenate(list(p)) for p in zip(pre_encoded, pos_encoded)]

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)
        w1 = dy.parameter(self.attention_w1)
        input_mat = dy.concatenate_cols(encoded)
        w1dt = None

        last_output_embeddings = self.output_lookup[self.input2int[self.EOS]]
        entity_embedding = self.input_lookup[self.input2int[entity]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.STATE_SIZE*4), last_output_embeddings, entity_embedding]))

        out = ''
        count_EOS = 0
        for i in range(len(pre_context)*2):
            if count_EOS == 2: break
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings, entity_embedding])
            s = s.add_input(vector)
            out_vector = w * s.output() + b
            probs = dy.softmax(out_vector).vec_value()
            next_word = probs.index(max(probs))
            last_output_embeddings = self.output_lookup[next_word]
            if self.int2output[next_word] == self.EOS:
                count_EOS += 1
                continue

            out = out + self.int2output[next_word] + ' '
        return out.strip()


    def get_loss(self, pre_context, pos_context, refex, entity):
        # dy.renew_cg()
        embedded = self.embed_sentence(pre_context)
        pre_encoded = self.encode_sentence(self.encpre_fwd_lstm, self.encpre_bwd_lstm, embedded)

        embedded = self.embed_sentence(pos_context)
        pos_encoded = self.encode_sentence(self.encpos_fwd_lstm, self.encpos_bwd_lstm, embedded)

        encoded = [dy.concatenate(list(p)) for p in zip(pre_encoded, pos_encoded)]
        return self.decode(encoded, refex, entity)


    def validate(self, save=False):
        if save:
            fname = 'data/results/dev_' + str(self.LSTM_NUM_OF_LAYERS) + '_' + str(self.EMBEDDINGS_SIZE) + '_' + str(self.STATE_SIZE) + '_' + str(self.ATTENTION_SIZE) + '_' + str(self.DROPOUT).split('.')[1]
            f = open(fname, 'w')
        num, dem = 0.0, 0.0
        for i, devinst in enumerate(self.devset['refex']):
            pre_context = self.devset['pre_context'][i]
            pos_context = self.devset['pos_context'][i]
            refex = ' '.join(self.devset['refex'][i]).replace('eos', '').strip()
            entity = self.devset['entity'][i]
            output = self.generate(pre_context, pos_context, entity)
            output = output.replace('eos', '').strip()
            if refex == output:
                num += 1
            dem += 1

            if i < 20:
                print ("Refex: ", refex, "\t Output: ", output)
                print(10 * '-')

            if save:
                f.write(output)
                f.write('\n')


            if i % 40:
                dy.renew_cg()
        print("Dev: ", str(num/dem))

        if save:
            f.close()
        return num, dem


    def test(self):
        dy.renew_cg()
        fname = 'data/results/test_' + str(self.LSTM_NUM_OF_LAYERS) + '_' + str(self.EMBEDDINGS_SIZE) + '_' + str(self.STATE_SIZE) + '_' + str(self.ATTENTION_SIZE) + '_' + str(self.DROPOUT).split('.')[1]
        f = open(fname, 'w')
        for i, testinst in enumerate(self.testset['refex']):
            pre_context = self.testset['pre_context'][i]
            pos_context = self.testset['pos_context'][i]
            # refex = ' '.join(testset['refex'][i]).replace('eos', '').strip()
            entity = self.testset['entity'][i]

            output = self.generate(pre_context, pos_context, entity)
            output = output.replace('eos', '').strip()

            if i % 40:
                dy.renew_cg()

            f.write(output)
            f.write('\n')
        f.close()

    def train(self, config):
        self.init(config)

        # trainer = dy.SimpleSGDTrainer(model)
        trainer = dy.AdadeltaTrainer(self.model)

        prev_acc, repeat = 0.0, 0
        for epoch in range(50):
            dy.renew_cg()
            losses = []
            closs = 0.0
            for i, traininst in enumerate(self.trainset['refex']):
                pre_context = self.trainset['pre_context'][i]
                pos_context = self.trainset['pos_context'][i]
                refex = self.trainset['refex'][i]
                entity = self.trainset['entity'][i]
                loss = self.get_loss(pre_context, pos_context, refex, entity)
                losses.append(loss)

                if len(losses) == 40:
                    loss = dy.esum(losses)
                    closs += loss.value()
                    loss.backward()
                    trainer.update()
                    dy.renew_cg()

                    print("Epoch: {0} \t Loss: {1}".format(epoch, (closs / 40)), end='     \r')
                    losses = []
                    closs = 0.0

            num, dem = self.validate()

            if round(num/dem, 2) == prev_acc:
                repeat += 1
            if repeat == 20:
                break
            prev_acc = round(num/dem, 2)

        self.validate(True)
        self.test()
        fname = 'data/models/' + str(self.LSTM_NUM_OF_LAYERS) + '_' + str(self.EMBEDDINGS_SIZE) + '_' + str(self.STATE_SIZE) + '_' + str(self.ATTENTION_SIZE) + '_' + str(self.DROPOUT).split('.')[1]
        self.model.save(fname)


if __name__ == '__main__':
    configs = [
        {'LSTM_NUM_OF_LAYERS':1, 'EMBEDDINGS_SIZE':300, 'STATE_SIZE':1024, 'ATTENTION_SIZE':1024, 'DROPOUT':0.2},
        {'LSTM_NUM_OF_LAYERS':1, 'EMBEDDINGS_SIZE':300, 'STATE_SIZE':1024, 'ATTENTION_SIZE':1024, 'DROPOUT':0.3},
        {'LSTM_NUM_OF_LAYERS':1, 'EMBEDDINGS_SIZE':256, 'STATE_SIZE':1024, 'ATTENTION_SIZE':1024, 'DROPOUT':0.2},
        {'LSTM_NUM_OF_LAYERS':1, 'EMBEDDINGS_SIZE':256, 'STATE_SIZE':1024, 'ATTENTION_SIZE':1024, 'DROPOUT':0.3},
        {'LSTM_NUM_OF_LAYERS':2, 'EMBEDDINGS_SIZE':300, 'STATE_SIZE':1024, 'ATTENTION_SIZE':1024, 'DROPOUT':0.2},
        {'LSTM_NUM_OF_LAYERS':2, 'EMBEDDINGS_SIZE':300, 'STATE_SIZE':1024, 'ATTENTION_SIZE':1024, 'DROPOUT':0.3},
        {'LSTM_NUM_OF_LAYERS':2, 'EMBEDDINGS_SIZE':256, 'STATE_SIZE':1024, 'ATTENTION_SIZE':1024, 'DROPOUT':0.2},
        {'LSTM_NUM_OF_LAYERS':2, 'EMBEDDINGS_SIZE':256, 'STATE_SIZE':1024, 'ATTENTION_SIZE':1024, 'DROPOUT':0.3},
        {'LSTM_NUM_OF_LAYERS':3, 'EMBEDDINGS_SIZE':300, 'STATE_SIZE':1024, 'ATTENTION_SIZE':1024, 'DROPOUT':0.2},
        {'LSTM_NUM_OF_LAYERS':3, 'EMBEDDINGS_SIZE':300, 'STATE_SIZE':1024, 'ATTENTION_SIZE':1024, 'DROPOUT':0.3},
        {'LSTM_NUM_OF_LAYERS':3, 'EMBEDDINGS_SIZE':256, 'STATE_SIZE':1024, 'ATTENTION_SIZE':1024, 'DROPOUT':0.2},
        {'LSTM_NUM_OF_LAYERS':3, 'EMBEDDINGS_SIZE':256, 'STATE_SIZE':1024, 'ATTENTION_SIZE':1024, 'DROPOUT':0.3}
    ]

    Generator(configs)