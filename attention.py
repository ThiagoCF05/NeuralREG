__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 25/11/2017
Description:
    NeuralREG+CAtt model concatenating the attention contexts from pre- and pos-contexts

    Based on https://github.com/clab/dynet/blob/master/examples/sequence-to-sequence/attention.py

    Attention()
        :param config
            LSTM_NUM_OF_LAYERS: number of LSTM layers
            EMBEDDINGS_SIZE: embedding dimensions
            STATE_SIZE: dimension of decoding output
            ATTENTION_SIZE: dimension of attention representations
            DROPOUT: dropout probabilities on the encoder and decoder LSTMs
            CHARACTER: character- (True) or word-based decoder
            GENERATION: max output limit
            BEAM_SIZE: beam search size

        train()
            :param fdir
                Directory to save best results and model

    PYTHON VERSION: 3

    DEPENDENCIES:
        Dynet: https://github.com/clab/dynet
        NumPy: http://www.numpy.org/

    UPDATE CONSTANTS:
        FDIR: directory to save results and trained models
"""

import dynet as dy
import load_data
import numpy as np
import os

class Config:
    def __init__(self, config):
        self.lstm_depth = config['LSTM_NUM_OF_LAYERS']
        self.embedding_dim = config['EMBEDDINGS_SIZE']
        self.state_dim = config['STATE_SIZE']
        self.attention_dim = config['ATTENTION_SIZE']
        self.dropout = config['DROPOUT']
        self.max_len = config['GENERATION']
        self.beam = config['BEAM_SIZE']
        self.batch = config['BATCH_SIZE']
        self.early_stop = config['EARLY_STOP']
        self.epochs = config['EPOCHS']

class Attention:
    def __init__(self, config):
        self.config = Config(config=config)
        self.character = False

        self.EOS = "eos"
        self.vocab, self.trainset, self.devset, self.testset = load_data.run(self.character)

        self.input_vocab = self.vocab['input']
        self.output_vocab = self.vocab['output']

        self.int2input = list(self.input_vocab)
        self.input2int = {c: i for i, c in enumerate(self.input_vocab)}

        self.int2output = list(self.output_vocab)
        self.output2int = {c: i for i, c in enumerate(self.output_vocab)}

        self.init()

    def init(self):
        dy.renew_cg()

        self.INPUT_VOCAB_SIZE = len(self.input_vocab)
        self.OUTPUT_VOCAB_SIZE = len(self.output_vocab)


        self.model = dy.Model()

        # ENCODERS
        self.encpre_fwd_lstm = dy.LSTMBuilder(self.config.lstm_depth, self.config.embedding_dim, self.config.state_dim, self.model)
        self.encpre_bwd_lstm = dy.LSTMBuilder(self.config.lstm_depth, self.config.embedding_dim, self.config.state_dim, self.model)
        self.encpre_fwd_lstm.set_dropout(self.config.dropout)
        self.encpre_bwd_lstm.set_dropout(self.config.dropout)

        self.encpos_fwd_lstm = dy.LSTMBuilder(self.config.lstm_depth, self.config.embedding_dim, self.config.state_dim, self.model)
        self.encpos_bwd_lstm = dy.LSTMBuilder(self.config.lstm_depth, self.config.embedding_dim, self.config.state_dim, self.model)
        self.encpos_fwd_lstm.set_dropout(self.config.dropout)
        self.encpos_bwd_lstm.set_dropout(self.config.dropout)

        self.encentity_fwd_lstm = dy.LSTMBuilder(self.config.lstm_depth, self.config.embedding_dim, self.config.state_dim, self.model)
        self.encentity_bwd_lstm = dy.LSTMBuilder(self.config.lstm_depth, self.config.embedding_dim, self.config.state_dim, self.model)
        self.encentity_fwd_lstm.set_dropout(self.config.dropout)
        self.encentity_bwd_lstm.set_dropout(self.config.dropout)

        # DECODER
        self.dec_lstm = dy.LSTMBuilder(self.config.lstm_depth, (self.config.state_dim * 6) + (self.config.embedding_dim), self.config.state_dim, self.model)
        self.dec_lstm.set_dropout(self.config.dropout)

        # EMBEDDINGS
        self.input_lookup = self.model.add_lookup_parameters((self.INPUT_VOCAB_SIZE, self.config.embedding_dim))
        self.output_lookup = self.model.add_lookup_parameters((self.OUTPUT_VOCAB_SIZE, self.config.embedding_dim))

        # ATTENTION
        self.attention_w1_pre = self.model.add_parameters((self.config.attention_dim, self.config.state_dim * 2))
        self.attention_w2_pre = self.model.add_parameters((self.config.attention_dim, self.config.state_dim * self.config.lstm_depth * 2))
        self.attention_v_pre = self.model.add_parameters((1, self.config.attention_dim))

        self.attention_w1_pos = self.model.add_parameters((self.config.attention_dim, self.config.state_dim * 2))
        self.attention_w2_pos = self.model.add_parameters((self.config.attention_dim, self.config.state_dim * self.config.lstm_depth * 2))
        self.attention_v_pos = self.model.add_parameters((1, self.config.attention_dim))

        self.attention_w1_entity = self.model.add_parameters((self.config.attention_dim, self.config.state_dim * 2))
        self.attention_w2_entity = self.model.add_parameters((self.config.attention_dim, self.config.state_dim * self.config.lstm_depth * 2))
        self.attention_v_entity = self.model.add_parameters((1, self.config.attention_dim))

        # COPY
        self.copy_x = self.model.add_parameters((1, self.config.embedding_dim))
        self.copy_decoder = self.model.add_parameters((1, self.config.state_dim * self.config.lstm_depth * 2))
        self.copy_context = self.model.add_parameters((1, self.config.state_dim * 4))
        self.copy_b = self.model.add_parameters((1))

        # self.copy_entity = entity embedding or -- second thoughts: entity separated from the context?
        self.copy_entity = self.model.add_parameters((1, self.config.state_dim * 2))

        # SOFTMAX
        self.decoder_w = self.model.add_parameters((self.OUTPUT_VOCAB_SIZE, self.config.state_dim))
        self.decoder_b = self.model.add_parameters((self.OUTPUT_VOCAB_SIZE))

    def embed_sentence(self, sentence):
        _sentence = list(sentence)
        sentence = []
        for w in _sentence:
            try:
                sentence.append(self.input2int[w])
            except:
                sentence.append(self.input2int[self.EOS])

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
        if enc_bwd_lstm:
            bwd_vectors = self.run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
            bwd_vectors = list(reversed(bwd_vectors))
            vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
            return vectors

        return fwd_vectors

    def attend(self, h, state, w1dt, attention_w2, attention_v):
        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = attention_w2 * dy.concatenate(list(state.s()))
        # att_weights: (seqlen,) row vector
        unnormalized = dy.transpose(attention_v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized)
        # context: (encoder_state)
        context = h * att_weights
        return context, att_weights

    def copy(self, x, decoder_state, entity):
        state = dy.concatenate(list(decoder_state.s()))
        return dy.logistic((self.copy_entity * entity) + (self.copy_decoder * state) + (self.copy_x * x) + self.copy_b)[0]

    def copy_with_context(self, x, decoder_state, context, entity):
        state = dy.concatenate(list(decoder_state.s()))
        return dy.logistic((self.copy_context * context) + (self.copy_entity * entity) + (self.copy_decoder * state) + (self.copy_x * x) + self.copy_b)[0]

    def decode(self, pre_encoded, pos_encoded, entity_encoded, output, entity):
        output = list(output)
        output = [self.output2int[c] for c in output]
        _entity = entity.replace('\"', '').replace('\'', '').replace(',', '').lower().split('_')

        h_pre = dy.concatenate_cols(pre_encoded)
        w1dt_pre = None

        h_pos = dy.concatenate_cols(pos_encoded)
        w1dt_pos = None

        h_entity = dy.concatenate_cols(entity_encoded)
        w1dt_entity = None

        last_output_embeddings = self.output_lookup[self.output2int[self.EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.config.state_dim * 6), last_output_embeddings]))
        loss = []

        for word in output:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt_pre = w1dt_pre or self.attention_w1_pre * h_pre
            w1dt_pos = w1dt_pos or self.attention_w1_pos * h_pos
            w1dt_entity = w1dt_entity or self.attention_w1_entity * h_entity

            attention_pre, _ = self.attend(h_pre, s, w1dt_pre, self.attention_w2_pre, self.attention_v_pre)
            attention_pos, _ = self.attend(h_pos, s, w1dt_pos, self.attention_w2_pos, self.attention_v_pos)
            attention_entity, att_weights = self.attend(h_entity, s, w1dt_entity, self.attention_w2_entity, self.attention_v_entity)

            p_gen = self.copy(last_output_embeddings, s, attention_entity)

            entity_prob = dy.scalarInput(0)
            lookup_word = self.int2output[word].lower()
            if lookup_word in entity:
                idx = entity.index(lookup_word)
                entity_prob = dy.pick(att_weights, idx)

            vector = dy.concatenate([attention_pre, attention_pos, attention_entity, last_output_embeddings])
            s = s.add_input(vector)
            out_vector = self.decoder_w * s.output() + self.decoder_b
            probs = dy.softmax(out_vector)
            context_prob = dy.pick(probs, word)
            last_output_embeddings = self.output_lookup[word]

            prob = dy.cmult(p_gen, context_prob) + dy.cmult(1 - p_gen, entity_prob)
            loss.append(-dy.log(prob))

        loss = dy.esum(loss)
        return loss

    def generate(self, pre_context, pos_context, entity):
        embedded = self.embed_sentence(pre_context)
        pre_encoded = self.encode_sentence(self.encpre_fwd_lstm, self.encpre_bwd_lstm, embedded)

        embedded = self.embed_sentence(pos_context)
        pos_encoded = self.encode_sentence(self.encpos_fwd_lstm, self.encpos_bwd_lstm, embedded)

        # _entity =
        embedded = self.embed_sentence(entity)
        entity_encoded = self.encode_sentence(self.encentity_fwd_lstm, self.encentity_bwd_lstm, embedded)

        h_pre = dy.concatenate_cols(pre_encoded)
        w1dt_pre = None

        h_pos = dy.concatenate_cols(pos_encoded)
        w1dt_pos = None

        h_entity = dy.concatenate_cols(entity_encoded)
        w1dt_entity = None

        last_output_embeddings = self.output_lookup[self.output2int[self.EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.config.state_dim * 6), last_output_embeddings]))

        out = []
        count_EOS = 0
        for i in range(self.config.max_len):
            if count_EOS == 2: break
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt_pre = w1dt_pre or self.attention_w1_pre * h_pre
            w1dt_pos = w1dt_pos or self.attention_w1_pos * h_pos
            w1dt_entity = w1dt_entity or self.attention_w1_entity * h_entity

            attention_pre, _ = self.attend(h_pre, s, w1dt_pre, self.attention_w2_pre,
                                                           self.attention_v_pre)
            attention_pos, _ = self.attend(h_pos, s, w1dt_pos, self.attention_w2_pos, self.attention_v_pos)
            attention_entity, att_weights = self.attend(h_entity, s, w1dt_entity, self.attention_w2_entity,
                                              self.attention_v_entity)

            p_gen = self.copy(last_output_embeddings, s, attention_entity)

            input_probs = dy.cmult(att_weights, 1 - p_gen).vec_value()
            input_prob_max = max(input_probs)
            input_next_word = input_probs.index(input_prob_max)

            vector = dy.concatenate([attention_pre, attention_pos, attention_entity, last_output_embeddings])
            s = s.add_input(vector)
            out_vector = self.decoder_w * s.output() + self.decoder_b
            probs = dy.cmult(dy.softmax(out_vector), p_gen).vec_value()

            for i, token in enumerate(entity):
                if token in self.output_vocab:
                    probs[self.output2int[token]] += input_probs[i]
            vocab_prob_max = max(probs)
            vocab_next_word = probs.index(vocab_prob_max)

            # If probability of input greater than the vocabulary
            if input_prob_max > vocab_prob_max:
                word = entity[input_next_word]
                try:
                    last_output_embeddings = self.output_lookup[self.output2int[word]]
                except:
                    last_output_embeddings = self.output_lookup[self.output2int[self.EOS]]
            else:
                last_output_embeddings = self.output_lookup[vocab_next_word]
                word = self.int2output[vocab_next_word]

            if word == self.EOS:
                count_EOS += 1
                continue

            out.append(word)

        return out

    def beam_search(self, pre_context, pos_context, entity):
        embedded = self.embed_sentence(pre_context)
        pre_encoded = self.encode_sentence(self.encpre_fwd_lstm, self.encpre_bwd_lstm, embedded)

        embedded = self.embed_sentence(pos_context)
        pos_encoded = self.encode_sentence(self.encpos_fwd_lstm, self.encpos_bwd_lstm, embedded)

        embedded = self.embed_sentence(entity)
        entity_encoded = self.encode_sentence(self.encentity_fwd_lstm, self.encentity_bwd_lstm, embedded)

        h_pre = dy.concatenate_cols(pre_encoded)
        w1dt_pre = None

        h_pos = dy.concatenate_cols(pos_encoded)
        w1dt_pos = None

        h_entity = dy.concatenate_cols(entity_encoded)
        w1dt_entity = None


        last_output_embeddings = self.output_lookup[self.output2int[self.EOS]]

        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.config.state_dim * 6), last_output_embeddings]))
        candidates = [{'sentence': [self.EOS], 'prob': 0.0, 'count_EOS': 0, 's': s}]
        outputs = []

        i = 0
        alpha = 0.6
        while i < self.config.max_len and len(outputs) < self.config.beam:
            new_candidates = []
            for candidate in candidates:
                if candidate['count_EOS'] == 2:
                    outputs.append(candidate)

                    if len(outputs) == self.config.beam: break
                else:
                    # w1dt can be computed and cached once for the entire decoding phase
                    w1dt_pre = w1dt_pre or self.attention_w1_pre * h_pre
                    w1dt_pos = w1dt_pos or self.attention_w1_pos * h_pos
                    w1dt_entity = w1dt_entity or self.attention_w1_entity * h_entity

                    attention_pre, _ = self.attend(h_pre, candidate['s'], w1dt_pre, self.attention_w2_pre, self.attention_v_pre)
                    attention_pos, _ = self.attend(h_pos, candidate['s'], w1dt_pos, self.attention_w2_pos, self.attention_v_pos)
                    attention_entity, att_weights = self.attend(h_entity, candidate['s'], w1dt_entity, self.attention_w2_entity, self.attention_v_entity)

                    try:
                        last_output_embeddings = self.output_lookup[self.output2int[candidate['sentence'][-1]]]
                    except:
                        last_output_embeddings = self.output_lookup[self.output2int[self.EOS]]

                    p_gen = self.copy(last_output_embeddings, s, attention_entity)

                    # INPUT WORDS
                    input_probs = dy.cmult(att_weights, 1 - p_gen).vec_value()
                    input_next_words = [{'prob': e, 'word': entity[input_probs.index(e)]}
                                        for e in sorted(input_probs, reverse=True)]


                    # VOCABULARY WORDS
                    vector = dy.concatenate([attention_pre, attention_pos, attention_entity, last_output_embeddings])
                    s = candidate['s'].add_input(vector)
                    out_vector = self.decoder_w * s.output() + self.decoder_b
                    probs = dy.cmult(dy.softmax(out_vector), p_gen).vec_value()

                    for i, token in enumerate(entity):
                        if token in self.output_vocab:
                            probs[self.output2int[token]] += input_probs[i]

                    vocab_next_words = [{'prob': e, 'word': self.int2output[probs.index(e)]}
                                        for e in sorted(probs, reverse=True)]

                    next_words = sorted(input_next_words + vocab_next_words, key=lambda x: x['prob'], reverse=True)[self.config.beam]
                    for next_word in next_words:
                        word = next_word['word']

                        new_candidate = {
                            'sentence': candidate['sentence'] + [word],
                            'prob': candidate['prob'] + np.log(next_word['prob']),
                            'count_EOS': candidate['count_EOS'],
                            's': s
                        }

                        # length normalization
                        length = len(new_candidate['sentence'])
                        lp_y = ((5.0 + length) ** alpha) / ((5.0 + 1.0) ** alpha)
                        new_candidate['prob'] = new_candidate['prob'] / lp_y

                        if word == self.EOS:
                            new_candidate['count_EOS'] += 1

                        new_candidates.append(new_candidate)
            candidates = sorted(new_candidates, key=lambda x: x['prob'], reverse=True)[:self.config.beam]
            i += 1

        if len(outputs) == 0:
            outputs = candidates

        # Length Normalization
        alpha = 0.6
        for output in outputs:
            length = len(output['sentence'])
            lp_y = ((5.0 + length) ** alpha) / ((5.0 + 1.0) ** alpha)

            output['prob'] = output['prob'] / lp_y

        outputs = sorted(outputs, key=lambda x: x['prob'], reverse=True)
        return list(map(lambda x: x['sentence'], outputs))

    def get_loss(self, pre_context, pos_context, refex, entity):
        embedded = self.embed_sentence(pre_context)
        pre_encoded = self.encode_sentence(self.encpre_fwd_lstm, self.encpre_bwd_lstm, embedded)

        embedded = self.embed_sentence(pos_context)
        pos_encoded = self.encode_sentence(self.encpos_fwd_lstm, self.encpos_bwd_lstm, embedded)

        embedded = self.embed_sentence(entity)
        entity_encoded = self.encode_sentence(self.encentity_fwd_lstm, self.encentity_bwd_lstm, embedded)

        return self.decode(pre_encoded, pos_encoded, entity_encoded, refex, entity)

    def write(self, fname, outputs):
        if not os.path.exists(fname):
            os.mkdir(fname)

        for i in range(self.config.beam):
            f = open(os.path.join(fname, str(i)), 'w')
            for output in outputs:
                if i < len(output):
                    f.write(output[i])
                f.write('\n')

            f.close()

    def validate(self):
        results = []
        num, dem = 0.0, 0.0
        for i, devinst in enumerate(self.devset['refex']):
            pre_context = self.devset['pre_context'][i]
            pos_context = self.devset['pos_context'][i]
            entity = self.devset['entity'][i].replace('\"', '').replace('\'', '').replace(',', '').split('_')
            if self.config.beam == 1:
                outputs = [self.generate(pre_context, pos_context, entity)]
            else:
                outputs = self.beam_search(pre_context, pos_context, entity)

            delimiter = ' '
            if self.character:
                delimiter = ''
            for j, output in enumerate(outputs):
                outputs[j] = delimiter.join(output).replace('eos', '').strip()
            refex = delimiter.join(self.devset['refex'][i]).replace('eos', '').strip()

            best_candidate = outputs[0]
            if refex.lower().strip() == best_candidate.lower().strip():
                num += 1
            dem += 1

            if i < 20:
                print("Refex: ", refex, "\t Output: ", best_candidate)
                print(10 * '-')

            results.append(outputs)

            if i % self.config.batch == 0:
                dy.renew_cg()

        return results, num, dem

    def test(self, fin, fout):
        self.model.populate(fin)
        results = []

        dy.renew_cg()
        for i, testinst in enumerate(self.testset['refex']):
            pre_context = self.testset['pre_context'][i]
            pos_context = self.testset['pos_context'][i]
            # refex = ' '.join(testset['refex'][i]).replace('eos', '').strip()
            entity = self.testset['entity'][i].replace('\"', '').replace('\'', '').replace(',', '').split('_')

            if self.config.beam == 1:
                outputs = [self.generate(pre_context, pos_context, entity)]
            else:
                outputs = self.beam_search(pre_context, pos_context, entity)
            delimiter = ' '
            if self.character:
                delimiter = ''
            for j, output in enumerate(outputs):
                outputs[j] = delimiter.join(output).replace('eos', '').strip()

            if i % self.config.batch == 0:
                dy.renew_cg()

            results.append(outputs)
        self.write(fout, results)

    def train(self):
        trainer = dy.AdadeltaTrainer(self.model)

        best_acc, repeat = 0.0, 0
        for epoch in range(self.config.epochs):
            dy.renew_cg()
            losses = []
            closs = 0.0
            size = len(self.trainset['refex'])
            for i, traininst in enumerate(self.trainset['refex']):
                pre_context = self.trainset['pre_context'][i]
                pos_context = self.trainset['pos_context'][i]
                refex = self.trainset['refex'][i]
                entity = self.trainset['entity'][i].replace('\"', '').replace('\'', '').replace(',', '').split('_')
                loss = self.get_loss(pre_context, pos_context, refex, entity)
                losses.append(loss)

                if len(losses) == self.config.batch:
                    loss = dy.esum(losses)
                    closs += loss.value()
                    loss.backward()
                    trainer.update()
                    dy.renew_cg()

                    print("Epoch: {0} \t Loss: {1} \t Progress: {2}".
                          format(epoch, round(closs / self.config.batch, 2), round(i / size, 2)), end='       \r')
                    losses = []
                    closs = 0.0

            outputs, num, dem = self.validate()
            acc = round(float(num) / dem, 2)

            print("Dev acc: {0} \t Best acc: {1}".format(str(num / dem), best_acc))

            # Saving the model with best accuracy
            if best_acc == 0.0 or acc > best_acc:
                best_acc = acc

                # fresults = os.path.join(fdir, 'results')
                # if not os.path.exists(fresults):
                #     os.mkdir(fresults)
                # fname = 'dev_best'
                # self.write(os.path.join(fresults, fname), outputs)
                #
                # fmodels = os.path.join(fdir, 'models')
                # if not os.path.exists(fmodels):
                #     os.mkdir(fmodels)
                # fname = 'best'
                # self.model.save(os.path.join(fmodels, fname))

                repeat = 0
            else:
                repeat += 1

            # In case the accuracy does not increase in 20 epochs, break the process
            if repeat == self.config.early_stop:
                break

        # fmodels = os.path.join(fdir, 'models')
        # fname = 'model'
        # self.model.save(os.path.join(fmodels, fname))


if __name__ == '__main__':
    config = {
        'LSTM_NUM_OF_LAYERS':1,
        'EMBEDDINGS_SIZE':128,
        'STATE_SIZE':256,
        'ATTENTION_SIZE':256,
        'DROPOUT':0.2,
        'GENERATION':30,
        'BEAM_SIZE':1,
        'BATCH_SIZE': 80,
        'EPOCHS': 60,
        'EARLY_STOP': 10
    }

    # DIRECTORY TO SAVE RESULTS AND TRAINED MODELS
    FDIR = 'data/v1.0/att'
    if not os.path.exists(FDIR):
        os.mkdir(FDIR)

    h = Attention(config)
    h.train()
