__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 25/11/2017
Description:
    NeuralREG model with a hierarchical attention to combine the attention contexts from pre- and pos-
    contexts

    Based on https://github.com/clab/dynet/blob/master/examples/sequence-to-sequence/attention.py

    HierAttention()
        :param config
        LSTM_NUM_OF_LAYERS: number of LSTM layers
        EMBEDDINGS_SIZE: embedding dimensions
        STATE_SIZE: dimension of decoding output
        ATTENTION_SIZE: dimension of attention representations
        DROPOUT: dropout probabilities on the encoder and decoder LSTMs
        CHARACTER: character- (True) or word-based decoder
        GENERATION: max output limit
        BEAM_SIZE: beam search size
"""

import dynet as dy
import load_data
import numpy as np
import os

class HierAttention():
    def __init__(self, config):
        self.config = config
        self.character = config['CHARACTER']

        self.EOS = "eos"
        self.vocab, self.trainset, self.devset, self.testset = load_data.run(self.character)

        self.int2input = list(self.vocab['input'])
        self.input2int = {c:i for i, c in enumerate(self.vocab['input'])}

        self.int2output = list(self.vocab['output'])
        self.output2int = {c:i for i, c in enumerate(self.vocab['output'])}

        self.init(config)


    def init(self, config):
        dy.renew_cg()

        self.INPUT_VOCAB_SIZE = len(self.vocab['input'])
        self.OUTPUT_VOCAB_SIZE = len(self.vocab['output'])

        self.LSTM_NUM_OF_LAYERS = config['LSTM_NUM_OF_LAYERS']
        self.EMBEDDINGS_SIZE = config['EMBEDDINGS_SIZE']
        self.STATE_SIZE = config['STATE_SIZE']
        self.ATTENTION_SIZE = config['ATTENTION_SIZE']
        self.DROPOUT = config['DROPOUT']
        self.BEAM = config['BEAM_SIZE']

        self.model = dy.Model()

        # ENCODERS
        self.encpre_fwd_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, self.model)
        self.encpre_bwd_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, self.model)
        self.encpre_fwd_lstm.set_dropout(self.DROPOUT)
        self.encpre_bwd_lstm.set_dropout(self.DROPOUT)

        self.encpos_fwd_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, self.model)
        self.encpos_bwd_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, self.model)
        self.encpos_fwd_lstm.set_dropout(self.DROPOUT)
        self.encpos_bwd_lstm.set_dropout(self.DROPOUT)

        # DECODER
        self.dec_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, (self.STATE_SIZE*2)+(self.EMBEDDINGS_SIZE*2), self.STATE_SIZE, self.model)
        self.dec_lstm.set_dropout(self.DROPOUT)

        # EMBEDDINGS
        self.input_lookup = self.model.add_lookup_parameters((self.INPUT_VOCAB_SIZE, self.EMBEDDINGS_SIZE))
        self.output_lookup = self.model.add_lookup_parameters((self.OUTPUT_VOCAB_SIZE, self.EMBEDDINGS_SIZE))

        # ATTENTION
        self.attention_w1_pre = self.model.add_parameters((self.ATTENTION_SIZE, self.STATE_SIZE * 2))
        self.attention_w2_pre = self.model.add_parameters((self.ATTENTION_SIZE, self.STATE_SIZE * self.LSTM_NUM_OF_LAYERS * 2))
        self.attention_v_pre = self.model.add_parameters((1, self.ATTENTION_SIZE))

        self.attention_w1_pos = self.model.add_parameters((self.ATTENTION_SIZE, self.STATE_SIZE * 2))
        self.attention_w2_pos = self.model.add_parameters((self.ATTENTION_SIZE, self.STATE_SIZE * self.LSTM_NUM_OF_LAYERS * 2))
        self.attention_v_pos = self.model.add_parameters((1, self.ATTENTION_SIZE))

        # HIERARCHICAL ATTENTION
        self.hier_w1_pre = self.model.add_parameters((self.ATTENTION_SIZE, self.STATE_SIZE * 2))
        self.hier_w_pre = self.model.add_parameters((self.STATE_SIZE * 2, self.STATE_SIZE * 2))

        self.hier_w1_pos = self.model.add_parameters((self.ATTENTION_SIZE, self.STATE_SIZE * 2))
        self.hier_w_pos = self.model.add_parameters((self.STATE_SIZE * 2, self.STATE_SIZE * 2))

        self.hier_w2 = self.model.add_parameters((self.ATTENTION_SIZE, self.STATE_SIZE * self.LSTM_NUM_OF_LAYERS * 2))
        self.hier_v = self.model.add_parameters((1, self.ATTENTION_SIZE))

        # SOFTMAX
        self.decoder_w = self.model.add_parameters((self.OUTPUT_VOCAB_SIZE, self.STATE_SIZE))
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
        bwd_vectors = self.run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

        return vectors


    def attend(self, h, state, w1dt, attention_w2, attention_v):
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
        context = h * att_weights
        return context


    def hier_attend(self, context_pre, context_pos, state):
        w2 = dy.parameter(self.hier_w2)
        v = dy.parameter(self.hier_v)

        w2dt = w2 * dy.concatenate(list(state.s()))

        # context_pre
        w1_pre = dy.parameter(self.hier_w1_pre)
        w1dt_pre = w1_pre * context_pre
        energy_pre = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt_pre, w2dt)))

        w_pre = dy.parameter(self.hier_w_pre)
        wdt_pre = w_pre * context_pre

        # context_pos
        w1_pos = dy.parameter(self.hier_w1_pos)
        w1dt_pos = w1_pos * context_pos
        energy_pos = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt_pos, w2dt)))

        w_pos = dy.parameter(self.hier_w_pos)
        wdt_pos = w_pos * context_pos

        beta = dy.softmax(dy.concatenate([energy_pre, energy_pos]))
        wdt = dy.concatenate_cols([wdt_pre, wdt_pos])
        context = wdt * beta
        return context


    def decode(self, pre_encoded, pos_encoded, output, entity):
        output = list(output)
        output = [self.output2int[c] for c in output]

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)

        w1_pre = dy.parameter(self.attention_w1_pre)
        h_pre = dy.concatenate_cols(pre_encoded)
        w1dt_pre = None

        w1_pos = dy.parameter(self.attention_w1_pos)
        h_pos = dy.concatenate_cols(pos_encoded)
        w1dt_pos = None

        last_output_embeddings = self.output_lookup[self.output2int[self.EOS]]
        entity_embedding = self.input_lookup[self.input2int[entity]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.STATE_SIZE*2), last_output_embeddings, entity_embedding]))
        loss = []

        for word in output:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt_pre = w1dt_pre or w1_pre * h_pre
            w1dt_pos = w1dt_pos or w1_pos * h_pos

            attention_pre = self.attend(h_pre, s, w1dt_pre, self.attention_w2_pre, self.attention_v_pre)
            attention_pos = self.attend(h_pos, s, w1dt_pos, self.attention_w2_pos, self.attention_v_pos)

            vector = dy.concatenate([self.hier_attend(attention_pre, attention_pos, s), last_output_embeddings, entity_embedding])
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

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)

        w1_pre = dy.parameter(self.attention_w1_pre)
        h_pre = dy.concatenate_cols(pre_encoded)
        w1dt_pre = None

        w1_pos = dy.parameter(self.attention_w1_pos)
        h_pos = dy.concatenate_cols(pos_encoded)
        w1dt_pos = None

        last_output_embeddings = self.output_lookup[self.output2int[self.EOS]]
        try:
            entity_embedding = self.input_lookup[self.input2int[entity]]
        except:
            entity_embedding = self.input_lookup[self.input2int[self.EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.STATE_SIZE*2), last_output_embeddings, entity_embedding]))

        out = []
        count_EOS = 0
        for i in range(self.config['GENERATION']):
            if count_EOS == 2: break
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt_pre = w1dt_pre or w1_pre * h_pre
            w1dt_pos = w1dt_pos or w1_pos * h_pos

            attention_pre = self.attend(h_pre, s, w1dt_pre, self.attention_w2_pre, self.attention_v_pre)
            attention_pos = self.attend(h_pos, s, w1dt_pos, self.attention_w2_pos, self.attention_v_pos)

            vector = dy.concatenate([self.hier_attend(attention_pre, attention_pos, s), last_output_embeddings, entity_embedding])
            s = s.add_input(vector)
            out_vector = w * s.output() + b
            probs = dy.softmax(out_vector).vec_value()
            next_word = probs.index(max(probs))
            last_output_embeddings = self.output_lookup[next_word]
            if self.int2output[next_word] == self.EOS:
                count_EOS += 1
                continue

            out.append(self.int2output[next_word])

        return out


    def beam_search(self, pre_context, pos_context, entity, beam):
        embedded = self.embed_sentence(pre_context)
        pre_encoded = self.encode_sentence(self.encpre_fwd_lstm, self.encpre_bwd_lstm, embedded)

        embedded = self.embed_sentence(pos_context)
        pos_encoded = self.encode_sentence(self.encpos_fwd_lstm, self.encpos_bwd_lstm, embedded)

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)

        w1_pre = dy.parameter(self.attention_w1_pre)
        h_pre = dy.concatenate_cols(pre_encoded)
        w1dt_pre = None

        w1_pos = dy.parameter(self.attention_w1_pos)
        h_pos = dy.concatenate_cols(pos_encoded)
        w1dt_pos = None

        entity_embedding = self.input_lookup[self.input2int[entity]]
        last_output_embeddings = self.output_lookup[self.output2int[self.EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.STATE_SIZE*2), last_output_embeddings, entity_embedding]))
        candidates = [{'sentence':[self.EOS], 'prob':0.0, 'count_EOS':0, 's':s}]
        outputs = []

        i = 0
        while i < self.config['GENERATION'] and len(outputs) < beam:
            new_candidates = []
            for candidate in candidates:
                if candidate['count_EOS'] == 2:
                    outputs.append(candidate)

                    if len(outputs) == beam: break
                else:
                    # w1dt can be computed and cached once for the entire decoding phase
                    w1dt_pre = w1dt_pre or w1_pre * h_pre
                    w1dt_pos = w1dt_pos or w1_pos * h_pos

                    attention_pre = self.attend(h_pre, candidate['s'], w1dt_pre, self.attention_w2_pre, self.attention_v_pre)
                    attention_pos = self.attend(h_pos, candidate['s'], w1dt_pos, self.attention_w2_pos, self.attention_v_pos)

                    last_output_embeddings = self.output_lookup[self.output2int[candidate['sentence'][-1]]]
                    vector = dy.concatenate([self.hier_attend(attention_pre, attention_pos, candidate['s']), last_output_embeddings, entity_embedding])
                    s = candidate['s'].add_input(vector)
                    out_vector = w * s.output() + b
                    probs = dy.softmax(out_vector).vec_value()
                    next_words = [{'prob':e, 'index':probs.index(e)} for e in sorted(probs, reverse=True)[:beam]]

                    for next_word in next_words:
                        word = self.int2output[next_word['index']]

                        new_candidate = {
                            'sentence': candidate['sentence'] + [word],
                            'prob': candidate['prob'] + np.log(next_word['prob']),
                            'count_EOS': candidate['count_EOS'],
                            's':s
                        }

                        if word == self.EOS:
                            new_candidate['count_EOS'] += 1

                        new_candidates.append(new_candidate)
            candidates = sorted(new_candidates, key=lambda x: x['prob'], reverse=True)[:beam]
            i += 1

        if len(outputs) == 0:
            outputs = candidates

        # Length Normalization
        alpha = 0.6
        for output in outputs:
            length = len(output['sentence'])
            lp_y = ((5.0 + length)**alpha) / ((5.0+1.0)**alpha)

            output['prob'] = output['prob'] / lp_y

        outputs = sorted(outputs, key=lambda x: x['prob'], reverse=True)
        return list(map(lambda x: x['sentence'], outputs))


    def get_loss(self, pre_context, pos_context, refex, entity):
        # dy.renew_cg()
        embedded = self.embed_sentence(pre_context)
        pre_encoded = self.encode_sentence(self.encpre_fwd_lstm, self.encpre_bwd_lstm, embedded)

        embedded = self.embed_sentence(pos_context)
        pos_encoded = self.encode_sentence(self.encpos_fwd_lstm, self.encpos_bwd_lstm, embedded)

        return self.decode(pre_encoded, pos_encoded, refex, entity)


    def write(self, fname, outputs):
        if not os.path.exists(fname):
            os.mkdir(fname)

        for i in range(self.BEAM):
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
            entity = self.devset['entity'][i]
            if self.BEAM == 1:
                outputs = [self.generate(pre_context, pos_context, entity)]
            else:
                outputs = self.beam_search(pre_context, pos_context, entity, self.BEAM)

            delimiter = ' '
            if self.character:
                delimiter = ''
            for j, output in enumerate(outputs):
                outputs[j] = delimiter.join(output).replace('eos', '').strip()
            refex = delimiter.join(self.devset['refex'][i]).replace('eos', '').strip()

            best_candidate = outputs[0]
            if refex == best_candidate:
                num += 1
            dem += 1

            if i < 20:
                print ("Refex: ", refex, "\t Output: ", best_candidate)
                print(10 * '-')

            results.append(outputs)

            if i % 40:
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
            entity = self.testset['entity'][i]

            if self.BEAM == 1:
                outputs = [self.generate(pre_context, pos_context, entity)]
            else:
                outputs = self.beam_search(pre_context, pos_context, entity, self.BEAM)
            delimiter = ' '
            if self.character:
                delimiter = ''
            for j, output in enumerate(outputs):
                outputs[j] = delimiter.join(output).replace('eos', '').strip()

            if i % 40:
                dy.renew_cg()

            results.append(outputs)
        self.write(fout, results)


    def train(self, fdir):
        # self.init(config)

        trainer = dy.AdadeltaTrainer(self.model)

        best_acc, repeat = 0.0, 0
        batch = 40
        for epoch in range(60):
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

                if len(losses) == batch:
                    loss = dy.esum(losses)
                    closs += loss.value()
                    loss.backward()
                    trainer.update()
                    dy.renew_cg()

                    print("Epoch: {0} \t Loss: {1}".format(epoch, (closs / batch)), end='       \r')
                    losses = []
                    closs = 0.0

            outputs, num, dem = self.validate()
            acc = round(float(num) / dem, 2)

            print("Dev acc: {0} \t Best acc: {1}".format(str(num/dem), best_acc))

            # Saving the model with best accuracy
            if best_acc == 0.0 or acc > best_acc:
                best_acc = acc

                fresults = os.path.join(fdir, 'results')
                if not os.path.exists(fresults):
                    os.mkdir(fresults)
                fname = 'dev_best_' + \
                        str(self.LSTM_NUM_OF_LAYERS) + '_' + \
                        str(self.EMBEDDINGS_SIZE) + '_' + \
                        str(self.STATE_SIZE) + '_' + \
                        str(self.ATTENTION_SIZE) + '_' + \
                        str(self.DROPOUT).split('.')[1] + '_' + \
                        str(self.character) + '_' + \
                        str(self.BEAM)
                self.write(os.path.join(fresults, fname), outputs)

                fmodels = os.path.join(fdir, 'models')
                if not os.path.exists(fmodels):
                    os.mkdir(fmodels)
                fname = 'best_' + \
                        str(self.LSTM_NUM_OF_LAYERS) + '_' + \
                        str(self.EMBEDDINGS_SIZE) + '_' + \
                        str(self.STATE_SIZE) + '_' + \
                        str(self.ATTENTION_SIZE) + '_' + \
                        str(self.DROPOUT).split('.')[1] + '_' + \
                        str(self.character) + '_' + \
                        str(self.BEAM)
                self.model.save(os.path.join(fmodels, fname))

                repeat = 0
            else:
                repeat += 1

            # In case the accuracy does not increase in 20 epochs, break the process
            if repeat == 20:
                break

        fmodels = os.path.join(fdir, 'models')
        fname = str(self.LSTM_NUM_OF_LAYERS) + '_' + \
                str(self.EMBEDDINGS_SIZE) + '_' + \
                str(self.STATE_SIZE) + '_' + \
                str(self.ATTENTION_SIZE) + '_' + \
                str(self.DROPOUT).split('.')[1] + '_' + \
                str(self.character) + '_' + \
                str(self.BEAM)
        self.model.save(os.path.join(fmodels, fname))


if __name__ == '__main__':
    configs = [
        {'LSTM_NUM_OF_LAYERS':1, 'EMBEDDINGS_SIZE':300, 'STATE_SIZE':512, 'ATTENTION_SIZE':512, 'DROPOUT':0.3, 'CHARACTER':False, 'GENERATION':30, 'BEAM_SIZE':1},
        {'LSTM_NUM_OF_LAYERS':1, 'EMBEDDINGS_SIZE':300, 'STATE_SIZE':512, 'ATTENTION_SIZE':512, 'DROPOUT':0.2, 'CHARACTER':False, 'GENERATION':30, 'BEAM_SIZE':1},
        {'LSTM_NUM_OF_LAYERS':1, 'EMBEDDINGS_SIZE':300, 'STATE_SIZE':512, 'ATTENTION_SIZE':512, 'DROPOUT':0.3, 'CHARACTER':False, 'GENERATION':30, 'BEAM_SIZE':5},
        {'LSTM_NUM_OF_LAYERS':1, 'EMBEDDINGS_SIZE':300, 'STATE_SIZE':512, 'ATTENTION_SIZE':512, 'DROPOUT':0.2, 'CHARACTER':False, 'GENERATION':30, 'BEAM_SIZE':5}
    ]

    fdir = 'data/hier'

    for config in configs:
        h = HierAttention(config)
        h.train(fdir)

        fmodels = os.path.join(fdir, 'models')
        fname = 'best_' + \
              str(config['LSTM_NUM_OF_LAYERS']) + '_' + \
              str(config['EMBEDDINGS_SIZE']) + '_' + \
              str(config['STATE_SIZE']) + '_' + \
              str(config['ATTENTION_SIZE']) + '_' + \
              str(config['DROPOUT']).split('.')[1] + '_' + \
              str(config['character']) + '_' + \
              str(config['BEAM'])
        fin = os.path.join(fmodels, fname)

        fresults = os.path.join(fdir, 'results')
        fname = 'test_best_' + \
                str(config['LSTM_NUM_OF_LAYERS']) + '_' + \
                str(config['EMBEDDINGS_SIZE']) + '_' + \
                str(config['STATE_SIZE']) + '_' + \
                str(config['ATTENTION_SIZE']) + '_' + \
                str(config['DROPOUT']).split('.')[1] + '_' + \
                str(config['character']) + '_' + \
                str(config['BEAM'])
        fout = os.path.join(fresults, fname)
        h.test(fin, fout)