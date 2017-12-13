__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 25/11/2017
Description:
    NeuralREG model concatenating the attention contexts from pre- and pos-contexts

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
"""

import dynet as dy
import load_data
import os

import pickle as p

class Attention():
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
        self.dec_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, (self.STATE_SIZE*4)+(self.EMBEDDINGS_SIZE*2), self.STATE_SIZE, self.model)
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
        # sentence = [self.input2int[c] for c in sentence]

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
        return att_weights, context


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
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.STATE_SIZE*4), last_output_embeddings, entity_embedding]))
        loss = []

        for word in output:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt_pre = w1dt_pre or w1_pre * h_pre
            w1dt_pos = w1dt_pos or w1_pos * h_pos

            attention_pre = self.attend(h_pre, s, w1dt_pre, self.attention_w2_pre, self.attention_v_pre)
            attention_pos = self.attend(h_pos, s, w1dt_pos, self.attention_w2_pos, self.attention_v_pos)

            vector = dy.concatenate([attention_pre, attention_pos, last_output_embeddings, entity_embedding])
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
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.STATE_SIZE*4), last_output_embeddings, entity_embedding]))

        pre_weights, out, pos_weights = [], [], []
        count_EOS = 0
        for i in range(self.config['GENERATION']):
            if count_EOS == 2: break
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt_pre = w1dt_pre or w1_pre * h_pre
            w1dt_pos = w1dt_pos or w1_pos * h_pos

            att_weights_pre, attention_pre = self.attend(h_pre, s, w1dt_pre, self.attention_w2_pre, self.attention_v_pre)
            att_weights_pos, attention_pos = self.attend(h_pos, s, w1dt_pos, self.attention_w2_pos, self.attention_v_pos)

            vector = dy.concatenate([attention_pre, attention_pos, last_output_embeddings, entity_embedding])
            s = s.add_input(vector)
            out_vector = w * s.output() + b
            probs = dy.softmax(out_vector).vec_value()
            next_word = probs.index(max(probs))
            last_output_embeddings = self.output_lookup[next_word]
            if self.int2output[next_word] == self.EOS:
                count_EOS += 1
                continue

            pre_weights.append(att_weights_pre.value())
            out.append(self.int2output[next_word])
            pos_weights.append(att_weights_pos.value())

        return pre_weights, out, pos_weights


    def extract(self, fin, fout):
        self.model.populate(fin)
        results = []

        dy.renew_cg()
        for i, testinst in enumerate(self.testset['refex']):
            pre_context = self.testset['pre_context'][i]
            pos_context = self.testset['pos_context'][i]
            # refex = ' '.join(testset['refex'][i]).replace('eos', '').strip()
            entity = self.testset['entity'][i]

            pre_weights, output, pos_weights = self.generate(pre_context, pos_context, entity)

            if i % 40:
                dy.renew_cg()

            result = {
                'pre_context':pre_context,
                'pre_weights':pre_weights,
                'entity':entity,
                'output':output,
                'pos_context':pos_context,
                'pos_weights':pos_weights
            }
            results.append(result)
        p.dump(results, open(fout, 'wb'), protocol=p.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    configs = [
        # {'LSTM_NUM_OF_LAYERS':1, 'EMBEDDINGS_SIZE':300, 'STATE_SIZE':512, 'ATTENTION_SIZE':512, 'DROPOUT':0.2, 'CHARACTER':False, 'GENERATION':30, 'BEAM_SIZE':1},
        # {'LSTM_NUM_OF_LAYERS':1, 'EMBEDDINGS_SIZE':300, 'STATE_SIZE':512, 'ATTENTION_SIZE':512, 'DROPOUT':0.3, 'CHARACTER':False, 'GENERATION':30, 'BEAM_SIZE':1},
        # {'LSTM_NUM_OF_LAYERS':1, 'EMBEDDINGS_SIZE':300, 'STATE_SIZE':512, 'ATTENTION_SIZE':512, 'DROPOUT':0.2, 'CHARACTER':False, 'GENERATION':30, 'BEAM_SIZE':5},
        {'LSTM_NUM_OF_LAYERS':1, 'EMBEDDINGS_SIZE':300, 'STATE_SIZE':512, 'ATTENTION_SIZE':512, 'DROPOUT':0.3, 'CHARACTER':False, 'GENERATION':30, 'BEAM_SIZE':5},
    ]

    fdir = 'data/att'

    for config in configs:
        h = Attention(config)

        fmodels = os.path.join(fdir, 'models')
        fname = 'best_' + \
                str(config['LSTM_NUM_OF_LAYERS']) + '_' + \
                str(config['EMBEDDINGS_SIZE']) + '_' + \
                str(config['STATE_SIZE']) + '_' + \
                str(config['ATTENTION_SIZE']) + '_' + \
                str(config['DROPOUT']).split('.')[1] + '_' + \
                str(config['CHARACTER']) + '_' + \
                str(config['BEAM_SIZE'])
        fin = os.path.join(fmodels, fname)

        fresults = os.path.join(fdir, 'results')
        fname = 'test_attweights_' + \
                str(config['LSTM_NUM_OF_LAYERS']) + '_' + \
                str(config['EMBEDDINGS_SIZE']) + '_' + \
                str(config['STATE_SIZE']) + '_' + \
                str(config['ATTENTION_SIZE']) + '_' + \
                str(config['DROPOUT']).split('.')[1] + '_' + \
                str(config['CHARACTER']) + '_' + \
                str(config['BEAM_SIZE'])
        fout = os.path.join(fresults, fname)
        h.extract(fin, fout)
