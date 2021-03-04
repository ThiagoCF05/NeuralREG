import torch
import torch.nn as nn
import torch.optim as optim

import json
import os

class NeuralREGCOLING(nn.Module):
  def __init__(self, layers, 
               emb_size, 
               hidden_size, 
               attention_size,
               w2id, id2w,
               dropout=0.1, max_len=50, device='cuda'):
    super(NeuralREGCOLING, self).__init__()
    self.layers = layers
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.max_len = max_len
    self.device = device
    self.eps = torch.tensor([1e-7]).to(device)
    
    self.bos, self.eos, self.pad = 'bos', 'eos', 'pad'
    self.id2w = id2w
    self.w2id = w2id

    self.lookup = nn.Embedding(len(w2id), emb_size)
    self.encoder = nn.ModuleDict({
        'entity': nn.LSTM(input_size=emb_size,
                          hidden_size=hidden_size,
                          num_layers=layers,
                          dropout=dropout,
                          bidirectional=True),
        'precontext': nn.LSTM(input_size=emb_size,
                              hidden_size=hidden_size,
                              num_layers=layers,
                              dropout=dropout,
                              bidirectional=True),
        'postcontext': nn.LSTM(input_size=emb_size,
                              hidden_size=hidden_size,
                              num_layers=layers,
                              dropout=dropout,
                              bidirectional=True)})

    self.attention = nn.ModuleDict({
        'entity_Wenc': nn.Linear(2*hidden_size, attention_size),
        'entity_Wdec': nn.Linear(hidden_size, attention_size),
        'entity_tanh': nn.Tanh(),
        'entity_v': nn.Linear(attention_size, 1),
        'entity_softmax': nn.Softmax(dim=1),
        'precontext_Wenc': nn.Linear(2*hidden_size, attention_size),
        'precontext_Wdec': nn.Linear(hidden_size, attention_size),
        'precontext_tanh': nn.Tanh(),
        'precontext_v': nn.Linear(attention_size, 1),
        'precontext_softmax': nn.Softmax(dim=1),
        'postcontext_Wenc': nn.Linear(2*hidden_size, attention_size),
        'postcontext_Wdec': nn.Linear(hidden_size, attention_size),
        'postcontext_tanh': nn.Tanh(),
        'postcontext_v': nn.Linear(attention_size, 1),
        'postcontext_softmax': nn.Softmax(dim=1)
    })

    self.copy_mech = nn.ModuleDict({
        'Wcontext': nn.Linear(2*hidden_size, 1),
        'Wdecoder': nn.Linear(hidden_size, 1),
        'Wword': nn.Linear(emb_size, 1),
        'sigmoid': nn.Sigmoid()
    })

    self.decoder = nn.LSTM(input_size=(6*hidden_size)+emb_size,
                  hidden_size=hidden_size,
                  num_layers=layers,
                  dropout=dropout)
    
    self.linear = nn.Linear(hidden_size, len(self.w2id))
    self.softmax = nn.LogSoftmax(dim=2)
  

  def encode(self, entities, pre_contexts, post_contexts):
    emb_pre = self.lookup(pre_contexts).to(self.device)
    emb_post = self.lookup(post_contexts).to(self.device)
    emb_entities = self.lookup(entities).to(self.device)

    encoded_ent, _ = self.encoder['entity'](emb_entities)
    encoded_pre, _ = self.encoder['precontext'](emb_pre)
    encoded_post, _ = self.encoder['postcontext'](emb_post)
    return encoded_ent, encoded_pre, encoded_post
  

  def attend(self, encoded, dec_state, Wenc, Wdec, tanh, v, softmax):
    # attention encoder math
    energy_enc = Wenc(encoded)
    # attention decoder math
    energy_dec = Wdec(dec_state)
    # energies
    energies = v(tanh(energy_enc + energy_dec)).transpose(0, 1)
    # attention weights
    alpha = softmax(energies)
    # context
    context = torch.bmm(alpha.transpose(1, 2), encoded.transpose(0, 1))
    context = context.transpose(0, 1)
    return context, alpha

  
  def copy(self, ent_context, dec_state, w_t, Wcontext, Wdecoder, Wword, sigmoid):
    mcontext = Wcontext(ent_context)
    mdecoder = Wdecoder(dec_state)
    mw_t = Wword(w_t)
    return sigmoid(mcontext + mdecoder + mw_t)


  def decode(self, entities, encoded_ent, encoded_pre, encoded_post, refexes):
    emb_refexes = self.lookup(refexes).to(self.device)
    batch_size = encoded_pre.size()[1]

    # instantiate entity context weights
    ent_Wenc = self.attention['entity_Wenc']
    ent_Wdec = self.attention['entity_Wdec']
    ent_tanh = self.attention['entity_tanh']
    ent_v = self.attention['entity_v']
    ent_softmax = self.attention['entity_softmax']
    
    # instantiate pre context weights
    pre_Wenc = self.attention['precontext_Wenc']
    pre_Wdec = self.attention['precontext_Wdec']
    pre_tanh = self.attention['precontext_tanh']
    pre_v = self.attention['precontext_v']
    pre_softmax = self.attention['precontext_softmax']

    # instantiate post context weights
    post_Wenc = self.attention['postcontext_Wenc']
    post_Wdec = self.attention['postcontext_Wdec']
    post_tanh = self.attention['postcontext_tanh']
    post_v = self.attention['postcontext_v']
    post_softmax = self.attention['postcontext_softmax']

    # instantiate copy weights
    Wcontext = self.copy_mech['Wcontext']
    Wdecoder = self.copy_mech['Wdecoder']
    Wword = self.copy_mech['Wword']
    sigmoid = self.copy_mech['sigmoid']

    seq_len = emb_refexes.size()[0] # target sequence size
    s_t = torch.zeros((self.layers, batch_size, self.hidden_size)).to(self.device) # initial decoder state
    c_t = torch.zeros((self.layers, batch_size, self.hidden_size)).to(self.device) # initial decoder cell state
    w_t = emb_refexes[0].unsqueeze(0) # initial word embeddings of the refex (<bos>)

    probs = []
    for i in range(1, seq_len):
      # ent_context attention math
      ent_context, attention_weights = self.attend(encoded_ent, s_t, 
                                                   ent_Wenc, ent_Wdec, 
                                                   ent_tanh, ent_v, ent_softmax)

      # pre_context attention math
      pre_context, _ = self.attend(encoded_pre, s_t, 
                                   pre_Wenc, pre_Wdec, 
                                   pre_tanh, pre_v, pre_softmax)

      # post context attention math
      post_context, _ = self.attend(encoded_post, s_t, 
                                    post_Wenc, post_Wdec, 
                                    post_tanh, post_v, post_softmax)
      
      # final attention context
      context = torch.cat([ent_context, pre_context, post_context, w_t], 2)

      # decoding step
      output, (s_t, c_t) = self.decoder(context, (s_t, c_t))
      context_probs = self.softmax(self.linear(output))

      # copy 
      p_gen = self.copy(ent_context, s_t, w_t, Wcontext, Wdecoder, Wword, sigmoid)
      
      # computing probability
      logits = torch.log(p_gen) + context_probs

      for batch_idx in range(entities.size()[1]):
        logits[0, batch_idx, entities[:, batch_idx]] += torch.log(1-p_gen[0, batch_idx, 0] + attention_weights[batch_idx, :, 0] + self.eps)

      probs.append(logits)

      # update last word based on teacher forcing
      w_t = emb_refexes[i].unsqueeze(0)
    return torch.cat(probs, 0), refexes[1:, :]


  def generate(self, entities, encoded_ent, encoded_pre, encoded_post):
    batch_size = encoded_pre.size()[1]
    
    # instantiate entity context weights
    ent_Wenc = self.attention['entity_Wenc']
    ent_Wdec = self.attention['entity_Wdec']
    ent_tanh = self.attention['entity_tanh']
    ent_v = self.attention['entity_v']
    ent_softmax = self.attention['entity_softmax']

    # instantiate pre context weights
    pre_Wenc = self.attention['precontext_Wenc']
    pre_Wdec = self.attention['precontext_Wdec']
    pre_tanh = self.attention['precontext_tanh']
    pre_v = self.attention['precontext_v']
    pre_softmax = self.attention['precontext_softmax']

    # instantiate post context weights
    post_Wenc = self.attention['postcontext_Wenc']
    post_Wdec = self.attention['postcontext_Wdec']
    post_tanh = self.attention['postcontext_tanh']
    post_v = self.attention['postcontext_v']
    post_softmax = self.attention['postcontext_softmax']

    # instantiate copy weights
    Wcontext = self.copy_mech['Wcontext']
    Wdecoder = self.copy_mech['Wdecoder']
    Wword = self.copy_mech['Wword']
    sigmoid = self.copy_mech['sigmoid']

    s_t = torch.zeros((self.layers, batch_size, self.hidden_size)).to(self.device) # initial decoder state
    c_t = torch.zeros((self.layers, batch_size, self.hidden_size)).to(self.device) # initial decoder cell state
    w_t = self.lookup(torch.tensor(batch_size * [self.w2id['bos']]).to(self.device)).unsqueeze(0).to(self.device) # initial word embeddings of the refex (<bos>)

    # refexes = torch.zeros((self.max_len, batch_size)).to(self.device)
    refexes = torch.full((self.max_len, batch_size), torch.scalar_tensor(self.w2id['eos'])).to(self.device)
    for i in range(1, self.max_len):
      # ent_context attention math
      ent_context, attention_weights = self.attend(encoded_ent, s_t, 
                                                   ent_Wenc, ent_Wdec, 
                                                   ent_tanh, ent_v, ent_softmax)

      # pre_context attention math
      pre_context, _ = self.attend(encoded_pre, s_t, 
                                   pre_Wenc, pre_Wdec, 
                                   pre_tanh, pre_v, pre_softmax)

      # post context attention math
      post_context, _ = self.attend(encoded_post, s_t, 
                                    post_Wenc, post_Wdec, 
                                    post_tanh, post_v, post_softmax)
      
      # final attention context
      context = torch.cat([ent_context, pre_context, post_context, w_t], 2)

      # decoding step
      output, (s_t, c_t) = self.decoder(context, (s_t, c_t))
      context_probs = self.softmax(self.linear(output))

      # copy 
      p_gen = self.copy(ent_context, s_t, w_t, Wcontext, Wdecoder, Wword, sigmoid)

      # computing probability
      logits = torch.log(p_gen) + context_probs
      for batch_idx in range(entities.size()[1]):
        logits[0, batch_idx, entities[:, batch_idx]] += torch.log(1-p_gen[0, batch_idx, 0] + attention_weights[batch_idx, :, 0] + self.eps)

      logits = logits.squeeze(0)

      words = torch.argmax(logits, 1)
      for j, w in enumerate(words):
        if self.w2id['<unk>'] == int(w):
          idx = torch.argmax(attention_weights[j, :, 0], 0)
          refexes[i,j] = entities[idx, j]
        else:
          refexes[i,j] = w
  
      # finish process in case all the refexes were generated
      unique = torch.unique(refexes[i, :].long()).cpu()
      if unique.size()[0] == 1 and unique.eq(torch.tensor([self.w2id['eos']])):
        break

      # update last word based on predictions
      w_t = self.lookup(refexes[i, :].long()).unsqueeze(0)
    return refexes


  def forward(self, entities, pre_contexts, post_contexts, refexes=None):
    encoded_ent, encoded_pre, encoded_post = self.encode(entities, pre_contexts, post_contexts)

    if refexes != None:
      probs = self.decode(entities, encoded_ent, encoded_pre, encoded_post, refexes)
      return probs#.transpose(0, 1)
    else:
      refexes = self.generate(entities, encoded_ent, encoded_pre, encoded_post)
      return refexes.transpose(0, 1)


def seq2idx(sequence, w2id, unk='<unk>'):
  result = []
  for w in sequence:
    try:
      result.append(w2id[w])
    except:
      result.append(w2id[unk])
  return torch.tensor(result)


def evaluate(model, devdata, batch_size, device):
  entities = []
  pre_contexts = []
  post_contexts = []
  refexes = []

  pred = []
  for batch_idx, inp in enumerate(devdata):
    entities.append(seq2idx(inp['entity'].split('_'), model.w2id))
    pre_contexts.append(seq2idx(['bos'] + inp['pre_context'], model.w2id))
    post_contexts.append(seq2idx(inp['pos_context'] + ['eos'], model.w2id))
    refexes.append(seq2idx(['bos'] + inp['refex'] + ['eos'], model.w2id))

    if (batch_idx+1) % batch_size == 0:
      # Padded sequence to device
      pre_contexts = torch.nn.utils.rnn.pad_sequence(pre_contexts, padding_value=model.w2id['pad']).to(device)
      post_contexts = torch.nn.utils.rnn.pad_sequence(post_contexts, padding_value=model.w2id['eos']).to(device)
      entities = torch.nn.utils.rnn.pad_sequence(entities, padding_value=model.w2id['eos']).to(device)
      refexes = torch.nn.utils.rnn.pad_sequence(refexes, padding_value=model.w2id['eos']).to(device)

      # Predict
      output = model(entities, pre_contexts, post_contexts)
      pred.extend(output.tolist())

      entities = []
      pre_contexts = []
      post_contexts = []
      refexes = []

  pre_contexts = torch.nn.utils.rnn.pad_sequence(pre_contexts, padding_value=model.w2id['pad']).to(device)
  post_contexts = torch.nn.utils.rnn.pad_sequence(post_contexts, padding_value=model.w2id['eos']).to(device)
  entities = torch.nn.utils.rnn.pad_sequence(entities, padding_value=model.w2id['eos']).to(device)
  refexes = torch.nn.utils.rnn.pad_sequence(refexes, padding_value=model.w2id['eos']).to(device)

  # Predict
  output = model(entities, pre_contexts, post_contexts)
  pred.extend(output.tolist())

  results = []
  acc, num = 0.0, 0.0
  for i, snt_pred in enumerate(pred):
    out = [model.id2w[idx] for idx in snt_pred]
    out = []
    for idx in snt_pred[1:]:
      if model.id2w[idx] == 'eos':
        break
      out.append(model.id2w[idx])
    refex = devdata[i]['refex']
    results.append(' '.join(out))
    if i < 10:
      print(' '.join(out), refex)
    if ' '.join(out) == ' '.join(refex):
      acc += 1
  return results, acc / len(pred)

def train(model, traindata, devdata, criterion, optimizer, epochs, batch_size, device, early_stop=5):
  batch_status, max_acc, repeat = 1024, 0, 0
  model.train()
  for epoch in range(epochs):
    losses = []
    refexes = []
    entities = []
    pre_contexts = []
    post_contexts = []

    for batch_idx, inp in enumerate(traindata):
      entities.append(seq2idx(inp['entity'].split('_'), model.w2id))
      pre_contexts.append(seq2idx(['bos'] + inp['pre_context'], model.w2id))
      post_contexts.append(seq2idx(inp['pos_context'] + ['eos'], model.w2id))
      refexes.append(seq2idx(['bos'] + inp['refex'] + ['eos'], model.w2id))

      if (batch_idx+1) % batch_size == 0:
        # Init
        optimizer.zero_grad()

        # Padded sequence to device
        pre_contexts = torch.nn.utils.rnn.pad_sequence(pre_contexts, padding_value=model.w2id['pad']).to(device)
        post_contexts = torch.nn.utils.rnn.pad_sequence(post_contexts, padding_value=model.w2id['eos']).to(device)
        # entities = torch.tensor(entities).to(device)
        entities = torch.nn.utils.rnn.pad_sequence(entities, padding_value=model.w2id['eos']).to(device)
        refexes = torch.nn.utils.rnn.pad_sequence(refexes, padding_value=model.w2id['eos']).to(device)

        # Predict
        probs, output = model(entities, pre_contexts, post_contexts, refexes)

        # Calculate loss
        loss = 0
        for l in [criterion(probs[i], output[i]) for i in range(probs.size()[0])]:
            loss += l
        # loss /= probs.size()[0]
        losses.append(float(loss))

        # Backpropagation
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 2.0)
        optimizer.step()

        refexes = []
        entities = []
        pre_contexts = []
        post_contexts = []

      # Display
      if (batch_idx+1) % batch_status == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTotal Loss: {:.6f}'.format(
          epoch, batch_idx+1, len(traindata),
          100. * batch_idx / len(traindata), float(loss), round(sum(losses) / len(losses), 5)))
    
    print()
    _, acc = evaluate(model, devdata, batch_size, 'cuda')
    print('Accuracy: ', acc)
    if acc > max_acc:
      max_acc = acc
      repeat = 0
      torch.save(model.state_dict(), 'neuralreg.pt')
    else:
      repeat += 1

    if repeat == early_stop:
      break

if __name__ == '__main__':
  path = 'data/v1.5'

  trainset = json.load(open(os.path.join(path, 'train.json')))
  devset = json.load(open(os.path.join(path, 'dev.json')))
  testset = json.load(open(os.path.join(path, 'test.json')))
  vocab = json.load(open(os.path.join(path, 'vocab.json')))
  new_vocab = json.load(open(os.path.join(path, 'new_vocab.json')))

  w2id = new_vocab['token2int']
  id2w = new_vocab['int2token']

  vocab = new_vocab['vocab']

  id2w = {i:c for i, c in enumerate(vocab)}
  w2id = {c:i for i, c in enumerate(vocab)}

  emb_size, hidden_size, attention_size, layers, dropout = 128, 256, 256, 1, 0.2
  model = NeuralREGCOLING(layers, emb_size, hidden_size, \
      attention_size, w2id, id2w, dropout=dropout, max_len=60)

  optimizer = optim.Adadelta(model.parameters())
  criterion = nn.NLLLoss()
  # model.load_state_dict(torch.load('neuralreg.pt'))
  model.to('cuda')
  # model.eval()
  # pass
  train(model, trainset, devset, criterion, optimizer, 32, 64, 'cuda')