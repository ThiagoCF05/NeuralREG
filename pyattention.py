__author__='thiagocastroferreira'

import load_data
import torch
import torch.nn as nn

class NeuralREG(nn.Module):
  def __init__(self, layers, 
               emb_size, 
               hidden_size, 
               attention_size,
               w2id, id2w,
               dropout=0.1, max_len=50, device='cuda'):
    super(NeuralREG, self).__init__()
    self.layers = layers
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.max_len = max_len
    self.device = device
    
    self.bos, self.eos, self.pad = 'bos', 'eos', 'pad'
    self.id2w = id2w
    self.w2id = w2id

    self.lookup = nn.Embedding(len(w2id), emb_size)
    self.encoder = nn.ModuleDict({
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

    self.decoder = nn.LSTM(input_size=(4*hidden_size)+(2*emb_size),
                  hidden_size=hidden_size,
                  num_layers=layers,
                  dropout=dropout)
    
    self.linear = nn.Linear(hidden_size, len(self.w2id))
    self.softmax = nn.LogSoftmax(dim=2)
  

  def encode(self, entities, pre_contexts, post_contexts):
    emb_pre = self.lookup(pre_contexts).to(self.device)
    emb_post = self.lookup(post_contexts).to(self.device)
    emb_entities = self.lookup(entities).to(self.device)

    encoded_pre, _ = self.encoder['precontext'](emb_pre)
    encoded_post, _ = self.encoder['postcontext'](emb_post)
    return emb_entities, encoded_pre, encoded_post
  

  def decode(self, emb_entities, encoded_pre, encoded_post, refexes):
    emb_refexes = self.lookup(refexes).to(self.device)
    batch_size = encoded_pre.size()[1]

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

    seq_len = emb_refexes.size()[0] # target sequence size
    s_t = torch.zeros((self.layers, batch_size, self.hidden_size)).to(self.device) # initial decoder state
    c_t = torch.zeros((self.layers, batch_size, self.hidden_size)).to(self.device) # initial decoder cell state
    w_t = emb_refexes[0].unsqueeze(0) # initial word embeddings of the refex (<bos>)
    emb_entities = emb_entities.unsqueeze(0) # entity embeddings

    # attention encoder math
    pre_energy_enc = pre_Wenc(encoded_pre)
    post_energy_enc = post_Wenc(encoded_post)

    probs = []
    for i in range(1, seq_len):
      # pre_context attention math
      pre_energy_dec = pre_Wdec(s_t)
      pre_energies = pre_v(pre_tanh(pre_energy_enc + pre_energy_dec)).transpose(0, 1)
      pre_alpha = pre_softmax(pre_energies)
      pre_context = torch.bmm(pre_alpha.transpose(1, 2), encoded_pre.transpose(0, 1))
      pre_context = pre_context.transpose(0, 1)

      # post context attention math
      post_energy_dec = post_Wdec(s_t)
      post_energies = post_v(post_tanh(post_energy_enc + post_energy_dec)).transpose(0, 1)
      post_alpha = post_softmax(post_energies)
      post_context = torch.bmm(post_alpha.transpose(1, 2), encoded_post.transpose(0, 1))
      post_context = post_context.transpose(0, 1)
      
      # final attention context
      context = torch.cat([pre_context, post_context, w_t, emb_entities], 2)

      # decoding step
      output, (s_t, c_t) = self.decoder(context, (s_t, c_t))
      logits = self.softmax(self.linear(output))
      probs.append(logits)

      # update last word based on teacher forcing
      w_t = emb_refexes[i].unsqueeze(0)
    return torch.cat(probs, 0), refexes[1:, :]


  def generate(self, emb_entities, encoded_pre, encoded_post):
    batch_size = encoded_pre.size()[1]

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

    s_t = torch.zeros((self.layers, batch_size, self.hidden_size)).to(self.device) # initial decoder state
    c_t = torch.zeros((self.layers, batch_size, self.hidden_size)).to(self.device) # initial decoder cell state
    w_t = self.lookup(torch.tensor(batch_size * [self.w2id['bos']]).to(self.device)).unsqueeze(0).to(self.device) # initial word embeddings of the refex (<bos>)
    emb_entities = emb_entities.unsqueeze(0) # entity embeddings

    # attention encoder math
    pre_energy_enc = pre_Wenc(encoded_pre)
    post_energy_enc = post_Wenc(encoded_post)

    refexes = torch.zeros((self.max_len, batch_size)).to(self.device)
    for i in range(1, self.max_len):
      # pre_context
      pre_energy_dec = pre_Wdec(s_t)
      pre_energies = pre_v(pre_tanh(pre_energy_enc + pre_energy_dec)).transpose(0, 1)
      pre_alpha = pre_softmax(pre_energies)
      pre_context = torch.bmm(pre_alpha.transpose(1, 2), encoded_pre.transpose(0, 1))
      pre_context = pre_context.transpose(0, 1)

      # post context
      post_energy_dec = post_Wdec(s_t)
      post_energies = post_v(post_tanh(post_energy_enc + post_energy_dec)).transpose(0, 1)
      post_alpha = post_softmax(post_energies)
      post_context = torch.bmm(post_alpha.transpose(1, 2), encoded_post.transpose(0, 1))
      post_context = post_context.transpose(0, 1)
      
      # final attention context
      context = torch.cat([pre_context, post_context, w_t, emb_entities], 2)

      # decoding step
      output, (s_t, c_t) = self.decoder(context, (s_t, c_t))
      logits = self.softmax(self.linear(output)).squeeze(0)

      words = torch.argmax(logits, 1)
      for j, w in enumerate(words):
        refexes[i,j] = w
  
      # finish process in case all the refexes were generated
      unique = torch.unique(words).cpu()
      if unique.size()[0] == 1 and unique.eq(torch.tensor([self.w2id['eos']])):
        break

      # update last word based on predictions
      w_t = self.lookup(words).unsqueeze(0)
    return refexes


  def forward(self, entities, pre_contexts, post_contexts, refexes=None):
    emb_entities, encoded_pre, encoded_post = self.encode(entities, pre_contexts, post_contexts)

    if refexes != None:
      probs = self.decode(emb_entities, encoded_pre, encoded_post, refexes)
      return probs#.transpose(0, 1)
    else:
      refexes = self.generate(emb_entities, encoded_pre, encoded_post)
      return refexes.transpose(0, 1)

if __name__ == "__main__":
    pass