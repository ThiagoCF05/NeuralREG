__author__='thiagocastroferreira'

from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn

class REGBERT(nn.Module):
    def __init__(self, nhead, layers, hidden_size, dropout=0.1, max_len=50, device='cuda'):
        super(REGBERT, self).__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.device = device
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        config = BertConfig.from_pretrained("bert-base-cased")
        config.is_decoder = True
        config.add_cross_attention = True
        self.seq2seq = BertModel.from_pretrained('bert-base-cased', config=config)
        self.lookup = self.seq2seq.get_input_embeddings()
        self.softmax = nn.LogSoftmax(dim=2)
    
    def encode(self, entities, contexts):
        entity_tok = self.tokenizer(entities, padding=True, return_tensors="pt").to(self.device)
        context_tok = self.tokenizer(contexts, padding=True, return_tensors="pt").to(self.device)

        # encode
        entity_enc = self.encoder(**entity_tok)['last_hidden_state']
        context_enc = self.encoder(**context_tok)['last_hidden_state']

        # get cls of entities encoding
        entity_cls = entity_enc[:, 0, :]

        # find mask ids in the text encodings
        mask_id = self.tokenizer.mask_token_id
        idxs = (context_tok['input_ids'] == mask_id).nonzero(as_tuple=False)
        text_masks = context_enc[idxs[:, 0], idxs[:, 1]]

        # sum cls with text masks
        encoded = text_masks + entity_cls
        return encoded.unsqueeze(1)


    def decode(self, encoded, refexes):
        refex_tok = self.tokenizer(refexes, padding=True, return_tensors="pt").to(self.device)
        seq_len = refex_tok['input_ids'].size()[-1]

        probs = []
        last_hidden_state, past_key_values = None, None
        for i in range(seq_len-1):
            inp = {
                'input_ids': refex_tok['input_ids'][:, i].unsqueeze(1),
                'token_type_ids': refex_tok['token_type_ids'][:, i].unsqueeze(1),
                'attention_mask': refex_tok['attention_mask'][:, i].unsqueeze(1)
            }
            decoded = self.seq2seq(**inp, encoder_hidden_states=encoded, past_key_values=past_key_values)
            last_hidden_state = decoded['last_hidden_state']
            past_key_values = decoded['past_key_values']

            logits = torch.matmul(last_hidden_state, lookup.weight.transpose(0, 1))
            logits = softmax(logits)
            probs.append(logits)

        return torch.cat(probs, 1), refex_tok['input_ids'][:, 1:]


    def generate(self, encoded):
        batch_size = encoded.size()[1]
        cls_id = torch.tensor([self.tokenizer.cls_token_id]).to(self.device)
        words = torch.tensor(batch_size * [cls_id]).unsqueeze(1).to(self.device)

        refexes = torch.zeros((self.max_len, batch_size)).to(self.device)
        last_hidden_state, past_key_values = None, None
        for i in range(self.max_len):
            inp = {
                'input_ids': words,
                'token_type_ids': torch.ones((batch_size, 1), dtype=torch.int),
                'attention_mask': torch.zeros((batch_size, 1), dtype=torch.int),
            }
            decoded = self.seq2seq(**inp, encoder_hidden_states=encoded, past_key_values=past_key_values)
            last_hidden_state = decoded['last_hidden_state']
            past_key_values = decoded['past_key_values']
            # softmax
            logits = self.softmax(torch.matmul(decoded, self.lookup.weight.transpose(0, 1))).squeeze(0)

            words = torch.argmax(logits, 2)
            for j, w in enumerate(words):
                refexes[i,j] = w

            unique = torch.unique(words).cpu()
            if unique.size()[0] == 1 and unique.eq(torch.tensor([self.tokenizer.sep_token_id])):
                break

        return refexes.transpose(0, 1)
    
    def forward(self, entities, contexts, refexes=None):
        encoded = self.encode(entities, contexts)

        if refexes:
            return self.decode(encoded, refexes)
        else:
            return self.generate(encoded)
        

def evaluate(model, devdata, batch_size, device):
    entities = []
    pre_contexts = []
    post_contexts = []
    refexes = []

    pred = []
    for batch_idx, inp in enumerate(devdata):
        context = inp['pre_context'] + ' ' + '[MASK]' + ' ' + inp['pos_context'].strip()
        entities.append(inp['entity'])
        contexts.append(context)
        refexes.append(inp['refex'])

        if (batch_idx+1) % batch_size == 0:
            # Predict
            output = model(entities, contexts)
            pred.extend(output.tolist())

            entities = []
            contexts = []
            refexes = []

    # Predict
    output = model(entities, contexts)
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
    contexts = []

    for batch_idx, inp in enumerate(traindata):
        context = inp['pre_context'] + ' ' + '[MASK]' + ' ' + inp['pos_context'].strip()
        entities.append(inp['entity'])
        contexts.append(context)
        refexes.append(inp['refex'])

        if (batch_idx+1) % batch_size == 0:
            # Init
            optimizer.zero_grad()

            # Predict
            probs, output = model(entities, contexts, refexes)

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
            contexts = []

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