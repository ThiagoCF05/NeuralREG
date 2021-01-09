__author__='thiagocastroferreira'

import load_data
import torch
import torch.nn as nn
from torch import optim

from pyattention import NeuralREG

def parse(data, w2id):
    r = []
    for i in range(len(data['entity'])):
        data['pre_context'][i][0] = 'bos'
    
        precontext = []
        for w in data['pre_context'][i]:
            try:
                precontext.append(w2id[w])
            except:
                precontext.append(w2id['<unk>'])
        
        poscontext = []
        for w in data['pos_context'][i]:
            try:
                poscontext.append(w2id[w])
            except:
                poscontext.append(w2id['<unk>'])

        data['refex'][i][0] = 'bos'    
        refex = []
        for w in data['refex'][i]:
            try:
                refex.append(w2id[w])
            except:
                refex.append(w2id['<unk>'])

        entity = w2id[data['entity'][i]] if data['entity'][i] in w2id else w2id['<unk>']

        r.append({
            'pre_context': torch.tensor(precontext),
            'pos_context': torch.tensor(poscontext),
            'entity': torch.tensor([entity]),
            'refex': torch.tensor(refex),
            'size': data['size'][i],
        })
    return r

def evaluate(model, devdata, batch_size, device):
    entities = []
    pre_contexts = []
    post_contexts = []
    refexes = []

    pred = []
    for batch_idx, inp in enumerate(devdata):
        entities.append(inp['entity'])
        pre_contexts.append(inp['pre_context'])
        post_contexts.append(inp['pos_context'])
        refexes.append(inp['refex'])

        if (batch_idx+1) % batch_size == 0:
            # Padded sequence to device
            pre_contexts = torch.nn.utils.rnn.pad_sequence(pre_contexts, padding_value=model.w2id['pad']).to(device)
            post_contexts = torch.nn.utils.rnn.pad_sequence(post_contexts, padding_value=model.w2id['eos']).to(device)
            entities = torch.tensor(entities).to(device)
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
    entities = torch.tensor(entities).to(device)
    refexes = torch.nn.utils.rnn.pad_sequence(refexes, padding_value=model.w2id['eos']).to(device)

    # Predict
    output = model(entities, pre_contexts, post_contexts)
    pred.extend(output.tolist())

    acc = 0.0
    for i, snt_pred in enumerate(pred):
        out = [id2w[idx] for idx in snt_pred]
        out = []
        for idx in snt_pred[1:]:
            if model.id2w[idx] == 'eos':
                break
            out.append(model.id2w[idx])
        refex = [model.id2w[idx] for idx in devset[i]['refex'].tolist()]
        if ' '.join(out) == ' '.join(refex[1:-1]):
            acc += 1
    return acc / len(pred)

def train(model, traindata, criterion, optimizer, epochs, batch_size, device, early_stop=5):
    batch_status, min_loss, repeat = 2048, 100, 0
    model.train()
    for epoch in range(epochs):
        losses = []
        refexes = []
        entities = []
        pre_contexts = []
        post_contexts = []

        for batch_idx, inp in enumerate(traindata):
            entities.append(inp['entity'])
            pre_contexts.append(inp['pre_context'])
            post_contexts.append(inp['pos_context'])
            refexes.append(inp['refex'])

            if (batch_idx+1) % batch_size == 0:
                # Init
                optimizer.zero_grad()

                # Padded sequence to device
                pre_contexts = torch.nn.utils.rnn.pad_sequence(pre_contexts, padding_value=w2id['pad']).to(device)
                post_contexts = torch.nn.utils.rnn.pad_sequence(post_contexts, padding_value=w2id['eos']).to(device)
                entities = torch.tensor(entities).to(device)
                refexes = torch.nn.utils.rnn.pad_sequence(refexes, padding_value=w2id['eos']).to(device)

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
        acc = evaluate(model, devset, batch_size, 'cuda')
        print('Accuracy: ', acc)
        if sum(losses) / len(losses) < min_loss:
            min_loss = sum(losses) / len(losses)
            repeat = 0
        else:
            repeat += 1

        if repeat == early_stop:
          break

if __name__ == "__main__":
    vocab, trainset, devset, testset = load_data.run(False)
    vocab['input'] += ['<unk>', '<mask>', 'bos', 'pad']

    vocab = set(vocab['input']).union(vocab['output'])
    id2w = {i:c for i, c in enumerate(vocab)}
    w2id = {c:i for i, c in enumerate(vocab)}

    trainset = parse(trainset, w2id)
    devset = parse(devset, w2id)
    testset = parse(testset, w2id)

    emb_size, hidden_size, attention_size, layers, dropout = 128, 256, 256, 1, 0.2
    model = NeuralREG(layers, emb_size, hidden_size, \
        attention_size, w2id, id2w, dropout=dropout, max_len=50)

    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    model.to('cuda')
    train(model, trainset, criterion, optimizer, 30, 80, 'cuda')
    
