__author__='thiagocastroferreira'

from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

class REGBERT(nn.Module):
    def __init__(self, nhead, layers, hidden_size, dropout=0.1, max_len=50, device='cuda'):
        super(REGBERT, self).__init__()
        self.layers = layers
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.device = device
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.encoder = BertModel.from_pretrained('bert-base-cased')

        self.lookup = self.encoder.get_input_embeddings()
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.softmax = nn.LogSoftmax(dim=2)
    
    def encode(self, entities, contexts):
        entity_tok = self.tokenizer(entity, padding=True, return_tensors="pt")
        context_tok = self.tokenizer(contexts, padding=True, return_tensors="pt")

        # encode
        entity_enc = self.model(**entity_tok)['last_hidden_state']
        context_enc = self.model(**context_tok)['last_hidden_state']

        # get cls of entities encoding
        entity_cls = entity_enc[:, 0, :]

        # find mask ids in the text encodings
        mask_id = tokenizer.mask_token_id
        idxs = (text_tok['input_ids'] == mask_id).nonzero(as_tuple=False)
        text_masks = text_enc[idxs[:, 0], idxs[:, 1]]

        # sum cls with text masks
        encoded = text_masks + entity_cls
        return encoded.unsqueeze(0)

    def decode(self, encoded, refexes):
        refex_tok = self.tokenizer(refexes, padding=True, return_tensors="pt")
        
        # masking
        sz = refex_tok['input_ids'].size()[1]
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float()

        # decoding
        tgt = self.lookup(refex_tok['input_ids']).transpose(0, 1)
        decoded = self.decoder(tgt=tgt, memory=encoded, tgt_mask=mask)
        # softmax
        logits = self.softmax(torch.matmul(o, self.lookup.weight.transpose(0, 1)))
        return logits.transpose(0, 1), refex_tok['input_ids']

    def generate(self, encoded):
        batch_size = encoded.size()[1]
        cls_id = torch.tensor([self.tokenizer.cls_token_id])
        tgt = self.lookup(cls_id).unsqueeze(0)
        tgt = torch.cat(batch_size * [tgt], 1)

        refexes = torch.zeros((self.max_len, batch_size)).to(self.device)
        for i in range(1, self.max_len):
            decoded = self.decoder(tgt=tgt, memory=encoded)
            # softmax
            logits = self.softmax(torch.matmul(o, self.lookup.weight.transpose(0, 1))).squeeze(0)

            words = torch.argmax(logits, 1)
            for j, w in enumerate(words):
                refexes[i,j] = w

            unique = torch.unique(words).cpu()
            if unique.size()[0] == 1 and unique.eq(torch.tensor([self.tokenizer.sep_token_id])):
                break

            # update last word based on predictions
            tgt = self.lookup(words).unsqueeze(0)
        return refexes

    
    def forward(self, entities, contexts, refexes=None):
        encoded = self.encode(entities, contexts)

        if refexes:
            return self.decode(encoded, refexes)
        else:
            return self.generate(encoded)
        