import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads

        assert (self.embed_size % self.heads == 0), "Embed size needs to be div by heads"

        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, values, keys, query, mask):
        batch_dim = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads piece (embed_size) -> (heads, head_dim)
        values = values.reshape(batch_dim, value_len, self.heads, self.head_dim)
        keys = values.reshape(batch_dim, key_len, self.heads, self.head_dim)
        queries = values.reshape(batch_dim, query_len, self.heads, self.head_dim)

        # queries shape: (batch_dim, query_len, heads, heads_dim)
        # keys shape: (batch_dim, key_len, heads, heads_dim)
        # energy shape: (batch_dim, heads, query_len, key_len)
        energy = torch.einsum("bqhd,bknd->bhqk", [queries, keys])

        if mask == None:
            # if the element of the mask equals 0
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        #Softmax normalized in key_len (input/output)
        attention = torch.softmax(energy / (self.embed_size) ** (1/2), dim=3)

        # attention shape: (batch_dim, heads, query_len, key_len)
        # values shape: (batch_dim, value_len, heads, heads_dim)
        # (batch_dim, query_len, heads, head_dim)
        out = torch.einsum("bhql,blhd->bqhd", [attention, values])
        out = out.reshape(batch_dim, query_len, self.embed_size)

        out = self.fc(out)

        return out