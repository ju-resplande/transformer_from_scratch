import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).init()

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
            keys = values.reshape(batch_dim, keys_len, self.heads, self.head_dim)
            queries = values.reshape(batch_dim, query_len, self.heads, self.head_dim)

            # queries shape: (batch_dim, query_len, heads, key_len)
            # keys shape: (batch_dim, key_len, heads, key_len)
            # energy shape: (batch_dim, heads, query_len, key_len)
            energy = torch.eisum("bqhd,bk->bh", [queries, keys])

            if 