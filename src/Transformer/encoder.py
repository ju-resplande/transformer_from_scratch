import torch
import torch.nn as nn

from transformer import TransformerBlock

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embeeding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        batch_dim, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(batch_dim, seq_length).to(self.device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embeeding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out