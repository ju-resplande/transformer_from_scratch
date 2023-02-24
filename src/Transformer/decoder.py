import torch
import torch.nn as nn

from self_attention import SelfAttention
from transformer import TransformerBlock

class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        forward_expansion,
        dropout,
        device
    ):

    super(DecoderBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm = nn.LayerNorm(embed_size)
    self.transformer_block = TransformerBlock(
        embed_size, heads, dropout, forward_expansion
    )

    self.dropout = nn.Dropout(dropout)