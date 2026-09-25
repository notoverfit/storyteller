import torch
import torch.nn as nn

from typing import List

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, hidden_size: int, dropout=0.0):
        super().__init__()

        self.layer_norm_one = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.layer_norm_two = nn.LayerNorm(d_model)
        # generally a wider layer than the embedding dimension is used here
        self.feed_forward = nn.Sequential(nn.Linear(d_model, hidden_size), nn.GELU(), nn.Linear(hidden_size, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        # x is (batch length, sequence length, embedding dimension)
        # sequence length (the number of tokens in the sequence) is what the transformer attends on
        T = x.size(1)

        # to remove the upper right triangle, which contains information about the future for row tokens
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        # self-attention
        residual = x
        x = self.layer_norm_one(x)

        # causal attention, removing forward looking attention and does not return weights
        attn_out, _ = self.attention(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = residual + attn_out

        # feed forward
        residual = x
        x = self.feed_forward(self.layer_norm_two(x))
        x = residual + x

        return x

class GPTNeo(nn.Module):
    def __init__(self, n_layers: int, vocab_size: int, d_model: int, max_seq_len: int, hidden_size: List[int], n_heads: List[int], dropout: List[int]):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        assert n_layers == len(hidden_size) == len(n_heads)
        self.blocks =  nn.ModuleList([DecoderBlock(d_model, n_heads[i], hidden_size[i], dropout[i]) for i in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.logit = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        x_seq_len = x.size(1)
        x_positions = torch.arange(x_seq_len, device=x.device).unsqueeze(0)

        embedding = self.token_embedding(x) + self.pos_embedding(x_positions)

        for block in self.blocks:
            embedding = block(embedding)

        logits = self.logit(self.final_norm(embedding))

        return logits