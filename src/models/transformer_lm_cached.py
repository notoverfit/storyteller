import torch
import torch.nn as nn

from typing import List

KVCache = tuple[torch.Tensor, torch.Tensor]

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # one layer produces the query, key and value projections stacked together
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None):
        """
        the key difference in this forward pass is that it only infers on
        new tokens, thus we do not need to attend on older sequences
        """
        B, T, d_model = x.shape

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        # (batches, heads, tokens, head dimension)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # append on the newly created key and value for the new tokens
        if cache is not None:
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        T_total = k.size(2)

        scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5
        causal_mask = torch.triu(torch.ones(T_total, T_total, device=x.device, dtype=torch.bool), diagonal=1)[-T:]
        scores = scores.masked_fill(causal_mask, float('-inf'))

        weights = scores.softmax(dim=-1)
        out = weights @ v

        # collapse the heads into embedding dimension
        out = out.transpose(1, 2).reshape(B, T, d_model)

        # k and v now cover every token so far, and are the cache for the next step
        return self.out_proj(out), (k, v)

class DecoderBlockV2(nn.Module):
    def __init__(self, d_model: int, n_heads: int, hidden_size: int, dropout=0.0):
        super().__init__()

        self.layer_norm_one = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads)
        self.layer_norm_two = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, hidden_size), nn.GELU(), nn.Linear(hidden_size, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, cache: KVCache | None = None):
        residual = x
        attn_out, cache = self.attention(self.layer_norm_one(x), cache)
        x = residual + attn_out

        residual = x
        x = self.feed_forward(self.layer_norm_two(x))
        x = residual + x

        return x, cache

class GPTNeoV2(nn.Module):
    def __init__(self, n_layers: int, vocab_size: int, d_model: int, max_seq_len: int, hidden_size: List[int], n_heads: List[int], dropout: List[int]):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        assert n_layers == len(hidden_size) == len(n_heads)
        self.blocks =  nn.ModuleList([DecoderBlockV2(d_model, n_heads[i], hidden_size[i], dropout[i]) for i in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.logit = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, cache: List[KVCache] | None = None):
        x_seq_len = x.size(1)

        # if the cache is passed as none, the cache is generated
        past_len = 0 if cache is None else cache[0][0].size(2)
        x_positions = torch.arange(past_len, past_len + x_seq_len, device=x.device).unsqueeze(0)

        embedding = self.token_embedding(x) + self.pos_embedding(x_positions)

        # each block has its own cache, as its keys and values come from the block below
        new_cache = []
        for i, block in enumerate(self.blocks):
            embedding, block_cache = block(embedding, None if cache is None else cache[i])
            new_cache.append(block_cache)

        logits = self.logit(self.final_norm(embedding))

        return logits, new_cache