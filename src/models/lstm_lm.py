import torch.nn as nn
from typing import List

class LSTMLangModel(nn.Module):
    def __init__(self, hidden_sizes: List[int], vocab_size: int, embedding_size: int):
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # hidden layers
        tmp_size = embedding_size
        self.layers: nn.ModuleList = nn.ModuleList()
        for layer in range(len(hidden_sizes)):
            self.layers.append(nn.LSTM(tmp_size, hidden_sizes[layer], 1, batch_first=True))
            tmp_size = hidden_sizes[layer]

        # linear output
        self.output = nn.Linear(tmp_size, vocab_size)
    
    def forward(self, token_ids):
        x = self.embedding(token_ids)

        for layer in self.layers: x, _ = layer(x)
        logits = self.output(x)

        return logits