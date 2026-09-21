import torch
import numpy as np
from torch.utils.data import Dataset

class LMDataset(Dataset):
    def __init__(self, path: str, seq_len: int):
        self.tokens = np.memmap(path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len

    def __len__(self):
        # returns a whole number of the amount of sequences with seq_len length.
        # doesn't include a possible remainder sequence
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start+self.seq_len+1].astype(np.int64)
        # below is essentially shifted; so token[i] vs token[i+1]
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return x, y