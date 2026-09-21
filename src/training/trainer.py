import torch.nn as nn
from torch.utils.data import DataLoader

def train_model(model: nn.Module, epochs: int, tr_data: DataLoader, tst_data: DataLoader) -> None:
    