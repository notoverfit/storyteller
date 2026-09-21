from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    pass

@dataclass
class ModelConfig(Config):
    pass

@dataclass
class LSTMConfig(ModelConfig):
    hidden_sizes: List[int]
    vocab_size: int
    embedding_size: int

@dataclass
class TrainConfig(Config):
    batch_size: int
    learning_rate: float