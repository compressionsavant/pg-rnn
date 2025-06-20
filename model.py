import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
# from rnn import Layer
from rnn import TLayer as Layer

@dataclass
class Config:
  vocab_size: int = 50304
  n_embd: int = 256
  n_layer: int = 2
  n_hidden: int = 512
  dropout: float = 0.2


class RNNLM(nn.Module):
  """RNN Language Model"""
  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    self.rnn = nn.ModuleDict(
      dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),
        h = nn.ModuleList([Layer(config.n_embd, config.n_hidden) if i == 0 else Layer(config.n_hidden, config.n_hidden) for i in range(config.n_layer)]),
        dropout = nn.Dropout(config.dropout)
      )
    )
    self.lm_head = nn.Linear(config.n_hidden, config.vocab_size)
    self.apply(self._init_weights)
  
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.xavier_normal_(module.weight)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.xavier_normal_(module.weight)

  def forward(self, idx, targets=None):
    x = self.rnn.wte(idx)
    B, T, C = x.shape
    for layer in self.rnn.h:
      x, _ = layer(x)
    x = self.rnn.dropout(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = x.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss