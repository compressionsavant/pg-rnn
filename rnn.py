import torch
import torch.nn as nn
import torch.nn.functional as F

class Cell(nn.Module):
  """RNN Cell using Linear Layers"""
  def __init__(self, input_size: int, hidden_size: int):
    super().__init__()
    self.i2h = nn.Linear(input_size, hidden_size)
    self.h2h = nn.Linear(hidden_size, hidden_size)

  def forward(self, x, hidden_state):
    state = self.i2h(x) + self.h2h(x)
    return F.tanh(state)
  

class Layer(nn.Module):
  """RNN Layer using Cell"""
  def __init__(self, input_size: int, hidden_size: int):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.cell = Cell(input_size, hidden_size)

  def forward(self, x, hidden_state=None):
    B, T, C = x.shape
    if hidden_state is None:
      h = torch.zeros(B, self.hidden_size)
    else:
      h = hidden_state
    
    hidden_states = []
    for seq in range(T):
      h = self.cell(x[:, seq, :])
      hidden_states.append(h)
    
    # B, C -> B, T, C
    hidden_states = torch.stack(hidden_states, dim=1)
    return hidden_states, h

class TLayer(nn.Module):
  """Truncated RNN Layer using Cell"""
  def __init__(self, input_size: int, hidden_size: int, truncated_size: int = 25):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.truncated_size = truncated_size
    self.cell = Cell(input_size, hidden_size)

  def forward(self, x, hidden_state=None):
    B, T, C = x.shape
    if hidden_state is None:
      h = torch.zeros(B, self.hidden_size)
    else:
      h = hidden_state
    
    hidden_states = []
    for start in range(0, T, self.truncated_size):
      end = min(start + self.truncated_size, T)
      chunk = x[:, start:end, :]
      h = h.detach()

      chunk_states = []
      for seq in range(chunk.size(1)):
        h = self.cell(chunk[:, seq, :], h)
        chunk_states.append(h)
      
      hidden_states.extend(chunk_states)
    
    # B, C -> B, T, C
    hidden_states = torch.stack(hidden_states, dim=1)
    return hidden_states, h


