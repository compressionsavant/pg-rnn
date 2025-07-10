import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from datasets import load_dataset
from itertools import chain
import tiktoken
from model import RNNLM, Config

# ----------------------------------------------
ctx: int = 50
batch_size: int = 32
lr: float = 2e-3
its: int = 12000
eval_it: int = 500
n_eval: int = 200
device: str = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------

torch.set_float32_matmul_precision("high")
torch.manual_seed(42)
if device == "cuda":
  torch.cuda.manual_seed(42)

model = RNNLM(Config())
model = torch.compile(model)
m = model.to(device)

class PGStream(IterableDataset):
  def __init__(self, B: int, T: int, dataset: str = "emozilla/PaulGrahamEssays", tokenizer: str = "gpt2"):
    super().__init__()
    self.B = B
    self.T = T
    self.pos = 0
    self.enc = tiktoken.get_encoding(tokenizer)
    data = load_dataset(dataset, split="train")
    self.data = data.map(self.encode, batched=True, remove_columns=["text"], batch_size=128)
    self.tokens = list(chain.from_iterable(self.data["idx"]))
    

  def encode(self, x):
    x["idx"] = self.enc.encode_batch(x["text"])
    return x
  
  def shuffle(self):
      self.data = self.data.shuffle()
      self.tokens = list(chain.from_iterable(self.data["idx"]))
      self.pos = 0

  def __iter__(self):
    while True:
      if self.pos + self.B * self.T + 1 > len(self.tokens):
        self.shuffle()
      
      token_buf = torch.tensor(self.tokens[self.pos : self.pos + self.B * self.T + 1])
      x = token_buf[:-1].view(self.B, self.T)
      y = token_buf[1:].view(self.B, self.T)

      self.pos += self.B * self.T
      x, y = x.to(device), y.to(device)
      yield x, y


print(f"model params: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
optim = torch.optim.AdamW(m.parameters(), lr=lr)

dataset = PGStream(B=batch_size, T=ctx)
dataloader = iter(dataset)

for it in range(its):
  x, y = next(dataloader)
  optim.zero_grad()
  with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = m(x, y)
  loss.backward()
  nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
  optim.step()
  torch.cuda.synchronize()

  if it % eval_it == 0 or it == its - 1:
    print(f"Iteration: {it} loss: {loss.item()}")

torch.save(m.state_dict(), "model.pth")

def generate(idx, ctx_window):
  for token in range(ctx_window):
    idxc = idx[:, -ctx:]
    logits, _ = m(idxc)
    logits = logits[:, -1, :]
    p = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(p, num_samples=1)
    idx = torch.cat([idx, idx_next], dim=1)
  return idx

context = torch.zeros((1,1), dtype=torch.long, device=device)
with open("pg.txt", "w") as f:
  f.write(dataset.enc.decode(generate(context, ctx_window=1000)[0].tolist()))
