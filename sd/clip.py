import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


#clip encoder (like the gpt transformer block)

class CLIPEmbedding(nn.Module):
  def __init__(self, n_vocab, n_embd, n_token):
    super().__init__()
    self.token_embedding = nn.Embedding(n_vocab, n_embd)
    self.position_embedding = nn.Parameter(torch.zeros(n_token, n_embd))

  def forward(self, tokens):
    x = self.token_embedding(tokens)
    x += self.position_embedding
    return x

#like MLP
class CLIPlayer(nn.Module):
  def __init__(self, n_head, n_embd):
    super().__init__()
    self.layernorm_1 = nn.LayerNorm(n_embd)
    self.attention = SelfAttention(n_head, n_embd)
    self.layernorm_2 = nn.LayerNorm(n_embd)
    self.linear_1 = nn.Linear(n_embd, 4 * n_embd) # same as in gpt
    self.linear_2 = nn.Linear(4 *n_embd, n_embd)

  def forward(self, x):
    residue = x
    # self attention
    x = self.layernorm_1(x)
    x = self.attention(x, causal_mask = True)
    x += residue
    # FF layer
    residue = x

    x = self.layernorm_2(x)
    x = self.linear_1(x)
    x = x * torch.sigmoid(1.702 * x) #quickGeLU some paper thing
    x = self.linear_2(x)
    x += residue
    return x

class CLIP(nn.Module):
  def __init__(self):
    super().__init__()
    self.embedding = CLIPEmbedding(49408, 768, 77)

    self.layers = nn.ModuleList([
        CLIPlayer(12, 768) for i in range(12)
    ])

    self.layernorm = nn.LayerNorm(768)

  def forward(self, tokens: torch.LongTensor):
    tokens = tokens.type(torch.long)

    #(batch_size, seq_len) -> (batch_size, seq_len, Dim)
    state = self.embedding(tokens)

    for layer in self.layers:
      state = layer(state)

    #(batch_size, Seq_len, dim)
    output = self.layernorm(state)

    return output

    