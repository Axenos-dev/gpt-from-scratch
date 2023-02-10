import torch
import torch.functional as F
from torch import nn


class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        
        # 384 - embedding size
        self.query = nn.Linear(384, head_size)
        self.key = nn.Linear(384, head_size)
        self.value = nn.Linear(384, head_size)

        self.register_buffer("tril", torch.tril(torch.ones(256, 256)))

        # setting dropout to prevent overfitting
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _, timestep, _ = x.shape

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        # attention = softmax((Q * K.T) / sqrt(keys of dimmention)) * V
        attention = query @ key.transpose(-2, -1) * key.shape[-1]**-0.5
        attention = attention.masked_fill(self.tril[:timestep, :timestep], float('-inf')) # applying mask
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention) # dropout to destroy some nodes
        attention = attention @ value

        return attention

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()

        self.heads = nn.ModuleList([ Head(head_size=head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size*num_heads, 384)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        output = torch.cat([ head(x) for head in self.heads ], dim=-1)
        output = self.dropout(self.projection(output))

        return output

class FeedForwardBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(384, 4 * 384),
            nn.ReLU(),
            nn.Linear(4 * 384, 384),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.block(x)

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int):
        head_size = emb_dim // num_heads