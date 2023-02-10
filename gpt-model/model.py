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

# masked multi-head attention block
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
    def __init__(self, emb_dim: int):
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
        super().__init__()
        
        head_size = emb_dim // num_heads
        
        self.mha = MultiHeadAttention(num_heads, head_size)
        self.ff = FeedForwardBlock(emb_dim)
        
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        
    def forward(self, x):
        # add + norm
        x = x + self.mha(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        
        return x
    
class GPT(nn.Module):
    def __init__(self, emb_dim: int, vocab_size: int, block_size: int, num_heads: int, num_layers: int):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
        self.position_embadding_table = nn.Embedding(block_size, emb_dim)
        
        self.blocks = nn.Sequential(*[ TransformerBlock(emb_dim, num_heads) for _ in range(num_layers)])
        
        self.norm = nn.LayerNorm(emb_dim)
        self.output = nn.Linear(emb_dim, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)
            
            if module.bias is None:
                nn.init.zeros_(module.bias)    
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)

    def forward(self, idx, targets=None):
        batch, timestep = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embadding_table(torch.arange(timestep))
        
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.norm(x)
        
        logits = self.output(x)
        loss = None
        
        if targets is not None:
            batch, timestep, channels = logits.shape
            
            logits = logits.view(batch*timestep, channels)
            targets = targets.view(batch*timestep)
            
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_tokens: int):
        for _ in range(max_tokens):
            idx_cond = idx[:, 1, :]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx