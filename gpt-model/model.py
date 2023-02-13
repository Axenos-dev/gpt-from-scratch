import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.values = nn.Linear(emb_dim, emb_dim)
        self.keys = nn.Linear(emb_dim, emb_dim)
        self.query = nn.Linear(emb_dim, emb_dim)

        self.out = nn.Linear(emb_dim, emb_dim)

    def forward(self, values, keys, query, mask=None):
        batch_size = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        query = self.query(query)

        values = values.reshape(batch_size, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, key_len, self.num_heads, self.head_dim)
        query = query.reshape(batch_size, query_len, self.num_heads, self.head_dim)

        #attention = query @ keys.transpose(-1, -2) * self.head_dim**-0.5

        attention = torch.einsum("BQHD, BKHD -> BHQK", [query, keys])

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))

        attention = attention ** -0.5

        attention = F.softmax(attention, dim=3)
        attention = torch.einsum("BHQK, BVHD -> BQHD", [attention, values])

        attention = attention.view(batch_size, query_len, self.emb_dim)

        return self.out(attention)

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super(FeedForward, self).__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            nn.ReLU(),
            nn.Linear(4*emb_dim, emb_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.feedforward(x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int):
        super(TransformerBlock, self).__init__()

        self.mha = MultiHeadAttention(emb_dim=emb_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(emb_dim)

        self.feedforward = FeedForward(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, values, keys, query, mask=None):
        attention = self.mha(values, keys, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feedforward(x)

        return self.dropout(self.norm2(forward + x))


class Encoder(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, num_layers: int, vocab_size: int, max_length: int):
        super(Encoder, self).__init__()

        self.emb_dim = emb_dim
        
        self.word_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_length, emb_dim)

        self.blocks = nn.ModuleList([ TransformerBlock(emb_dim, num_heads) for _ in range(num_layers) ])

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape

        positions = torch.arange(0, seq_len).expand(batch_size, seq_len)

        word_emb = self.word_emb(x)
        pos_emb = self.pos_emb(positions)

        out = self.dropout(word_emb + pos_emb)

        for block in self.blocks:
            out = block(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int):
        super(DecoderBlock, self).__init__()
        
        self.mha = MultiHeadAttention(emb_dim=emb_dim,num_heads=num_heads)
        self.block = TransformerBlock(emb_dim, num_heads)
        
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, value, key, src_mask, trg_mask):
        attention = self.mha(query, query, query, trg_mask)
        query = self.dropout(self.norm(attention + query))
        
        out = self.block(value, key, query, src_mask)
        
        return out
    

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_layers: int, num_heads: int, max_length: int):
        super(Decoder, self).__init__()
        
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        
        self.blocks = nn.ModuleList([
            DecoderBlock(emb_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.out = nn.Linear(emb_dim, vocab_size)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, encoder_output, src_mask, trg_mask):
        batch_size, seq_len = x.shape
        
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len)
        
        word_emb = self.word_embedding(x)
        pos_emb = self.pos_embedding(positions)
        
        x = self.dropout(pos_emb + word_emb)
        
        for block in self.blocks:
            x = block(x, encoder_output, encoder_output, src_mask, trg_mask)
            
        out = self.out(x)
        
        return out


class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, trg_vocab_size: int, emb_dim: int, num_layers: int, num_heads: int, max_length: int):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            vocab_size=src_vocab_size,
            max_length=max_length
        )
        
        self.decoder = Decoder(
            emb_dim=emb_dim,
            num_heads=num_layers,
            num_layers=num_layers,
            vocab_size=trg_vocab_size,
            max_length=max_length
        )
        
        self.mask = torch.tril(torch.ones(num_heads, num_heads)) # tril mask shape: (block_size x block_size)
    
    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask
    
    def forward(self, x, y):
        encoder_output = self.encoder(x)
        out = self.decoder(y, encoder_output, self.make_src_mask(x), self.make_trg_mask(y))
        
        return out
        