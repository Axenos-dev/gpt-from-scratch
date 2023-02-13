import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int):
        super(SelfAttention, self).__init__()

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

        print(key_len, attention.shape)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))

        attention = attention ** -0.5

        attention = F.softmax(attention, dim=3)
        attention = torch.einsum("BHQK, BVHD -> BQHD", [attention, values])

        attention = attention.view(batch_size, query_len, self.emb_dim)

        return self.out(attention)

# class SelfAttention(nn.Module):
#     def __init__(self, embed_size, heads):
#         super(SelfAttention, self).__init__()
#         self.embed_size = embed_size
#         self.heads = heads
#         self.head_dim = embed_size // heads

#         assert (
#             self.head_dim * heads == embed_size
#         ), "Embedding size needs to be divisible by heads"

#         self.values = nn.Linear(embed_size, embed_size)
#         self.keys = nn.Linear(embed_size, embed_size)
#         self.queries = nn.Linear(embed_size, embed_size)
#         self.fc_out = nn.Linear(embed_size, embed_size)

#     def forward(self, values, keys, query, mask):
#         # Get number of training examples
#         N = query.shape[0]

#         value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

#         values = self.values(values)  # (N, value_len, embed_size)
#         keys = self.keys(keys)  # (N, key_len, embed_size)
#         queries = self.queries(query)  # (N, query_len, embed_size)

#         # Split the embedding into self.heads different pieces
#         values = values.reshape(N, value_len, self.heads, self.head_dim)
#         keys = keys.reshape(N, key_len, self.heads, self.head_dim)
#         queries = queries.reshape(N, query_len, self.heads, self.head_dim)

#         # Einsum does matrix mult. for query*keys for each training example
#         # with every other training example, don't be confused by einsum
#         # it's just how I like doing matrix multiplication & bmm

#         energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
#         # queries shape: (N, query_len, heads, heads_dim),
#         # keys shape: (N, key_len, heads, heads_dim)
#         # energy: (N, heads, query_len, key_len)

#         # Mask padded indices so their weights become 0
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, float("-1e20"))

#         # Normalize energy values similarly to seq2seq + attention
#         # so that they sum to 1. Also divide by scaling factor for
#         # better stability
#         attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
#         # attention shape: (N, heads, query_len, key_len)

#         out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
#             N, query_len, self.heads * self.head_dim
#         )
#         # attention shape: (N, heads, query_len, key_len)
#         # values shape: (N, value_len, heads, heads_dim)
#         # out after matrix multiply: (N, query_len, heads, head_dim), then
#         # we reshape and flatten the last two dimensions.

#         out = self.fc_out(out)
#         # Linear layer doesn't modify the shape, final shape will be
#         # (N, query_len, embed_size)

#         return out

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
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = FeedForward(embed_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int):
        super(DecoderBlock, self).__init__()
        
        self.mha = SelfAttention(emb_dim=emb_dim, num_heads=num_heads)
        self.block = TransformerBlock(emb_dim, num_heads)
        
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, value, key, src_mask, trg_mask):
        attention = self.mha(query, query, query, trg_mask)
        query = self.dropout(self.norm(attention + query))
        
        out = self.block(value, key, query, src_mask)
        
        return out

# class DecoderBlock(nn.Module):
#     def __init__(self, embed_size, heads):
#         super(DecoderBlock, self).__init__()
#         self.norm = nn.LayerNorm(embed_size)
#         self.attention = SelfAttention(embed_size, heads)
#         self.transformer_block = TransformerBlock(
#             embed_size, heads
#         )
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x, value, key, src_mask, trg_mask):
#         attention = self.attention(x, x, x, trg_mask)
#         query = self.dropout(self.norm(attention + x))
#         out = self.transformer_block(value, key, query, src_mask)
#         return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(emb_dim=embed_size, num_heads=heads)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        heads=8,
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4], [1, 8, 7, 3]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 1, 1, 1], [1, 5, 1, 1, 1]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(
        device
    )
    out = model(x, trg)
    print(out.shape)