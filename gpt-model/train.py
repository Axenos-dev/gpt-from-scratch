from model import GPT
import torch
from torch import nn
import os

with open("./data/text.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
chars = sorted(set(text))
vocab_size = len(chars)

index_to_char = { i: char for i, char in enumerate(chars) }
char_to_index = { char: i for i, char in enumerate(chars) }

def encode(s: str):
    return [ char_to_index[char] for char in s ]

def decode(idx: list):
    return ''.join([ index_to_char[i] for i in idx ])

data = torch.tensor(encode(text), dtype=torch.long)

split = len(data) * 0.9
train_data = data[:split]
valid_data = data[split:]