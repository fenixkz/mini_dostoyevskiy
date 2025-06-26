"""
Original decoder-style transformer with MultiHeadAttention. Only difference is using RMSNorm instead of LayerNorm.
Inspired by Karpathy nanoGPT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from typing import Tuple

class RMSNorm(nn.Module):
    '''
    Improving training speed and stability in comparison to LayerNorm. 
    Source: https://arxiv.org/pdf/1910.07467
    '''
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x * 1/sqrt( E[x^2] + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Cast to float32 for stability if using mixed precision, then cast back
        input_dtype = x.dtype
        x = x.float()
        output = self._norm(x).type(input_dtype)
        return output * self.weight

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim, context_length, head_size, dropout):
        super(SingleHeadAttention, self).__init__()
        self.k = nn.Linear(embedding_dim, head_size, bias=False)
        self.q = nn.Linear(embedding_dim, head_size, bias=False)
        self.v = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)     
        self.head_size = head_size
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        # x это наш ряд слов уже встроенный в вектор
        # размер х - (размер батча B, размер ряда слов T, размер embedding С
        q = self.q(x) # (B, T, A)
        k = self.k(x) # (B, T, A) A - размер блока внимания (head_size)
        v = self.v(x) # (B, T, A)

        scores = q @ torch.transpose(k, -2, -1) * self.head_size**-0.5
        scores = scores.masked_fill(self.tril[:k.shape[1], :k.shape[1]] == 0, float('-inf'))
        scores = F.softmax(scores, dim = -1)
        scores = self.dropout(scores)
        return scores @ v

class MultiHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim, context_length, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(embedding_dim, context_length, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim * 2),
            nn.GLU(dim=-1),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, embedding_dim, context_length, num_heads, dropout):
        assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        # embedding_dim: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()
        head_size = embedding_dim // num_heads
        self.sa = MultiHeadAttention(embedding_dim, context_length, num_heads, head_size, dropout)
        self.ffwd = FeedForward(embedding_dim, dropout)
        self.ln1 = RMSNorm(embedding_dim)
        self.ln2 = RMSNorm(embedding_dim)

    def forward(self, x):
        x_norm1 = self.ln1(x)
        x = x + self.sa(x_norm1)
        x_norm2 = self.ln2(x)
        x = x + self.ffwd(x_norm2)
        return x

class GPT(nn.Module):

    def __init__(self, vocab_size, context_length, embedding_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        
        self.blocks = nn.Sequential(*[Block(embedding_dim, context_length, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = RMSNorm(embedding_dim) # final layer norm
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.lm_head.weight = self.embedding_layer.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear) and module == self.lm_head:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))

    def forward(self, x, targets=None):
        B, T = x.shape
        assert T <= self.context_length, f"Sequence length {T} exceeds context length {self.context_length}"

        # x and targets are both (B,T) tensor of integers
        tok_emb = self.embedding_layer(x) # (B,T,C)
        # Create position indices
        pos = torch.arange(0, T, dtype=torch.long, device=x.device) # (T,)
        pos_emb = self.position_embedding(pos) # (T,C)
        
        x = tok_emb + pos_emb  # (B,T,C) + (T,C) -> (B,T,C) via broadcasting
        x = self.dropout(x)
        
        for block in self.blocks: # Iterate through ModuleList
            x = block(x) # No need to pass freqs_cis anymore
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B_logits, T_logits, C_logits = logits.shape # Use different var names to avoid clash
            logits_for_loss = logits.view(B_logits * T_logits, C_logits)
            targets_for_loss = targets.view(B_logits * T_logits)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)

        return logits, loss