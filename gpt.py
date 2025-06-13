import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6): # Added eps
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # This is gamma

    def _norm(self, x):
        # x * 1/sqrt( E[x^2] + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Cast to float32 for stability if using mixed precision, then cast back
        input_dtype = x.dtype
        x = x.float()
        output = self._norm(x).type(input_dtype)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the complex frequency terms (cis = cos + i*sin) for RoPE.
    dim: The dimension of the head (head_size). RoPE is applied to pairs, so dim must be even.
    end: The maximum sequence length (context_length).
    theta: A base period for the frequencies.
    """
    assert dim % 2 == 0, "Dimension must be even for RoPE"
    # freqs for each pair of dimensions
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # t for each position
    t = torch.arange(end, device=freqs.device)  # Positions m
    # freqs_m = m * freqs_j
    freqs = torch.outer(t, freqs)  # Shape (end, dim / 2)
    # freqs_cis = e^(i * m * freqs_j) = cos(m * freqs_j) + i * sin(m * freqs_j)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Shape (end, dim / 2)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor, # Query tensor (B, T, head_size) or (B, T, num_heads, head_size_per_head)
    xk: torch.Tensor, # Key tensor
    freqs_cis: torch.Tensor, # (T_slice, head_size / 2)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embedding to query and key tensors.
    """
    # Reshape xq and xk to view pairs of dimensions as complex numbers
    # xq: (B, T, head_size) -> (B, T, head_size/2, 2)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # Convert to complex
    # xq_complex: (B, T, head_size/2)
    xq_c = torch.view_as_complex(xq_)
    xk_c = torch.view_as_complex(xk_)
    
    # freqs_cis: (T_slice, head_size/2) -> needs to be broadcastable for (B, T, head_size/2)
    # Add batch dim for broadcasting, if necessary, or ensure freqs_cis is already [None, :, None] etc.
    # Here, freqs_cis is (T_slice, head_size/2). We expect xq_c is (B, T_slice, head_size/2)
    # So, freqs_cis should be shaped [None, T_slice, head_size/2] for broadcasting
    freqs_cis = freqs_cis.unsqueeze(0) # Now (1, T_slice, head_size/2)

    # Apply rotation: q_new = q_complex * freqs_cis
    xq_out_c = xq_c * freqs_cis
    xk_out_c = xk_c * freqs_cis
    
    # Convert back to real
    # xq_out: (B, T, head_size/2, 2)
    xq_out = torch.view_as_real(xq_out_c)
    xk_out = torch.view_as_real(xk_out_c)
    
    # Reshape back to original
    # xq_out: (B, T, head_size)
    xq_out = xq_out.flatten(2)
    xk_out = xk_out.flatten(2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim, context_length, head_size, dropout):
        super(SingleHeadAttention, self).__init__()
        self.k = nn.Linear(embedding_dim, head_size, bias=False)
        self.q = nn.Linear(embedding_dim, head_size, bias=False)
        self.v = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)     
        self.head_size = head_size
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x, freqs_cis_slice):
        # x это наш ряд слов уже встроенный в вектор
        # размер х - (размер батча B, размер ряда слов T, размер embedding С
        q = self.q(x) # (B, T, A)
        k = self.k(x) # (B, T, A) A - размер блока внимания (head_size)
        v = self.v(x) # (B, T, A)
        # Apply RoPE to q and k
        # freqs_cis_slice should correspond to the current sequence length T
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis_slice)

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

    def forward(self, x, freqs_cis):
        out = torch.cat([h(x, freqs_cis) for h in self.heads], dim=-1)
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

    def forward(self, x, freqs_cis):
        x_norm1 = self.ln1(x)
        x = x + self.sa(x_norm1, freqs_cis)
        x_norm2 = self.ln2(x)
        x = x + self.ffwd(x_norm2)
        return x

class GPT(nn.Module):

    def __init__(self, vocab_size, context_length, embedding_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        # Precompute RoPE frequencies for the head dimension
        self.head_size = embedding_dim // num_heads
        assert self.head_size % 2 == 0, "Head size must be even for RoPE"
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_size, context_length), persistent=False)
        
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

        # x and targets are both (B,T) tensor of integers
        tok_emb = self.embedding_layer(x) # (B,T,C)
        x = tok_emb  # (B,T,C)
        x = self.dropout(x)
        # Get the slice of freqs_cis for the current sequence length T
        current_freqs_cis = self.freqs_cis[:T]
        for block in self.blocks: # Iterate through ModuleList
            x = block(x, current_freqs_cis) # Pass freqs_cis to each block
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B_logits, T_logits, C_logits = logits.shape # Use different var names to avoid clash
            logits_for_loss = logits.view(B_logits * T_logits, C_logits)
            targets_for_loss = targets.view(B_logits * T_logits)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)

        return logits, loss