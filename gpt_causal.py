import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from typing import Tuple
import inspect

class RMSNorm(nn.Module):
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

class CausalAttention(nn.Module):

    def __init__(self, embedding_dim, context_length, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        # K,Q,V matrices in one triple size one
        self.c_attn = nn.Linear(embedding_dim, 3*embedding_dim, bias=False)
        # Output projection
        self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("tril", torch.tril(torch.ones(context_length, context_length))
                                        .view(1, 1, context_length, context_length))

    def forward(self, x):
        B, T, C = x.size() # Batch size, context length, embedding dimensions

        q, k, v = self.c_attn(x).split(C, dim = 2)
        k = k.view(B,T, self.n_heads, C // self.n_heads).transpose(1, 2) # Shape: B x n_heads x T x head_size
        q = q.view(B,T, self.n_heads, C // self.n_heads).transpose(1, 2) # Shape: B x n_heads x T x head_size
        v = v.view(B,T, self.n_heads, C // self.n_heads).transpose(1, 2) # Shape: B x n_heads x T x head_size

        # Causal self-attention: (B, nh, T, hs) x (B, nh, ns, T) -> (B, nh, T, T)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # Manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v 
        
        y = y.transpose(1,2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side
        y = self.dropout(self.proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim, bias = False),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim, bias = False),
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
        self.sa = CausalAttention(embedding_dim, context_length, num_heads, dropout)
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Taken from Karpathy source files
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    
