import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import torch
import torch.nn.functional as F
from gpt import GPT
from data import get_dataset
from tokenizers import ByteLevelBPETokenizer, Tokenizer
import json

def encode(tokenizer: ByteLevelBPETokenizer, text):
    return tokenizer.encode(text).ids

def decode(tokenizer: ByteLevelBPETokenizer, ids):
    return tokenizer.decode(ids)

def write(model, initial_token_ids, max_new_tokens, context_length,
          temperature=1.0, top_p=0.9, top_k=None, # Added top_k back as an option
          repetition_penalty=1.0, # Standard name, 1.0 means no penalty
          no_repeat_ngram_size=0):
    """
    Generates text using a model, applying various sampling strategies.

    Args:
        model: The PyTorch model for text generation.
        tokenizer: The tokenizer used for encoding/decoding.
        initial_token_ids: A 2D tensor (batch_size, seq_len) of initial token IDs.
        max_new_tokens: The maximum number of new tokens to generate.
        context_length: The maximum context length the model can handle.
        temperature (float): Controls randomness. Lower is more deterministic.
        top_p (float, optional): Nucleus sampling threshold.
        top_k (int, optional): Top-k sampling threshold.
        repetition_penalty (float): Penalizes tokens based on their frequency in the context.
                                   Values > 1.0 discourage repetition. 1.0 means no penalty.
        no_repeat_ngram_size (int): If > 0, n-grams of this size cannot be repeated.
    
    Returns:
        A 2D tensor (batch_size, seq_len + max_new_tokens) of token IDs.
    """
    model.eval()
    x = initial_token_ids
    device = next(model.parameters()).device # Get model's device

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # 1. Determine current context
            if x.size(1) > context_length:
                current_context = x[:, -context_length:]
            else:
                current_context = x
            
            # 2. Get model logits
            logits, _ = model(current_context)
            # Get logits for the very last token prediction
            next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

            # 3. Apply repetition penalty (before temperature)
            # This discourages repeating tokens that have appeared in the input `x`
            if repetition_penalty != 1.0 and x.size(1) > 0:
                for i in range(x.size(0)): # Iterate over batch
                    # Create a score for each token in the vocabulary
                    score = torch.gather(next_token_logits[i], 0, x[i])
                    # If score < 0, it means the token is already penalized (e.g. by a previous penalty)
                    # so we divide by penalty. Otherwise, we multiply.
                    # This is a common way to implement repetition penalty.
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    next_token_logits[i].scatter_(0, x[i], score)
            
            # 4. Apply temperature
            # Temperature must be positive.
            # Smaller temperature -> more confident, larger -> more random
            if temperature <= 0:
                temperature = 1.0 # Default to no effect if invalid
            next_token_logits = next_token_logits / max(temperature, 1e-6) # Avoid division by zero

            # 5. Apply top-k filtering (optional)
            # This keeps only the top_k most likely tokens
            if top_k is not None and top_k > 0:
                k = min(top_k, next_token_logits.size(-1)) # Ensure k is not larger than vocab size
                if k > 0:
                    # Get the k-th score (smallest among the top-k)
                    top_k_logits, _ = torch.topk(next_token_logits, k, dim=-1)
                    min_logit_in_top_k = top_k_logits[:, -1, None] # (B, 1)
                    # Set logits of tokens not in top-k to -infinity
                    indices_to_remove = next_token_logits < min_logit_in_top_k
                    next_token_logits[indices_to_remove] = -float('Inf')

            # 6. Apply top-p (nucleus) filtering (optional)
            # This keeps the smallest set of tokens whose cumulative probability exceeds top_p
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Create a mask for tokens to remove
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the mask: keep the first token that exceeds top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0 # Never remove the most probable token by this rule
                
                # Create a mask in the original logit order
                indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('Inf')

            # 7. Prevent repeating n-grams (optional)
            if no_repeat_ngram_size > 0 and x.size(1) >= no_repeat_ngram_size:
                for i in range(x.size(0)): # Iterate over batch
                    # Get previously generated n-grams
                    generated_ngrams = {}
                    current_sequence = x[i].tolist()
                    for j in range(len(current_sequence) - no_repeat_ngram_size + 1):
                        ngram = tuple(current_sequence[j : j + no_repeat_ngram_size])
                        if ngram in generated_ngrams:
                            generated_ngrams[ngram] += 1
                        else:
                            generated_ngrams[ngram] = 1
                    
                    # Get the prefix for the next potential n-gram
                    prefix_ngram_tokens = current_sequence[-(no_repeat_ngram_size - 1):] if no_repeat_ngram_size > 1 else []

                    # Ban tokens that would complete a repeated n-gram
                    for token_to_check_id in range(next_token_logits.size(-1)):
                        potential_ngram = tuple(prefix_ngram_tokens + [token_to_check_id])
                        if potential_ngram in generated_ngrams:
                            next_token_logits[i, token_to_check_id] = -float('Inf')
            
            # 8. Get probabilities and sample
            # Safety check for NaNs/Infs in logits before softmax
            if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).all(dim=-1, keepdim=True).any():
                print("Warning: NaNs or all Infs detected in logits before softmax. Attempting to recover.")
                # If all logits for an item are -Inf, softmax will produce NaNs.
                # Set one logit to 0 to allow softmax to produce a distribution (e.g., pick token 0).
                all_inf_mask = torch.isinf(next_token_logits).all(dim=-1)
                if all_inf_mask.any():
                    # For rows where all logits are -Inf, set the logit of token 0 to 0.0
                    # This makes token 0 the only possible choice for these rows.
                    next_token_logits[all_inf_mask, 0] = 0.0 
                
                next_token_logits = torch.nan_to_num(next_token_logits, nan=0.0, posinf=1e4, neginf=-1e4)


            probs = F.softmax(next_token_logits, dim=-1)

            # Final safety check for probabilities (e.g. if all logits were -Inf)
            # If probs still sum to 0 or contain NaN (e.g., if all logits became -Inf and recovery failed)
            # For each item in batch, check if sum of probs is close to 1
            probs_sum_is_valid = probs.sum(dim=-1).isclose(torch.tensor(1.0, device=device))
            if not torch.all(probs_sum_is_valid):
                print("Warning: Invalid probabilities for multinomial after softmax. Applying fallback.")
                for i in range(probs.size(0)):
                    if not probs_sum_is_valid[i] or torch.isnan(probs[i]).any() or torch.isinf(probs[i]).any():
                        # If probs for this batch item are invalid, set to uniform or pick most likely from original logits
                        # Fallback to picking the token with the highest original logit if possible
                        # This is a desperate measure.
                        print(f"Fallback for batch item {i}")
                        # Try to recover from original logits if they were not all -inf
                        original_logits_for_item = logits[i, -1, :]
                        if not torch.isinf(original_logits_for_item).all():
                            fallback_token = torch.argmax(original_logits_for_item).unsqueeze(0)
                            probs[i].zero_() # Zero out current bad probs
                            probs[i, fallback_token] = 1.0
                        else: # All original logits were also -inf, pick token 0
                            probs[i].zero_()
                            probs[i, 0] = 1.0 
            
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 9. Append chosen token and continue
            x = torch.cat((x, next_token), dim=1)
    return x


print("Hello! This is a mini-Dostoyevskiy GPT based model. Please to evaluate the model, write the vocabulary size: ")
vocab_size = int(input())

tokenizer = Tokenizer.from_file(f'data/tokenizer/tokenizer_{vocab_size}.json')

best_val_loss = float('inf')
# Iterate over all folders in path models/
models_path = 'models/'
for folder_name in os.listdir(models_path):
    folder_path = os.path.join(models_path, folder_name)
    if os.path.isdir(folder_path):
        # Read vocab_size from config.json in the folder
        config_path = os.path.join(folder_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        folder_vocab_size = config.get("vocab_size")
        if folder_vocab_size == vocab_size:
            # Read best_val_loss from torch checkpoint
            checkpoint_path = os.path.join(folder_path, "checkpoint.pth")
            checkpoint = torch.load(checkpoint_path)
            folder_best_val_loss = checkpoint.get("best_val_loss")
            if folder_best_val_loss < best_val_loss:
                best_val_loss = folder_best_val_loss
                best_model_path = folder_path
print(f"The best model with vocab size {vocab_size} is in folder: {best_model_path} with validation loss: {best_val_loss:.4f}")

config = json.load(open(os.path.join(best_model_path, "config.json"), "r", encoding="utf-8"))
context_length = config.get("context_length")
n_embedding = config.get("embedding_dim")
n_heads = config.get("num_heads")
n_layers = config.get("num_layers")
dropout = config.get("dropout")

#### Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = True if device == 'cuda' else False
####

model = GPT(vocab_size=vocab_size, context_length = context_length, 
            embedding_dim = n_embedding, num_heads = n_heads, 
            num_layers = n_layers, dropout=dropout).to(device)

# print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
model.load_state_dict(torch.load(os.path.join(best_model_path, "best_model.pth")))

# Generate text
max_words = 200
text = "Родина "
encoded = encode(tokenizer, text)
text = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0).to(device)
generated = write(
    model, text, max_words, context_length,
    temperature=0.7,
    top_p=0.8,     
    repetition_penalty = 1.2    
)
generated_ids = generated[0].tolist() # Assuming batch size is 1
print("".join(decode(tokenizer, generated_ids)))
print()