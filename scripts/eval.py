import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt import GPT
from tokenizers import ByteLevelBPETokenizer, Tokenizer
import argparse

def encode(tokenizer: ByteLevelBPETokenizer, text):
    return tokenizer.encode(text).ids

def decode(tokenizer: ByteLevelBPETokenizer, ids):
    return tokenizer.decode(ids)

# A context manager to disable gradient tracking
@torch.inference_mode()
def write(model: GPT, 
          tokenizer: ByteLevelBPETokenizer, 
          initial_text: str, 
          max_new_tokens: int, 
          context_length: int,
          temperature=1.0, 
          top_p=None, 
          top_k=None, 
          min_p = None,
          repetition_penalty=1.0, 
          no_repeat_ngram_size=0):
    """
    Using a pre-trained model to generate text. Applies the following sampling techniques:
        1. Repetition penalty -- A parameter that penalizes the model from repeating the same tokens. Operates by decreasing scores (logit values) of the tokens that are already in the context. 
           Default value is 1.0, means no penalization. Value < 1.0 makes the model make more repetitions. Value > 1.0 makes the model make less repetitions. 
        2. Temperature -- A parameter that controls how deterministic the model is. Works by dividing raw logits by a scalar. Because of non-linear exponential operation, the resulting probability distribution changes.
           Default value is 1.0, means no modifications. Value < 1.0 makes the model more deterministic (increasing probabilities of highly probable tokens). Value > 1.0 makes the model more stochastic (decreasing probabilities of highly probable tokens).
        3. Top-K filtering -- Sorting the logits based on the value and keeping only first K logits. The remaining logits are set to -inf to result in 0% chance of picking.
           Default value is None, means no filtering. The higher value the higher is the number of logits that are kept. Value cannot be higher than size of the vocabulary.
        4. Top-P filtering -- A filtering technique to keep only some amount of top logits whose cumulative probability is higher than the top-p parameter. 
           Default value is 0.0, means no filtering. The higher the value is the lower amount of logits that are kept. Parameter cannot be higher than 1.0.  
        6. Min-P filtering -- Keeps tokens with probability >= min_p * (probability of the most likely token).
           Default value is None, which means no filtering. The lower the value is the more strict the technique is. 
        5. N-gram repetition penalty -- A filtering technique to filter the tokens that would generate a already existing N-gram in the text. Works by computing all N-grams in already generated text, and then checking if the last N-gram with new token would be already in the sequence.
           Default value is 0, means no filtering. The higher the value is the less strict the technique is.

    All filtering techniques are implemented for experimentation. Using all of them together can lead to all tokens being filtered out.
    """
    # For simplicity, using this notation: V -- size of vocabulary, T -- context length
    model.eval() # Ensure model is in evaluation mode

    # Get the device of the model
    device = next(model.parameters()).device 
    
    # Encode text and prepare the initial tensor of token IDs
    encoded_text = encode(tokenizer, initial_text)
    # The running sequence of token IDs, with padded batch dimensions, shape (1, seq_len)
    x = torch.tensor(encoded_text, dtype=torch.long, device=device).unsqueeze(0)

    # A loop to generate tokens one by one until max_new_tokens
    for _ in range(max_new_tokens):
        # 1. Determine current context, crop to the latest T tokens (as model is trained only on T size)
        current_context = x[:, -context_length:]
        
        # 2. Forward pass, get the raw logits, shape: (1, T, V)
        logits, _ = model(current_context)
        # We only care about the logits for the very last token, get the logits of the last token, shape: (V, 1)
        next_token_logits = logits[0, -1, :]  

        # --- SAMPLING LOGIC ---

        # 3. Apply repetition penalty
        if repetition_penalty != 1.0 and x.size(1) > 0:
            # Get ids of the tokens already generated (provided)
            tokens_in_context = x.squeeze(0)
            # Get raw logits (scores) of the tokens that were already generated
            scores = torch.gather(next_token_logits, 0, tokens_in_context)
            # Penalize these scores by diving by repetition_penalty if the score is positive, or multiplying by it if the score is negative
            scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
            # Replace the original logits by the modified
            next_token_logits.scatter_(0, tokens_in_context, scores)

        # 8. Prevent repeating n-grams
        if no_repeat_ngram_size > 0 and x.size(1) >= no_repeat_ngram_size:
            current_sequence = x.squeeze(0).tolist()
            generated_ngrams = {tuple(current_sequence[j:j+no_repeat_ngram_size]) for j in range(len(current_sequence) - no_repeat_ngram_size + 1)}
            prefix_tokens = current_sequence[-(no_repeat_ngram_size - 1):] if no_repeat_ngram_size > 1 else []
            
            for token_id in range(next_token_logits.size(-1)):
                potential_ngram = tuple(prefix_tokens + [token_id])
                if potential_ngram in generated_ngrams:
                    # Set the logit to -infinity to effectively ban this token
                    next_token_logits[token_id] = -float('Inf')

        # 4. Apply temperature
        next_token_logits = next_token_logits / max(temperature, 1e-6)

        # 5. Apply top-k filtering
        if top_k is not None and top_k > 0:
            # K cannot be higher than V
            k = min(top_k, next_token_logits.size(-1))
            # Get values of top_k logits 
            top_k_logits, _ = torch.topk(next_token_logits, k)
            # Get the value of the last top-k logit (the smaller one that we keep)
            min_logit_to_keep = top_k_logits[-1]
            # Set all logits below this threshold to -infinity
            next_token_logits[next_token_logits < min_logit_to_keep] = -float('Inf')

        # 6. Apply top-p (nucleus) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            # Sort all logits in descending order
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            # Get a cumulative sum of all tokens
            # NOTE: provides a tensor of size (V), where each element is the sum of all logits before it.
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Find indices to remove, i.e. check after which logit our cumsum got higher than the top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # As we want to keep the logit who tipped the cumsum over a threshold we have to shift the indices by 1 to the right
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0

            # Create a mask for the original unsorted logits
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            # Set logits that 
            next_token_logits[indices_to_remove] = -float('Inf')

        # 7. Apply Min-P filtering
        if min_p is not None and 0.0 < min_p < 1.0:
            # IMPORTANT: We calculate probabilities here just for filtering, not for final sampling.
            # We use the logits as they are *after* top-k and top-p have already been applied.
            probs_for_filter = F.softmax(next_token_logits, dim=-1)
            
            # Find the probability of the most likely token
            p_highest = torch.max(probs_for_filter)
            
            # Calculate the cutoff threshold
            threshold = min_p * p_highest
            
            # Create a mask for tokens with probabilities below the threshold
            indices_to_remove = probs_for_filter < threshold
            
            # Apply this mask to the logits, setting them to -infinity
            next_token_logits[indices_to_remove] = -float('Inf')

        
        
        # --- FINAL SAMPLING ---
        
        # 9. Check if after all filtering we have no non-inf logits.
        if torch.isinf(next_token_logits).all():
            print("\n--- [Warning] ---")
            print("All potential tokens were banned by sampling rules.")
            print("Recovering by deterministically picking the token with the highest original logit.")
            # Fallback to the single most likely token from the model's raw output
            # This ignores all sampling rules for this one step to avoid crashing.
            recovery_token = torch.argmax(logits[0, -1, :]).unsqueeze(0)
            next_token = recovery_token
        else:
            # Convert filtered logits to probabilities and sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        # 10. Append chosen token and continue
        x = torch.cat((x, next_token.unsqueeze(0)), dim=1) # Reshape next_token to (1, 1)

    return x

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument('--model_dir', type=str,
                        help='Name of the directory with a saved model, without models/ path.')

    parser.add_argument('--vocab_size', type=int, default=15000, help='Size of the vocabulary')
    
    return parser.parse_args()


if __name__ == '__main__':
    # This block is for standalone testing.
    # Replace with your actual model and tokenizer loading logic.
    try:
        args = get_args()
        PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tokenizer_path = f'{PROJECT_DIR}/data/tokenizer/tokenizer_{args.vocab_size}.json'
        model_dir = f'{PROJECT_DIR}/models/{args.model_dir}' 

        print(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        
        print(f"Loading config from: {model_dir}")
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Initialize model from your config
        model = GPT(
            vocab_size=config.get("vocab_size", 15000), 
            context_length=config.get("context_length"), 
            embedding_dim=config.get("n_embedding"), 
            num_heads=config.get("n_heads"), 
            num_layers=config.get("n_layers"), 
            dropout=config.get("dropout", 0.0), # Use 0.0 for inference
            drop_path_rate=config.get("drop_path_rate", 0.0) # Use 0.0 for inference
        ).to(device)

        # Load the state dictionary
        print("Loading model weights...")
        state_dict_path = os.path.join(model_dir, "best_model.pth")
        
        checkpoint = torch.load(state_dict_path, map_location=device)
        state_dict = checkpoint.get('model', checkpoint) 

        # Remove '_orig_mod.' prefix if model was compiled
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            print("Compiled model detected. Removing '_orig_mod.' prefix from state_dict keys.")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")

        # --- GENERATE TEXT ---
        max_new_tokens = 512
        initial_text = "Кто" # Your initial text
        
        print(f"\n--- Generating text from initial prompt: '{initial_text}' ---")
        generated_tensor = write(
            model=model,
            tokenizer=tokenizer,
            initial_text=initial_text,
            max_new_tokens=max_new_tokens,
            context_length=config["context_length"],
            temperature=1.5,
            min_p=0.2,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

        generated_ids = generated_tensor.squeeze(0).tolist()
        generated_text = decode(tokenizer, generated_ids)
        print("\n--- Generated Output ---")
        print(generated_text)
        print("\n--- End of Generation ---")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your model path, tokenizer path, and config are correct.")



