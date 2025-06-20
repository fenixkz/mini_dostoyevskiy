import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from gpt_causal import GPT
from data import get_dataset, TextDataset
import cProfile
import pstats
import io
import os
from contextlib import nullcontext
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import time
import json
import numpy as np

# Set tokenizer parallelism to true
os.environ["TOKENIZERS_PARALLELISM"] = "true"

CONTINUE_TRAINING = False

def encode(tokenizer: ByteLevelBPETokenizer, text):
    return tokenizer.encode(text).ids

def decode(tokenizer: ByteLevelBPETokenizer, ids):
    return tokenizer.decode(ids)

def write(model, x, max_words, context_length):
    model.eval()
    with torch.no_grad():
        for _ in range(max_words):
            if x.size(1) > context_length:
                logits, loss = model.forward(x[:, -context_length:]) # (B, V)
            else:
                logits, loss = model.forward(x) # (B, V)
            logits = logits[:,-1, :]
            probs = F.softmax(logits, dim=-1)
            next_word = torch.multinomial(probs, num_samples = 1)
            # Concatenate the next word to the input sequence
            x = torch.cat((x, next_word), dim=1)  # (B, T+1)
    model.train()
    return x

def validate_loss(model, val_loader, device):
    model.eval()
    local_loss_sum = 0.0
    local_samples_count = 0
    
    val_pbar = tqdm(total=len(val_loader), desc="Validation", leave=False, position=1)

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            
            # Multiply batch loss by batch size to get total loss for the batch
            local_loss_sum += loss.item() * x.size(0)
            local_samples_count += x.size(0)
            
            val_pbar.set_postfix({'val_loss_rank0': f'{loss.item():.4f}'})
            val_pbar.update(1)
    
    val_pbar.close()

    
    avg_loss = local_loss_sum / local_samples_count
        
    model.train()
    return avg_loss

def train_tokenizer(train_data, vocab_size):
    cleaned_book_filepath = 'data/books_for_tokenizer.txt'

    if not os.path.exists('data'):
        os.makedirs('data')

    with open(cleaned_book_filepath, 'w', encoding='utf-8') as f:
        f.write(train_data) 

    # --- Шаг 2: Инициализация и обучение ByteLevelBPETokenizer ---

    # Инициализируем ByteLevelBPETokenizer
    # add_prefix_space=True - важный параметр, помогает в реконструкции предложений
    #                       (добавляет пробел в начало строки перед токенизацией)
    byte_level_tokenizer = ByteLevelBPETokenizer(add_prefix_space=True, lowercase=False) # lowercase=False, если вы хотите сохранить регистр. Можно поставить True.

    # Обучаем токенизатор
    # vocab_size - желаемый размер словаря
    # min_frequency - минимальная частота встречаемости
    # special_tokens - список специальных токенов
    print(f"Начинаем обучение ByteLevelBPETokenizer на файле: {cleaned_book_filepath}")
    byte_level_tokenizer.train(
        files=[cleaned_book_filepath],
        vocab_size=vocab_size, 
        min_frequency=3,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    print("Обучение ByteLevelBPETokenizer завершено.")

    # Директория, куда мы хотим сохранить токенизатор
    save_directory = "data/tokenizer"
    # Создаем эту директорию, если она не существует
    os.makedirs(save_directory, exist_ok=True)

    # Сохраняем токенизатор в один JSON файл внутри этой директории
    tokenizer_filepath = os.path.join(save_directory, f"tokenizer_{vocab_size}.json")
    byte_level_tokenizer.save(tokenizer_filepath)
    print(f"Токенизатор сохранен в: {tokenizer_filepath}")

    return byte_level_tokenizer

def main():


    # --- DATA PREPARATION (ONLY ON RANK 0) ---
    vocab_size = 15000
    tokenizer_path = f'data/tokenizer/tokenizer_{vocab_size}.json'
    # Get raw text data for training and validation
    train_data_raw, val_data_raw = get_dataset()
    # Train tokenizer if it does not exist 
    if not os.path.exists(tokenizer_path):
        train_tokenizer(train_data_raw, vocab_size)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    # Encode text data
    train_data = encode(tokenizer, train_data_raw)
    val_data = encode(tokenizer, val_data_raw)

    
    print(f'Training set contains: {len(train_data)} tokens')
    print(f'Validation set contains: {len(val_data)} tokens')

    ############ Hyperparameters ################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    context_length = 256
    batch_size = 32
    n_embedding = 128
    n_heads = 4
    n_layers = 4
    dropout = 0.2
    max_epoch = 20000
    eval_interval = 2500 # Not epochs, but iterations in training (within epoch)
    learning_rate = 3e-4
    patience = 5
    no_improve = 0
    current_iter = 0
    use_amp = True if device == 'cuda' else False
    grad_accum_steps = 4
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=pt_dtype) # Context for autocast or none if cpu
    ##############################################
    path = f"models/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(path, exist_ok=True)
    # Save hyperparameters to a json file
    # Create hyperparameters dictionary
    hyperparameters = {
        'device': device,
        'context_length': context_length,
        'batch_size': batch_size,
        'vocab_size': vocab_size,
        'n_embedding': n_embedding,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'dropout': dropout,
        'max_epoch': max_epoch,
        'eval_interval': eval_interval,
        'learning_rate': learning_rate,
        'patience': patience,
        'no_improve': no_improve,
        'current_iter': current_iter,
        'use_amp': use_amp,
        'grad_accum_steps': grad_accum_steps
    }
    
    # Save hyperparameters to a json file
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=2)


    ##################################### Dataloader initialization #############################################
    train_data = torch.tensor(train_data, dtype=torch.long)  # Keep on CPU
    val_data = torch.tensor(val_data, dtype=torch.long)      # Keep on CPU
    train_dataset = TextDataset(train_data, context_length)

    val_dataset = TextDataset(val_data, context_length)
    
    train_loader = DataLoader(
                                train_dataset,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True,
                                shuffle=True 
                             )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)
    ############################################################################################################


    model = GPT(vocab_size=vocab_size, context_length = context_length, 
                embedding_dim = n_embedding, num_heads = n_heads, 
                num_layers = n_layers, dropout=dropout).to(device)
    print("Compiling the model... (this may take a moment)")
    model = torch.compile(model)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    
    optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=learning_rate, betas=(0.9, 0.95), device_type=device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                                                        optimizer,
                                                                        T_0=5000, # Period of lr cycling, per single iteration  
                                                                        T_mult=1,                  
                                                                        eta_min=3e-6               
                                                                    )
    
    enable_scaler = use_amp and dtype == 'float16'
    scaler = GradScaler(enabled=enable_scaler)

    
    best_val_loss = 1e9
    val_loss = 0
    epoch_loss = 1e9
    model.to(device)

    # Create the main training loop with improved progress tracking
    main_pbar = tqdm(total=max_epoch-current_iter, desc="Overall Training Progress", position=0)

    def save_checkpoint():
        checkpoint_data = {
                            'model': model.state_dict(),
                            'optim': optimizer.state_dict(),
                            'current_iter': e,
                            'best_val_loss': best_val_loss,
                            'train_loss': epoch_loss,
                            }
        if scheduler is not None:
            checkpoint_data['scheduler'] = scheduler.state_dict()
        if use_amp:
            checkpoint_data['scaler'] = scaler.state_dict()
        torch.save(checkpoint_data, os.path.join(path, 'checkpoint.pth'))

    def train_epoch(e, epoch_pbar=None):
        nonlocal best_val_loss, no_improve, val_loss

        total_loss = 0
        num_batches = len(train_loader)
        stop_training = False
        # Create epoch progress bar only for rank 0
        if  epoch_pbar is None:
            epoch_pbar = tqdm(total=num_batches, desc=f"Epoch {e+1}", leave=False, position=1)
        
        for i, (x, y) in enumerate(train_loader):
            # Move data to device
            x, y = x.to(device), y.to(device)
            
            # Forward pass with autocasting
            with ctx:
                _, loss = model(x, y)
            
            loss = loss / grad_accum_steps
            total_loss += loss.item() * grad_accum_steps  # Store unscaled loss for logging
            
            # scaler.scale() is a no-op if scaler is disabled
            scaler.scale(loss).backward()
            

            if (i+1) % grad_accum_steps == 0: # Accumulate gradients over multiple iterations and step the optimizer
                # scaler.unscale_ is a no-op if scaler is disabled
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                # scaler.step is a no-op if scaler is disabled
                scaler.step(optimizer)
                # scaler.update is a no-op if scaler is disabled
                scaler.update()
                
                optimizer.zero_grad(set_to_none=True)
                scheduler.step() # Update learning rate each iteration
            
            if i % eval_interval == 0 and i != 0:
                val_loss = validate_loss(model, val_loader, device=device)
                save_checkpoint()
                # Save validation loss as the best if it is the best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                    torch.save(model.state_dict(), os.path.join(path, 'best_model.pth'))
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        main_pbar.set_description(f"Training (Early stopping at epoch {e+1})")
                        main_pbar.close()
                        stop_training = True
                        print(f"Stopping training at epoch {e} with best validation loss: {best_val_loss} and current validation loss {val_loss}")
                        break
            
            # Update epoch progress bar
            if epoch_pbar is not None:
                epoch_pbar.set_postfix({
                        'batch_loss': f'{(loss.item() * grad_accum_steps):.4f}',  # Show unscaled loss
                        'avg_loss': f'{total_loss/(i+1):.4f}',
                        'val_loss': f'{val_loss:.4f}'
                    })
                epoch_pbar.update(1)
                
        if epoch_pbar is not None:
            epoch_pbar.close()
            
        return total_loss / num_batches, stop_training

    if CONTINUE_TRAINING:
        checkpoint = torch.load(os.path.join(path, 'checkpoint.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        current_iter = checkpoint['current_iter']
        best_val_loss = checkpoint['best_val_loss']
        if use_amp and 'scaler' in checkpoint: # Load scaler state
            scaler.load_state_dict(checkpoint['scaler'])

    
    for e in range(current_iter, max_epoch):
        
        # Train one epoch
        epoch_loss, stop_training = train_epoch(e)
        if stop_training:
            break        
        # Update main progress bar and save checkpoint
        current_lr = optimizer.param_groups[0]['lr']
        main_pbar.set_postfix({
            'epoch': f'{e+1}/{max_epoch}',
            'train_loss': f'{epoch_loss:.4f}',
            'lr': f'{current_lr:.2e}'
        })
        main_pbar.update(1)
        
    # Close main progress bar if training completed normally
    main_pbar.close()
                        
if __name__ == '__main__':
    main()