import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from gpt import GPT
from data import get_dataset, TextDataset
import cProfile
import pstats
import io
import os
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
os.environ["TOKENIZERS_PARALLELISM"] = "true"

CONTINUE_TRAINING = False

def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    # Get the rank and local rank from environment variables set by torchrun
    # local_rank is the GPU ID for this specific process
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    print(f"Started DDP process on rank {dist.get_rank()} for GPU {local_rank}.")



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
    total_val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            # When calling validate_loss, you pass the ddp_model.
            # DDP transparently handles the forward pass on each GPU.
            _, loss = model(x, y)
            total_val_loss += loss.item()
    model.train()
    return total_val_loss / len(val_loader)

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
    setup_ddp()

    # Get the local rank for device placement
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    vocab_size = 10000
    train_data, val_data = get_dataset()
    if dist.get_rank() == 0:
        # This code will only be run by the main process
        if not os.path.exists(f'data/tokenizer/tokenizer_{vocab_size}.json'):
            # Ensure the data directory exists before any process needs it
            os.makedirs('data/tokenizer', exist_ok=True)
            train_tokenizer(train_data, vocab_size)

    # All processes wait here until rank 0 has finished preparing the data
    dist.barrier()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Now, all processes can safely load the tokenizer file
    tokenizer = Tokenizer.from_file(f'data/tokenizer/tokenizer_{vocab_size}.json')
    train_data = encode(tokenizer, train_data)
    val_data = encode(tokenizer, val_data)
    
    print(f'Training set contains: {len(train_data)} tokens')
    print(f'Validation set contains: {len(val_data)} tokens')

    #### Hyperparameters
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    context_length = 256
    batch_size = 128
    vocab_size = tokenizer.get_vocab_size()
    n_embedding = 128
    n_heads = 4
    n_layers = 8
    dropout = 0.1
    max_epoch = 20000
    eval_interval = 500
    eval_iters = 1
    learning_rate = 3e-4
    patience = 50
    no_improve = 0
    current_iter = 0
    use_amp = True if device == 'cuda' else False
    grad_accum_steps = 4
    ####

    train_data = torch.tensor(train_data, dtype=torch.long)  # Keep on CPU
    val_data = torch.tensor(val_data, dtype=torch.long)      # Keep on CPU
    train_dataset = TextDataset(train_data, context_length)
    train_sampler = DistributedSampler(train_dataset)

    val_dataset = TextDataset(val_data, context_length)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
                                train_dataset,
                                batch_size=batch_size,
                                sampler=train_sampler,
                                num_workers=16,
                                pin_memory=True,
                                shuffle=False # shuffle=False is required when using a sampler
                            )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=16)

    model = GPT(vocab_size=vocab_size, context_length = context_length, 
                embedding_dim = n_embedding, num_heads = n_heads, 
                num_layers = n_layers, dropout=dropout).to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2500,  
        T_mult=1,                  
        eta_min=3e-6               
    )
    if use_amp:
        scaler = GradScaler()
    val_losses = []
    best_val_loss = 1e9
    
    model.to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    def train_epoch(e):
        total_loss = 0 
        num_batches = len(train_loader)
        print(f"Epoch {e+1}/{max_epoch}, Batches: {num_batches}")
        for x,y in train_loader:
            # Move data to device
            x, y = x.to(device), y.to(device)
            # Forward pass with autocasting
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp): # autocast marks regions for AMP
                _, loss = ddp_model(x, y)
            loss = loss / grad_accum_steps 
            total_loss += loss.item() # Accumulate unscaled loss for logging
            if (e+1)% grad_accum_steps == 0:
                if use_amp:
                    scaler.scale(loss).backward()  # Scales loss, calls backward on scaled loss
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients after backward, before unscaling
                    scaler.step(optimizer)         # Unscales gradients and calls optimizer.step()
                    scaler.update()                # Updates the scale for next iteration
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        return total_loss / num_batches

    if not os.path.exists('models'):
        os.makedirs('models')
    path = f'models/checkpoint_{n_embedding}_{context_length}_{n_heads}_{n_layers}.pth'

    if CONTINUE_TRAINING:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        current_iter = checkpoint['current_iter']
        best_val_loss = checkpoint['best_val_loss']
        if use_amp and 'scaler' in checkpoint: # Load scaler state
            scaler.load_state_dict(checkpoint['scaler'])

    rank = dist.get_rank()

    # Create the main training loop iterable
    training_range = range(current_iter, max_epoch)

    # Wrap it in tqdm ONLY for the main process (rank 0)
    if rank == 0:
        training_range = tqdm(training_range, desc="Training")

    for e in training_range:
        # Set epoch for distributed samplers
        train_sampler.set_epoch(e)
        val_sampler.set_epoch(e)
        
        loss = train_epoch(e)
        scheduler.step()
        if e % eval_interval == 0:
            val_loss = validate_loss(ddp_model, val_loader, n_epoch=eval_iters, device=device)
            # Update the tqdm progress bar with the current loss
            val_losses.append(val_loss)
            # Check if learning rate was updated
            current_lr = optimizer.param_groups[0]['lr'] 
            training_range.set_postfix({"val_loss": val_loss, "lr": current_lr})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                # Only save checkpoint on rank 0
                if dist.get_rank() == 0:
                    checkpoint_data = {
                        'model': ddp_model.module.state_dict(),
                        'optim': optimizer.state_dict(),
                        'current_iter': e,
                        'best_val_loss': best_val_loss,
                        'tokenizer_vocab_size': vocab_size
                    }
                    if scheduler is not None:
                        checkpoint_data['scheduler'] = scheduler.state_dict()
                    if use_amp:
                        checkpoint_data['scaler'] = scaler.state_dict()
                    torch.save(checkpoint_data, path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {e}")
                    break
                        


    # Only do inference on rank 0
    if dist.get_rank() == 0:
        print(f"Final validation loss: {val_loss.item()}")
        print(f"Best validation loss: {best_val_loss}")

        # load the best model
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        max_words = 250
        text = torch.tensor([encode(tokenizer, "Я")], dtype=torch.long).view(1,1).to(device)
        print(''.join(decode(tokenizer, write(model, text, max_words, context_length)[0].tolist())))
    
    # Cleanup DDP
    dist.destroy_process_group()

    # plot the val lossses
    # import matplotlib.pyplot as plt
    # plt.plot(val_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Validation Loss')
    # plt.title('Validation Loss evaluated every 500 iterations')
    # plt.show()


if __name__ == '__main__':
    main()