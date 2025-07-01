import os
import time
import json
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from gpt import GPT
from data import TextDataset, get_dataset
from contextlib import nullcontext
from tokenizer_utils import RuTokenizer
import inspect

# Set tokenizer parallelism to true
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune a GPT model on a custom dataset.")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory where the pre-trained model is stored (folder name, not full path).')
    parser.add_argument('--save_dir', type=str, default=f"models/fine_tune/{time.strftime('%Y-%m-%d_%H-%M-%S')}",
                        help="Directory to save the fine-tuned model.")
    parser.add_argument('--resume', type=bool, default=False, help="To resume training from some checkpoint")

    # Regularization for fine-tuning
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate for fine-tuning.")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="Stochastic depth rate.")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="Weight decay for the optimizer.")

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size per GPU.")
    parser.add_argument('--max_epochs', type=int, default=10, help="Maximum number of epochs.")
    parser.add_argument('--learning_rate', type=float, default=3e-5, help="Learning rate for fine-tuning.")
    parser.add_argument('--grad_accum_steps', type=int, default=4, help="Steps for gradient accumulation.")
    
    # Validation & Checkpointing
    parser.add_argument('--eval_interval', type=int, default=200, help="Validate every N optimizer steps.")
    parser.add_argument('--eval_iters', type=int, default=1000, help="Total validation batches to run.") # FIX: Added this argument
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience.")

    # DDP & Hardware
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader.")
    
    # Fine-tuning Hyperparameters
    parser.add_argument('--layers_to_freeze', type=float, default=0.5, help="Percentage of layers to freeze, 0-1 range.")

    return parser.parse_args()

def setup_ddp():
    is_ddp = 'WORLD_SIZE' in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        print(f"Started DDP on rank {dist.get_rank()} for GPU {local_rank}.")
        return True, local_rank, device
    else:
        print("DDP not detected. Running in single-process mode.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return False, 0, device

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

@torch.inference_mode()
def validate(model, val_loader, device, config, is_ddp):
    model.eval()
    
    if is_ddp:
        world_size = dist.get_world_size()
        steps_per_gpu = config.eval_iters // world_size
        if config.eval_iters % world_size != 0 and dist.get_rank() == 0:
            print(f"Warning: eval_iters ({config.eval_iters}) isn't divisible by world_size ({world_size}).")
    else:
        steps_per_gpu = config.eval_iters

    local_loss_sum = 0.0
    local_samples_count = 0
    
    is_main_process = not is_ddp or dist.get_rank() == 0
    val_pbar = tqdm(total=steps_per_gpu, desc="Validation", leave=False, position=1) if is_main_process else None

    for i, (x, y) in enumerate(val_loader):
        if i >= steps_per_gpu:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        
        local_loss_sum += loss.item() * x.size(0)
        local_samples_count += x.size(0)
        
        if is_main_process:
            val_pbar.update(1)
    
    if is_main_process: 
        val_pbar.close()

    if is_ddp:
        totals = torch.tensor([local_loss_sum, local_samples_count], dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        global_loss_sum, global_samples_count = totals[0].item(), totals[1].item()
    else:
        global_loss_sum, global_samples_count = local_loss_sum, local_samples_count
    
    model.train()
    if global_samples_count == 0: return float('inf')
    return global_loss_sum / global_samples_count

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = get_args()
    is_ddp, local_rank, device = setup_ddp()
    
    is_main_process = not is_ddp or dist.get_rank() == 0

    # --- SETUP PATHS, LOGGING ---
    load_dir = os.path.join(project_dir, 'models', config.model_dir)
    save_dir = os.path.join(project_dir, config.save_dir)
    writer = None
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
        print(f"Loading pre-trained model from: {load_dir}")
        print(f"Fine-tuned model will be saved in: {save_dir}")

    # --- LOAD CONFIG AND TOKENIZER FROM PRE-TRAINED MODEL ---
    with open(os.path.join(load_dir, "config.json"), "r", encoding="utf-8") as f:
        config_json = json.load(f)
    
    tokenizer_path = os.path.join(project_dir, 'data', 'tokenizer', f'tokenizer_{config_json["vocab_size"]}.json')
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # --- SAVE THE NEW FINE-TUNING CONFIGURATION ---
    if is_main_process:
        # This dictionary holds all the hyperparameters for this specific fine-tuning run
        finetune_config = {
            "base_model": config.model_dir,
            "dropout": config.dropout,
            "drop_path_rate": config.drop_path_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "max_epochs": config.max_epochs,
            "learning_rate": config.learning_rate,
            "grad_accum_steps": config.grad_accum_steps,
            "eval_interval": config.eval_interval,
            "eval_iters": config.eval_iters,
            "patience": config.patience,
            "layers_to_freeze": config.layers_to_freeze
        }
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(finetune_config, f, indent=2)
        print("Fine-tuning configuration saved.")

    # --- DATA PREPARATION (Fine-tuning dataset) ---
    train_data_raw, val_data_raw = get_dataset() 
    # Note: Reusing the pre-trained tokenizer is correct!
    train_data = torch.tensor(RuTokenizer.encode(tokenizer, train_data_raw))
    val_data = torch.tensor(RuTokenizer.encode(tokenizer, val_data_raw))

    train_dataset = TextDataset(train_data, config_json["context_length"])
    val_dataset = TextDataset(val_data, config_json["context_length"])
    
    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=config.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler, num_workers=config.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    # --- MODEL INITIALIZATION AND LOADING ---
    model = GPT(
        vocab_size=config_json["vocab_size"], 
        context_length=config_json["context_length"], 
        embedding_dim=config_json["n_embedding"], 
        num_heads=config_json["n_heads"], 
        num_layers=config_json["n_layers"], 
        dropout=config.dropout, # Use new dropout for fine-tuning
        drop_path_rate=config.drop_path_rate # Use new drop_path for fine-tuning
    ).to(device)

    # --- LOAD PRE-TRAINED WEIGHTS ---
    print("Loading pre-trained model weights...")
    ckpt_path = os.path.join(load_dir, 'best_model.pth')
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get('model', checkpoint) # Flexible loading
    
    # Clean state_dict if it was from a compiled model
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    
    model.load_state_dict(state_dict)
    print("Pre-trained weights loaded successfully.")
    
    # --- FREEZING LAYERS FOR FINE-TUNING ---

    print("Freezing model layers for fine-tuning...")

    # 1. Freeze the embedding layers
    for param in model.embedding_layer.parameters():
        param.requires_grad = False
    for param in model.position_embedding.parameters():
        param.requires_grad = False

    # 2. Freeze some part of layers (blocks). 
    num_layers_to_freeze = int(model.num_layers * config.layers_to_freeze)
    for i in range(num_layers_to_freeze):
        for param in model.blocks[i].parameters():
            param.requires_grad = False

    print(f"Froze token/position embeddings and the first {num_layers_to_freeze} Transformer blocks.")

    # --- CREATE OPTIMIZER WITH ONLY TRAINABLE PARAMETERS ---
    # This is a crucial step! We must create a new optimizer that only sees
    # the parameters we left unfrozen (those with requires_grad=True).

    # Create a list of parameters that still require gradients
    params_to_update = []
    print("Parameters to be fine-tuned:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print(f"\t{name}")

    if is_main_process:
        print("Compiling the model... (this may take a moment)")
    model = torch.compile(model)
    
    # --- OPTIMIZER, SCALER, DDP WRAPPER ---
    print("Creating optimizer for fine-tuning...")

    # Separate the trainable parameters into those with and without weight decay
    decay_params = [p for p in params_to_update if p.dim() >= 2]
    nodecay_params = [p for p in params_to_update if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"Created optimizer with {len(decay_params)} decayed parameter tensors ({num_decay_params:,} parameters)")
    print(f"Created optimizer with {len(nodecay_params)} non-decayed parameter tensors ({num_nodecay_params:,} parameters)")

    # Create AdamW optimizer and use the fused version if available
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device.type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95), **extra_args)
    print(f"Using fused AdamW: {use_fused}")
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    ctx = torch.amp.autocast(device_type='cuda', dtype={'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype])
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    # --- TRAINING LOOP ---
    optimizer_step = 0
    best_val_loss = float('inf')
    no_improve_count = 0
    for epoch in range(config.max_epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        model.train()
        
        epoch_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config.max_epochs}", leave=False) if is_main_process else None
            
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            with ctx:
                _, loss = model(x, y)
                loss = loss / config.grad_accum_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((model.module.parameters() if is_ddp else model.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if is_main_process and writer:
                    writer.add_scalar('Loss/train_batch', loss.item() * config.grad_accum_steps, optimizer_step)

                if optimizer_step > 0 and optimizer_step % config.eval_interval == 0:
                    val_loss = validate(model, val_loader, device, config, is_ddp)
                    
                    if is_main_process:
                        print(f"\nStep {optimizer_step}: Validation Loss: {val_loss:.4f}")
                        if writer: writer.add_scalar('Loss/validation', val_loss, optimizer_step)
                        
                        is_best = val_loss < best_val_loss
                        if is_best:
                            best_val_loss = val_loss
                            no_improve_count = 0
                            # Save the best model
                            model_to_save = model.module if is_ddp else model
                            torch.save({'model': model_to_save.state_dict()}, os.path.join(save_dir, 'best_model.pth'))
                            # Save the best val loss in the .txt file
                            with open(os.path.join(save_dir, "best_val_loss.txt"), "w") as f:
                                f.write(str(best_val_loss))
                            print(f"New best model saved with val_loss: {best_val_loss:.4f}")
                        else:
                            no_improve_count += 1
                    
                    # Synchronize early stopping decision
                    if is_ddp:
                        stop_tensor = torch.tensor([1 if no_improve_count >= config.patience else 0], device=device)
                        dist.broadcast(stop_tensor, src=0)
                        if stop_tensor.item() == 1: break
                    elif no_improve_count >= config.patience:
                        break
            
            if is_main_process: epoch_pbar.update(1)
        
        # Check for early stopping after batch loop
        if no_improve_count >= config.patience:
            if is_main_process: print(f"Early stopping triggered after {config.patience} validations with no improvement.")
            break
            
        if is_main_process: epoch_pbar.close()

    if is_main_process:
        print("Fine-tuning finished.")
        if writer: writer.close()
    
    if is_ddp:
        cleanup_ddp()

if __name__ == '__main__':
    main()