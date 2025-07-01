import os
import time
import json
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from gpt import GPT
from data import LargeTextDataset
from contextlib import nullcontext

# Set tokenizer parallelism to true
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def get_args():
    parser = argparse.ArgumentParser(description="Train a GPT model on Wikipedia.")
    parser.add_argument('--exp_dir', type=str, default=f"models/{time.strftime('%Y-%m-%d_%H-%M-%S')}",
                        help='Directory to save experiment artifacts (checkpoints, logs).')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to an experiment directory to resume training from.')
    
    # Model Hyperparameters
    parser.add_argument('--context_length', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=15000)
    parser.add_argument('--n_embedding', type=int, default=400)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size per GPU.")
    parser.add_argument('--max_epochs', type=int, default=20000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--scheduler_lr_cycling', type=int, default=1000)

    # Validation & Checkpointing
    parser.add_argument('--eval_interval', type=int, default=5000, help="Validate every N optimizer steps.")
    parser.add_argument('--eval_iters', type=int, default=1000, help="Total validation batches to run across all GPUs.")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience.")

    # DDP & Hardware
    parser.add_argument('--num_workers', type=int, default=16)
    
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
    ''' 
    A function to validate the model on the validation set.
    '''
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
        
        # Multiply batch loss by batch size to get total loss for the batch
        local_loss_sum += loss.item() * x.size(0)
        local_samples_count += x.size(0)
        
        if is_main_process: val_pbar.update(1)
    
    if is_main_process: val_pbar.close()

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
    config = get_args()
    is_ddp, local_rank, device = setup_ddp()
    
    # Is it a main process (rank 0 or a single process)
    is_main_process = not is_ddp or dist.get_rank() == 0


    # --- SETUP (LOGGING, PATHS, ETC.) ---
    # Determine experiment path: resume or new
    exp_path = config.resume if config.resume else config.exp_dir
    writer = None
    if is_main_process:
        os.makedirs(exp_path, exist_ok=True)
        # SOTA logging with TensorBoard
        writer = SummaryWriter(log_dir=os.path.join(exp_path, 'logs'))
        print(f"Experiment artifacts will be saved in: {exp_path}")
    # --- SETUP (LOGGING, PATHS, ETC.) ---


    # --- DATA PREPARATION  ---
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_bin_path = f'{project_dir}/data/wiki/train.bin'
    val_bin_path = f'{project_dir}/data/wiki/val.bin'

    train_dataset = LargeTextDataset(train_bin_path, config.context_length)
    val_dataset = LargeTextDataset(val_bin_path, config.context_length)

    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=config.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler, num_workers=config.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    # --- DATA PREPARATION ---

    # --- MODEL, OPTIMIZER, SCHEDULER ---
    # All processes initialize the model, DDP will synchronize it
    model = GPT(vocab_size=config.vocab_size, context_length=config.context_length, 
                embedding_dim=config.n_embedding, num_heads=config.n_heads, 
                num_layers=config.n_layers, dropout=config.dropout, 
                drop_path_rate=config.drop_path_rate).to(device)
    if is_main_process:
        print("Compiling the model... (this may take a moment)")
    model = torch.compile(model)
    if is_main_process:
        print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    optimizer = model.configure_optimizers(weight_decay=config.weight_decay, learning_rate=config.learning_rate, 
                                           betas=(0.9, 0.95), device_type=device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                                                        optimizer,
                                                                        T_0=config.scheduler_lr_cycling, # Period of lr cycling, per single iteration  
                                                                        T_mult=1,                  
                                                                        eta_min=3e-6               
                                                                    )
    
    # Mixed Precision Scaler
    dtype = 'float16' # Default for older GPUs
    if torch.cuda.is_available():
        # V100 is 7.0, A100 is 8.0. bfloat16 is supported on 8.0+
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            dtype = 'bfloat16'
            if is_main_process: print("Using bfloat16 for Ampere or newer GPU.")
        else:
            if is_main_process: print("Using float16 for pre-Ampere GPU.")

    pt_dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type='cuda', dtype=pt_dtype)
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # Wrap model with DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
    # --- MODEL, OPTIMIZER, SCHEDULER ---


    # --- CHECKPOINTING & RESUME LOGIC ---
    start_epoch = 0
    optimizer_step = 0
    best_val_loss = float('inf')
    
    if config.resume:
        ckpt_path = os.path.join(exp_path, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            # All ranks load the checkpoint to stay in sync
            loc = f'cuda:{local_rank}'
            checkpoint = torch.load(ckpt_path, map_location=loc)
            state_dict = checkpoint.get('model', checkpoint)
            if is_ddp:
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch'] + 1
            optimizer_step = checkpoint['optimizer_step']
            best_val_loss = checkpoint['best_val_loss']
            if is_main_process:
                print(f"Resumed training from epoch {start_epoch} at step {optimizer_step}.")
        elif is_main_process:
            print(f"Resume specified, but checkpoint not found at {ckpt_path}. Starting from scratch.")

    if is_main_process:
        # Save config only on rank 0 after potential resume logic
        with open(os.path.join(exp_path, 'config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2)
    # --- CHECKPOINTING & RESUME LOGIC ---

    # --- TRAINING LOOP ---
    no_improve_count = 0
    for epoch in range(start_epoch, config.max_epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        
        model.train()
        
        epoch_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config.max_epochs}", leave=False) if is_main_process else None
            
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass under autocast context
            with ctx:
                _, loss = model(x, y)
                loss = loss / config.grad_accum_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Optimizer step (occurs every grad_accum_steps)
            if (i + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((model.module.parameters() if is_ddp else model.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1

                # Logging (only for rank 0)
                if is_main_process and writer:
                    writer.add_scalar('Loss/train_batch', loss.item() * config.grad_accum_steps, optimizer_step)
                    writer.add_scalar('LR/learning_rate', scheduler.get_last_lr()[0], optimizer_step)

                # Validation run
                if optimizer_step > 0 and optimizer_step % config.eval_interval == 0:
                    if is_ddp:
                        val_sampler.set_epoch(optimizer_step) # Use step to ensure new shuffle
                    val_loss = validate(model, val_loader, device, config, is_ddp)
                    
                    if is_main_process:
                        print(f"\nStep {optimizer_step}: Validation Loss: {val_loss:.4f}")
                        if writer:
                            writer.add_scalar('Loss/validation', val_loss, optimizer_step)
                        
                        # Checkpointing and Early Stopping Logic
                        is_best = val_loss < best_val_loss
                        if is_best:
                            best_val_loss = val_loss
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                        model_to_save = model.module if is_ddp else model
                        # Save checkpoint
                        checkpoint_data = {
                            'model': model_to_save.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict(),
                            'epoch': epoch,
                            'optimizer_step': optimizer_step,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        torch.save(checkpoint_data, os.path.join(exp_path, 'checkpoint.pth'))
                        if is_best:
                            torch.save(checkpoint_data, os.path.join(exp_path, 'best_model.pth'))
                            print(f"New best model saved with val_loss: {best_val_loss:.4f}")
                            # Save the best loss in a text file for better visualization
                            with open(os.path.join(exp_path, 'best_val_loss.txt'), 'w') as f:
                                f.write(f"{best_val_loss:.6f}\n")
                    
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
        print("Training finished.")
        if writer: writer.close()

    if is_ddp: cleanup_ddp()

if __name__ == '__main__':
    main()