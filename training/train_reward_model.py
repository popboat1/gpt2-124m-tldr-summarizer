import os
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from src.models.model import GPTConfig
from src.models.reward_model import GPTRewardModel
from src.utils.dataloader_reward import DataLoaderReward

# --- 1. DDP SETUP ---
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    master_process = True

if master_process:
    print(f"Spawning Reward Model Trainer on {device}")

# --- 2. HYPERPARAMETERS ---
# We use a lower learning rate for RM training so we don't destroy the SFT linguistic weights
total_batch_size_pairs = 64
B = 2 # Pairs per GPU (4 total sequences of 1024)
T = 1024 
grad_accum_steps = total_batch_size_pairs // (B * ddp_world_size)

max_lr = 1e-5
min_lr = 1e-6
warmup_steps = 100
max_steps = 1500 # Approx 1 epoch over the 92k training pairs

# --- 3. MODEL INITIALIZATION & SFT WEIGHT INJECTION ---
config = GPTConfig(vocab_size=50304)
raw_model = GPTRewardModel(config)

# Load SFT Weights
sft_checkpoint_path = "log/model_best.pt"
if os.path.exists(sft_checkpoint_path):
    if master_process: print(f"Injecting SFT Brain from {sft_checkpoint_path}...")
    checkpoint = torch.load(sft_checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    # Filter out the lm_head from the SFT model (since RM doesn't have one)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("lm_head.")}
    
    # Load into Reward Model (strict=False allows the new v_head to remain randomly initialized)
    missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)
    if master_process: 
        print(f"Expected missing keys (v_head): {missing}")
else:
    if master_process: print("WARNING: No SFT checkpoint found. Starting from scratch!")

raw_model.to(device)
if ddp: 
    model = DDP(raw_model, device_ids=[ddp_local_rank])
else:
    model = raw_model

# --- 4. OPTIMIZER & DATALOADERS ---
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)
train_loader = DataLoaderReward(B, T, ddp_rank, ddp_world_size, 'train', master_process)
val_loader = DataLoaderReward(B, T, ddp_rank, ddp_world_size, 'val', master_process)

def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# --- 5. TRAINING LOOP ---
log_dir = "log_rm"
os.makedirs(log_dir, exist_ok=True)
if master_process:
    open(os.path.join(log_dir, "rm_log.txt"), "w").close()

for step in range(max_steps):
    t0 = time.time()
    
    # Validation Loop
    if step % 100 == 0 or step == max_steps - 1:
        model.eval()
        val_loss_accum = 0.0
        val_acc_accum = 0.0
        val_steps = 20
        with torch.no_grad():
            for _ in range(val_steps):
                x_chosen, x_rejected = val_loader.next_batch()
                x_chosen, x_rejected = x_chosen.to(device), x_rejected.to(device)
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Forward pass both sequences
                    r_chosen = model(x_chosen)
                    r_rejected = model(x_rejected)
                    
                    # Pairwise Ranking Loss
                    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
                    
                    # Calculate Accuracy (Did the chosen summary get a higher score?)
                    acc = (r_chosen > r_rejected).float().mean()
                
                val_loss_accum += loss.detach() / val_steps
                val_acc_accum += acc.detach() / val_steps
                
        if ddp:
            torch.distributed.all_reduce(val_loss_accum, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(val_acc_accum, op=torch.distributed.ReduceOp.AVG)
            
        if master_process:
            print(f"step {step:4d} | val loss: {val_loss_accum.item():.4f} | val acc: {val_acc_accum.item()*100:.2f}%")
            with open(os.path.join(log_dir, "rm_log.txt"), "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f} {val_acc_accum.item():.4f}\n")
                
            checkpoint = {
                'step': step,
                'model_state_dict': raw_model.state_dict(),
                'val_loss': val_loss_accum.item(),
                'val_acc': val_acc_accum.item()
            }
            torch.save(checkpoint, os.path.join(log_dir, f"rm_latest.pt"))

    # Training Loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    acc_accum = 0.0
    
    for micro_step in range(grad_accum_steps):
        x_chosen, x_rejected = train_loader.next_batch()
        x_chosen, x_rejected = x_chosen.to(device), x_rejected.to(device)
        
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            r_chosen = model(x_chosen)
            r_rejected = model(x_rejected)
            loss = -F.logsigmoid(r_chosen - r_rejected).mean()
            loss = loss / grad_accum_steps
            acc = (r_chosen > r_rejected).float().mean() / grad_accum_steps
            
        loss_accum += loss.detach()
        acc_accum += acc.detach()
        loss.backward()

    if ddp:
        torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(acc_accum, op=torch.distributed.ReduceOp.AVG)
        
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    
    t1 = time.time()
    dt = t1 - t0
    
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.4f} | acc: {acc_accum.item()*100:.2f}% | lr: {lr:.4e} | norm: {norm:.2f} | dt: {dt*1000:.2f}ms")
        with open(os.path.join(log_dir, "rm_log.txt"), "a") as f:
            f.write(f"{step} train {loss_accum.item():.4f} {acc_accum.item():.4f}\n")

if ddp:
    destroy_process_group()