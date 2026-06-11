import os
import time
import math
import torch
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken

# Removed hellaswag import as it is primarily for general pre-training evaluation
from src.models.model import GPT, GPTConfig
from src.utils.dataloader import DataLoaderMasked

# set up DDP (distributed data parallel).
ddp = int(os.environ.get('RANK', -1)) != -1 
if ddp:
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
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
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

# --- SFT BATCH SIZE ---
# Reduced from 524,288 to 65,536 for smaller dataset fine-tuning
total_batch_size = 65536 
B = 2
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B*T*ddp_world_size"
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# NOTE: Using the new masked dataloader for proper loss computation
train_loader = DataLoaderMasked(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train', master_process=master_process)
val_loader = DataLoaderMasked(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val', master_process=master_process)

torch.set_float32_matmul_precision('high')

# --- LOAD PRE-TRAINED WEIGHTS ---
# We MUST load the 124M weights so we are fine-tuning, not starting from scratch
# --- LOAD CUSTOM LOCAL WEIGHTS ---
if master_process:
    print("Loading custom 10B Fineweb-Edu weights for fine-tuning...")

# 1. Initialize the blank architecture using YOUR padded vocab size
model = GPT(GPTConfig(vocab_size=50304))

# 2. Load your saved local checkpoint from the pre-training log directory
checkpoint_path = "/kaggle/input/notebooks/mikellmf/gpt-2-training/log/model_latest.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint['model_state_dict']

# 3. Strip the "_orig_mod." prefix that torch.compile adds to saved weights
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

# 4. Inject your trained weights into the model
model.load_state_dict(state_dict)
model.to(device)

use_compile = True
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# --- SFT HYPERPARAMETERS ---
max_lr = 5e-5      
min_lr = max_lr * 0.1
warmup_steps = 360   # 10% of total training path
max_steps = 3616     # 2 complete epochs over 116k samples

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (max_lr - min_lr)

raw_model = model.module if ddp else model
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type, master_process=master_process)
scaler = torch.amp.GradScaler()

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"sft_log.txt")

start_step = 0
resume_run = False 

best_val_loss = float('inf')

if resume_run:
    checkpoint_path = os.path.join(log_dir, "model_latest.pt")
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_step = checkpoint['step'] + 1

        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
            

for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # evaluate validation loss
    if (step % 50 == 0) or last_step:
        model.eval()
        val_loader.reset()
        with torch.inference_mode():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                current_val_loss = val_loss_accum.item()
                print(f"validation loss: {current_val_loss:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {current_val_loss:.6f}\n")

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    print(f"New best validation loss: {best_val_loss:.4f}! Saving model_best.pt...")
                    best_checkpoint = {
                        'step': step,
                        'model_state_dict': raw_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'val_loss': best_val_loss,
                    }
                    torch.save(best_checkpoint, os.path.join(log_dir, "model_best.pt"))

    # Generate from the model
    if ((step > 0 and step % 50 == 0) or last_step):
        gen_model = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
        gen_model.eval()
        num_return_sequences = 1
        max_length = 200
        
        sample_post = "My roommate keeps eating my food without asking. I've tried talking to him, but he says it's not a big deal because we split the grocery bill. But I buy specific snacks for my lunches, and they are always gone when I need them. It's driving me crazy and I don't know how to confront him again without starting a huge fight."
        prompt = f"Post: {sample_post}\n\nTL;DR: "
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        
        if master_process:
            print("\n--- GENERATION TEST ---")
            
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    logits, _ = gen_model(xgen) 
                logits = logits[:, -1, :] 
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) 
                xcol = torch.gather(topk_indices, -1, ix) 
                xgen = torch.cat((xgen, xcol), dim=1)
                
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            if master_process:
                print(f"{decoded}")
                print("-----------------------\n")
    
    # Save Checkpoint every 250 steps
    if (step > 0 and step % 250 == 0 or last_step) and master_process:
        checkpoint = {
            'step': step,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(log_dir, f"model_latest.pt"))
        print(f"Saved checkpoint at step {step}")

    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x, y)
        loss /= grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        scaler.scale(loss).backward()
        
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
        
    scaler.step(optimizer)
    scaler.update()
    
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) 
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    
    if master_process:
        print(f'step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | time: {dt:.2f}s | tok/sec: {tokens_per_sec:.2f}')
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
            
if ddp:
    destroy_process_group()