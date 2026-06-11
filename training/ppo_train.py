import os
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from src.models.model import GPT, GPTConfig
from src.models.reward_model import GPTRewardModel
from src.utils.dataloader_ppo import DataLoaderPPO

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

# --- 2. HYPERPARAMETERS ---
B = 4 # Batch size per GPU
gen_len = 64 # Max tokens to generate for the summary
max_steps = 1000
lr = 1e-6 # VERY small learning rate for PPO
kl_beta = 0.1 # How strictly to punish grammar destruction
clip_ratio = 0.2 # PPO clipping bound to prevent catastrophic updates

# --- 3. MODEL ORCHESTRATION ---
config = GPTConfig(vocab_size=50304)

# 1. Load the SFT dictionary and strip the compiled prefix
if master_process: print("Loading SFT Base Weights...")
checkpoint = torch.load("log/model_best.pt", map_location='cpu')
state_dict = checkpoint['model_state_dict']

unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

# 2. Inject stripped weights into the Actor
if master_process: print("Loading Actor Model (Trainable)...")
actor = GPT(config)
actor.load_state_dict(state_dict)
actor.to(device)

# 3. Inject stripped weights into the Reference Model
if master_process: print("Loading Reference Model (Frozen)...")
ref_model = GPT(config)
ref_model.load_state_dict(state_dict)
ref_model.to(device)
ref_model.eval() # Freeze

# 4. Load the Critic
if master_process: print("Loading Critic / Reward Model (Frozen)...")
critic = GPTRewardModel(config)
critic_state = torch.load("log_rm/rm_latest.pt", map_location='cpu')['model_state_dict']
critic.load_state_dict(critic_state, strict=False)
critic.to(device)
critic.eval() # Freeze

if ddp: 
    actor = DDP(actor, device_ids=[ddp_local_rank])

optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=0.01)
train_loader = DataLoaderPPO(B, ddp_rank, ddp_world_size, 'train', master_process)

def get_log_probs(logits, tokens):
    # Extracts the log probability of the exact tokens that were generated
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, 2, tokens.unsqueeze(2)).squeeze(2)

# --- 4. THE PPO LOOP ---
if master_process:
    os.makedirs("log_ppo", exist_ok=True)
    open("log_ppo/ppo_log.txt", "w").close()

for step in range(max_steps):
    t0 = time.time()
    
    # 1. Fetch Prompts (Shape: B, 512)
    prompts = train_loader.next_batch().to(device)
    
    # ==========================================
    # PHASE 1: ROLLOUT (GENERATION)
    # ==========================================
    actor.eval() # Eval mode for clean generation
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen_seqs = prompts
            # Autoregressive generation loop
            for _ in range(gen_len):
                logits, _ = actor(gen_seqs)
                next_token_logits = logits[:, -1, :]
                # Greedily sample for stability in simple PPO
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                gen_seqs = torch.cat((gen_seqs, next_token), dim=1)
                
            # gen_seqs now contains [Prompt + Generated Summary]
            generated_tokens_only = gen_seqs[:, prompts.size(1):]
    
    # ==========================================
    # PHASE 2: SCORING & KL PENALTY
    # ==========================================
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # A. Get Critic Score
            reward_scores = critic(gen_seqs) # Shape: (B)
            
            # B. Get Log-Probs from Frozen Reference Model
            ref_logits, _ = ref_model(gen_seqs)
            ref_log_probs = get_log_probs(ref_logits[:, prompts.size(1)-1:-1, :], generated_tokens_only)
            
            # C. Get Log-Probs from Actor (Old Policy before update)
            old_actor_logits, _ = actor(gen_seqs)
            old_log_probs = get_log_probs(old_actor_logits[:, prompts.size(1)-1:-1, :], generated_tokens_only)
            
            # D. Calculate KL Divergence (Did the actor destroy its English?)
            kl_div = torch.clamp(old_log_probs - ref_log_probs, min=0.0)
            kl_penalty = kl_div.sum(dim=1) # Sum penalty across generated tokens
            
            # E. Final Advantage = Base Reward - (Beta * KL Penalty)
            advantages = reward_scores - (kl_beta * kl_penalty)
            
            # Normalize Advantages across the batch
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            normalized_advantages = (advantages - adv_mean) / adv_std

    # ==========================================
    # PHASE 3: PPO CLIPPED OPTIMIZATION
    # ==========================================
    actor.train()
    optimizer.zero_grad()
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Forward pass through the Actor again to get gradients
        new_logits, _ = actor(gen_seqs)
        new_log_probs = get_log_probs(new_logits[:, prompts.size(1)-1:-1, :], generated_tokens_only)
        
        # Calculate PPO Ratio
        ratio = torch.exp(new_log_probs - old_log_probs) # Shape: (B, gen_len)
        
        # Expand advantages to match generated token shape
        adv_expanded = normalized_advantages.unsqueeze(1).expand_as(ratio)
        
        # Surrogate Loss Objective
        surr1 = ratio * adv_expanded
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_expanded
        
        # We maximize advantage, meaning we minimize negative advantage
        ppo_loss = -torch.min(surr1, surr2).mean()

    ppo_loss.backward()
    
    if ddp:
        torch.distributed.all_reduce(ppo_loss, op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(reward_scores, op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(kl_penalty, op=torch.distributed.ReduceOp.AVG)

    # Clip gradients tightly to prevent alignment collapse
    norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()
    
    t1 = time.time()
    dt = t1 - t0
    
    if master_process:
        print(f"step {step:4d} | reward: {reward_scores.mean().item():.4f} | KL: {kl_penalty.mean().item():.4f} | loss: {ppo_loss.item():.4f} | norm: {norm:.2f} | dt: {dt*1000:.0f}ms")
        with open("log_ppo/ppo_log.txt", "a") as f:
            f.write(f"{step} {reward_scores.mean().item():.4f} {kl_penalty.mean().item():.4f} {ppo_loss.item():.4f}\n")
            
        if step > 0 and step % 100 == 0:
            torch.save({'model_state_dict': actor.module.state_dict() if ddp else actor.state_dict()}, f"log_ppo/ppo_latest.pt")

if ddp: destroy_process_group()