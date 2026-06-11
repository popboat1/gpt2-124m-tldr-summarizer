import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.model import GPT, GPTConfig
from src.models.reward_model import GPTRewardModel
from src.utils.dataloader import DataLoaderLite
from src.utils.dataloader_masked import DataLoaderMasked
from src.utils.dataloader_reward import DataLoaderReward
from src.utils.dataloader_ppo import DataLoaderPPO

def get_lr(step, mode, max_steps):
    # conservative rate to prevent policy shattering during rl phase
    if mode == "ppo":
        return 1e-7

    # cosine decay boundaries based on the architectural training phase
    if mode in {"pretrain", "sft"}:
        max_lr = 1e-4
        min_lr = 1e-5
        warmup_steps = 100
    elif mode == "reward":
        max_lr = 1e-5
        min_lr = 1e-6
        warmup_steps = 50

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def load_weights(model, checkpoint_path, master_process):
    # inject stripped base weights seamlessly into new network graphs
    if master_process:
        print(f"loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict, strict=False)

def get_log_probs(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, 2, tokens.unsqueeze(2)).squeeze(2)

def generate(model, prompts, gen_len, temperature=0.7, top_k=40):
    model.eval()
    with torch.no_grad():
        gen_seqs = prompts
        for _ in range(gen_len):
            logits, _ = model(gen_seqs)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            gen_seqs = torch.cat((gen_seqs, next_token), dim=1)
    return gen_seqs

def train_step(mode, models, loader, device, ddp, grad_accum_steps, scaler=None):
    loss_accum = 0.0
    metrics = {}
    
    if mode in {"pretrain", "sft"}:
        model = models["model"]
        for micro_step in range(grad_accum_steps):
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        metrics["loss"] = loss_accum
        return metrics

    elif mode == "reward":
        model = models["model"]
        for micro_step in range(grad_accum_steps):
            x_chosen, x_rejected = loader.next_batch()
            x_chosen, x_rejected = x_chosen.to(device), x_rejected.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                r_chosen = model(x_chosen)
                r_rejected = model(x_rejected)
                loss = -F.logsigmoid(r_chosen - r_rejected).mean()
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
            
            acc = (r_chosen > r_rejected).float().mean()
            metrics["acc"] = acc.detach()
            
        metrics["loss"] = loss_accum
        return metrics

    elif mode == "ppo":
        actor = models["actor"]
        ref_model = models["ref_model"]
        critic = models["critic"]
        
        prompts = loader.next_batch().to(device)
        gen_seqs = generate(actor, prompts, gen_len=128)
        generated_tokens_only = gen_seqs[:, prompts.size(1):]
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                reward_scores = critic(gen_seqs)
                ref_logits, _ = ref_model(gen_seqs)
                ref_log_probs = get_log_probs(ref_logits[:, prompts.size(1)-1:-1, :], generated_tokens_only)
                old_actor_logits, _ = actor(gen_seqs)
                old_log_probs = get_log_probs(old_actor_logits[:, prompts.size(1)-1:-1, :], generated_tokens_only)
                kl_div = torch.clamp(old_log_probs - ref_log_probs, min=0.0)
                kl_penalty = kl_div.sum(dim=1)
                advantages = reward_scores - (0.3 * kl_penalty)
                normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actor.train()
        for micro_step in range(grad_accum_steps):
            if ddp:
                actor.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                new_logits, _ = actor(gen_seqs)
                new_log_probs = get_log_probs(new_logits[:, prompts.size(1)-1:-1, :], generated_tokens_only)
                ratio = torch.exp(new_log_probs - old_log_probs)
                adv_expanded = normalized_advantages.unsqueeze(1).expand_as(ratio)
                surr1 = ratio * adv_expanded
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv_expanded
                ppo_loss = -torch.min(surr1, surr2).mean()
                ppo_loss = ppo_loss / grad_accum_steps
                loss_accum += ppo_loss.detach()
                ppo_loss.backward()
                
        metrics["loss"] = loss_accum
        metrics["reward"] = reward_scores.mean().detach()
        metrics["kl"] = kl_penalty.mean().detach()
        return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="pretrain", choices=["pretrain", "sft", "reward", "ppo"])
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    args = parser.parse_args()

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        master_process = True

    config = GPTConfig(vocab_size=50304)
    models = {}
    
    # orchestrate parameter layout mapping and inject specific stage weights
    if args.mode in {"pretrain", "sft"}:
        models["model"] = GPT(config).to(device)
        if args.mode == "sft" and os.path.exists("log/pretrained_model.pt"):
            load_weights(models["model"], "log/pretrained_model.pt", master_process)
        if ddp:
            models["model"] = DDP(models["model"], device_ids=[ddp_local_rank])
            
        optimizer = torch.optim.AdamW(models["model"].parameters(), lr=1e-4)
        loader = DataLoaderLite(args.B, args.T, ddp_rank, ddp_world_size, 'train', master_process) if args.mode == "pretrain" else DataLoaderMasked(args.B, args.T, ddp_rank, ddp_world_size, 'train', master_process)
        
    elif args.mode == "reward":
        models["model"] = GPTRewardModel(config).to(device)
        if os.path.exists("log/model_best.pt"):
            load_weights(models["model"], "log/model_best.pt", master_process)
        if ddp:
            models["model"] = DDP(models["model"], device_ids=[ddp_local_rank])
            
        optimizer = torch.optim.AdamW(models["model"].parameters(), lr=1e-5)
        loader = DataLoaderReward(args.B, args.T, ddp_rank, ddp_world_size, 'train', master_process)
        
    elif args.mode == "ppo":
        models["actor"] = GPT(config).to(device)
        models["ref_model"] = GPT(config).to(device)
        models["critic"] = GPTRewardModel(config).to(device)
        
        if os.path.exists("log/model_best.pt"):
            load_weights(models["actor"], "log/model_best.pt", master_process)
            load_weights(models["ref_model"], "log/model_best.pt", master_process)
        if os.path.exists("log/rm_latest.pt"):
            load_weights(models["critic"], "log/rm_latest.pt", master_process)
            
        models["ref_model"].eval()
        models["critic"].eval()
        if ddp:
            models["actor"] = DDP(models["actor"], device_ids=[ddp_local_rank])
            
        optimizer = torch.optim.AdamW(models["actor"].parameters(), lr=1e-7)
        loader = DataLoaderPPO(args.B, ddp_rank, ddp_world_size, 'train', master_process)

    scaler = torch.cuda.amp.GradScaler() if args.mode in {"pretrain", "sft"} else None
    
    log_file = f"log/{args.mode}_pipeline_log.txt"
    if master_process:
        os.makedirs("log", exist_ok=True)
        open(log_file, "w").close()

    for step in range(args.max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        
        metrics = train_step(args.mode, models, loader, device, ddp, args.grad_accum_steps, scaler=scaler)
        
        if args.mode in {"pretrain", "sft"}:
            tokens_step = args.B * args.T * args.grad_accum_steps * ddp_world_size
            active_model = models["model"]
        elif args.mode == "reward":
            tokens_step = args.B * args.T * 2 * args.grad_accum_steps * ddp_world_size
            active_model = models["model"]
        elif args.mode == "ppo":
            tokens_step = args.B * (512 + 128) * args.grad_accum_steps * ddp_world_size
            active_model = models["actor"]

        if ddp:
            dist.all_reduce(metrics["loss"], op=dist.ReduceOp.AVG)
            if args.mode == "reward":
                dist.all_reduce(metrics["acc"], op=dist.ReduceOp.AVG)
            elif args.mode == "ppo":
                dist.all_reduce(metrics["reward"], op=dist.ReduceOp.AVG)
                dist.all_reduce(metrics["kl"], op=dist.ReduceOp.AVG)

        if scaler:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(active_model.parameters(), 1.0)
            
            lr = get_lr(step, args.mode, args.max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            scaler.step(optimizer)
            scaler.update()
        else:
            norm = torch.nn.utils.clip_grad_norm_(active_model.parameters(), 1.0)
            
            lr = get_lr(step, args.mode, args.max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = tokens_step / dt

        if master_process:
            if args.mode == "ppo":
                print(f"step {step:4d} | reward: {metrics['reward'].item():.4f} | kl: {metrics['kl'].item():.4f} | loss: {metrics['loss'].item():.4f} | norm: {norm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tokens_per_sec:.0f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} {metrics['reward'].item():.4f} {metrics['kl'].item():.4f} {metrics['loss'].item():.4f} {norm:.2f} {dt*1000:.0f} {tokens_per_sec:.0f}\n")
            elif args.mode == "reward":
                print(f"step {step:4d} | loss: {metrics['loss'].item():.4f} | acc: {metrics['acc'].item():.4f} | norm: {norm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tokens_per_sec:.0f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} {metrics['loss'].item():.4f} {metrics['acc'].item():.4f} {norm:.2f} {dt*1000:.0f} {tokens_per_sec:.0f}\n")
            else:
                print(f"step {step:4d} | loss: {metrics['loss'].item():.4f} | norm: {norm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tokens_per_sec:.0f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} {metrics['loss'].item():.4f} {norm:.2f} {dt*1000:.0f} {tokens_per_sec:.0f}\n")

            # periodic checkpoint flushing to disk to preserve state evolution
            if step > 0 and step % 100 == 0 or step == args.max_steps - 1:
                checkpoint_dict = {'model_state_dict': active_model.module.state_dict() if ddp else active_model.state_dict()}
                if args.mode == "pretrain":
                    torch.save(checkpoint_dict, "log/pretraining/pretrained_latest.pt")
                elif args.mode == "sft":
                    torch.save(checkpoint_dict, "log/sft/final_sft.pt")
                elif args.mode == "reward":
                    torch.save(checkpoint_dict, "log/reward-model/rm_latest.pt")
                elif args.mode == "ppo":
                    torch.save(checkpoint_dict, "log/ppo/ppo_latest.pt")

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()