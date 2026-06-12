import torch
import torch.nn as nn
from .model import GPT, GPTConfig

class GPTRewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        base_gpt = GPT(config)
        self.transformer = base_gpt.transformer
        
        # instead of predicting the next word out of 50,257 tokens, 
        # this linear layer projects the 768-dimensional embedding down to 1 number.
        self.v_head = nn.Linear(config.n_embd, 1, bias=False)
        
        torch.nn.init.normal_(self.v_head.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        # idx is shape (B, T) - e.g., (2, 1024)
        B, T = idx.size()
        
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x) # shape: (B, T, n_embd)
        
        # reward extraction
        # because GPT-2 uses causal attention (looking left), the final token in the sequence 
        # contains the aggregated mathematical context of the ENTIRE prompt and summary.
        # we slice out just the last token's embeddings across the batch.
        x_last = x[:, -1, :] # shape: (B, n_embd)
        
        # project to a single scalar reward
        value = self.v_head(x_last) # shape: (B, 1)
        
        return value.squeeze(-1) # shape: (B,)