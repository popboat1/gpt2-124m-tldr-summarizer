import os
import sys
import torch
import pytest

# Add parent directory to sys.path to resolve 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.model import GPT, GPTConfig

def test_kv_cache_equivalence():
    """
    Test that using KV caching produces the exact same logits 
    as passing the full sequence.
    """
    torch.manual_seed(42)
    
    # Tiny model for fast testing
    config = GPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=16)
    model = GPT(config)
    model.eval()
    
    # 1. Standard forward pass (full sequence)
    idx_full = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    with torch.no_grad():
        logits_full, _ = model(idx_full)
    
    # The logit predictions for the final token (given 10, 20, 30, 40, 50)
    target_logits = logits_full[:, -1, :]
    
    # 2. KV Cached forward pass
    idx_context = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    idx_next = torch.tensor([[50]], dtype=torch.long)
    
    with torch.no_grad():
        # Step A: Get past_key_values from the context
        _, _, past_key_values = model(idx_context, use_cache=True)
        
        # Step B: Pass ONLY the newest token + past_key_values
        logits_cached, _, _ = model(idx_next, past_key_values=past_key_values, use_cache=True)
        
    cached_target_logits = logits_cached[:, -1, :]
    
    # 3. Assert exact mathematical equivalence
    assert torch.allclose(target_logits, cached_target_logits, atol=1e-5), "KV Cached logits do not match standard logits!"

if __name__ == "__main__":
    test_kv_cache_equivalence()
    print("Test passed!")
