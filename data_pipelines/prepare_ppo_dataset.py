import os
import numpy as np
import tiktoken
from datasets import load_dataset

output_dir = "data/ppo_dataset"
os.makedirs(output_dir, exist_ok=True)

def process_split(dataset_split, split_name):
    enc = tiktoken.get_encoding("gpt2")
    eot_token = 50256 
    
    # We cap prompts at 512 tokens to leave 512 tokens for generation
    # and to prevent VRAM Out-of-Memory (OOM) errors during the PPO rollout phase.
    max_prompt_len = 512 

    prompts = []
    
    print(f"Processing {split_name} split for PPO...")
    
    for row in dataset_split:
        post_text = row['prompt'].strip()
        
        # We only need the prompt and the trigger syntax. No summaries!
        prompt_str = f"Post: {post_text}\n\nTL;DR: "
        tokens = enc.encode_ordinary(prompt_str)
        
        # If the post is too long, we slice off the BEGINNING, not the end.
        # We must preserve the "TL;DR: " text at the very end.
        if len(tokens) > max_prompt_len:
            tokens = tokens[-max_prompt_len:]
            
        # Left-padding: We pad with EOT tokens on the left side.
        # This ensures the actual prompt tokens are physically flush against 
        # the right side of the matrix right before generation begins.
        pad_len = max_prompt_len - len(tokens)
        padded = [eot_token] * pad_len + tokens
        
        prompts.append(padded)

    total_prompts = len(prompts)
    print(f"Total valid PPO prompts for {split_name}: {total_prompts:,}")
    
    # Save as 2D NumPy array: (Num_Prompts, 512)
    prompts_np = np.array(prompts, dtype=np.int32)
    output_path = os.path.join(output_dir, f"ppo_{split_name}.npy")
    np.save(output_path, prompts_np)
    print(f"Saved to: {output_path}")

def main():
    print("Downloading CarperAI/openai_summarize_tldr dataset...")
    # We return to the original dataset because we only need prompts, not comparisons
    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    
    process_split(dataset['train'], "train")

if __name__ == "__main__":
    main()