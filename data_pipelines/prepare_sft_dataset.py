import os
import numpy as np
import tiktoken
from datasets import load_dataset

output_dir = "data/sft_dataset"
os.makedirs(output_dir, exist_ok=True)

def process_split(dataset_split, split_name):
    enc = tiktoken.get_encoding("gpt2")
    eot_token = 50256 
    max_seq_len = 1024 

    blocks_tokens = []
    blocks_masks = []
    
    print(f"processing {split_name} split...")
    
    for row in dataset_split:
        post_text = row['prompt'].strip()
        summary_text = row['label'].strip()
        
        # format prompt with strict structural delimiters
        prompt_str = f"Post: {post_text}\n\nTL;DR: "
        prompt_tokens = enc.encode_ordinary(prompt_str)
        ans_tokens = enc.encode_ordinary(summary_text)
        
        # skip examples that exceed maximum physical context window
        if len(prompt_tokens) + len(ans_tokens) + 1 > max_seq_len:
            continue
            
        sample_tokens = prompt_tokens + ans_tokens + [eot_token]
        sample_masks = ([0] * len(prompt_tokens)) + ([1] * len(ans_tokens)) + [1]
        
        # pad the remainder of the block to exactly 1024 tokens
        padding_len = max_seq_len - len(sample_tokens)
        padded_tokens = sample_tokens + [eot_token] * padding_len
        padded_masks = sample_masks + [0] * padding_len
        
        blocks_tokens.extend(padded_tokens)
        blocks_masks.extend(padded_masks)

    total_elements = len(blocks_tokens)
    print(f"total tokens for {split_name} (including padding): {total_elements:,}")
    
    # safe allocation directly into 32-bit integer arrays to avoid memory spikes
    tokens_np = np.empty((total_elements, 2), dtype=np.int32)
    tokens_np[:, 0] = np.array(blocks_tokens, dtype=np.int32)
    tokens_np[:, 1] = np.array(blocks_masks, dtype=np.int32)
    
    output_path = os.path.join(output_dir, f"sft_{split_name}_0000.npy")
    np.save(output_path, tokens_np)
    print(f"saved to: {output_path}")
    
    # free memory
    del blocks_tokens
    del blocks_masks

def main():
    print("downloading full carperai/openai_summarize_tldr dataset...")
    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    
    # process entire splits
    process_split(dataset['train'], "train")
    process_split(dataset['valid'], "val")

if __name__ == "__main__":
    main()