import os
import numpy as np
import tiktoken
from datasets import load_dataset

output_dir = "data/rm_dataset"
os.makedirs(output_dir, exist_ok=True)

def process_split(dataset_split, split_name):
    enc = tiktoken.get_encoding("gpt2")
    eot_token = 50256 
    max_seq_len = 1024 

    # we will store pairs of sequences: [chosen_tokens, rejected_tokens]
    paired_blocks = []
    
    print(f"processing {split_name} split...")
    
    for row in dataset_split:
        post_text = row['prompt'].strip()
        chosen_text = row['chosen'].strip()
        rejected_text = row['rejected'].strip()
        
        # format both exactly as the SFT model saw them
        chosen_str = f"Post: {post_text}\n\nTL;DR: {chosen_text}"
        rejected_str = f"Post: {post_text}\n\nTL;DR: {rejected_text}"
        
        chosen_tokens = enc.encode_ordinary(chosen_str) + [eot_token]
        rejected_tokens = enc.encode_ordinary(rejected_str) + [eot_token]
        
        # skip if either sequence blows past our 1024 physical limit
        if len(chosen_tokens) > max_seq_len or len(rejected_tokens) > max_seq_len:
            continue
            
        # pad both sequences up to exactly 1024 with eot_token.
        # because gpt-2 uses causal attention, the final token at position 1023 
        # will naturally aggregate the context of the entire sequence.
        chosen_padded = chosen_tokens + [eot_token] * (max_seq_len - len(chosen_tokens))
        rejected_padded = rejected_tokens + [eot_token] * (max_seq_len - len(rejected_tokens))
        
        paired_blocks.append([chosen_padded, rejected_padded])

    total_pairs = len(paired_blocks)
    print(f"total valid pairs for {split_name}: {total_pairs:,}")
    
    # allocate 3D array: (Num_Pairs, 2, 1024)
    tokens_np = np.empty((total_pairs, 2, max_seq_len), dtype=np.int32)
    tokens_np[:] = np.array(paired_blocks, dtype=np.int32)
    
    output_path = os.path.join(output_dir, f"rm_{split_name}_0000.npy")
    np.save(output_path, tokens_np)
    print(f"saved to: {output_path}")
    
    del paired_blocks

def main():
    print("downloading carperai/openai_summarize_comparisons dataset...")
    dataset = load_dataset("CarperAI/openai_summarize_comparisons")
    
    process_split(dataset['train'], "train")
    
    process_split(dataset['test'], "val") 

if __name__ == "__main__":
    main()