import os
import numpy as np
import torch

def load_data(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)

class DataLoaderPPO:
    def __init__(self, B, process_rank, num_processes, split, master_process):
        self.B = B
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        data_root = "data/ppo_dataset"
        shards = [os.path.join(data_root, s) for s in os.listdir(data_root) if split in s]
        self.shards = sorted(shards)
        assert len(self.shards) > 0, f"no shards found for split {split}"
        
        if master_process:
            print(f"found {len(self.shards)} ppo shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.data = load_data(self.shards[self.current_shard])
        self.current_position = self.B * self.process_rank
    
    def next_batch(self):
        B = self.B
        # isolate the exact prompt token batch chunks out of static arrays
        prompts = self.data[self.current_position : self.current_position + B]
        
        self.current_position += B * self.num_processes
        if self.current_position + B * self.num_processes > len(self.data):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.data = load_data(self.shards[self.current_shard])
            self.current_position = B * self.process_rank
        return prompts