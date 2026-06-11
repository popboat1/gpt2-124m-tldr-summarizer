import os
import numpy as np
import torch

def load_data(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)

class DataLoaderReward:
    def __init__(self, B, T, process_rank, num_processes, split, master_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        
        data_root = "data/rm_dataset"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} reward shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.data = load_data(self.shards[self.current_shard])
        self.current_position = self.B * self.process_rank
    
    def next_batch(self):
        B = self.B
        buf = self.data[self.current_position : self.current_position + B]
        
        # split custom preference sequence records into winning and losing tensors
        x_chosen = buf[:, 0, :]   
        x_rejected = buf[:, 1, :] 
        
        self.current_position += B * self.num_processes
        if self.current_position + B * self.num_processes > len(self.data):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.data = load_data(self.shards[self.current_shard])
            self.current_position = B * self.process_rank
        return x_chosen, x_rejected