import os
import numpy as np
import torch

def load_data(filename):
    # extract two column array containing tracking tokens and target masks
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderMasked:
    def __init__(self, B, T, process_rank, num_processes, split, master_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        
        data_root = "data/sft_dataset"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.data = load_data(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.data[self.current_position : self.current_position + B * T + 1]
        
        # partition raw text streams away from structural response mask columns
        x = buf[:-1, 0].view(B, T) 
        y = buf[1:, 0].view(B, T)  
        m = buf[1:, 1].view(B, T)  
        
        # replace unmasked prompt context fields with pytorch ignore token index
        y = y.clone()
        y[m == 0] = -100
        
        self.current_position += B * T * self.num_processes
        if self.current_position + B * T * self.num_processes + 1 > len(self.data):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.data = load_data(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y