import os
import sys
import shutil
import unittest
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestDataLoaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # build independent temporary local mock datasets to validate execution states
        os.makedirs("data/edu_fineweb10B", exist_ok=True)
        os.makedirs("data/sft_dataset", exist_ok=True)
        os.makedirs("data/ppo_dataset", exist_ok=True)
        os.makedirs("data/rm_dataset", exist_ok=True)
        
        np.save("data/edu_fineweb10B/test_shard_train.npy", np.arange(100, dtype=np.int32))
        np.save("data/sft_dataset/test_masked_train.npy", np.ones((100, 2), dtype=np.int32))
        np.save("data/ppo_dataset/test_ppo_train.npy", np.ones((10, 512), dtype=np.int32))
        np.save("data/rm_dataset/test_rm_train.npy", np.ones((10, 2, 1024), dtype=np.int32))

    @classmethod
    def tearDownClass(cls):
        # tear down temporary folders cleanly from local filesystem arrays
        shutil.rmtree("data/edu_fineweb10B", ignore_errors=True)
        shutil.rmtree("data/sft_dataset", ignore_errors=True)
        shutil.rmtree("data/ppo_dataset", ignore_errors=True)
        shutil.rmtree("data/rm_dataset", ignore_errors=True)

    def test_lite_dataloader_dimensions(self):
        from src.utils.dataloader import DataLoaderLite
        loader = DataLoaderLite(B=2, T=4, process_rank=0, num_processes=1, split="train", master_process=False)
        x, y = loader.next_batch()
        self.assertEqual(x.shape, (2, 4))
        self.assertEqual(y.shape, (2, 4))

    def test_masked_dataloader_outputs(self):
        from src.utils.dataloader_masked import DataLoaderMasked
        loader = DataLoaderMasked(B=2, T=5, process_rank=0, num_processes=1, split="train", master_process=False)
        x, y = loader.next_batch()
        self.assertEqual(x.shape, (2, 5))
        self.assertEqual(y.shape, (2, 5))

    def test_ppo_dataloader_prompt_extraction(self):
        from src.utils.dataloader_ppo import DataLoaderPPO
        loader = DataLoaderPPO(B=2, process_rank=0, num_processes=1, split="train", master_process=False)
        prompts = loader.next_batch()
        self.assertEqual(prompts.shape, (2, 512))

    def test_reward_dataloader_pairwise_split(self):
        from src.utils.dataloader_reward import DataLoaderReward
        loader = DataLoaderReward(B=2, T=1024, process_rank=0, num_processes=1, split="train", master_process=False)
        chosen, rejected = loader.next_batch()
        self.assertEqual(chosen.shape, (2, 1024))
        self.assertEqual(rejected.shape, (2, 1024))

if __name__ == "__main__":
    unittest.main()