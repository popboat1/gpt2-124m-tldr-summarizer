import os
import sys
import shutil
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestUnifiedPipelineLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # isolate independent dataset shards inside custom local data scopes
        os.makedirs("data/edu_fineweb10B", exist_ok=True)
        os.makedirs("data/sft_dataset", exist_ok=True)
        os.makedirs("data/ppo_dataset", exist_ok=True)
        os.makedirs("data/rm_dataset", exist_ok=True)
        
        np.save("data/edu_fineweb10B/test_shard_train.npy", np.arange(50, dtype=np.int32))
        np.save("data/sft_dataset/test_masked_train.npy", np.ones((50, 2), dtype=np.int32))
        np.save("data/ppo_dataset/test_ppo_train.npy", np.ones((5, 512), dtype=np.int32))
        np.save("data/rm_dataset/test_rm_train.npy", np.ones((5, 2, 1024), dtype=np.int32))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("data/edu_fineweb10B", ignore_errors=True)
        shutil.rmtree("data/sft_dataset", ignore_errors=True)
        shutil.rmtree("data/ppo_dataset", ignore_errors=True)
        shutil.rmtree("data/rm_dataset", ignore_errors=True)

    def test_pretrain_step_execution(self):
        from training.train_pipeline import train_step
        from src.utils.dataloader import DataLoaderLite
        
        # enforce gradient tracking on the mock tensor to satisfy loss backward calls
        mock_loss = torch.tensor(1.5, requires_grad=True)
        mock_model = MagicMock(return_value=(None, mock_loss))
        models = {"model": mock_model}
        loader = DataLoaderLite(B=1, T=4, process_rank=0, num_processes=1, split="train", master_process=False)
        
        metrics = train_step("pretrain", models, loader, "cpu", ddp=False, grad_accum_steps=1)
        self.assertAlmostEqual(metrics["loss"].item(), 1.5)

    def test_generation_token_concatenation(self):
        from training.train_pipeline import generate
        
        mock_model = MagicMock()
        mock_logits = torch.randn(1, 1, 50304)
        mock_model.return_value = (mock_logits, None)
        prompts = torch.ones((1, 8), dtype=torch.long)
        
        output = generate(mock_model, prompts, gen_len=4)
        self.assertEqual(output.shape, (1, 12))

if __name__ == "__main__":
    unittest.main()