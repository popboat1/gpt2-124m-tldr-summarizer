import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import torch
import tiktoken

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestLLMJudgeEval(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.enc = tiktoken.get_encoding("gpt2")
        
    def test_summary_generation_logic(self):
        from evaluation.llm_judge_eval import generate_summary
        
        # implement small mocked network structure mimicking gpt architecture
        mock_model = MagicMock()
        mock_logits = torch.randn(1, 1, 50304)
        mock_model.return_value = (mock_logits, None)
        
        prompt = "sample text for processing test loops"
        summary = generate_summary(mock_model, prompt, self.enc, self.device)
        
        # check output content forms a clean text presentation
        self.assertIsInstance(summary, str)

    @patch('requests.post')
    def test_openrouter_api_judge_decisions(self, mock_post):
        from evaluation.llm_judge_eval import call_openrouter_judge
        
        # design dynamic mocked request response structure to mimic api outputs
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "evaluation steps finished\nWINNER: A",
                    "reasoning_details": "clear choice context found"
                }
            }]
        }
        mock_post.return_value = mock_response
        
        result = call_openrouter_judge("post data", "summary choice a", "summary choice b", "fake_key")
        self.assertIn("WINNER: A", result)
        self.assertIn("Reasoning Details:", result)

if __name__ == "__main__":
    unittest.main()