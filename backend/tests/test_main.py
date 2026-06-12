import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch

client = TestClient(app)

def test_generate_endpoint():
    with patch("main.generate_text", return_value=("Mock summary", 1.2, 50.0)) as mock_gen:
        # Default PPO Aligned
        response = client.post(
            "/api/generate",
            json={"text": "Test input", "model": "PPO Aligned", "temperature": 0.5, "top_k": 40}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Mock summary"
        assert data["time"] == 1.2
        assert data["tps"] == 50.0
        
        mock_gen.assert_called_once()
        kwargs = mock_gen.call_args[1]
        assert "post: Test input" in kwargs["prompt"]
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_k"] == 40
        assert kwargs["model_path"] == "models/ppo_latest.pt"
        assert kwargs["repetition_penalty"] == 1.3
        assert kwargs["max_new_tokens"] == 128

    with patch("main.generate_text", return_value=("SFT summary", 1.0, 60.0)) as mock_gen_sft:
        # SFT Baseline
        response = client.post(
            "/api/generate",
            json={"text": "Test input", "model": "SFT Baseline", "temperature": 0.7, "top_k": 40}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "SFT summary"
        
        mock_gen_sft.assert_called_once()
        kwargs = mock_gen_sft.call_args[1]
        assert kwargs["model_path"] == "models/model_best.pt"
        assert kwargs["repetition_penalty"] == 1.0
        assert kwargs["max_new_tokens"] == 64

def test_reddit_bucket():
    with patch("main.requests.get") as mock_get:
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "children": [
                    {"data": {"title": "Test Title", "selftext": "Test body text that is reasonably long enough. This makes it over 50 characters.", "stickied": False}}
                ]
            }
        }
        
        response = client.get("/api/reddit/tifu")
        assert response.status_code == 200
        assert "Test Title" in response.json()["text"]
