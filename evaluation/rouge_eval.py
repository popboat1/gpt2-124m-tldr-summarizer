import os
import sys
import json
import time
import torch
from torch.nn import functional as F
import tiktoken
from datasets import load_dataset
from rouge_score import rouge_scorer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.model import GPT, GPTConfig

def load_eval_model(checkpoint_path, device):
    config = GPTConfig(vocab_size=50304)
    model = GPT(config)
    
    print(f"loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def generate_summary(model, text, enc, device, max_new_tokens=64):
    formatted_prompt = f"post: {text}\n\ntl;dr: "
    tokens = enc.encode_ordinary(formatted_prompt)
    
    max_context = 1024
    if len(tokens) > (max_context - max_new_tokens):
        tokens = tokens[-(max_context - max_new_tokens):]
        
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
            for _ in range(max_new_tokens):
                logits, _ = model(x)
                next_token_logits = logits[:, -1, :]
                
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                x = torch.cat((x, next_token), dim=1)
                
                if next_token.item() == 50256:
                    break
                    
    output_tokens = x[0, len(tokens):].tolist()
    return enc.decode(output_tokens).replace("<|endoftext|>", "").strip()

def run_rouge_evaluation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = tiktoken.get_encoding("gpt2")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    print("loading validation dataset...")
    dataset = load_dataset("CarperAI/openai_summarize_tldr", split="valid")
    num_samples = 200
    
    # inject the raw pretraining checkpoint alongside the sft weights
    checkpoints = {
        "pretrained_model": "log/pretraining/pretrained_model.pt",
        "best_model": "log/sft/best_sft.pt",
        "latest_model": "log/sft/final_sft.pt"
    }
    
    results = {}
    
    for model_name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"skipping {model_name}, file not found at {ckpt_path}")
            continue
            
        model = load_eval_model(ckpt_path, device)
        scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        print(f"evaluating {model_name} across {num_samples} samples...")
        t0 = time.time()
        
        for i in range(num_samples):
            post = dataset[i]['prompt']
            reference = dataset[i]['label']
            
            prediction = generate_summary(model, post, enc, device)
            item_scores = scorer.score(reference, prediction)
            
            scores['rouge1'] += item_scores['rouge1'].fmeasure
            scores['rouge2'] += item_scores['rouge2'].fmeasure
            scores['rougeL'] += item_scores['rougeL'].fmeasure
            
            if (i + 1) % 50 == 0:
                print(f"processed {i + 1}/{num_samples} samples")
                
        for k in scores.keys():
            scores[k] = scores[k] / num_samples
            
        results[model_name] = {
            "rouge-1": round(scores['rouge1'], 4),
            "rouge-2": round(scores['rouge2'], 4),
            "rouge-l": round(scores['rougeL'], 4)
        }
        
        t1 = time.time()
        print(f"completed {model_name} evaluation in {t1 - t0:.1f}s\n")
        
        del model
        torch.cuda.empty_cache()
        
    os.makedirs("log/sft", exist_ok=True)
    with open("log/sft/sft_rouge_scores.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("evaluation complete. results saved to log/sft/sft_rouge_scores.json")

if __name__ == "__main__":
    run_rouge_evaluation()