import os
import torch
import torch.nn.functional as F
import tiktoken
import random
import json
import time
import requests
from dotenv import load_dotenv
from datasets import load_dataset
from src.models.model import GPT, GPTConfig

# global workspace variables and constants
max_context = 1024
eot_token = 50256

def load_model(checkpoint_path, device):
    # build config matching the base gpt structure and inject weights safely
    config = GPTConfig(vocab_size=50304)
    model = GPT(config)
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

def generate_summary(model, prompt_text, enc, device):
    # format prompt context into target sequence boundaries with soft length constraint
    prompt_str = f"Post: {prompt_text}\n\nTL;DR: "
    tokens = enc.encode_ordinary(prompt_str)
    
    if len(tokens) > (max_context - 64):
        tokens = tokens[-(max_context - 64):]
        
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    temperature = 0.7
    top_k = 40
    repetition_penalty = 1.3
    
    # run autoregressive decoding sequence with localized token loop inhibition
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            for _ in range(64):
                logits, _ = model(x)
                next_token_logits = logits[:, -1, :]
                
                for token_id in set(x[0].tolist()):
                    if token_id in next_token_logits[0]:
                        if next_token_logits[0, token_id] > 0:
                            next_token_logits[0, token_id] /= repetition_penalty
                        else:
                            next_token_logits[0, token_id] *= repetition_penalty

                next_token_logits = next_token_logits / temperature
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                x = torch.cat((x, next_token), dim=1)
                if next_token.item() == eot_token:
                    break
                    
    generated_tokens = x[0, len(tokens):].tolist()
    return enc.decode(generated_tokens).replace("<|endoftext|>", "").strip()

def call_openrouter_judge(post, summary_a, summary_b, api_key):
    # construct systemic prompt layout to pass along to openrouter endpoint
    judge_prompt = f"You are an impartial expert evaluating AI-generated summaries of Reddit posts.\nEvaluate which summary is better based on accuracy, conciseness, and fluency.\n\nOriginal Post:\n{post}\n\nSummary A:\n{summary_a}\n\nSummary B:\n{summary_b}\n\nOutput your reasoning steps first, and then on the very last line, output exactly one of the following formats:\nWINNER: A\nWINNER: B\nTIE"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openai/gpt-oss-120b:free",
        "messages": [{"role": "user", "content": judge_prompt}],
        "reasoning": {"enabled": True}
    }

    # backoff network call orchestration suite with custom exception capture
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            if response.status_code == 429:
                sleep_time = 2 ** attempt
                print(f"rate limited. retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                continue
                
            response.raise_for_status()
            res_json = response.json()
            
            if 'choices' in res_json and len(res_json['choices']) > 0:
                res_msg = res_json['choices'][0]['message']
                content = res_msg.get('content', '')
                reasoning = res_msg.get('reasoning_details', '')
                
                full_output = ""
                if reasoning:
                    full_output += f"Reasoning Details:\n{reasoning}\n\n"
                full_output += content
                
                if full_output.strip():
                    return full_output
                    
            print("empty payload received from openrouter. retrying...")
            time.sleep(2 ** attempt)
            
        except Exception as e:
            print(f"openrouter api exception: {e}. retrying...")
            time.sleep(2 ** attempt)
            
    print("failed to get response after max retries. defaulting to tie.")
    return "WINNER: TIE"

def run_evaluation_pipeline():
    # primary interface loop executing model analysis cycles across split entries
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("missing openrouter_api_key! check your .env file.")

    num_eval_samples = 50
    enc = tiktoken.get_encoding("gpt2")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    judge_history_file = os.path.join(log_dir, "128tokens_judge_arena_results.txt")

    with open(judge_history_file, "w", encoding="utf-8") as f:
        f.write("LLM JUDGE ARENA RUN LOG\n\n")

    print("loading sft baseline model...")
    sft_model = load_model("log/model_best.pt", device)

    print("loading ppo aligned model...")
    ppo_model = load_model("log/ppo_128tokens_latest.pt", device)

    print("fetching validation dataset split...")
    dataset = load_dataset("CarperAI/openai_summarize_tldr", split="valid")

    sft_wins = 0
    ppo_wins = 0
    ties = 0

    print(f"evaluating head-to-head across {num_eval_samples} samples...")
    for i in range(num_eval_samples):
        print(f"processing sample {i + 1} of {num_eval_samples}...")
        post = dataset[i]['prompt']
        
        summary_sft = generate_summary(sft_model, post, enc, device)
        summary_ppo = generate_summary(ppo_model, post, enc, device)
        
        swap = random.choice([True, False])
        if swap:
            sum_a, sum_b = summary_ppo, summary_sft
        else:
            sum_a, sum_b = summary_sft, summary_ppo
            
        judge_response = call_openrouter_judge(post, sum_a, sum_b, api_key)
        
        lines = [line.strip() for line in judge_response.strip().split('\n') if line.strip()]
        last_line = lines[-1].upper() if lines else "TIE"
        
        if "WINNER: A" in last_line:
            verdict = "ppo aligned" if swap else "sft baseline"
            if swap: ppo_wins += 1
            else: sft_wins += 1
        elif "WINNER: B" in last_line:
            verdict = "sft baseline" if swap else "ppo aligned"
            if swap: sft_wins += 1
            else: ppo_wins += 1
        else:
            verdict = "tie"
            ties += 1

        with open(judge_history_file, "a", encoding="utf-8") as f:
            f.write(f"Sample {i + 1}\n")
            f.write(f"Original Post:\n{post}\n\n")
            f.write(f"Summary A ({'PPO' if swap else 'SFT'}):\n{sum_a}\n\n")
            f.write(f"Summary B ({'SFT' if swap else 'PPO'}):\n{sum_b}\n\n")
            f.write(f"Judge Output:\n{judge_response}\n\n")
            f.write(f"Resolved Verdict: {verdict}\n")
            f.write("\n")

    print("\nOPENROUTER-AS-A-JUDGE ARENA REPORT")
    print(f"PPO Aligned Model Wins: {ppo_wins}")
    print(f"SFT Baseline Model Wins: {sft_wins}")
    print(f"Ties/Failures: {ties}")

    total_decisive = ppo_wins + sft_wins
    if total_decisive > 0:
        win_rate = (ppo_wins / total_decisive) * 100
        print(f"Adjusted PPO Win Rate: {win_rate:.2f}% (excluding ties)")
    else:
        print("no decisive wins recorded.")
    print("evaluation complete")

if __name__ == "__main__":
    run_evaluation_pipeline()