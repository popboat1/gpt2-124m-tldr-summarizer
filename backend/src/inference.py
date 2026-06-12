import time
import torch
import tiktoken
import torch.nn.functional as F
from .models.model import GPT, GPTConfig

def generate_text_stream(prompt, model_path, max_new_tokens=50, temperature=0.5, top_k=40, repetition_penalty=1.0, stop_on_eot=False, num_probs=0, device='cpu'):
    # load checkpoint safely onto the requested hardware device
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # parse the internal state dict regardless of how the training script packaged it
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # strip out any compilation prefixes leftover from pytorch two point zero optimizations
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    # initialize config and restore any custom saved hyper parameters
    config = GPTConfig()
    if 'config' in checkpoint:
        for k, v in checkpoint['config'].items():
            setattr(config, k, v)
            
    # dynamically read the vocab size from the checkpoint embedding weights
    # this gracefully handles our padding trick without hardcoding numbers
    vocab_size = state_dict['transformer.wte.weight'].shape[0]
    config.vocab_size = vocab_size
    
    # map states to the architecture and freeze gradients
    model = GPT(config)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # setup tokenizer and define structural boundary tokens
    enc = tiktoken.get_encoding('gpt2')
    eot_token = 50256
    input_tokens = enc.encode(prompt)
    idx = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # cache initial input length to calculate exact generation volume later
    input_len = idx.size(1)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    generated_text_so_far = ""
    
    # start the autoregressive decoding loop
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
            for i in range(max_new_tokens):
                # crop context to max block size if we exceed the physical attention window
                idx_cond = idx[:, -config.block_size:]
                logits, _ = model(idx_cond)
                
                # isolate the raw activations at the final time step
                next_token_logits = logits[:, -1, :]
                
                # apply the repetition penalty to heavily suppress tokens we already generated
                if repetition_penalty != 1.0:
                    for token_id in set(idx[0, input_len:].tolist()):
                        if next_token_logits[0, token_id] > 0:
                            next_token_logits[0, token_id] /= repetition_penalty
                        else:
                            next_token_logits[0, token_id] *= repetition_penalty
                                
                # scale by temperature to control the deterministic boundary
                next_token_logits = next_token_logits / temperature
                
                # truncate long tail probabilities via top k filtering
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                    
                probs = F.softmax(next_token_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                top_tokens = []
                if num_probs > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, min(num_probs, probs.size(-1)))
                    top_k_probs_list = top_k_probs[0].tolist()
                    top_k_indices_list = top_k_indices[0].tolist()
                    for p, t_id in zip(top_k_probs_list, top_k_indices_list):
                        try:
                            token_bytes = enc.decode_single_token_bytes(t_id)
                            token_str = token_bytes.decode('utf-8', errors='replace')
                        except Exception:
                            token_str = str(t_id)
                        top_tokens.append({"token": token_str, "prob": p})
                
                idx = torch.cat((idx, idx_next), dim=1)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                actual_tokens_generated = i + 1
                inference_time = t1 - t0
                tokens_per_sec = actual_tokens_generated / inference_time if inference_time > 0 else 0
                
                # slice the tensor to ONLY decode the newly generated tokens, ignoring the prompt
                output_tokens = idx[0, input_len:].tolist()
                current_text = enc.decode(output_tokens).replace("<|endoftext|>", "")
                
                new_chunk = current_text[len(generated_text_so_far):]
                generated_text_so_far = current_text
                
                is_done = (i == max_new_tokens - 1) or (stop_on_eot and idx_next.item() == eot_token)
                
                yield {
                    "text": new_chunk,
                    "time": inference_time,
                    "tps": tokens_per_sec,
                    "length": actual_tokens_generated,
                    "top_tokens": top_tokens,
                    "is_done": is_done
                }
                
                # trigger early termination if the model emits an end of text token
                if stop_on_eot and idx_next.item() == eot_token:
                    break