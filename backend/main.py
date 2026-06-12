import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.inference import generate_text
import requests
import random

app = FastAPI()

# define what the frontend will send us
class GenerateRequest(BaseModel):
    text: str
    model: str = "PPO Aligned"
    temperature: float = 0.7
    top_k: int = 40

@app.post("/api/generate")
def generate(req: GenerateRequest):
    prompt = f"post: {req.text}\n\ntl;dr: "
    
    if req.model == "SFT Baseline":
        model_path = "models/model_best.pt"
        max_tokens = 64
        rep_pen = 1.0
    else:
        model_path = "models/ppo_latest.pt"
        max_tokens = 128
        rep_pen = 1.3
        
    out_text, out_time, out_tps = generate_text(
        prompt=prompt, 
        model_path=model_path, 
        max_new_tokens=max_tokens, 
        temperature=req.temperature,
        top_k=req.top_k,
        repetition_penalty=rep_pen,
        stop_on_eot=True
    )

    return {
        "text": out_text, 
        "time": out_time, 
        "tps": out_tps
    }

@app.get("/api/reddit/{subreddit}")
def get_reddit_post(subreddit: str):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 GPT2Summarizer/1.0'}
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=15"
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return {"error": "Failed to fetch from Reddit", "text": ""}
            
        data = response.json()
        valid_posts = []
        
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            if not post.get("stickied") and len(post.get("selftext", "")) > 50:
                valid_posts.append(f"Title: {post.get('title')}\n\n{post.get('selftext')}")
                
        if not valid_posts:
            return {"error": "No valid text posts found", "text": ""}
            
        return {"text": random.choice(valid_posts)}
    except Exception as e:
        return {"error": str(e), "text": ""}

# mount the react build folder so fastapi serves the ui
app.mount("/", StaticFiles(directory="static", html=True), name="static")