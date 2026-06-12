import os
import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
import random
from huggingface_hub import hf_hub_download
try:
    # When running from the project root (e.g., Docker / Hugging Face)
    from backend.src.inference import generate_text_stream
except ModuleNotFoundError:
    # When running locally from inside the backend/ folder
    from src.inference import generate_text_stream

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# define what the frontend will send us
class GenerateRequest(BaseModel):
    text: str
    model: str = "PPO Aligned"
    temperature: float = 0.7
    top_k: int = 40
    num_probs: int = 5

@app.post("/api/generate")
def generate(req: GenerateRequest):
    print(f"Incoming request -> Model: {req.model} | Temp: {req.temperature} | Top K: {req.top_k} | Probs: {req.num_probs} | Text length: {len(req.text)}")
    prompt = f"Post: {req.text}\n\nTL;DR: "
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if req.model == "SFT Baseline":
        model_filename = "best_sft.pt"
        max_tokens = 64
        rep_pen = 1.0
    else:
        model_filename = "ppo_latest.pt"
        max_tokens = 128
        rep_pen = 1.3
        
    try:
        # Dynamically download the model from the Hugging Face Model Hub
        model_path = hf_hub_download(repo_id="popboat1/gpt2-summarizer-models", filename=model_filename)
    except Exception as e:
        print(f"Failed to download from HF Hub: {e}")
        # Fallback to local paths if running locally without internet
        if req.model == "SFT Baseline":
            model_path = os.path.join(base_dir, "log", "sft", "best_sft.pt")
        else:
            model_path = os.path.join(base_dir, "log", "ppo", "ppo_latest.pt")
        
    def event_stream():
        for chunk in generate_text_stream(
            prompt=prompt, 
            model_path=model_path, 
            max_new_tokens=max_tokens, 
            temperature=req.temperature,
            top_k=req.top_k,
            repetition_penalty=rep_pen,
            stop_on_eot=True,
            num_probs=req.num_probs
        ):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

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

# Mount the react build folder if it exists (local monolithic development)
backend_dir = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(backend_dir, "static")
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
elif os.path.exists(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
else:
    @app.get("/")
    def read_root():
        return {"status": "online", "message": "API is running. Frontend is hosted separately."}