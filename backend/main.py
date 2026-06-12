import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.inference import generate_text

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

# mount the react build folder so fastapi serves the ui
app.mount("/", StaticFiles(directory="static", html=True), name="static")