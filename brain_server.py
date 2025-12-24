import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

app = FastAPI(title="ArkTech Brain (Groq)")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("GROQ_API_KEY loaded:", bool(GROQ_API_KEY))

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"  # if this fails, we’ll swap to another Groq model you have

class Ask(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask(req: Ask):
    if not GROQ_API_KEY:
        return {"type": "speech", "text": "ArkTech is missing GROQ API key."}

    # Simple router (cheap rules first)
    t = req.text.strip().lower()

    # ACTION examples (we will expand later)
    if t.startswith("open "):
        app_name = req.text.strip()[5:]
        return {"type": "action", "name": "open_app", "app": app_name}

    if "set alarm" in t or t.startswith("alarm "):
        return {"type": "action", "name": "set_alarm", "time": req.text}

    # Otherwise CHAT via Groq
    try:
        r = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are ArkTech. Answer clearly in 2-4 sentences."},
                    {"role": "user", "content": req.text},
                ],
                "temperature": 0.4,
            },
            timeout=60,
        )

        if r.status_code != 200:
            return {"type": "speech", "text": f"Groq error {r.status_code}: {r.text}"}

        data = r.json()
        answer = data["choices"][0]["message"]["content"]
        return {"type": "speech", "text": answer}

    except Exception as e:
        return {"type": "speech", "text": f"Server exception: {repr(e)}"}
