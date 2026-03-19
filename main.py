from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class chat_message(BaseModel):
    message: str
OLLAMA_URL = "http://localhost:11434/api/chat"

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
def chat(req: chat_message):
    payload = {
        "model": "phi3",
        "messages": [
            {"role": "user", "content": req.message}
        ],
        "stream": False,
        "options": {
            "num_predict": 50
        }
    }
    response = requests.post(OLLAMA_URL, json=payload)
    result = response.json()

    return {'response': result['message']['content']}


