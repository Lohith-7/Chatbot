from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class chat_message(BaseModel):
    message: str

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

symptom_db = FAISS.load_local("vector_db/symptom_index", embeddings, allow_dangerous_deserialization=True)

medquad_db = FAISS.load_local("vector_db/medquad_index", embeddings, allow_dangerous_deserialization=True)

OLLAMA_URL = "http://localhost:11434/api/chat"

#Getting data from Vector DB
def get_top_diseases(query):
    docs = symptom_db.similarity_search(query, k=3)

    results = []
    for doc in docs:
        text = doc.page_content

        if "→ Disease:" in text:
            disease = text.split("→ Disease:")[-1].strip()
            results.append(disease)

    return list(set(results))

def get_medical_context(query, diseases):
    context = ""

    for disease in diseases:
        docs = medquad_db.similarity_search(disease, k=1)

        for doc in docs:
            context += doc.page_content + "\n\n"

    return context[:1500]


#routes
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
def chat(req: chat_message):
    diseases = get_top_diseases(req.message)

    context = get_medical_context(req.message, diseases)

    system_prompt = f"""
    You are a medical assistant.

    User symptoms: {req.message}

    Possible diseases:
    {", ".join(diseases)}

    Medical knowledge:
    {context}
    
    Instructions:
    - If they say hi greet them first do not expect a disease or create one to say them
    - Suggest likely conditions
    - Explain briefly
    - Do NOT give final diagnosis
    - Always recommend consulting a doctor
    """

    payload = {
        "model": "phi3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ],
        "stream": False,
        "options": {
            "num_predict": 200
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        result = response.json()

        return {
            "response": result["message"]["content"]
                        + "\n\n\n This is not medical advice. Consult a doctor."
        }

    except Exception as e:
        return {
            "response": f"Error: {str(e)}"
        }
