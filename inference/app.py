from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# 📦 Charger le modèle fine-tuné avec LoRA
base_model_name = "distilgpt2"
model_path = "./lora-model"  # Dossier contenant le modèle fine-tuné

tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

# Préparer un pipeline de génération
device = 0 if torch.cuda.is_available() else -1
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# 🎯 Format des requêtes
class Query(BaseModel):
    prompt: str

# 📍 API principale
@app.post("/predict")
def predict(query: Query):
    input_text = f"Question : {query.prompt}\nRéponse :"
    output = generator(input_text, max_length=100, do_sample=True, top_p=0.9, temperature=0.8)
    answer = output[0]["generated_text"].split("Réponse :")[-1].strip()
    return {"response": answer}

# 🌐 Interface web
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
