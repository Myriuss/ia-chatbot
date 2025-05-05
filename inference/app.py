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
model_path = "./lora-model"

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

# ✅ Fix pad token si nécessaire
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 📌 Pipeline avec paramètres affinés
device = 0 if torch.cuda.is_available() else -1
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
)

# 🎯 Format des requêtes
class Query(BaseModel):
    prompt: str

@app.post("/predict")
def predict(query: Query):
    input_text = f"Q: {query.prompt.strip()}\nA:"
    try:
        output = generator(
            input_text,
            max_new_tokens=60,
            do_sample=True,
            top_p=0.9,
            temperature=0.5,
            repetition_penalty=1.3,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_text = output[0]["generated_text"]
        answer = generated_text.split("A:")[-1].strip()
    except Exception as e:
        answer = f"❌ Error generating response: {e}"

    return {"response": answer}

# 🌐 Interface web
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
