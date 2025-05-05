from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

#charger le modèle fine-tuné avec LoRA
base_model_name = "distilgpt2"
model_path = "./lora-model"

base_model = AutoModelForCausalLM.from_pretrained(base_model_name) #chargement du modèle de base
tokenizer = AutoTokenizer.from_pretrained(base_model_name) #chargement du tokenizer associé

#application de la configuration LoRA au modèle de base
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()#mode évaluation, désactive dropout etcccc

## configuration du token de padding
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

#création d’un pipeline Hugging Face pour la génération de texte
#le pipeline est configuré pour utiliser le GPU si disponible, sinon le CPU
device = 0 if torch.cuda.is_available() else -1
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
)

#Format des requêtes
class Query(BaseModel):
    prompt: str #une seule clé "prompt" de type string

#definition de la route POST /predict
#cette route reçoit une requête JSON contenant un prompt, génère une réponse et la retourne
@app.post("/predict")
def predict(query: Query):
    #format d’entrée pour le modèle
    input_text = f"Q: {query.prompt.strip()}\nA:"
    try:
        #génération du texte avec plusieurs paramètres pour contrôler la créativité
        output = generator(
            input_text,
            max_new_tokens=60, #limite de tokens générés
            do_sample=True, #activation de l’échantillonnage
            top_p=0.9,#nucleus sampling
            temperature=0.5,#contrôle de la diversité
            repetition_penalty=1.3,#pénalise les répétitions
            eos_token_id=tokenizer.eos_token_id,#token de fin
        )
        #extraction de la reponse textuelle aprss le préfixe "A:"
        generated_text = output[0]["generated_text"]
        answer = generated_text.split("A:")[-1].strip()
    except Exception as e:
        #voir si jamais ça echoue
        answer = f"❌ !!! Error generating response: {e}"
    #reponse renvoyée sous forme de dictionnaire JSON
    return {"response": answer}

#interface web
app.mount("/static", StaticFiles(directory="static"), name="static")

#route GET pour la page d’accueil
def root():
    return FileResponse("static/index.html")
