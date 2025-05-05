import os
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

#fonction utilitaire pour charger un fichier JSON et le convertir en dataset
def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    texts = [{"text": f"Question: {item['prompt']}\nAnswer: {item['response']}"} for item in raw_data]
    return Dataset.from_list(texts)

#fonction pour tokeniser un échantillon de texte
def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

#fonction principale contenant toutes les étapes du pipeline
def main():
    #affiche les fichiers présents dans le dossier courant du conteneur docker
    print("Fichiers présents dans /app :", os.listdir(), flush=True)

    #nom du modèle pré-entraîné utilisé comme base
    model_name = "distilgpt2"
    data_path = "sport_qa.json" #le dataset

    #chargement du tokenizer et du modèle depuis Hugging Face
    print("Téléchargement du tokenizer et modèle", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    #configuration du token de padding (obligatoire pour DistilGPT2)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    print("Chargement du dataset", flush=True)
    dataset = load_dataset(data_path) #chargement et préparation du dataset
    #tokenisation de toutes les entrées du dataset
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    #separation des données en jeu d'entraînement et de validation
    print("Split train/validation", flush=True)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_set = split_dataset["train"]
    eval_set = split_dataset["test"]

    #configuration de LoRA pour faire un fine-tuning allégé du modèle
    print("Configuration LoRA en cours", flush=True)
    lora_config = LoraConfig(
        r=8, #taille du rang pour la décomposition
        lora_alpha=16, #facteur de scaling
        target_modules=["c_attn"], #cible les couches d'attention du modèle
        lora_dropout=0.1,#taux de dropout pour éviter l'overfitting
        bias="none", #Pas de biais ajouté
        task_type=TaskType.CAUSAL_LM #Type de tâche (modèle de langage causal)
    )
    model = get_peft_model(model, lora_config)

    #paramètres d'entraînement
    print("Configuration de l'entraînement en cours", flush=True)
    training_args = TrainingArguments(
        output_dir="./lora-output", #dossier de sortie pour les checkpoints
        per_device_train_batch_size=2,#taille de batch
        num_train_epochs=4,
        evaluation_strategy="epoch",#evaluation a chaque fin d’époque
        save_strategy="epoch",#sauvegarde de chaque epoque
        save_total_limit=1,#garder seulement le dernier modele
        logging_dir="./logs",#dossier le logs
        logging_steps=50,
        learning_rate=5e-5, #taux d’apprentissage
        report_to="none",#pas de reporting vers wandb ou autres
        remove_unused_columns=False # necessaire pour peft, sinon certains champs sont supprimer automatiquement
    )

    #création d’un data collator pour la génération de texte (sans masque MLM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    #creation de l'objet Trainer de Hugging Face
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        tokenizer=tokenizer,
        data_collator=data_collator
    )


    print("Lancement du fine-tuning", flush=True)
    trainer.train()#entraînement du modèle
    print("✅✅✅✅Fine-tuning terminé.", flush=True)
    #puis sauvegarde
    print("Sauvegarde du modèle...", flush=True)
    model.save_pretrained("./lora-model")
    tokenizer.save_pretrained("./lora-model")
    print("Modèle sauvegardé dans ./lora-model", flush=True)

if __name__ == "__main__":
    main()
