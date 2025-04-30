import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

# 1. Charger les données
def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    # Fusionner prompt + réponse comme texte à prédire
    texts = [{"text": f"Question : {item['prompt']}\nRéponse : {item['response']}"} for item in raw_data]
    return Dataset.from_list(texts)

# 2. Tokenization
def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

# 3. Entraînement LoRA
def main():
    model_name = "distilgpt2"
    data_path = "sport_qa.json" # Chemin relatif

    # Charger modèle et tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Ajouter token pad si besoin
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Préparer dataset
    dataset = load_dataset(data_path)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Configurer LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # Paramètres d'entraînement
    training_args = TrainingArguments(
        output_dir="./lora-output",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        report_to="none"
    )

    # Préparer trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Lancer l'entraînement
    trainer.train()
    model.save_pretrained("./lora-model")
    tokenizer.save_pretrained("./lora-model")
    print("✅ Fine-tuning terminé et modèle sauvegardé dans ./lora-model")

if __name__ == "__main__":
    main()
