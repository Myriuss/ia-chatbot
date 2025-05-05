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

def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    texts = [{"text": f"Question: {item['prompt']}\nAnswer: {item['response']}"} for item in raw_data]
    return Dataset.from_list(texts)

def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

def main():
    print("📁 Fichiers présents dans /app :", os.listdir(), flush=True)

    model_name = "distilgpt2"
    data_path = "sport_qa.json"

    print("🚀 Téléchargement du tokenizer et modèle...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    print("📚 Chargement du dataset...", flush=True)
    dataset = load_dataset(data_path)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    print("🔀 Split train/validation...", flush=True)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_set = split_dataset["train"]
    eval_set = split_dataset["test"]

    print("🔧 Configuration LoRA...", flush=True)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    print("🧪 Configuration de l'entraînement...", flush=True)
    training_args = TrainingArguments(
        output_dir="./lora-output",
        per_device_train_batch_size=2,
        num_train_epochs=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=50,
        learning_rate=5e-5,
        report_to="none",
        remove_unused_columns=False
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("🔥 Lancement du fine-tuning...", flush=True)
    trainer.train()
    print("✅ Fine-tuning terminé.", flush=True)

    print("💾 Sauvegarde du modèle...", flush=True)
    model.save_pretrained("./lora-model")
    tokenizer.save_pretrained("./lora-model")
    print("🎉 Modèle sauvegardé dans ./lora-model", flush=True)

if __name__ == "__main__":
    main()
