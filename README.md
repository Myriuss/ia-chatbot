# 🏋️‍♀️ SportBot — Fine-tuned Chatbot using LoRA & GPT-2

This project implements a lightweight, domain-specific chatbot focused on **sports and fitness**, built by fine-tuning `distilgpt2` with **LoRA (Low-Rank Adaptation)** and deploying it via a **FastAPI** web service. It also includes a simple web interface to test the chatbot.

---

## Features

-  Fine-tuning of `distilgpt2` with LoRA using PEFT
-  Custom sport/fitness Q&A dataset
-  Exposed API via FastAPI
-  Simple HTML frontend to ask questions
-  Fully containerized (Docker)

---

##  Project structure

```
project/
├── training/
│   ├── train.py
│   ├── sport_qa.json        # Dataset: 30 sport-related Q&As (EN)
│   └── lora-model/          # Fine-tuned model output
├── inference/
│   ├── app.py               # FastAPI app
│   ├── Dockerfile
│   ├── static/
│   │   └── index.html       # Web UI
├── docker-compose.yml       # (optional)
└── README.md
```

---

##  Requirements

- Docker + Colima (on macOS)
- ~5 GB free disk space

---

##  1. Fine-tune the model

```bash
cd training
docker build -t lora-train .
docker run --rm -v $(pwd)/lora-model:/app/lora-model lora-train
```

➡ Model saved in `training/lora-model/`

---

##  2. Launch the API

```bash
cp -r training/lora-model inference/
cd inference
docker build -t lora-api .
docker run -p 8001:8000 lora-api
```

 Open in browser: [http://localhost:8000(http://localhost:8000)

---

##  3. Test via cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How can I build muscle quickly?"}'
```

---

##  Dataset format

`training/sport_qa.json`:
```json
[
  {
    "prompt": "What are the best exercises for abs?",
    "response": "Crunches, leg raises, planks, and mountain climbers are very effective."
  },
  ...
]
```

---

##  (Optional) Run with docker-compose

```bash
docker-compose build
docker-compose up
```

---

## 👩‍💻 Built by Mariam

Apprentice Software Engineer passionate about AI & MLOps  
Fine-tuning LoRA + API + interface web made with 💪 & ☕  
