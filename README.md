# 🏋️‍♀️ SportBot — Fine-tuned Chatbot using LoRA, FastAPI, Docker & Kubernetes

Un projet complet d'IA embarquant un modèle de langage léger (`distilgpt2`) fine-tuné via **LoRA**, déployé via **FastAPI**, conteneurisé avec **Docker**, et orchestré avec **Kubernetes (kind)**.

---

##  Objectif

Fournir un chatbot thématique sur le sport, capable de :

- Être fine-tuné via des données personnalisées
- Servir les réponses via une API REST
- S'adapter à la demande grâce à une architecture scalable

---

##  Stack technique

| Composant       | Tech utilisée             |
|------------------|----------------------------|
| 📘 Modèle         | `distilgpt2` + LoRA via `peft` |
| 🏋️ Dataset        | Fichier JSON Q&R sportives |
| 🔧 Entraînement   | Script Python (`train.py`) |
| 🧠 Inférence      | FastAPI + Transformers     |
| 🐳 Conteneurisation | Docker                    |
| ☸️ Orchestration   | Kubernetes avec `kind`     |
| 🌐 Frontend       | Interface HTML minimaliste |

---

##  Structure du projet

```
TROPHENIX_Exercice/
├── training/
│   ├── train.py
│   ├── sport_qa.json
├── inference/
│   ├── app.py
│   ├── lora-model/           # modèle fine-tuné
│   ├── static/index.html
├── k8s/                      # Fichiers YAML Kubernetes
│   ├── lora-api-deployment.yaml
│   ├── lora-api-service.yaml
│   └── lora-train-job.yaml
├── deploy.sh                 # Script de build + apply + port-forward
└── README.md
```

---

##  Lancer le projet (en une commande)

```bash
bash deploy.sh
```

Ce script :
- build les containers `lora-api` et `lora-train`
- les charge dans le cluster `kind`
- déploie les manifests K8s
- effectue un `port-forward` local
- lance l’application sur [http://localhost:8000](http://localhost:8000)

---

##  Pour re-fine-tuner le modèle (à la demande)

```bash
kubectl apply -f k8s/lora-train-job.yaml
```

---

##  Fonctionnalités

- Chatbot sport via FastAPI + LoRA
- Interface HTML simple
- Entraînement personnalisé
- Orchestration Kubernetes (kind)
- Prêt à déployer dans un cluster réel

---

## 👩‍💻 Réalisé par **Mariam**

Projet d'entraînement à l'ingénierie IA et MLOps pour un test.

---

