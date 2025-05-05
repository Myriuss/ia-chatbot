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

| Composant                         | Tech utilisée                                                |
|-----------------------------------|--------------------------------------------------------------|
| Modèle                            | `distilgpt2` + LoRA via `peft` (pour le fine tuning)         |
| Dataset                           | Fichier JSON Q&R sportives                                   |
| Entraînement                      | Script Python (`train.py`)                                   |
| Inférence                         | FastAPI + Transformers  hugging face (traitement de langage) |
| Conteneurisation des composants   | Docker / colima                                              |
| Orchestration et deploement local | Kubernetes avec `kind`                                       |
| Frontend                          | Interface HTML minimaliste                                   
 Metrics Server + HPA              | scalabilité automatique                                      

---

##  Structure du projet

```
TROPHENIX_Exercice/
├── inference/               # API FastAPI + interface web
│   ├── app.py               # API REST pour prédiction
│   ├── static/              # Frontend HTML/CSS/JS
│   │   ├── index.html       # Interface du chatbot
│   │   ├── style.css        # Design (custom ou template)
│   │   └── script.js        # Requête à l’API
│   ├── Dockerfile           # Image Docker de l'API
│   └── lora-model/          # Modèle fine-tuné (copié après entraînement)
│
├── training/                # Fine-tuning du modèle
│   ├── train.py             # Script d’entraînement LoRA
│   ├── sport_qa.json        # Dataset QA Sport (format JSON)
│   └── Dockerfile           # Image Docker pour entraîner
│
├── k8s/                     # Déploiement Kubernetes
│   ├── lora-api-deployment.yaml     # Déploiement API
│   ├── lora-api-service.yaml        # Service (NodePort)
│   ├── lora-api-hpa.yaml            # Autoscaler HPA
│   ├── lora-train-job.yaml          # Job unique d'entraînement
│   └── components.yaml              # Metrics Server (HPA)
│
├── deploy.sh               # Script pour build et déployer toute l’app
└── README.md               
```

---

##  Lancer le projet (en une commande)

```bash
bash deploy.sh
```

Ce script :
- build les containers `lora-inference` et `lora-train`
- créer un cluster et les charge dans le cluster `kind`
- déploie les manifests K8s
- effectue un `port-forward` local
- après on peut lancer l’application sur [http://localhost:8000](http://localhost:8000)

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

