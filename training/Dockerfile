# Étape de base : image Python officielle
FROM python:3.10-slim

# Créer le dossier de travail
WORKDIR /app

# Installer git et autres dépendances système
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copier requirements et les fichiers
COPY train.py ./
COPY sport_qa.json ./

# Installer les bibliothèques Python
RUN pip install --no-cache-dir \
    torch \
    transformers \
    datasets \
    peft \
    accelerate

# Lancer le script d'entraînement
CMD ["python", "train.py"]
