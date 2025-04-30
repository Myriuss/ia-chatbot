FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# ✅ Copie le fichier API et le modèle fine-tuné
COPY app.py ./
COPY lora-model ./lora-model
COPY static ./static

# ✅ Installe les dépendances
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    torch \
    transformers \
    peft

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
