#!/bin/bash

set -e

echo "🔁 Redémarrage de Colima..."
colima stop || true
colima start

echo "🔥 Suppression du cluster KIND existant (s'il existe)..."
kind delete cluster || true

echo "📦 Création d’un nouveau cluster KIND..."
kind create cluster

echo "🐳 Chargement de l'image Docker locale 'lora-api' dans KIND..."
kind load docker-image lora-api

echo "🚀 Déploiement de l'API FastAPI + HPA..."
kubectl apply -f k8s/lora-api-deployment.yaml
kubectl apply -f k8s/lora-api-service.yaml
kubectl apply -f k8s/lora-api-hpa.yaml

echo "✅ Cluster redémarré et application déployée avec succès !"
echo "🌐 Tu peux maintenant lancer : kubectl port-forward deployment/lora-api 8000:8000"
