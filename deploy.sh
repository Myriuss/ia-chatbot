#!/bin/bash

set -e

# 1. Construire les images Docker
docker build -t lora-train ./training
docker build -t lora-api ./inference

echo " Redémarrage de Colima..."
colima stop || true
colima start

echo " Suppression du cluster KIND (si existant)..."
kind delete cluster || true

echo "📦 Création d’un nouveau cluster KIND..."
kind create cluster

echo "Reconstruction de l'image Docker locale 'lora-api'..."
docker build -t lora-api ./inference

echo "Chargement de l'image 'lora-api' dans KIND..."
kind load docker-image lora-api

echo "🚀 Déploiement de l'application et du scaling HPA..."
kubectl apply -f k8s/lora-api-deployment.yaml
kubectl apply -f k8s/lora-api-service.yaml
kubectl apply -f k8s/lora-api-hpa.yaml
kubectl apply -f components.yaml

#echo "🧹 Nettoyage des anciens pods (au cas où)..."
#kubectl delete pod -l app=lora-api || true

echo "✅ Déploiement terminé. Attends quelques secondes..."
echo "🌐 Tu peux maintenant exécuter : kubectl port-forward deployment/lora-api 8000:8000"
