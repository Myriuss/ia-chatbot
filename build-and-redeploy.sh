#!/bin/bash

# === build-and-redeploy.sh ===
# Objectif : reconstruire uniquement si nécessaire, recharger dans KIND, redéployer rapidement

set -e

APP_NAME="lora-api"
IMAGE_NAME="$APP_NAME"
INFERENCE_DIR="./inference"
K8S_DIR="./k8s"

# 📅 Timestamp pour l'image
BUILD_TAG=$(date +%s)

echo "🔧 Étape 1 : Build de l'image Docker ($IMAGE_NAME:$BUILD_TAG)"
docker build -t $IMAGE_NAME:$BUILD_TAG $INFERENCE_DIR

echo "💵 Étape 2 : Chargement dans KIND"
kind load docker-image $IMAGE_NAME:$BUILD_TAG

#  Remplacer la version dans le déploiement YAML temporairement
TEMP_DEPLOY="$K8S_DIR/temp-deployment.yaml"
sed "s|image: $IMAGE_NAME:.*|image: $IMAGE_NAME:$BUILD_TAG|" $K8S_DIR/lora-api-deployment.yaml > $TEMP_DEPLOY

echo "🚀 Étape 3 : Déploiement"
kubectl apply -f $TEMP_DEPLOY
kubectl apply -f $K8S_DIR/lora-api-service.yaml
kubectl apply -f $K8S_DIR/lora-api-hpa.yaml

#  Nettoyer le temporaire
rm -f $TEMP_DEPLOY

echo "✅ Application déployée avec l'image taguée $BUILD_TAG"
echo "🌐 Accès : http://localhost:8000 (après port-forward)"
echo "🚪 Pour accéder : kubectl port-forward deployment/$APP_NAME 8000:8000"
