#!/bin/bash

# Build images
docker build -t lora-api ./inference
docker build -t lora-train ./training

# Load them into kind
kind load docker-image lora-api
kind load docker-image lora-train

# Apply Kubernetes configs
kubectl apply -f k8s/lora-api-deployment.yaml
kubectl apply -f k8s/lora-api-service.yaml
