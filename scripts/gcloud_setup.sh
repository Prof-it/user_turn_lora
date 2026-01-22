#!/bin/bash
# Setup GCloud project and deploy UserTurnLoRA
# Usage: ./scripts/gcloud_setup.sh

set -e

PROJECT_ID="user-turn-lora"
REGION="us-central1"
ZONE="us-central1-a"
IMAGE_NAME="userturn-lora"
ARTIFACT_REGISTRY="us-central1-docker.pkg.dev/${PROJECT_ID}/${IMAGE_NAME}/${IMAGE_NAME}"

echo "========================================"
echo "UserTurnLoRA GCloud Setup"
echo "========================================"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "========================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI not installed. Install from https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Create project (may fail if exists, that's ok)
echo "Creating project ${PROJECT_ID}..."
gcloud projects create ${PROJECT_ID} --name="User Turn LoRA" 2>/dev/null || echo "Project may already exist"

# Set project
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable compute.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Create Artifact Registry repository if it doesn't exist
echo "Creating Artifact Registry repository..."
gcloud artifacts repositories create ${IMAGE_NAME} \
    --repository-format=docker \
    --location=${REGION} \
    --description="UserTurnLoRA Docker images" 2>/dev/null || echo "Repository may already exist"

# Build and push Docker image using Cloud Build
echo "Building Docker image with Cloud Build..."
cd "$(dirname "$0")/.."
gcloud builds submit --config=cloudbuild.yaml --project=${PROJECT_ID} .

echo "Image built and pushed to Artifact Registry"

echo "========================================"
echo "Setup complete!"
echo "Image pushed to: ${ARTIFACT_REGISTRY}:latest"
echo ""
echo "Next: Run ./scripts/gcloud_run.sh to start a GPU instance"
echo "========================================"
