#!/bin/bash
# Run UserTurnLoRA on GCloud GPU instance
# Usage: ./scripts/gcloud_run.sh [model_name]

set -e

PROJECT_ID="user-turn-lora"
REGION="us-central1"
ZONE="us-central1-a"
INSTANCE_NAME="userturn-lora-a100"
MACHINE_TYPE="a2-highgpu-1g"
GPU_TYPE="nvidia-tesla-a100"
GPU_COUNT=1
IMAGE_NAME="userturn-lora"
ARTIFACT_REGISTRY="us-central1-docker.pkg.dev/${PROJECT_ID}/${IMAGE_NAME}/${IMAGE_NAME}"

MODEL=${1:-"LiquidAI/LFM2.5-1.2B-Instruct"}

echo "========================================"
echo "UserTurnLoRA GCloud GPU Run"
echo "========================================"
echo "Instance: ${INSTANCE_NAME}"
echo "GPU: ${GPU_TYPE} x ${GPU_COUNT}"
echo "Model: ${MODEL}"
echo "========================================"

# Set project
gcloud config set project ${PROJECT_ID}

# Check if instance exists
if gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} &>/dev/null; then
    echo "Instance ${INSTANCE_NAME} already exists. Starting it..."
    gcloud compute instances start ${INSTANCE_NAME} --zone=${ZONE}
else
    echo "Creating GPU instance ${INSTANCE_NAME}..."
    
    # Create instance with GPU
    gcloud compute instances create ${INSTANCE_NAME} \
        --zone=${ZONE} \
        --machine-type=${MACHINE_TYPE} \
        --accelerator=type=${GPU_TYPE},count=${GPU_COUNT} \
        --maintenance-policy=TERMINATE \
        --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=100GB \
        --boot-disk-type=pd-ssd \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata="install-nvidia-driver=True"
    
    echo "Waiting for instance to be ready..."
    sleep 60
fi

# Get instance IP
INSTANCE_IP=$(gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo "Instance IP: ${INSTANCE_IP}"

# Copy HF token if set
if [ -n "${HF_TOKEN}" ]; then
    echo "Setting up HuggingFace token..."
    gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="echo 'export HF_TOKEN=${HF_TOKEN}' >> ~/.bashrc"
fi

# Install Docker and NVIDIA Container Toolkit
echo "Setting up Docker and NVIDIA Container Toolkit..."
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y docker.io
        sudo systemctl start docker
        sudo usermod -aG docker \$USER
    fi
    
    # Install NVIDIA Container Toolkit
    if ! dpkg -l | grep -q nvidia-container-toolkit; then
        sudo rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        echo 'deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
    fi
"

# Configure Docker authentication and pull image
echo "Pulling Docker image..."
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="
    gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
    sudo gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
    sudo docker pull ${ARTIFACT_REGISTRY}:latest
"

# Run the pipeline
echo "Running pipeline on GPU instance..."
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="
    sudo docker run --gpus all \
        -v /home/\$USER/output:/app/output \
        -e HF_TOKEN=\${HF_TOKEN} \
        -e WANDB_API_KEY=\${WANDB_API_KEY} \
        -e HF_HUB_ENABLE_HF_TRANSFER=1 \
        ${ARTIFACT_REGISTRY}:latest \
        --model '${MODEL}' \
        --train-samples 100 \
        --eval-samples 10 \
        --epochs 1 \
        --no-wandb
"

echo "========================================"
echo "Pipeline complete!"
echo "Copying results..."
echo "========================================"

# Copy results back
gcloud compute scp --recurse ${INSTANCE_NAME}:/home/*/output ./output_gcloud --zone=${ZONE}

echo "Results saved to ./output_gcloud/"
echo ""
echo "To stop the instance (save costs):"
echo "  gcloud compute instances stop ${INSTANCE_NAME} --zone=${ZONE}"
echo ""
echo "To delete the instance:"
echo "  gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE}"
