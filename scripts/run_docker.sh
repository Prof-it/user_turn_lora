#!/bin/bash
# Run the pipeline in Docker
# Usage: ./scripts/run_docker.sh [model_name] [extra_args...]

set -e

MODEL=${1:-"Qwen/Qwen2.5-3B-Instruct"}
shift 2>/dev/null || true

echo "========================================"
echo "UserTurnLoRA Pipeline (Docker)"
echo "========================================"
echo "Model: $MODEL"
echo "========================================"

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Build if needed
docker build -t userturn-lora:latest .

# Run with GPU support
docker run --rm \
    --gpus all \
    -v "$(pwd)/output:/app/output" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    userturn-lora:latest \
    --model "$MODEL" "$@"
