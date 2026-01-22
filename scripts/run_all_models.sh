#!/bin/bash
# Run training for all 5 models sequentially on the VM
# Usage: Copy to VM and run with nohup

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY environment variable}"
export HF_HUB_ENABLE_HF_TRANSFER=1
IMAGE=us-central1-docker.pkg.dev/user-turn-lora/userturn-lora/userturn-lora:latest

LOG=/home/sebastianboehler/training.log
echo "Starting sequential training for 5 models..." > $LOG

MODELS=(
  "LiquidAI/LFM2.5-1.2B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "allenai/OLMo-3-7B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "mistralai/Ministral-3-3B-Instruct-2512"
)

for MODEL in "${MODELS[@]}"; do
  MODEL_SHORT=$(echo $MODEL | sed "s|/|-|g")
  echo "[$(date)] Starting $MODEL" >> $LOG
  sudo docker run --gpus all \
    -e "WANDB_API_KEY=$WANDB_API_KEY" \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -v /home/sebastianboehler/output:/app/output \
    $IMAGE --model "$MODEL" -o "output/${MODEL_SHORT}" 2>&1 | tee -a $LOG
  echo "[$(date)] Finished $MODEL" >> $LOG
done

echo "[$(date)] All models complete!" >> $LOG
