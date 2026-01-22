#!/bin/bash
# Quick test of the pipeline with minimal data
# Usage: ./scripts/test_pipeline.sh

set -e

echo "========================================"
echo "UserTurnLoRA Pipeline - Quick Test"
echo "========================================"

cd "$(dirname "$0")/.."

python -m src.main \
    --model "Qwen/Qwen2.5-3B-Instruct" \
    --train-samples 50 \
    --eval-samples 5 \
    --epochs 1 \
    --no-wandb \
    --output-dir output/test_run \
    "$@"

echo "========================================"
echo "Test complete! Check output/test_run/"
echo "========================================"
