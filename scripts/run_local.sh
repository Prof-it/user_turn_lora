#!/bin/bash
# Run the pipeline locally (without Docker)
# Usage: ./scripts/run_local.sh [model_name] [extra_args...]

set -e

MODEL=${1:-"Qwen/Qwen2.5-3B-Instruct"}
shift 2>/dev/null || true

echo "========================================"
echo "UserTurnLoRA Pipeline"
echo "========================================"
echo "Model: $MODEL"
echo "========================================"

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Run the pipeline
python -m src.main --model "$MODEL" "$@"
