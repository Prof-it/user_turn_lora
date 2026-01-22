# Ablation Study Report

## Study Information

- **Model**: LiquidAI/LFM2.5-1.2B-Instruct
- **Date**: 2026-01-22T12:21:03.957693
- **Duration**: 236.7 minutes
- **Seed**: 42

## System Information

- **Platform**: Linux-6.8.0-1045-gcp-x86_64-with-glibc2.35
- **GPU**: NVIDIA A100-SXM4-40GB
- **CUDA**: 12.8
- **PyTorch**: 2.10.0+cu128

## Dataset

- **Training samples**: 2000
- **Evaluation samples**: 200

## Stage 1: LoRA Architecture Search

Searched over:
- **Rank (r)**: [8, 16, 32, 64]
- **Alpha**: [16, 32, 64]
- **Dropout**: [0.0, 0.05]

Total experiments: 24

### Top 5 Configurations

| Rank | Alpha | Dropout | Best Eval Loss | Duration (s) |
|------|-------|---------|----------------|--------------|
| 8 | 64 | 0.0 | 2.8185 | 154.5 |
| 8 | 64 | 0.05 | 2.8290 | 156.2 |
| 32 | 64 | 0.0 | 2.8384 | 154.5 |
| 32 | 64 | 0.05 | 2.8427 | 155.9 |
| 64 | 64 | 0.0 | 2.8452 | 155.0 |


## Stage 2: Training Hyperparameter Search

Using best LoRA config: r=8, α=64, dropout=0.0

Searched over:
- **Learning rate**: [1e-05, 5e-05, 0.0001, 0.0002]
- **Epochs**: [1, 3]
- **Warmup ratio**: [0.03, 0.1]
- **Weight decay**: [0.0, 0.01]

Total experiments: 32

### Top 5 Configurations

| Learning Rate | Epochs | Warmup | Weight Decay | Best Eval Loss |
|---------------|--------|--------|--------------|----------------|
| 2e-04 | 3 | 0.1 | 0.0 | 1.8117 |
| 2e-04 | 3 | 0.1 | 0.01 | 1.8132 |
| 2e-04 | 3 | 0.03 | 0.0 | 1.8133 |
| 2e-04 | 3 | 0.03 | 0.01 | 1.8170 |
| 1e-04 | 3 | 0.03 | 0.01 | 1.9991 |


## Optimal Configuration

```json
{
  "lora_r": 8,
  "lora_alpha": 64,
  "lora_dropout": 0.0,
  "learning_rate": 0.0002,
  "num_epochs": 3,
  "warmup_ratio": 0.1,
  "weight_decay": 0.0,
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
  ]
}
```

## Reproducibility

To reproduce this study:

```bash
python -m src.ablation \
    --model LiquidAI/LFM2.5-1.2B-Instruct \
    --train-samples 2000 \
    --eval-samples 200 \
    --seed 42
```

## Files

- `ablation_manifest.json` - Full study metadata
- `summary_lora.csv` - Stage 1 results
- `summary_training.csv` - Stage 2 results
- `optimal_config.json` - Best configuration
- `stage1_lora/` - Individual experiment logs
- `stage2_training/` - Individual experiment logs
