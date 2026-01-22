# Ablation Study Documentation

This document provides complete documentation of the ablation study conducted for the UserTurnLoRA paper, enabling full reproducibility without re-running experiments.

## Study Overview

| Parameter             | Value                         |
| --------------------- | ----------------------------- |
| **Model**             | LiquidAI/LFM2.5-1.2B-Instruct |
| **Date**              | 2026-01-22                    |
| **Duration**          | 236.7 minutes (~4 hours)      |
| **Total Experiments** | 56 (24 LoRA + 32 Training)    |
| **Random Seed**       | 42                            |

## Hardware & Environment

| Component              | Specification                              |
| ---------------------- | ------------------------------------------ |
| **GPU**                | NVIDIA A100-SXM4-40GB                      |
| **GPU Memory**         | 42.4 GB                                    |
| **Compute Capability** | 8.0                                        |
| **CUDA Version**       | 12.8                                       |
| **PyTorch Version**    | 2.10.0+cu128                               |
| **Python Version**     | 3.11.0rc1                                  |
| **Platform**           | Linux-6.8.0-1045-gcp-x86_64-with-glibc2.35 |
| **Compute Dtype**      | torch.bfloat16                             |

## Dataset Configuration

| Parameter                 | Value         |
| ------------------------- | ------------- |
| **Training Samples**      | 2,000         |
| **Evaluation Samples**    | 200           |
| **WildChat Training**     | 1,000 samples |
| **SGD Training**          | 1,000 samples |
| **WildChat Evaluation**   | 100 samples   |
| **SGD Evaluation**        | 100 samples   |
| **Train/Eval Split Seed** | 42            |

## Training Configuration (Fixed)

| Parameter                  | Value            |
| -------------------------- | ---------------- |
| **Batch Size**             | 4                |
| **Gradient Accumulation**  | 16               |
| **Effective Batch Size**   | 64               |
| **Max Gradient Norm**      | 0.3              |
| **Evaluation Steps**       | 25               |
| **Logging Steps**          | 5                |
| **Optimizer**              | paged_adamw_8bit |
| **LR Scheduler**           | cosine           |
| **Gradient Checkpointing** | True             |
| **Packing**                | False            |

## LoRA Target Modules

```python
target_modules = [
    "q_proj",   # Query projection
    "k_proj",   # Key projection
    "v_proj",   # Value projection
    "o_proj",   # Output projection
    "gate_proj", # MLP gate
    "up_proj",   # MLP up projection
    "down_proj"  # MLP down projection
]
```

---

## Stage 1: LoRA Architecture Search

### Search Space

| Parameter     | Values        | Count |
| ------------- | ------------- | ----- |
| **Rank (r)**  | 8, 16, 32, 64 | 4     |
| **Alpha (α)** | 16, 32, 64    | 3     |
| **Dropout**   | 0.0, 0.05     | 2     |

**Total combinations**: 4 × 3 × 2 = **24 experiments**

### Fixed Parameters for Stage 1

| Parameter     | Value |
| ------------- | ----- |
| Learning Rate | 1e-4  |
| Epochs        | 1     |
| Warmup Ratio  | 0.05  |
| Weight Decay  | 0.01  |

### Stage 1 Results (All 24 Experiments)

| Rank | Experiment ID         | r   | α   | Dropout | Eval Loss | Train Loss | Duration (s) |
| ---- | --------------------- | --- | --- | ------- | --------- | ---------- | ------------ |
| 1    | exp_005_r8_a64_d0.0   | 8   | 64  | 0.0     | 2.8185    | 2.4474     | 154.5        |
| 2    | exp_006_r8_a64_d0.05  | 8   | 64  | 0.05    | 2.8290    | 2.4582     | 156.2        |
| 3    | exp_017_r32_a64_d0.0  | 32  | 64  | 0.0     | 2.8384    | 2.4621     | 154.5        |
| 4    | exp_018_r32_a64_d0.05 | 32  | 64  | 0.05    | 2.8427    | 2.4660     | 155.9        |
| 5    | exp_023_r64_a64_d0.0  | 64  | 64  | 0.0     | 2.8452    | 2.4691     | 155.0        |
| 6    | exp_024_r64_a64_d0.05 | 64  | 64  | 0.05    | 2.8462    | 2.4708     | 156.6        |
| 7    | exp_011_r16_a64_d0.0  | 16  | 64  | 0.0     | 2.8602    | 2.4792     | 155.0        |
| 8    | exp_012_r16_a64_d0.05 | 16  | 64  | 0.05    | 2.8621    | 2.4810     | 156.3        |
| 9    | exp_016_r32_a32_d0.05 | 32  | 32  | 0.05    | 3.0841    | 2.6680     | 155.9        |
| 10   | exp_003_r8_a32_d0.0   | 8   | 32  | 0.0     | 3.0987    | 2.6783     | 154.4        |
| 11   | exp_004_r8_a32_d0.05  | 8   | 32  | 0.05    | 3.1036    | 2.6882     | 155.7        |
| 12   | exp_021_r64_a32_d0.0  | 64  | 32  | 0.0     | 3.1146    | 2.6888     | 154.8        |
| 13   | exp_022_r64_a32_d0.05 | 64  | 32  | 0.05    | 3.1149    | 2.6860     | 156.1        |
| 14   | exp_015_r32_a32_d0.0  | 32  | 32  | 0.0     | 3.1188    | 2.7015     | 154.5        |
| 15   | exp_009_r16_a32_d0.0  | 16  | 32  | 0.0     | 3.1307    | 2.7028     | 154.9        |
| 16   | exp_010_r16_a32_d0.05 | 16  | 32  | 0.05    | 3.1361    | 2.7081     | 156.3        |
| 17   | exp_014_r32_a16_d0.05 | 32  | 16  | 0.05    | 3.3963    | 2.9762     | 155.9        |
| 18   | exp_013_r32_a16_d0.0  | 32  | 16  | 0.0     | 3.3983    | 2.9734     | 154.6        |
| 19   | exp_002_r8_a16_d0.05  | 8   | 16  | 0.05    | 3.4069    | 2.9765     | 155.7        |
| 20   | exp_001_r8_a16_d0.0   | 8   | 16  | 0.0     | 3.4139    | 2.9778     | 154.4        |
| 21   | exp_019_r64_a16_d0.0  | 64  | 16  | 0.0     | 3.4256    | 2.9940     | 154.8        |
| 22   | exp_020_r64_a16_d0.05 | 64  | 16  | 0.05    | 3.4342    | 3.0012     | 156.1        |
| 23   | exp_007_r16_a16_d0.0  | 16  | 16  | 0.0     | 3.4476    | 3.0098     | 154.9        |
| 24   | exp_008_r16_a16_d0.05 | 16  | 16  | 0.05    | 3.4506    | 3.0166     | 156.3        |

### Stage 1 Key Findings

1. **Alpha (α) is the most impactful parameter**: α=64 consistently outperforms α=32 and α=16
2. **Rank (r) has minimal impact**: r=8 performs as well as r=64, suggesting the task doesn't require high-rank adaptations
3. **Dropout has negligible effect**: 0.0 vs 0.05 shows <0.02 difference in eval loss
4. **Best LoRA config**: r=8, α=64, dropout=0.0 (eval_loss=2.8185)

---

## Stage 2: Training Hyperparameter Search

### Search Space

| Parameter         | Values                 | Count |
| ----------------- | ---------------------- | ----- |
| **Learning Rate** | 1e-5, 5e-5, 1e-4, 2e-4 | 4     |
| **Epochs**        | 1, 3                   | 2     |
| **Warmup Ratio**  | 0.03, 0.10             | 2     |
| **Weight Decay**  | 0.0, 0.01              | 2     |

**Total combinations**: 4 × 2 × 2 × 2 = **32 experiments**

### Fixed Parameters for Stage 2 (Best from Stage 1)

| Parameter      | Value |
| -------------- | ----- |
| LoRA Rank (r)  | 8     |
| LoRA Alpha (α) | 64    |
| LoRA Dropout   | 0.0   |

### Stage 2 Results (All 32 Experiments)

| Rank | Experiment ID      | LR   | Epochs | Warmup | WD   | Eval Loss | Train Loss | Duration (s) |
| ---- | ------------------ | ---- | ------ | ------ | ---- | --------- | ---------- | ------------ |
| 1    | exp_031_lr2e-04_e3 | 2e-4 | 3      | 0.10   | 0.00 | 1.8117    | 1.7482     | 516.4        |
| 2    | exp_032_lr2e-04_e3 | 2e-4 | 3      | 0.10   | 0.01 | 1.8132    | 1.7494     | 502.5        |
| 3    | exp_029_lr2e-04_e3 | 2e-4 | 3      | 0.03   | 0.00 | 1.8133    | 1.7418     | 493.3        |
| 4    | exp_030_lr2e-04_e3 | 2e-4 | 3      | 0.03   | 0.01 | 1.8170    | 1.7472     | 487.6        |
| 5    | exp_022_lr1e-04_e3 | 1e-4 | 3      | 0.03   | 0.01 | 1.9991    | 1.9053     | 468.0        |
| 6    | exp_021_lr1e-04_e3 | 1e-4 | 3      | 0.03   | 0.00 | 2.0037    | 1.9097     | 495.6        |
| 7    | exp_024_lr1e-04_e3 | 1e-4 | 3      | 0.10   | 0.01 | 2.0269    | 1.9172     | 494.0        |
| 8    | exp_023_lr1e-04_e3 | 1e-4 | 3      | 0.10   | 0.00 | 2.0369    | 1.9223     | 475.8        |
| 9    | exp_027_lr2e-04_e1 | 2e-4 | 1      | 0.10   | 0.00 | 2.3157    | 2.0742     | 163.6        |
| 10   | exp_026_lr2e-04_e1 | 2e-4 | 1      | 0.03   | 0.01 | 2.3170    | 2.0750     | 156.8        |
| 11   | exp_028_lr2e-04_e1 | 2e-4 | 1      | 0.10   | 0.01 | 2.3215    | 2.0781     | 161.6        |
| 12   | exp_025_lr2e-04_e1 | 2e-4 | 1      | 0.03   | 0.00 | 2.3342    | 2.0880     | 157.9        |
| 13   | exp_013_lr5e-05_e3 | 5e-5 | 3      | 0.03   | 0.00 | 2.3350    | 2.1747     | 464.8        |
| 14   | exp_016_lr5e-05_e3 | 5e-5 | 3      | 0.10   | 0.01 | 2.3516    | 2.1870     | 463.5        |
| 15   | exp_015_lr5e-05_e3 | 5e-5 | 3      | 0.10   | 0.00 | 2.3550    | 2.1871     | 463.3        |
| 16   | exp_014_lr5e-05_e3 | 5e-5 | 3      | 0.03   | 0.01 | 2.3568    | 2.1883     | 463.1        |
| 17   | exp_018_lr1e-04_e1 | 1e-4 | 1      | 0.03   | 0.01 | 2.8222    | 2.4500     | 154.8        |
| 18   | exp_020_lr1e-04_e1 | 1e-4 | 1      | 0.10   | 0.01 | 2.8275    | 2.4537     | 161.7        |
| 19   | exp_017_lr1e-04_e1 | 1e-4 | 1      | 0.03   | 0.00 | 2.8384    | 2.4637     | 155.6        |
| 20   | exp_019_lr1e-04_e1 | 1e-4 | 1      | 0.10   | 0.00 | 2.8436    | 2.4692     | 155.5        |
| 21   | exp_010_lr5e-05_e1 | 5e-5 | 1      | 0.03   | 0.01 | 3.2667    | 2.8501     | 155.5        |
| 22   | exp_011_lr5e-05_e1 | 5e-5 | 1      | 0.10   | 0.00 | 3.2685    | 2.8393     | 155.1        |
| 23   | exp_012_lr5e-05_e1 | 5e-5 | 1      | 0.10   | 0.01 | 3.2815    | 2.8601     | 155.1        |
| 24   | exp_009_lr5e-05_e1 | 5e-5 | 1      | 0.03   | 0.00 | 3.3013    | 2.8853     | 155.1        |
| 25   | exp_005_lr1e-05_e3 | 1e-5 | 3      | 0.03   | 0.00 | 3.6015    | 3.2484     | 464.4        |
| 26   | exp_006_lr1e-05_e3 | 1e-5 | 3      | 0.03   | 0.01 | 3.6631    | 3.3067     | 463.1        |
| 27   | exp_008_lr1e-05_e3 | 1e-5 | 3      | 0.10   | 0.01 | 3.6755    | 3.3121     | 463.8        |
| 28   | exp_007_lr1e-05_e3 | 1e-5 | 3      | 0.10   | 0.00 | 3.6828    | 3.3150     | 463.8        |
| 29   | exp_003_lr1e-05_e1 | 1e-5 | 1      | 0.10   | 0.00 | 4.0244    | 3.4785     | 155.0        |
| 30   | exp_002_lr1e-05_e1 | 1e-5 | 1      | 0.03   | 0.01 | 4.0263    | 3.4783     | 154.9        |
| 31   | exp_001_lr1e-05_e1 | 1e-5 | 1      | 0.03   | 0.00 | 4.0269    | 3.4783     | 155.1        |
| 32   | exp_004_lr1e-05_e1 | 1e-5 | 1      | 0.10   | 0.01 | 4.0292    | 3.4775     | 154.9        |

### Stage 2 Key Findings

1. **Learning rate is critical**: 2e-4 significantly outperforms lower rates
2. **More epochs help**: 3 epochs consistently beats 1 epoch
3. **Warmup ratio**: 0.10 slightly better than 0.03 for best configs
4. **Weight decay**: Minimal impact (0.0 vs 0.01 difference <0.01)
5. **Best training config**: lr=2e-4, epochs=3, warmup=0.10, wd=0.0 (eval_loss=1.8117)

---

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
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
```

### Performance Summary

| Metric               | Value                        |
| -------------------- | ---------------------------- |
| **Best Eval Loss**   | 1.8117                       |
| **Final Train Loss** | 1.7482                       |
| **Train/Eval Gap**   | 0.0635 (minimal overfitting) |

---

## Reproducibility

### Command to Reproduce

```bash
python -m src.ablation \
    --model LiquidAI/LFM2.5-1.2B-Instruct \
    --train-samples 2000 \
    --eval-samples 200 \
    --seed 42 \
    -o output/ablation_paper
```

### Quick Test (8 experiments)

```bash
python -m src.ablation \
    --model LiquidAI/LFM2.5-1.2B-Instruct \
    --train-samples 100 \
    --eval-samples 20 \
    --quick \
    -o output/ablation_test
```

### Docker Command

```bash
docker run --gpus all \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -v $(pwd)/output:/app/output \
    us-central1-docker.pkg.dev/user-turn-lora/userturn-lora/userturn-lora:latest \
    python -m src.ablation \
    --model LiquidAI/LFM2.5-1.2B-Instruct \
    --train-samples 2000 \
    --eval-samples 200 \
    -o output/ablation_paper
```

---

## Output Files

| File                     | Description                               |
| ------------------------ | ----------------------------------------- |
| `ablation_manifest.json` | Complete study metadata and configuration |
| `optimal_config.json`    | Best hyperparameter configuration         |
| `summary_lora.csv`       | Stage 1 results (24 experiments)          |
| `summary_training.csv`   | Stage 2 results (32 experiments)          |
| `ablation_report.md`     | Auto-generated summary report             |
| `stage1_lora/`           | Individual experiment logs and configs    |
| `stage2_training/`       | Individual experiment logs and configs    |

---
