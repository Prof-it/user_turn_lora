"""
Ablation Study Module for UserTurnLoRA
======================================

This module performs a two-stage ablation study:
1. LoRA Architecture Search: Finds optimal rank (r), alpha, and dropout
2. Training Hyperparameter Search: Finds optimal learning rate, epochs, etc.

All experiments are logged with:
- Full configuration details
- Training/eval loss curves
- Runtime statistics
- Hardware utilization
- Reproducibility seeds

Output Structure:
    ablation_results/
    ├── ablation_manifest.json      # Full study metadata
    ├── stage1_lora/
    │   ├── experiment_001/
    │   │   ├── config.json
    │   │   ├── metrics.json
    │   │   └── training_log.csv
    │   └── ...
    ├── stage2_training/
    │   └── ...
    ├── summary_lora.csv
    ├── summary_training.csv
    ├── optimal_config.json
    └── ablation_report.md

Usage:
    python -m src.ablation --model Qwen/Qwen2.5-3B-Instruct --train-samples 1000
"""

import gc
import itertools
import json
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from .config import PipelineConfig, get_config, SPECIAL_TOKENS
from .data import load_data, format_for_training
from .model import load_tokenizer, get_quantization_config


@dataclass
class AblationConfig:
    """
    Configuration for ablation study hyperparameter search space.
    
    This implements a two-stage grid search as recommended for academic papers:
    
    Stage 1 - LoRA Architecture Search:
        - Rank (r): Controls expressiveness vs efficiency tradeoff
        - Alpha: Scaling factor, typically 2x rank
        - Dropout: Regularization for overfitting prevention
        
    Stage 2 - Training Hyperparameter Search:
        - Learning rate: Most critical hyperparameter
        - Epochs: Training duration
        - Warmup ratio: Learning rate warmup schedule
        - Weight decay: L2 regularization
    
    Default search space yields:
        - Stage 1: 4 × 3 × 2 = 24 experiments (LoRA architecture)
        - Stage 2: 4 × 2 × 2 × 2 = 32 experiments (training params)
        - Total: 56 experiments (suitable for IEEE publication)
    
    For quick testing, use --skip-training or reduce parameter lists.
    
    Attributes:
        lora_r: List of LoRA rank values to test
        lora_alpha: List of LoRA alpha values to test  
        lora_dropout: List of dropout rates to test
        learning_rate: List of learning rates to test
        num_epochs: List of epoch counts to test
        warmup_ratio: List of warmup ratios to test
        weight_decay: List of weight decay values to test
        batch_size: Per-device batch size
        grad_accum: Gradient accumulation steps
        train_samples: Number of training samples for ablation
        eval_samples: Number of eval samples for ablation
        seed: Random seed for reproducibility
    """
    
    # LoRA parameters to search (Stage 1)
    # r: 8, 16, 32, 64 - covers low to high capacity
    # alpha: 16, 32, 64 - typically 1x to 2x rank
    # dropout: 0.0, 0.05 - with/without regularization
    lora_r: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    lora_alpha: List[int] = field(default_factory=lambda: [16, 32, 64])
    lora_dropout: List[float] = field(default_factory=lambda: [0.0, 0.05])
    
    # Training parameters to search (Stage 2)
    # learning_rate: 1e-5 to 3e-4 covers conservative to aggressive
    # epochs: 1-3 for ablation (full training uses more)
    # warmup_ratio: 0.03-0.1 standard range
    # weight_decay: 0.0-0.01 with/without regularization
    learning_rate: List[float] = field(default_factory=lambda: [1e-5, 5e-5, 1e-4, 2e-4])
    num_epochs: List[int] = field(default_factory=lambda: [1, 3])
    warmup_ratio: List[float] = field(default_factory=lambda: [0.03, 0.1])
    weight_decay: List[float] = field(default_factory=lambda: [0.0, 0.01])
    
    # Fixed training params
    batch_size: int = 4
    grad_accum: int = 16
    max_grad_norm: float = 0.3
    eval_steps: int = 25
    logging_steps: int = 5
    
    # Dataset sizes
    train_samples: int = 1000
    eval_samples: int = 100
    
    # Reproducibility
    seed: int = 42
    
    # LoRA target modules
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    def to_dict(self) -> Dict:
        return asdict(self)


class MetricsCallback(TrainerCallback):
    """Callback to track detailed training metrics for academic reporting."""
    
    def __init__(self):
        self.train_logs = []
        self.eval_logs = []
        self.start_time = None
        self.end_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        
    def on_train_end(self, args, state, control, **kwargs):
        self.end_time = datetime.now()
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                "timestamp": datetime.now().isoformat(),
            }
            
            if "loss" in logs:
                log_entry["train_loss"] = logs["loss"]
                log_entry["learning_rate"] = logs.get("learning_rate", 0)
                log_entry["grad_norm"] = logs.get("grad_norm", 0)
                self.train_logs.append(log_entry)
                
            if "eval_loss" in logs:
                eval_entry = {
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "eval_loss": logs["eval_loss"],
                    "timestamp": datetime.now().isoformat(),
                }
                self.eval_logs.append(eval_entry)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary metrics for reporting."""
        train_losses = [l["train_loss"] for l in self.train_logs if "train_loss" in l]
        eval_losses = [l["eval_loss"] for l in self.eval_logs if "eval_loss" in l]
        
        return {
            "train_losses": train_losses,
            "eval_losses": eval_losses,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_eval_loss": eval_losses[-1] if eval_losses else None,
            "best_eval_loss": min(eval_losses) if eval_losses else None,
            "min_train_loss": min(train_losses) if train_losses else None,
            "total_steps": len(train_losses),
            "eval_steps": len(eval_losses),
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
        }
    
    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Export logs as DataFrames."""
        train_df = pd.DataFrame(self.train_logs) if self.train_logs else pd.DataFrame()
        eval_df = pd.DataFrame(self.eval_logs) if self.eval_logs else pd.DataFrame()
        return train_df, eval_df


def get_system_info() -> Dict[str, Any]:
    """Collect system information for reproducibility."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["compute_capability"] = torch.cuda.get_device_capability(0)
    
    return info


def run_single_experiment(
    exp_id: str,
    config: Dict[str, Any],
    train_dataset: Dataset,
    eval_dataset: Dataset,
    model_name: str,
    bnb_config,
    compute_dtype: torch.dtype,
    ablation_cfg: AblationConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run a single ablation experiment with full documentation.
    
    Args:
        exp_id: Unique experiment identifier
        config: Hyperparameter configuration for this experiment
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        model_name: HuggingFace model name
        bnb_config: BitsAndBytes quantization config
        compute_dtype: Compute dtype (bfloat16 or float16)
        ablation_cfg: Ablation study configuration
        output_dir: Directory to save experiment results
    
    Returns:
        Dict with experiment results and metrics
    """
    exp_dir = output_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize result structure
    result = {
        "experiment_id": exp_id,
        "config": config,
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "metrics": {},
        "error": None,
    }
    
    # Save config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"  [{exp_id}] r={config['lora_r']}, α={config['lora_alpha']}, "
          f"lr={config['learning_rate']:.0e}", end=" ")
    
    try:
        # Load fresh model
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Use the compute_dtype passed in (already auto-detected for GPU)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        
        # LoRA config
        peft_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=ablation_cfg.target_modules,
        )
        
        # Training config
        # On T4 (compute capability < 8), disable mixed precision to avoid bfloat16 issues
        use_bf16 = compute_dtype == torch.bfloat16
        batch_size = ablation_cfg.batch_size if use_bf16 else 2
        grad_accum = ablation_cfg.grad_accum if use_bf16 else 32
        
        train_config = SFTConfig(
            output_dir=str(exp_dir / "checkpoints"),
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=config["learning_rate"],
            lr_scheduler_type="cosine",
            warmup_ratio=config["warmup_ratio"],
            weight_decay=config["weight_decay"],
            max_grad_norm=ablation_cfg.max_grad_norm,
            logging_steps=ablation_cfg.logging_steps,
            eval_strategy="steps",
            eval_steps=ablation_cfg.eval_steps,
            save_strategy="no",
            bf16=use_bf16,
            fp16=False,  # Disable fp16 AMP - model already quantized
            optim="paged_adamw_8bit",
            packing=False,
            gradient_checkpointing=True,
            report_to="none",
            dataloader_num_workers=0,
            seed=ablation_cfg.seed,
        )
        
        # Train with metrics callback
        metrics_cb = MetricsCallback()
        trainer = SFTTrainer(
            model=model,
            args=train_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            callbacks=[metrics_cb],
        )
        
        train_result = trainer.train()
        
        # Collect metrics
        summary = metrics_cb.get_summary()
        summary["train_runtime"] = train_result.metrics.get("train_runtime", 0)
        summary["samples_per_second"] = train_result.metrics.get("train_samples_per_second", 0)
        
        result["metrics"] = summary
        result["status"] = "completed"
        result["end_time"] = datetime.now().isoformat()
        
        # Save detailed logs
        train_df, eval_df = metrics_cb.to_dataframe()
        if not train_df.empty:
            train_df.to_csv(exp_dir / "training_log.csv", index=False)
        if not eval_df.empty:
            eval_df.to_csv(exp_dir / "eval_log.csv", index=False)
        
        best_loss = summary.get("best_eval_loss")
        print(f"→ best_eval_loss={best_loss:.4f}" if best_loss else "→ done")
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["end_time"] = datetime.now().isoformat()
        print(f"→ FAILED: {e}")
    
    finally:
        # Cleanup
        try:
            del trainer, model, tokenizer
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save result
    with open(exp_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    return result


def generate_lora_configs(cfg: AblationConfig) -> List[Dict[str, Any]]:
    """Generate LoRA parameter configurations for Stage 1."""
    configs = []
    for r, alpha, dropout in itertools.product(cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout):
        configs.append({
            "lora_r": r,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
            "learning_rate": 1e-4,  # Fixed for LoRA search
            "num_epochs": cfg.num_epochs[0],
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
        })
    return configs


def generate_training_configs(cfg: AblationConfig, best_lora: Dict) -> List[Dict[str, Any]]:
    """Generate training parameter configurations for Stage 2."""
    configs = []
    for lr, epochs, warmup, wd in itertools.product(
        cfg.learning_rate, cfg.num_epochs, cfg.warmup_ratio, cfg.weight_decay
    ):
        configs.append({
            "lora_r": best_lora["lora_r"],
            "lora_alpha": best_lora["lora_alpha"],
            "lora_dropout": best_lora["lora_dropout"],
            "learning_rate": lr,
            "num_epochs": epochs,
            "warmup_ratio": warmup,
            "weight_decay": wd,
        })
    return configs


def analyze_stage_results(
    results: List[Dict],
    stage_name: str,
    output_dir: Path,
) -> Tuple[Optional[Dict], pd.DataFrame]:
    """
    Analyze results from an ablation stage.
    
    Returns:
        Tuple of (best_config, summary_dataframe)
    """
    completed = [r for r in results if r["status"] == "completed"]
    
    if not completed:
        print(f"  No completed experiments in {stage_name}!")
        return None, pd.DataFrame()
    
    # Build summary DataFrame
    rows = []
    for r in completed:
        cfg = r["config"]
        m = r["metrics"]
        rows.append({
            "experiment_id": r["experiment_id"],
            "lora_r": cfg.get("lora_r"),
            "lora_alpha": cfg.get("lora_alpha"),
            "lora_dropout": cfg.get("lora_dropout"),
            "learning_rate": cfg.get("learning_rate"),
            "num_epochs": cfg.get("num_epochs"),
            "warmup_ratio": cfg.get("warmup_ratio"),
            "weight_decay": cfg.get("weight_decay"),
            "best_eval_loss": m.get("best_eval_loss"),
            "final_eval_loss": m.get("final_eval_loss"),
            "final_train_loss": m.get("final_train_loss"),
            "duration_seconds": m.get("duration_seconds"),
            "total_steps": m.get("total_steps"),
        })
    
    # Sort by best_eval_loss, falling back to final_train_loss for None values
    df = pd.DataFrame(rows)
    df["_sort_key"] = df["best_eval_loss"].fillna(df["final_train_loss"]).fillna(float("inf"))
    df = df.sort_values("_sort_key").drop(columns=["_sort_key"])
    df.to_csv(output_dir / f"summary_{stage_name}.csv", index=False)
    
    # Print top 5
    print(f"\n  Top 5 configurations ({stage_name}):")
    print(df.head(5).to_string(index=False))
    
    # Return best config - use best_eval_loss if available, otherwise final_train_loss
    def get_sort_key(x):
        metrics = x.get("metrics", {})
        eval_loss = metrics.get("best_eval_loss")
        train_loss = metrics.get("final_train_loss")
        # Prefer eval loss, fallback to train loss, then infinity
        if eval_loss is not None:
            return eval_loss
        if train_loss is not None:
            return train_loss
        return float("inf")
    
    best_result = sorted(completed, key=get_sort_key)[0]
    return best_result["config"], df


def generate_report(
    manifest: Dict,
    lora_df: pd.DataFrame,
    training_df: pd.DataFrame,
    optimal: Dict,
    output_dir: Path,
):
    """Generate a markdown report for academic documentation."""
    
    report = f"""# Ablation Study Report

## Study Information

- **Model**: {manifest['model_name']}
- **Date**: {manifest['start_time']}
- **Duration**: {manifest.get('duration_minutes', 'N/A'):.1f} minutes
- **Seed**: {manifest['ablation_config']['seed']}

## System Information

- **Platform**: {manifest['system_info']['platform']}
- **GPU**: {manifest['system_info'].get('gpu_name', 'N/A')}
- **CUDA**: {manifest['system_info'].get('cuda_version', 'N/A')}
- **PyTorch**: {manifest['system_info']['torch_version']}

## Dataset

- **Training samples**: {manifest['ablation_config']['train_samples']}
- **Evaluation samples**: {manifest['ablation_config']['eval_samples']}

## Stage 1: LoRA Architecture Search

Searched over:
- **Rank (r)**: {manifest['ablation_config']['lora_r']}
- **Alpha**: {manifest['ablation_config']['lora_alpha']}
- **Dropout**: {manifest['ablation_config']['lora_dropout']}

Total experiments: {len(lora_df)}

### Top 5 Configurations

| Rank | Alpha | Dropout | Best Eval Loss | Duration (s) |
|------|-------|---------|----------------|--------------|
"""
    
    for _, row in lora_df.head(5).iterrows():
        eval_loss = row['best_eval_loss'] if pd.notna(row['best_eval_loss']) else row['final_train_loss']
        eval_str = f"{eval_loss:.4f}" if pd.notna(eval_loss) else "N/A"
        dur_str = f"{row['duration_seconds']:.1f}" if pd.notna(row['duration_seconds']) else "N/A"
        report += f"| {row['lora_r']} | {row['lora_alpha']} | {row['lora_dropout']} | {eval_str} | {dur_str} |\n"
    
    report += f"""

## Stage 2: Training Hyperparameter Search

Using best LoRA config: r={optimal['lora_r']}, α={optimal['lora_alpha']}, dropout={optimal['lora_dropout']}

Searched over:
- **Learning rate**: {manifest['ablation_config']['learning_rate']}
- **Epochs**: {manifest['ablation_config']['num_epochs']}
- **Warmup ratio**: {manifest['ablation_config']['warmup_ratio']}
- **Weight decay**: {manifest['ablation_config']['weight_decay']}

Total experiments: {len(training_df)}

### Top 5 Configurations

| Learning Rate | Epochs | Warmup | Weight Decay | Best Eval Loss |
|---------------|--------|--------|--------------|----------------|
"""
    
    for _, row in training_df.head(5).iterrows():
        eval_loss = row['best_eval_loss'] if pd.notna(row['best_eval_loss']) else row['final_train_loss']
        eval_str = f"{eval_loss:.4f}" if pd.notna(eval_loss) else "N/A"
        report += f"| {row['learning_rate']:.0e} | {row['num_epochs']} | {row['warmup_ratio']} | {row['weight_decay']} | {eval_str} |\n"
    
    report += f"""

## Optimal Configuration

```json
{json.dumps(optimal, indent=2)}
```

## Reproducibility

To reproduce this study:

```bash
python -m src.ablation \\
    --model {manifest['model_name']} \\
    --train-samples {manifest['ablation_config']['train_samples']} \\
    --eval-samples {manifest['ablation_config']['eval_samples']} \\
    --seed {manifest['ablation_config']['seed']}
```

## Files

- `ablation_manifest.json` - Full study metadata
- `summary_lora.csv` - Stage 1 results
- `summary_training.csv` - Stage 2 results
- `optimal_config.json` - Best configuration
- `stage1_lora/` - Individual experiment logs
- `stage2_training/` - Individual experiment logs
"""
    
    with open(output_dir / "ablation_report.md", "w") as f:
        f.write(report)
    
    print(f"\nReport saved to {output_dir / 'ablation_report.md'}")


def run_ablation(
    model_name: str,
    output_dir: str = "ablation_results",
    ablation_config: Optional[AblationConfig] = None,
    skip_lora: bool = False,
    skip_training: bool = False,
) -> Dict[str, Any]:
    """
    Run full ablation study with academic documentation.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save all results
        ablation_config: Configuration for search space
        skip_lora: Skip LoRA parameter search (use defaults)
        skip_training: Skip training parameter search
    
    Returns:
        Dict with optimal hyperparameters
    """
    cfg = ablation_config or AblationConfig()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    torch.manual_seed(cfg.seed)
    
    # Collect system info
    system_info = get_system_info()
    
    # Auto-detect compute dtype based on GPU capability
    # - T4/V100 (compute capability < 8): float16
    # - A100/H100 (compute capability >= 8): bfloat16
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()[0]
        compute_dtype = torch.bfloat16 if compute_capability >= 8 else torch.float16
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Detected GPU: {gpu_name} (compute capability {compute_capability}.x)")
        print(f"Auto-selected dtype: {compute_dtype}")
    else:
        compute_dtype = torch.float16
        print("No GPU detected, using float16")
    
    # Create BnB config with correct compute dtype for this GPU
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    # Initialize manifest
    manifest = {
        "model_name": model_name,
        "start_time": datetime.now().isoformat(),
        "system_info": system_info,
        "ablation_config": cfg.to_dict(),
        "compute_dtype": str(compute_dtype),
        "stages": {},
    }
    
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Output: {out_dir}")
    print(f"Compute dtype: {compute_dtype}")
    print(f"Train samples: {cfg.train_samples}")
    print(f"Eval samples: {cfg.eval_samples}")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    pipeline_config = get_config(
        model_name=model_name,
        num_train_samples=cfg.train_samples,
        num_eval_samples=cfg.eval_samples,
    )
    train_pairs, eval_pairs = load_data(pipeline_config)
    
    tokenizer = load_tokenizer(pipeline_config)
    train_ds = format_for_training(train_pairs, tokenizer, pipeline_config)
    eval_ds = format_for_training(eval_pairs, tokenizer, pipeline_config)
    
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    
    # Stage 1: LoRA ablation
    if skip_lora:
        best_lora = {"lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05}
        lora_df = pd.DataFrame()
        print(f"\nSkipping LoRA ablation, using defaults: {best_lora}")
    else:
        lora_configs = generate_lora_configs(cfg)
        print(f"\n[Stage 1] LoRA Architecture Search: {len(lora_configs)} experiments")
        
        lora_results = []
        stage1_dir = out_dir / "stage1_lora"
        stage1_dir.mkdir(exist_ok=True)
        
        for i, exp_cfg in enumerate(lora_configs):
            exp_id = f"exp_{i+1:03d}_r{exp_cfg['lora_r']}_a{exp_cfg['lora_alpha']}_d{exp_cfg['lora_dropout']}"
            result = run_single_experiment(
                exp_id=exp_id,
                config=exp_cfg,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                model_name=model_name,
                bnb_config=bnb_config,
                compute_dtype=compute_dtype,
                ablation_cfg=cfg,
                output_dir=stage1_dir,
            )
            lora_results.append(result)
        
        best_lora, lora_df = analyze_stage_results(lora_results, "lora", out_dir)
        if best_lora is None:
            best_lora = {"lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05}
        
        manifest["stages"]["lora"] = {
            "num_experiments": len(lora_configs),
            "completed": len([r for r in lora_results if r["status"] == "completed"]),
            "best_config": best_lora,
        }
    
    # Stage 2: Training ablation
    if skip_training:
        best_training = {
            "learning_rate": 1e-4,
            "num_epochs": 3,
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
        }
        training_df = pd.DataFrame()
        print(f"\nSkipping training ablation, using defaults")
    else:
        training_configs = generate_training_configs(cfg, best_lora)
        print(f"\n[Stage 2] Training Hyperparameter Search: {len(training_configs)} experiments")
        
        training_results = []
        stage2_dir = out_dir / "stage2_training"
        stage2_dir.mkdir(exist_ok=True)
        
        for i, exp_cfg in enumerate(training_configs):
            exp_id = f"exp_{i+1:03d}_lr{exp_cfg['learning_rate']:.0e}_e{exp_cfg['num_epochs']}"
            result = run_single_experiment(
                exp_id=exp_id,
                config=exp_cfg,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                model_name=model_name,
                bnb_config=bnb_config,
                compute_dtype=compute_dtype,
                ablation_cfg=cfg,
                output_dir=stage2_dir,
            )
            training_results.append(result)
        
        best_training, training_df = analyze_stage_results(training_results, "training", out_dir)
        if best_training is None:
            best_training = best_lora.copy()
            best_training.update({
                "learning_rate": 1e-4,
                "num_epochs": 3,
                "warmup_ratio": 0.05,
                "weight_decay": 0.01,
            })
        
        manifest["stages"]["training"] = {
            "num_experiments": len(training_configs),
            "completed": len([r for r in training_results if r["status"] == "completed"]),
            "best_config": best_training,
        }
    
    # Combine optimal config
    optimal = {
        "lora_r": best_lora.get("lora_r", 32),
        "lora_alpha": best_lora.get("lora_alpha", 64),
        "lora_dropout": best_lora.get("lora_dropout", 0.05),
        "learning_rate": best_training.get("learning_rate", 1e-4),
        "num_epochs": best_training.get("num_epochs", 3),
        "warmup_ratio": best_training.get("warmup_ratio", 0.05),
        "weight_decay": best_training.get("weight_decay", 0.01),
        "target_modules": cfg.target_modules,
    }
    
    # Finalize manifest
    end_time = datetime.now()
    manifest["end_time"] = end_time.isoformat()
    manifest["duration_minutes"] = (end_time - datetime.fromisoformat(manifest["start_time"])).total_seconds() / 60
    manifest["optimal_config"] = optimal
    
    # Save outputs
    with open(out_dir / "ablation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    
    with open(out_dir / "optimal_config.json", "w") as f:
        json.dump(optimal, f, indent=2)
    
    # Generate report
    generate_report(manifest, lora_df, training_df, optimal, out_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATION")
    print("=" * 70)
    for k, v in optimal.items():
        if k != "target_modules":
            print(f"  {k}: {v}")
    print("=" * 70)
    print(f"Results saved to: {out_dir}")
    
    return optimal


def main():
    """CLI entry point for ablation study."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run ablation study for UserTurnLoRA hyperparameter search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full ablation study (56 experiments, IEEE quality)
    python -m src.ablation --model Qwen/Qwen2.5-3B-Instruct
    
    # Quick test (12 experiments)
    python -m src.ablation --model Qwen/Qwen2.5-3B-Instruct --quick
    
    # Custom search space
    python -m src.ablation --model Qwen/Qwen2.5-3B-Instruct \\
        --lora-r 16 32 64 --learning-rates 5e-5 1e-4 2e-4
    
    # Skip LoRA search, only tune training params
    python -m src.ablation --model Qwen/Qwen2.5-3B-Instruct --skip-lora

Search Space (default - IEEE ICETSIS quality):
    Stage 1 (LoRA): r=[8,16,32,64] × α=[16,32,64] × dropout=[0,0.05] = 24 experiments
    Stage 2 (Training): lr=[1e-5,5e-5,1e-4,2e-4] × epochs=[1,3] × warmup=[0.03,0.1] × wd=[0,0.01] = 32 experiments
    Total: 56 experiments

Quick mode (--quick):
    Stage 1: r=[16,32,64] × α=[32,64] × dropout=[0.05] = 6 experiments
    Stage 2: lr=[5e-5,1e-4] × epochs=[1] = 2 experiments
    Total: 8 experiments
        """
    )
    
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--output-dir", "-o", type=str, default="ablation_results",
                        help="Output directory")
    parser.add_argument("--train-samples", type=int, default=1000,
                        help="Training samples for ablation")
    parser.add_argument("--eval-samples", type=int, default=100,
                        help="Eval samples for ablation")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--skip-lora", action="store_true",
                        help="Skip LoRA parameter search")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training parameter search")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: reduced search space for testing (8 experiments)")
    
    # Custom search space (overrides defaults)
    parser.add_argument("--lora-r", type=int, nargs="+",
                        help="LoRA ranks to search (default: 8 16 32 64)")
    parser.add_argument("--lora-alpha", type=int, nargs="+",
                        help="LoRA alphas to search (default: 16 32 64)")
    parser.add_argument("--lora-dropout", type=float, nargs="+",
                        help="LoRA dropout rates (default: 0.0 0.05)")
    parser.add_argument("--learning-rates", type=float, nargs="+",
                        help="Learning rates to search (default: 1e-5 5e-5 1e-4 2e-4)")
    parser.add_argument("--epochs", type=int, nargs="+",
                        help="Epochs to search (default: 1 3)")
    
    args = parser.parse_args()
    
    # Build ablation config
    if args.quick:
        # Quick mode: minimal search space for testing
        ablation_cfg = AblationConfig(
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            lora_r=[16, 32, 64],
            lora_alpha=[32, 64],
            lora_dropout=[0.05],
            learning_rate=[5e-5, 1e-4],
            num_epochs=[1],
            warmup_ratio=[0.05],
            weight_decay=[0.01],
        )
    else:
        # Full search space (IEEE quality)
        ablation_cfg = AblationConfig(
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        # Override with custom values if provided
        if args.lora_r:
            ablation_cfg.lora_r = args.lora_r
        if args.lora_alpha:
            ablation_cfg.lora_alpha = args.lora_alpha
        if args.lora_dropout:
            ablation_cfg.lora_dropout = args.lora_dropout
        if args.learning_rates:
            ablation_cfg.learning_rate = args.learning_rates
        if args.epochs:
            ablation_cfg.num_epochs = args.epochs
    
    # Print search space summary
    n_lora = len(ablation_cfg.lora_r) * len(ablation_cfg.lora_alpha) * len(ablation_cfg.lora_dropout)
    n_training = len(ablation_cfg.learning_rate) * len(ablation_cfg.num_epochs) * len(ablation_cfg.warmup_ratio) * len(ablation_cfg.weight_decay)
    print(f"\nSearch space: {n_lora} LoRA + {n_training} training = {n_lora + n_training} total experiments")
    
    # Run ablation
    run_ablation(
        model_name=args.model,
        output_dir=args.output_dir,
        ablation_config=ablation_cfg,
        skip_lora=args.skip_lora,
        skip_training=args.skip_training,
    )


if __name__ == "__main__":
    main()
