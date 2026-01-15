"""
Ablation Study Module for UserTurnLoRA
======================================
Systematic hyperparameter search for LoRA fine-tuning.

Usage in notebook:
    from modules.ablation import run_ablation, AblationConfig
    
    best_config = run_ablation(
        model_name=MODEL_NAME,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        bnb_config=bnb_config,
        compute_dtype=COMPUTE_DTYPE,
    )
"""

import gc
import itertools
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


@dataclass
class AblationConfig:
    """Configuration for ablation study hyperparameter search space."""
    
    # LoRA parameters to search
    lora_r: List[int] = field(default_factory=lambda: [16, 32, 64])
    lora_alpha: List[int] = field(default_factory=lambda: [32, 64])
    lora_dropout: List[float] = field(default_factory=lambda: [0.0, 0.05])
    
    # Training parameters to search
    learning_rate: List[float] = field(default_factory=lambda: [5e-5, 1e-4, 2e-4])
    num_epochs: List[int] = field(default_factory=lambda: [3])
    warmup_ratio: List[float] = field(default_factory=lambda: [0.05])
    weight_decay: List[float] = field(default_factory=lambda: [0.01])
    
    # Fixed training params
    batch_size: int = 4
    grad_accum: int = 16
    max_grad_norm: float = 0.3
    eval_steps: int = 25
    logging_steps: int = 5
    
    # Ablation dataset sizes (subset for speed)
    train_samples: int = 1000
    eval_samples: int = 100
    
    # LoRA target modules
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


class _MetricsCallback(TrainerCallback):
    """Callback to track training metrics."""
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_steps_list = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                self.eval_steps_list.append(state.global_step)
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "eval_steps": self.eval_steps_list,
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_eval_loss": self.eval_losses[-1] if self.eval_losses else None,
            "best_eval_loss": min(self.eval_losses) if self.eval_losses else None,
        }


def _generate_lora_configs(cfg: AblationConfig) -> List[Dict[str, Any]]:
    """Generate LoRA parameter configurations."""
    configs = []
    for r, alpha, dropout in itertools.product(cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout):
        configs.append({
            "name": f"r{r}_a{alpha}_d{dropout}",
            "lora_r": r,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
            "learning_rate": 1e-4,
            "num_epochs": cfg.num_epochs[0],
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
        })
    return configs


def _generate_training_configs(cfg: AblationConfig, best_lora: Dict) -> List[Dict[str, Any]]:
    """Generate training parameter configurations using best LoRA config."""
    configs = []
    for lr, epochs, warmup, wd in itertools.product(
        cfg.learning_rate, cfg.num_epochs, cfg.warmup_ratio, cfg.weight_decay
    ):
        configs.append({
            "name": f"lr{lr}_e{epochs}_w{warmup}_wd{wd}",
            "lora_r": best_lora["lora_r"],
            "lora_alpha": best_lora["lora_alpha"],
            "lora_dropout": best_lora["lora_dropout"],
            "learning_rate": lr,
            "num_epochs": epochs,
            "warmup_ratio": warmup,
            "weight_decay": wd,
        })
    return configs


def _run_single_experiment(
    config: Dict[str, Any],
    train_dataset,
    eval_dataset,
    model_name: str,
    bnb_config,
    compute_dtype,
    target_modules: List[str],
    ablation_cfg: AblationConfig,
    output_dir: Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single ablation experiment."""
    
    exp_name = config["name"]
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"  Running: {exp_name} (r={config['lora_r']}, α={config['lora_alpha']}, "
              f"lr={config['learning_rate']})")
    
    result = {"config": config, "status": "running", "metrics": {}, "error": None}
    
    try:
        # Load fresh model
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
        )
        model = prepare_model_for_kbit_training(model)
        
        # LoRA config
        peft_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        
        # Training config
        use_bf16 = compute_dtype == torch.bfloat16
        train_config = SFTConfig(
            output_dir=str(exp_dir),
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=ablation_cfg.batch_size if use_bf16 else 2,
            gradient_accumulation_steps=ablation_cfg.grad_accum if use_bf16 else 32,
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
            fp16=not use_bf16,
            optim="paged_adamw_8bit",
            packing=False,
            gradient_checkpointing=True,
            report_to="none",
            dataloader_num_workers=0,
        )
        
        # Train
        metrics_cb = _MetricsCallback()
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
        result["metrics"] = metrics_cb.get_metrics()
        result["metrics"]["train_runtime"] = train_result.metrics.get("train_runtime", 0)
        result["status"] = "completed"
        
        if verbose:
            best_loss = result["metrics"].get("best_eval_loss")
            print(f"    ✓ best_eval_loss={best_loss:.4f}" if best_loss else "    ✓ done")
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        if verbose:
            print(f"    ✗ Failed: {e}")
    
    finally:
        # Cleanup
        try:
            del trainer, model, tokenizer
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
    
    return result


def _analyze_results(results: List[Dict], title: str, output_dir: Path) -> Optional[Dict]:
    """Analyze ablation results and create visualizations."""
    
    completed = [r for r in results if r["status"] == "completed"]
    if not completed:
        print("No completed experiments!")
        return None
    
    # Build summary
    data = []
    for r in completed:
        cfg = r["config"]
        m = r["metrics"]
        data.append({
            "name": cfg["name"],
            "lora_r": cfg.get("lora_r"),
            "lora_alpha": cfg.get("lora_alpha"),
            "lora_dropout": cfg.get("lora_dropout"),
            "learning_rate": cfg.get("learning_rate"),
            "best_eval_loss": m.get("best_eval_loss"),
            "final_eval_loss": m.get("final_eval_loss"),
            "runtime_sec": m.get("train_runtime", 0),
        })
    
    df = pd.DataFrame(data).sort_values("best_eval_loss")
    
    print(f"\n{'='*60}")
    print(f"TOP 5 - {title}")
    print(f"{'='*60}")
    print(df.head(5).to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    ax1 = axes[0]
    top_n = min(10, len(df))
    bars = ax1.barh(range(top_n), df["best_eval_loss"].head(top_n), color="steelblue")
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(df["name"].head(top_n), fontsize=8)
    ax1.set_xlabel("Best Eval Loss")
    ax1.set_title(f"Top {top_n} Configurations")
    ax1.invert_yaxis()
    bars[0].set_color("green")
    
    # Learning curves for top 3
    ax2 = axes[1]
    sorted_results = sorted(completed, key=lambda x: x["metrics"].get("best_eval_loss", float("inf")))
    for r in sorted_results[:3]:
        if r["metrics"].get("eval_losses"):
            steps = r["metrics"].get("eval_steps", range(len(r["metrics"]["eval_losses"])))
            ax2.plot(steps, r["metrics"]["eval_losses"], label=r["config"]["name"], marker="o", markersize=3)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Eval Loss")
    ax2.set_title("Learning Curves (Top 3)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"ablation_{title.lower().replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # Return best config
    best = sorted_results[0]
    return {
        "config": best["config"],
        "best_eval_loss": best["metrics"]["best_eval_loss"],
        "df": df,
    }


def run_ablation(
    model_name: str,
    train_dataset,
    eval_dataset,
    bnb_config,
    compute_dtype: torch.dtype,
    config: Optional[AblationConfig] = None,
    output_dir: str = "ablation_results",
    skip_lora: bool = False,
    skip_training: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run full ablation study and return optimal configuration.
    
    Args:
        model_name: HuggingFace model name
        train_dataset: Training dataset (will be subsampled)
        eval_dataset: Eval dataset (will be subsampled)
        bnb_config: BitsAndBytes config
        compute_dtype: torch.bfloat16 or torch.float16
        config: AblationConfig (uses defaults if None)
        output_dir: Directory to save results
        skip_lora: Skip LoRA ablation (use defaults)
        skip_training: Skip training param ablation
        verbose: Print progress
    
    Returns:
        Dict with optimal config: {lora_r, lora_alpha, lora_dropout, learning_rate, ...}
    """
    cfg = config or AblationConfig()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Subsample datasets
    train_size = min(cfg.train_samples, len(train_dataset))
    eval_size = min(cfg.eval_samples, len(eval_dataset))
    train_ds = train_dataset.select(range(train_size))
    eval_ds = eval_dataset.select(range(eval_size))
    
    if verbose:
        print(f"Ablation Study for {model_name}")
        print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")
        print(f"  Output: {out_dir}")
    
    results_file = out_dir / "ablation_results.json"
    
    # Stage 1: LoRA ablation
    if skip_lora:
        best_lora = {"lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05}
        if verbose:
            print(f"\nSkipping LoRA ablation, using defaults: {best_lora}")
    else:
        lora_configs = _generate_lora_configs(cfg)
        if verbose:
            print(f"\n[Stage 1] LoRA Ablation: {len(lora_configs)} experiments")
        
        lora_results = []
        for i, exp_cfg in enumerate(lora_configs):
            if verbose:
                print(f"[{i+1}/{len(lora_configs)}]", end="")
            result = _run_single_experiment(
                config=exp_cfg,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                model_name=model_name,
                bnb_config=bnb_config,
                compute_dtype=compute_dtype,
                target_modules=cfg.target_modules,
                ablation_cfg=cfg,
                output_dir=out_dir / "lora",
                verbose=verbose,
            )
            lora_results.append(result)
        
        analysis = _analyze_results(lora_results, "LoRA Parameters", out_dir)
        best_lora = analysis["config"] if analysis else {"lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05}
    
    # Stage 2: Training param ablation
    if skip_training:
        best_training = {"learning_rate": 1e-4, "num_epochs": 3, "warmup_ratio": 0.05, "weight_decay": 0.01}
        if verbose:
            print(f"\nSkipping training ablation, using defaults: {best_training}")
    else:
        training_configs = _generate_training_configs(cfg, best_lora)
        if verbose:
            print(f"\n[Stage 2] Training Ablation: {len(training_configs)} experiments")
        
        training_results = []
        for i, exp_cfg in enumerate(training_configs):
            if verbose:
                print(f"[{i+1}/{len(training_configs)}]", end="")
            result = _run_single_experiment(
                config=exp_cfg,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                model_name=model_name,
                bnb_config=bnb_config,
                compute_dtype=compute_dtype,
                target_modules=cfg.target_modules,
                ablation_cfg=cfg,
                output_dir=out_dir / "training",
                verbose=verbose,
            )
            training_results.append(result)
        
        analysis = _analyze_results(training_results, "Training Parameters", out_dir)
        best_training = analysis["config"] if analysis else best_lora
    
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
    
    # Save
    with open(out_dir / "optimal_config.json", "w") as f:
        json.dump(optimal, f, indent=2)
    
    if verbose:
        print(f"\n{'='*60}")
        print("OPTIMAL CONFIGURATION")
        print(f"{'='*60}")
        for k, v in optimal.items():
            if k != "target_modules":
                print(f"  {k}: {v}")
        print(f"{'='*60}")
        print(f"Saved to {out_dir / 'optimal_config.json'}")
    
    return optimal
