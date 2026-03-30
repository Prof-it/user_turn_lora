"""
Targeted per-model alpha/lr sweeps for reviewer-driven follow-up experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import pandas as pd


if TYPE_CHECKING:
    from .config import PipelineConfig


def load_pipeline_config(model_dir: Path) -> "PipelineConfig":
    """Load config.json from an existing model output directory."""
    from .config_loader import load_saved_pipeline_config

    return load_saved_pipeline_config(model_dir)


def load_saved_pairs(model_dir: Path, train_samples: int, eval_samples: int) -> Tuple[List[Dict], List[Dict]]:
    """Load saved train/eval pairs so sweeps reuse the same data split."""
    with open(model_dir / "training_pairs.json") as f:
        train_pairs = json.load(f)
    with open(model_dir / "chat_pairs.json") as f:
        eval_pairs = json.load(f)
    return train_pairs[:train_samples], eval_pairs[:eval_samples]


def run_targeted_sweep(
    model_dir: Path,
    alphas: List[int],
    learning_rates: List[float],
    lora_r: int | None = None,
    epochs: int | None = None,
    train_samples: int | None = None,
    eval_samples: int | None = None,
) -> pd.DataFrame:
    """Run a compact sweep over LoRA alpha and learning rate for one model."""
    from .config import PipelineConfig
    from .data import format_for_training
    from .evaluate import evaluate_finetuned
    from .model import load_tokenizer
    from .train import train

    base_config = load_pipeline_config(model_dir)
    train_samples = train_samples or base_config.num_train_samples
    eval_samples = eval_samples or base_config.num_eval_samples
    train_pairs, eval_pairs = load_saved_pairs(model_dir, train_samples, eval_samples)

    tokenizer = load_tokenizer(base_config)
    train_ds = format_for_training(train_pairs, tokenizer, base_config)
    eval_ds = format_for_training(eval_pairs, tokenizer, base_config)

    sweep_dir = model_dir / "targeted_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []

    for alpha in alphas:
        for lr in learning_rates:
            exp_name = f"alpha{alpha}_lr{lr:.0e}"
            exp_dir = sweep_dir / exp_name
            config = PipelineConfig(
                model_name=base_config.model_name,
                num_train_samples=train_samples,
                num_eval_samples=eval_samples,
                num_epochs=epochs or base_config.num_epochs,
                batch_size=base_config.batch_size,
                gradient_accumulation_steps=base_config.gradient_accumulation_steps,
                learning_rate=lr,
                warmup_ratio=base_config.warmup_ratio,
                weight_decay=base_config.weight_decay,
                max_grad_norm=base_config.max_grad_norm,
                lora_r=lora_r or base_config.lora_r,
                lora_alpha=alpha,
                lora_dropout=base_config.lora_dropout,
                target_modules=base_config.target_modules,
                output_dir=str(exp_dir),
                report_to="none",
            )
            config.batch_size = base_config.batch_size
            config.gradient_accumulation_steps = base_config.gradient_accumulation_steps
            config.learning_rate = lr
            config.lora_r = lora_r or base_config.lora_r
            config.lora_alpha = alpha
            config.num_epochs = epochs or base_config.num_epochs
            config.output_dir = str(exp_dir)
            config.report_to = "none"

            print(f"\nRunning {exp_name}...")
            adapter_path = train(config, train_ds, eval_ds)
            metrics = evaluate_finetuned(config, eval_pairs, adapter_path=adapter_path, output_dir=str(exp_dir))
            rows.append(
                {
                    "experiment": exp_name,
                    "lora_r": config.lora_r,
                    "lora_alpha": alpha,
                    "learning_rate": lr,
                    "num_epochs": config.num_epochs,
                    "bertscore_f1_macro": metrics["bertscore_f1_macro"],
                    "bleurt_macro": metrics["bleurt_macro"],
                    "ppl_macro": metrics["ppl_macro"],
                    "output_dir": str(exp_dir),
                }
            )

    df = pd.DataFrame(rows).sort_values(by=["bleurt_macro", "bertscore_f1_macro"], ascending=False)
    df.to_csv(sweep_dir / "summary.csv", index=False)
    if not df.empty:
        with open(sweep_dir / "best_config.json", "w") as f:
            json.dump(df.iloc[0].to_dict(), f, indent=2)
    return df


def main():
    parser = argparse.ArgumentParser(description="Run a compact alpha/lr sweep for one saved model directory")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4, 2e-4])
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    args = parser.parse_args()

    df = run_targeted_sweep(
        model_dir=args.model_dir,
        alphas=args.alphas,
        learning_rates=args.learning_rates,
        lora_r=args.lora_r,
        epochs=args.epochs,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
    )
    print(df.head())


if __name__ == "__main__":
    main()
