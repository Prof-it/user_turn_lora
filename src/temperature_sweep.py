"""
Temperature sweep for evaluation-only ablation.

This script runs inference at multiple temperatures on both baseline and 
fine-tuned models to find optimal generation temperature. Training is 
skipped since temperature only affects generation/inference.

Usage:
    python -m src.temperature_sweep --model-dir output/Qwen-Qwen2.5-3B-Instruct
    python -m src.temperature_sweep --model-dir output/Qwen-Qwen2.5-3B-Instruct --temps 0.3 0.4 0.5
"""

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import torch

from .config import PipelineConfig
from .data import load_data, build_eval_examples
from .model import load_base_model, load_finetuned_model, load_tokenizer, cleanup_model
from .evaluate import generate_predictions, compute_bertscore, compute_bleurt


DEFAULT_TEMPERATURES = [0.3, 0.4, 0.5, 0.6, 0.7]


def evaluate_at_temperature(
    model,
    tokenizer,
    eval_pairs: List[Dict],
    config: PipelineConfig,
    temperature: float,
    pred_key: str,
) -> Dict:
    """Run evaluation at a specific temperature."""
    # Update config temperature
    config.temperature = temperature
    
    # Generate predictions
    eval_pairs = generate_predictions(
        model, tokenizer, eval_pairs, config,
        pred_key=pred_key,
        verbose=True,
    )
    
    # Extract predictions and references
    predictions = [p.get(pred_key, "") for p in eval_pairs]
    references = [p.get("target_user", "") for p in eval_pairs]
    
    # Filter empty
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
    if not valid_pairs:
        return {"temperature": temperature, "bertscore_f1": 0.0, "bleurt": 0.0, "n_samples": 0}
    
    preds, refs = zip(*valid_pairs)
    preds, refs = list(preds), list(refs)
    
    # Compute metrics
    _, bertscore_f1 = compute_bertscore(preds, refs)
    _, bleurt_score = compute_bleurt(preds, refs)
    
    return {
        "temperature": temperature,
        "bertscore_f1": bertscore_f1,
        "bleurt": bleurt_score,
        "n_samples": len(preds),
    }


def run_temperature_sweep(
    model_dir: str,
    temperatures: List[float] = None,
    eval_samples: int = 400,
) -> pd.DataFrame:
    """
    Run temperature sweep on both baseline and fine-tuned models.
    
    Args:
        model_dir: Path to model output directory (contains config.json and adapter/)
        temperatures: List of temperatures to test
        eval_samples: Number of evaluation samples
        
    Returns:
        DataFrame with results for each temperature and model type
    """
    temperatures = temperatures or DEFAULT_TEMPERATURES
    model_dir = Path(model_dir)
    
    # Load config from model directory
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        config = PipelineConfig(**{k: v for k, v in config_dict.items() 
                                   if k in PipelineConfig.__dataclass_fields__})
    else:
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    config.num_eval_samples = eval_samples
    
    # Load evaluation data
    print(f"\nLoading evaluation data ({eval_samples} samples)...")
    chat_pairs_path = model_dir / "chat_pairs.json"
    if chat_pairs_path.exists():
        with open(chat_pairs_path) as f:
            eval_pairs = json.load(f)
        print(f"  Loaded {len(eval_pairs)} eval pairs from {chat_pairs_path}")
    else:
        _, eval_pairs = load_data(config)
        eval_pairs = build_eval_examples(eval_pairs)
    
    results = []
    
    # === Baseline evaluation ===
    print(f"\n{'='*60}")
    print("Baseline Model Temperature Sweep")
    print(f"{'='*60}")
    
    tokenizer = load_tokenizer(config)
    model = load_base_model(config, for_training=False)
    
    for temp in temperatures:
        print(f"\n  Temperature: {temp}")
        # Make a copy of eval_pairs for each run
        eval_pairs_copy = [dict(p) for p in eval_pairs]
        result = evaluate_at_temperature(
            model, tokenizer, eval_pairs_copy, config, temp, pred_key="pred_baseline"
        )
        result["model_type"] = "baseline"
        results.append(result)
        print(f"    BERTScore F1: {result['bertscore_f1']:.4f}")
        print(f"    BLEURT: {result['bleurt']:.4f}")
        
        # Clear cache
        torch.cuda.empty_cache()
    
    cleanup_model(model=model)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # === Fine-tuned evaluation ===
    print(f"\n{'='*60}")
    print("Fine-tuned Model Temperature Sweep")
    print(f"{'='*60}")
    
    adapter_path = model_dir / "adapter"
    if not adapter_path.exists():
        print(f"  WARNING: No adapter found at {adapter_path}, skipping fine-tuned eval")
    else:
        model, tokenizer = load_finetuned_model(config, str(adapter_path))
        
        for temp in temperatures:
            print(f"\n  Temperature: {temp}")
            eval_pairs_copy = [dict(p) for p in eval_pairs]
            result = evaluate_at_temperature(
                model, tokenizer, eval_pairs_copy, config, temp, pred_key="pred_finetuned"
            )
            result["model_type"] = "finetuned"
            results.append(result)
            print(f"    BERTScore F1: {result['bertscore_f1']:.4f}")
            print(f"    BLEURT: {result['bleurt']:.4f}")
            
            torch.cuda.empty_cache()
        
        cleanup_model(model=model)
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_path = model_dir / "temperature_sweep.csv"
    df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Temperature Sweep Summary")
    print(f"{'='*60}")
    
    for model_type in ["baseline", "finetuned"]:
        subset = df[df["model_type"] == model_type]
        if len(subset) == 0:
            continue
        best_bleurt = subset.loc[subset["bleurt"].idxmax()]
        best_bert = subset.loc[subset["bertscore_f1"].idxmax()]
        print(f"\n{model_type.upper()}:")
        print(f"  Best BLEURT: temp={best_bleurt['temperature']}, score={best_bleurt['bleurt']:.4f}")
        print(f"  Best BERTScore: temp={best_bert['temperature']}, score={best_bert['bertscore_f1']:.4f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Temperature sweep for evaluation")
    parser.add_argument(
        "--model-dir", "-d",
        required=True,
        help="Path to model output directory (contains config.json and adapter/)"
    )
    parser.add_argument(
        "--temps", "-t",
        nargs="+",
        type=float,
        default=DEFAULT_TEMPERATURES,
        help=f"Temperatures to test (default: {DEFAULT_TEMPERATURES})"
    )
    parser.add_argument(
        "--eval-samples", "-n",
        type=int,
        default=400,
        help="Number of evaluation samples (default: 400)"
    )
    
    args = parser.parse_args()
    
    run_temperature_sweep(
        model_dir=args.model_dir,
        temperatures=args.temps,
        eval_samples=args.eval_samples,
    )


if __name__ == "__main__":
    main()
