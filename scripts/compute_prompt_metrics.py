#!/usr/bin/env python
"""Recompute prompt-baseline metrics from a saved predictions file."""

import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import compute_bertscore, compute_bleurt


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compute_prompt_metrics.py <output_dir>")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    predictions_path = output_dir / "predictions.json"
    if not predictions_path.exists():
        candidates = sorted(output_dir.glob("*_predictions.json"))
        predictions_path = candidates[0] if candidates else predictions_path
    
    if not predictions_path.exists():
        print(f"Error: {predictions_path} not found")
        sys.exit(1)
    
    # Load predictions
    with open(predictions_path) as f:
        data = json.load(f)
    
    references = [item["target_user"] for item in data]
    predictions = [item.get("pred_prompt_baseline", "") for item in data]
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Compute metrics
    print("\nComputing BERTScore...")
    bertscore_f1_per_example, bertscore_f1_macro = compute_bertscore(predictions, references)
    
    print("\nComputing BLEURT...")
    bleurt_per_example, bleurt_macro = compute_bleurt(predictions, references)
    
    # Load config if exists
    config = {}
    for candidate in [output_dir / "metrics.json", output_dir / "config.json"]:
        if candidate.exists():
            with open(candidate) as f:
                config = json.load(f)
            break
    
    metrics = {
        **config,
        "bertscore_f1_macro": bertscore_f1_macro,
        "bleurt_macro": bleurt_macro,
        "bertscore_f1_per_example": bertscore_f1_per_example,
        "bleurt_per_example": bleurt_per_example,
    }
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Prompt Baseline Metrics")
    print(f"{'='*60}")
    print(f"BERTScore F1: {metrics['bertscore_f1_macro']:.4f}")
    print(f"BLEURT: {metrics['bleurt_macro']:.4f}")


if __name__ == "__main__":
    main()
