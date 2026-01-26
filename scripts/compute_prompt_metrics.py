#!/usr/bin/env python
"""
Compute BERTScore and BLEURT metrics for prompt baseline predictions.
"""

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
    bertscore_results = compute_bertscore(references, predictions)
    
    print("\nComputing BLEURT...")
    bleurt_results = compute_bleurt(references, predictions)
    
    # Load config if exists
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    
    metrics = {
        **config,
        "bertscore_f1": bertscore_results["f1"],
        "bertscore_precision": bertscore_results["precision"],
        "bertscore_recall": bertscore_results["recall"],
        "bleurt": bleurt_results["mean"],
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
    print(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")
    print(f"BLEURT: {metrics['bleurt']:.4f}")


if __name__ == "__main__":
    main()
