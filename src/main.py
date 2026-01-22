"""
Main entry point for UserTurnLoRA pipeline.
Runs the full training and evaluation workflow.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig, get_config
from src.data import load_data, format_for_training, build_eval_examples
from src.model import load_tokenizer, cleanup_model
from src.train import train
from src.evaluate import evaluate_baseline, evaluate_finetuned, compare_results


def run_pipeline(
    model_name: str = None,
    skip_baseline: bool = False,
    skip_training: bool = False,
    skip_finetuned_eval: bool = False,
    output_dir: str = None,
    **config_overrides,
):
    """
    Run the full UserTurnLoRA pipeline.
    
    Args:
        model_name: HuggingFace model name
        skip_baseline: Skip baseline evaluation
        skip_training: Skip training (use existing adapter)
        skip_finetuned_eval: Skip fine-tuned evaluation
        output_dir: Output directory
        **config_overrides: Override any config parameter
    """
    # Create config
    config = get_config(model_name, **config_overrides)
    
    if output_dir:
        config.output_dir = output_dir
    else:
        # Default output dir based on model name
        model_short = config.model_name.replace("/", "_")
        config.output_dir = f"output/{model_short}"
    
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("UserTurnLoRA Pipeline")
    print("="*70)
    print(f"Model: {config.model_name}")
    print(f"Output: {config.output_dir}")
    print(f"Train samples: {config.num_train_samples}")
    print(f"Eval samples: {config.num_eval_samples}")
    print("="*70 + "\n")
    
    # Save config
    config_path = Path(config.output_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Config saved to {config_path}")
    
    # Load data
    train_pairs, eval_pairs = load_data(config)
    
    # Save data for reproducibility
    data_path = Path(config.output_dir) / "chat_pairs.json"
    with open(data_path, "w") as f:
        json.dump(eval_pairs, f, indent=2)
    print(f"Eval pairs saved to {data_path}")
    
    training_path = Path(config.output_dir) / "training_pairs.json"
    with open(training_path, "w") as f:
        json.dump(train_pairs, f, indent=2)
    print(f"Training pairs saved to {training_path}")
    
    results = {}
    
    # Baseline evaluation
    if not skip_baseline:
        baseline_results = evaluate_baseline(config, eval_pairs, config.output_dir)
        results["baseline"] = baseline_results
    
    # Training
    if not skip_training:
        tokenizer = load_tokenizer(config)
        train_ds = format_for_training(train_pairs, tokenizer, config)
        eval_ds = format_for_training(eval_pairs, tokenizer, config)
        
        adapter_path = train(config, train_ds, eval_ds)
        results["adapter_path"] = adapter_path
    
    # Fine-tuned evaluation
    if not skip_finetuned_eval:
        finetuned_results = evaluate_finetuned(config, eval_pairs, output_dir=config.output_dir)
        results["finetuned"] = finetuned_results
    
    # Compare results
    if "baseline" in results and "finetuned" in results:
        deltas = compare_results(results["baseline"], results["finetuned"])
        results["deltas"] = deltas
        
        # Save comparison
        comparison_path = Path(config.output_dir) / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(deltas, f, indent=2)
        print(f"\nComparison saved to {comparison_path}")
    
    print("\n" + "="*70)
    print("Pipeline Complete!")
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="UserTurnLoRA: Fine-tune LLMs to predict user turns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with default model (Qwen2.5-3B-Instruct)
    python -m src.main
    
    # Specify a different model
    python -m src.main --model meta-llama/Llama-3.2-3B-Instruct
    
    # Skip baseline evaluation (faster)
    python -m src.main --skip-baseline
    
    # Only evaluate existing adapter
    python -m src.main --skip-training --adapter-path output/adapter
    
    # Custom training parameters
    python -m src.main --epochs 3 --lr 5e-5 --lora-r 16
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model name"
    )
    
    # Pipeline control
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--skip-training", action="store_true", help="Skip training")
    parser.add_argument("--skip-finetuned-eval", action="store_true", help="Skip fine-tuned evaluation")
    
    # Output
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lora-r", type=int, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, help="LoRA alpha")
    
    # Data parameters
    parser.add_argument("--train-samples", type=int, help="Number of training samples")
    parser.add_argument("--eval-samples", type=int, help="Number of evaluation samples")
    
    # Logging
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    
    args = parser.parse_args()
    
    # Build config overrides
    overrides = {}
    if args.epochs:
        overrides["num_epochs"] = args.epochs
    if args.lr:
        overrides["learning_rate"] = args.lr
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.lora_r:
        overrides["lora_r"] = args.lora_r
    if args.lora_alpha:
        overrides["lora_alpha"] = args.lora_alpha
    if args.train_samples:
        overrides["num_train_samples"] = args.train_samples
    if args.eval_samples:
        overrides["num_eval_samples"] = args.eval_samples
    if args.no_wandb:
        overrides["report_to"] = "none"
    
    # Run pipeline
    run_pipeline(
        model_name=args.model,
        skip_baseline=args.skip_baseline,
        skip_training=args.skip_training,
        skip_finetuned_eval=args.skip_finetuned_eval,
        output_dir=args.output_dir,
        **overrides,
    )


if __name__ == "__main__":
    main()
