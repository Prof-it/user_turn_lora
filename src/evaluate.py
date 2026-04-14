"""
Evaluation module for UserTurnLoRA pipeline.
Handles perplexity, BERTScore, and BLEURT metrics.
"""

import gc
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import PipelineConfig, SYSTEM_PROMPT
from .model import load_finetuned_model, load_base_model, load_tokenizer, predict_next_user, build_messages


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    item: Dict,
    config: PipelineConfig,
) -> float:
    """
    Compute perplexity on the target user turn only.
    
    The context (conversation history) is treated as conditioning,
    and we only compute loss on the target user tokens.
    """
    messages = build_messages(item)
    target = item["target_user"]
    
    # Add target user turn
    messages.append({"role": "user", "content": target})
    
    # Tokenize full conversation
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=False,
    ).to(model.device)
    
    # Tokenize context only
    context_msgs = messages[:-1]
    context_ids = tokenizer.apply_chat_template(
        context_msgs,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=False,
    ).to(model.device)
    
    context_len = context_ids.shape[1]
    
    # Truncate if needed
    if full_ids.shape[1] > config.max_context_len:
        full_ids = full_ids[:, :config.max_context_len]
    
    # Forward pass
    outputs = model(full_ids)
    logits = outputs.logits
    
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = full_ids[:, 1:].contiguous()
    
    # Compute loss only on target tokens (after context)
    target_start = max(0, context_len - 1)
    target_logits = shift_logits[:, target_start:, :]
    target_labels = shift_labels[:, target_start:]
    
    if target_labels.numel() == 0:
        return float('nan')
    
    # Cross entropy loss
    loss = F.cross_entropy(
        target_logits.view(-1, target_logits.size(-1)),
        target_labels.view(-1),
        reduction='mean'
    )
    
    return math.exp(loss.item())


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
) -> Tuple[List[float], float]:
    """
    Compute BERTScore F1 for predictions vs references.
    
    Uses raw BERTScore values (0-1 range), no rescaling.
    
    Returns:
        Tuple of (per_example_f1, macro_f1)
    """
    from bert_score import score as bertscore
    
    P, R, F1 = bertscore(predictions, references, lang=lang)
    f1_list = F1.tolist()
    macro_f1 = float(F1.mean().item())
    
    return f1_list, macro_f1


def compute_bleurt(
    predictions: List[str],
    references: List[str],
    checkpoint: str = "BLEURT-20",
) -> Tuple[List[float], float]:
    """
    Compute BLEURT scores for predictions vs references.
    
    Forces CPU execution to avoid GPU memory conflicts with PyTorch.
    
    Returns:
        Tuple of (per_example_scores, macro_score)
    """
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings
    checkpoint = os.environ.get("BLEURT_CHECKPOINT", checkpoint)
    
    import tensorflow as tf
    # Force TensorFlow to use CPU only to avoid GPU memory conflicts with PyTorch
    tf.config.set_visible_devices([], 'GPU')
    
    from bleurt import score as bleurt_score
    
    scorer = bleurt_score.BleurtScorer(checkpoint)
    scores = scorer.score(references=references, candidates=predictions)
    macro_score = float(np.mean(scores))
    
    return scores, macro_score


def generate_predictions(
    model,
    tokenizer,
    eval_pairs: List[Dict],
    config: PipelineConfig,
    pred_key: str = "pred_user",
    verbose: bool = True,
) -> List[Dict]:
    """
    Generate predictions for all evaluation pairs.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        eval_pairs: List of evaluation pairs
        config: Pipeline configuration
        pred_key: Key to store predictions under
        verbose: Print progress
    
    Returns:
        eval_pairs with predictions added
    """
    import sys
    print(f"Generating predictions for {len(eval_pairs)} examples...", flush=True)
    sys.stdout.flush()
    
    for i, item in enumerate(eval_pairs):
        try:
            pred = predict_next_user(model, tokenizer, item, config, verbose=False)
            item[pred_key] = pred
        except Exception as e:
            print(f"  Warning: Prediction failed for example {i}: {e}", flush=True)
            item[pred_key] = ""
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(eval_pairs)} completed", flush=True)
            sys.stdout.flush()
        
        # Clear cache periodically
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
    
    print(f"  Predictions complete")
    return eval_pairs


def evaluate_model(
    model,
    tokenizer,
    eval_pairs: List[Dict],
    config: PipelineConfig,
    pred_key: str = "pred_user",
    output_prefix: str = "eval",
    output_dir: Optional[str] = None,
    skip_generation: bool = False,
) -> Dict:
    """
    Full evaluation pipeline: generate predictions and compute metrics.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        eval_pairs: List of evaluation pairs
        config: Pipeline configuration
        pred_key: Key for predictions
        output_prefix: Prefix for output files
        output_dir: Directory to save results (uses config.output_dir if None)
        skip_generation: Skip generation if predictions already exist
    
    Returns:
        Dict with all metrics
    """
    output_dir = Path(output_dir or config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Evaluating Model")
    print(f"{'='*60}")
    
    # Generate predictions if needed
    if not skip_generation:
        eval_pairs = generate_predictions(model, tokenizer, eval_pairs, config, pred_key)
    
    # Extract refs and predictions
    refs, preds = [], []
    for item in eval_pairs:
        ref = (item.get("target_user") or "").strip()
        pred = (item.get(pred_key) or "").strip()
        if ref:
            refs.append(ref)
            preds.append(pred)
    
    print(f"Evaluating {len(refs)} examples...")
    
    # Compute perplexity
    print("  Computing perplexity...")
    ppl_vals = []
    for item in eval_pairs:
        ppl = compute_perplexity(model, tokenizer, item, config)
        ppl_vals.append(ppl)
    ppl_macro = float(np.nanmean(ppl_vals))
    
    # Compute BERTScore
    print("  Computing BERTScore...")
    bertscore_f1, bertscore_macro = compute_bertscore(preds, refs)
    
    # Compute BLEURT
    print("  Computing BLEURT...")
    bleurt_vals, bleurt_macro = compute_bleurt(preds, refs)
    
    # Build results
    results = {
        "ppl_macro": ppl_macro,
        "bertscore_f1_macro": bertscore_macro,
        "bleurt_macro": bleurt_macro,
        "num_examples": len(refs),
    }
    
    # Summary dataframe
    summary_df = pd.DataFrame([{
        "bertscore_f1_macro": bertscore_macro,
        "bleurt_macro": bleurt_macro,
        "ppl_content_macro": ppl_macro,
    }])
    
    # Per-example dataframe with full metadata (no joins needed for plotting)
    per_example_data = []
    for i, item in enumerate(eval_pairs):
        ref = (item.get("target_user") or "").strip()
        pred = (item.get(pred_key) or "").strip()
        if not ref:
            continue
        
        meta = item.get("meta", {})
        per_example_data.append({
            "ref": ref,
            "pred": pred,
            "bertscore_f1": bertscore_f1[len(per_example_data)] if len(per_example_data) < len(bertscore_f1) else None,
            "bleurt": bleurt_vals[len(per_example_data)] if len(per_example_data) < len(bleurt_vals) else None,
            "ppl_content": ppl_vals[i] if i < len(ppl_vals) else None,
            "dataset": meta.get("dataset", ""),
            "domain": "Open-domain" if "WildChat" in meta.get("dataset", "") else "Task-oriented",
            "language": meta.get("language", ""),
            "num_turns": meta.get("num_turns", 0),
            "conversation_hash": meta.get("conversation_hash", ""),
        })
    
    per_example_df = pd.DataFrame(per_example_data)
    
    # Save results
    summary_path = output_dir / f"{output_prefix}_bleurt_bertscore_summary.csv"
    per_example_path = output_dir / f"{output_prefix}_bleurt_bertscore_per_example.csv"
    
    summary_df.to_csv(summary_path, index=False)
    per_example_df.to_csv(per_example_path, index=False)
    
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"  Perplexity (macro): {ppl_macro:.4f}")
    print(f"  BERTScore F1 (macro): {bertscore_macro:.4f}")
    print(f"  BLEURT (macro): {bleurt_macro:.4f}")
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {per_example_path}")
    
    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def evaluate_baseline(
    config: PipelineConfig,
    eval_pairs: List[Dict],
    output_dir: Optional[str] = None,
) -> Dict:
    """Evaluate the base model (before fine-tuning)."""
    print("\n" + "="*60)
    print("Baseline Evaluation")
    print("="*60)
    
    tokenizer = load_tokenizer(config)
    model = load_base_model(config, for_training=False)
    
    try:
        results = evaluate_model(
            model, tokenizer, eval_pairs, config,
            pred_key="pred_user",
            output_prefix="eval",
            output_dir=output_dir,
        )
    finally:
        from .model import cleanup_model
        cleanup_model(model=model)
    
    return results


def evaluate_finetuned(
    config: PipelineConfig,
    eval_pairs: List[Dict],
    adapter_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """Evaluate the fine-tuned model."""
    print("\n" + "="*60)
    print("Fine-tuned Model Evaluation")
    print("="*60)
    
    model, tokenizer = load_finetuned_model(config, adapter_path)
    
    try:
        results = evaluate_model(
            model, tokenizer, eval_pairs, config,
            pred_key="pred_user_ft",
            output_prefix="eval_ft",
            output_dir=output_dir,
        )
    finally:
        from .model import cleanup_model
        cleanup_model(model=model)
    
    return results


def compare_results(
    baseline_results: Dict,
    finetuned_results: Dict,
) -> Dict:
    """Compare baseline and fine-tuned results, computing deltas."""
    deltas = {}
    
    for key in ["ppl_macro", "bertscore_f1_macro", "bleurt_macro"]:
        base_val = baseline_results.get(key, 0)
        ft_val = finetuned_results.get(key, 0)
        
        if key == "ppl_macro":
            # Lower is better for perplexity
            delta_pct = ((base_val - ft_val) / base_val * 100) if base_val else 0
            deltas[f"{key}_delta_pct"] = delta_pct
        else:
            # Higher is better for BERTScore and BLEURT
            delta_pct = ((ft_val - base_val) / abs(base_val) * 100) if base_val else 0
            deltas[f"{key}_delta_pct"] = delta_pct
        
        deltas[f"{key}_base"] = base_val
        deltas[f"{key}_ft"] = ft_val
    
    print("\n" + "="*60)
    print("Comparison: Base vs Fine-tuned")
    print("="*60)
    print(f"  Perplexity: {deltas['ppl_macro_base']:.2f} → {deltas['ppl_macro_ft']:.2f} "
          f"(Δ {deltas['ppl_macro_delta_pct']:+.1f}%)")
    print(f"  BERTScore:  {deltas['bertscore_f1_macro_base']:.4f} → {deltas['bertscore_f1_macro_ft']:.4f} "
          f"(Δ {deltas['bertscore_f1_macro_delta_pct']:+.1f}%)")
    print(f"  BLEURT:     {deltas['bleurt_macro_base']:.4f} → {deltas['bleurt_macro_ft']:.4f} "
          f"(Δ {deltas['bleurt_macro_delta_pct']:+.1f}%)")
    
    return deltas
