"""
Utilities for saving arbitrary evaluation conditions in a consistent format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def score_predictions(predictions: List[str], references: List[str]) -> Dict:
    """Compute shared text metrics for one evaluation condition."""
    from .evaluate import compute_bertscore, compute_bleurt

    bertscore_f1_per_example, bertscore_f1_macro = compute_bertscore(predictions, references)
    bleurt_per_example, bleurt_macro = compute_bleurt(predictions, references)
    return {
        "bertscore_f1_per_example": bertscore_f1_per_example,
        "bertscore_f1_macro": bertscore_f1_macro,
        "bleurt_per_example": bleurt_per_example,
        "bleurt_macro": bleurt_macro,
    }


def build_per_example_rows(
    eval_pairs: List[Dict],
    predictions: List[str],
    scores: Dict,
    ppl_values: Optional[List[Optional[float]]] = None,
) -> List[Dict]:
    """Build per-example rows aligned with existing evaluation CSVs."""
    rows = []
    ppl_values = ppl_values or [None] * len(predictions)

    for index, item in enumerate(eval_pairs):
        ref = (item.get("target_user") or "").strip()
        pred = (predictions[index] if index < len(predictions) else "").strip()
        if not ref:
            continue

        meta = item.get("meta", {})
        rows.append(
            {
                "ref": ref,
                "pred": pred,
                "bertscore_f1": scores["bertscore_f1_per_example"][len(rows)],
                "bleurt": scores["bleurt_per_example"][len(rows)],
                "ppl_content": ppl_values[index] if index < len(ppl_values) else None,
                "dataset": meta.get("dataset", ""),
                "domain": "Open-domain" if "WildChat" in meta.get("dataset", "") else "Task-oriented",
                "language": meta.get("language", ""),
                "num_turns": meta.get("num_turns", 0),
                "conversation_hash": meta.get("conversation_hash", ""),
            }
        )
    return rows


def save_condition_outputs(
    output_dir: Path,
    output_prefix: str,
    eval_pairs: List[Dict],
    predictions: List[str],
    metadata: Optional[Dict] = None,
    ppl_values: Optional[List[Optional[float]]] = None,
) -> Dict:
    """Persist predictions, per-example metrics, and a summary for one condition."""
    output_dir.mkdir(parents=True, exist_ok=True)
    references = [(item.get("target_user") or "").strip() for item in eval_pairs]
    scores = score_predictions(predictions, references)
    per_example_rows = build_per_example_rows(eval_pairs, predictions, scores, ppl_values=ppl_values)

    predictions_payload = []
    for item, pred in zip(eval_pairs, predictions):
        predictions_payload.append({**item, "pred_prompt_baseline": pred})

    with open(output_dir / f"{output_prefix}_predictions.json", "w") as f:
        json.dump(predictions_payload, f, indent=2)

    summary = {
        **(metadata or {}),
        "bertscore_f1_macro": scores["bertscore_f1_macro"],
        "bleurt_macro": scores["bleurt_macro"],
        "num_examples": len(per_example_rows),
    }
    with open(output_dir / f"{output_prefix}_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    summary_df = pd.DataFrame(
        [
            {
                "bertscore_f1_macro": scores["bertscore_f1_macro"],
                "bleurt_macro": scores["bleurt_macro"],
                "ppl_content_macro": None,
            }
        ]
    )
    per_example_df = pd.DataFrame(per_example_rows)
    summary_df.to_csv(output_dir / f"{output_prefix}_bleurt_bertscore_summary.csv", index=False)
    per_example_df.to_csv(output_dir / f"{output_prefix}_bleurt_bertscore_per_example.csv", index=False)
    return summary
