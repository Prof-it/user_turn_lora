"""
Metrics for multi-turn rollout evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional

def _compute_text_scores(predictions: List[str], references: List[str]) -> Dict[str, List[float] | float]:
    if not predictions:
        return {
            "bertscore_per_example": [],
            "bertscore_macro": 0.0,
            "bleurt_per_example": [],
            "bleurt_macro": 0.0,
        }

    from .evaluate import compute_bertscore, compute_bleurt

    bertscore_per_example, bertscore_macro = compute_bertscore(predictions, references)
    bleurt_per_example, bleurt_macro = compute_bleurt(predictions, references)
    return {
        "bertscore_per_example": bertscore_per_example,
        "bertscore_macro": bertscore_macro,
        "bleurt_per_example": bleurt_per_example,
        "bleurt_macro": bleurt_macro,
    }


def _compute_alignment_depths(
    step_rows: List[Dict],
    *,
    user_bertscore_threshold: Optional[float],
    user_bleurt_threshold: Optional[float],
) -> Dict[str, float]:
    if user_bertscore_threshold is None and user_bleurt_threshold is None:
        return {}

    by_dialogue: Dict[str, List[Dict]] = defaultdict(list)
    for row in step_rows:
        by_dialogue[row["dialogue_id"]].append(row)

    depths = []
    for dialogue_rows in by_dialogue.values():
        depth = 0
        for row in sorted(dialogue_rows, key=lambda item: item["step_index"]):
            if user_bertscore_threshold is not None and row["user_bertscore_f1"] < user_bertscore_threshold:
                break
            if user_bleurt_threshold is not None and row["user_bleurt"] < user_bleurt_threshold:
                break
            depth += 1
        depths.append(depth)

    return {"alignment_depth_mean": mean(depths) if depths else 0.0}


def compute_rollout_metrics(
    rollout_rows: List[Dict],
    dialogue_summaries: List[Dict],
    *,
    user_bertscore_threshold: Optional[float] = None,
    user_bleurt_threshold: Optional[float] = None,
) -> Dict:
    """Compute aggregate rollout metrics over all steps and dialogues."""
    user_predictions = [row["predicted_user"] for row in rollout_rows]
    user_references = [row["gold_user"] for row in rollout_rows]
    user_scores = _compute_text_scores(user_predictions, user_references)

    assistant_predictions = [row["generated_assistant"] for row in rollout_rows if row.get("generated_assistant")]
    assistant_references = [row["gold_assistant"] for row in rollout_rows if row.get("generated_assistant")]
    assistant_scores = _compute_text_scores(assistant_predictions, assistant_references)

    step_rows: List[Dict] = []
    per_step_user_bertscore: Dict[int, List[float]] = defaultdict(list)
    per_step_user_bleurt: Dict[int, List[float]] = defaultdict(list)
    per_step_assistant_bertscore: Dict[int, List[float]] = defaultdict(list)
    per_step_assistant_bleurt: Dict[int, List[float]] = defaultdict(list)

    for index, row in enumerate(rollout_rows):
        step_row = {
            **row,
            "user_bertscore_f1": user_scores["bertscore_per_example"][index],
            "user_bleurt": user_scores["bleurt_per_example"][index],
        }
        per_step_user_bertscore[row["step_index"]].append(step_row["user_bertscore_f1"])
        per_step_user_bleurt[row["step_index"]].append(step_row["user_bleurt"])
        step_rows.append(step_row)

    assistant_index = 0
    for row in step_rows:
        if row.get("generated_assistant"):
            row["assistant_bertscore_f1"] = assistant_scores["bertscore_per_example"][assistant_index]
            row["assistant_bleurt"] = assistant_scores["bleurt_per_example"][assistant_index]
            per_step_assistant_bertscore[row["step_index"]].append(row["assistant_bertscore_f1"])
            per_step_assistant_bleurt[row["step_index"]].append(row["assistant_bleurt"])
            assistant_index += 1
        else:
            row["assistant_bertscore_f1"] = None
            row["assistant_bleurt"] = None

    empty_user_rate = sum(1 for row in rollout_rows if not (row["predicted_user"] or "").strip()) / max(len(rollout_rows), 1)
    repeated_stop_rate = sum(1 for summary in dialogue_summaries if summary["stop_reason"] == "repeated_user") / max(len(dialogue_summaries), 1)
    collapse_rate = sum(
        1 for summary in dialogue_summaries if summary["stop_reason"] in {"empty_user", "repeated_user", "empty_assistant"}
    ) / max(len(dialogue_summaries), 1)

    per_step_summary = {
        "user_bertscore_f1_macro": {str(step): mean(values) for step, values in per_step_user_bertscore.items()},
        "user_bleurt_macro": {str(step): mean(values) for step, values in per_step_user_bleurt.items()},
    }
    if per_step_assistant_bertscore:
        per_step_summary["assistant_bertscore_f1_macro"] = {
            str(step): mean(values) for step, values in per_step_assistant_bertscore.items()
        }
        per_step_summary["assistant_bleurt_macro"] = {
            str(step): mean(values) for step, values in per_step_assistant_bleurt.items()
        }

    ordered_user_steps = sorted(per_step_summary["user_bertscore_f1_macro"].items(), key=lambda item: int(item[0]))
    degradation = 0.0
    if len(ordered_user_steps) >= 2:
        degradation = ordered_user_steps[0][1] - ordered_user_steps[-1][1]

    summary = {
        "num_dialogues": len(dialogue_summaries),
        "num_rollout_steps": len(rollout_rows),
        "average_completed_steps": mean(summary["completed_steps"] for summary in dialogue_summaries)
        if dialogue_summaries
        else 0.0,
        "user_bertscore_f1_macro": user_scores["bertscore_macro"],
        "user_bleurt_macro": user_scores["bleurt_macro"],
        "assistant_bertscore_f1_macro": assistant_scores["bertscore_macro"] if assistant_predictions else None,
        "assistant_bleurt_macro": assistant_scores["bleurt_macro"] if assistant_predictions else None,
        "empty_user_rate": empty_user_rate,
        "repeated_user_stop_rate": repeated_stop_rate,
        "collapse_rate": collapse_rate,
        "user_bertscore_degradation": degradation,
        "per_step": per_step_summary,
    }
    summary.update(
        _compute_alignment_depths(
            step_rows,
            user_bertscore_threshold=user_bertscore_threshold,
            user_bleurt_threshold=user_bleurt_threshold,
        )
    )
    return {"summary": summary, "step_rows": step_rows}
