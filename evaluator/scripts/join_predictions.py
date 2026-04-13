#!/usr/bin/env python3
"""
Join per-condition prediction CSVs with chat_pairs by ground truth text.
Creates a merged dataset for evaluator UIs and later analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


CONDITION_FILES = {
    "base": "eval_bleurt_bertscore_per_example.csv",
    "fine_tuned": "eval_ft_bleurt_bertscore_per_example.csv",
    "prompt_zero_shot": "eval_prompt_zero_shot_bleurt_bertscore_per_example.csv",
    "prompt_few_shot": "eval_prompt_few_shot_bleurt_bertscore_per_example.csv",
}


def json_value(value):
    """Convert pandas missing values into strict JSON-compatible nulls."""
    return None if pd.isna(value) else value


def json_float(value):
    """Convert numeric cells into JSON-compatible floats."""
    return None if pd.isna(value) else float(value)


def build_example_key(*, conversation_hash: str, ground_truth: str, dataset: str) -> str:
    """Build a stable merge key for one evaluation example."""
    return f"{dataset}::{conversation_hash}::{ground_truth.strip()}"


def load_chat_pairs(model_dir: Path) -> List[Dict]:
    with open(model_dir / "chat_pairs.json", "r") as f:
        data = json.load(f)
    records = []
    for index, item in enumerate(data):
        meta = item.get("meta", {})
        ground_truth = item["target_user"]
        records.append(
            {
                "index": index,
                "ground_truth": ground_truth,
                "dataset": meta.get("dataset", ""),
                "num_turns": meta.get("num_turns", len(item["conversation"])),
                "conversation_hash": meta.get("conversation_hash", ""),
                "example_key": build_example_key(
                    conversation_hash=meta.get("conversation_hash", ""),
                    ground_truth=ground_truth,
                    dataset=meta.get("dataset", ""),
                ),
            }
        )
    return records


def load_condition_map(model_dir: Path, filename: str) -> Dict[str, Dict]:
    """Load one condition file and map it by a stable example identifier."""
    csv_path = model_dir / filename
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    rows = {}
    for _, row in df.iterrows():
        key = build_example_key(
            conversation_hash=str(row.get("conversation_hash", "")).strip(),
            ground_truth=str(row["ref"]),
            dataset=str(row.get("dataset", "")).strip(),
        )
        rows[key] = {
            "pred": json_value(row.get("pred")),
            "metrics": {
                "bertscore_f1": json_float(row.get("bertscore_f1")),
                "bleurt": json_float(row.get("bleurt")),
                "ppl": json_float(row.get("ppl_content")),
            },
        }
    return rows


def merge_model(model_dir: Path) -> List[Dict]:
    """Merge all available condition files for one model directory."""
    chat_pairs = load_chat_pairs(model_dir)
    condition_maps = {
        condition_id: load_condition_map(model_dir, filename)
        for condition_id, filename in CONDITION_FILES.items()
    }

    merged_rows = []
    for row in chat_pairs:
        conditions = {}
        for condition_id, condition_map in condition_maps.items():
            if row["example_key"] in condition_map:
                conditions[condition_id] = condition_map[row["example_key"]]
        merged_rows.append({**row, "conditions": conditions})
    return merged_rows


def process_all_models(root_dir: Path) -> Dict[str, List[Dict]]:
    """Process all saved model output directories under outputs/."""
    outputs_dir = root_dir / "outputs"
    results: Dict[str, List[Dict]] = {}
    if not outputs_dir.exists():
        return results

    for model_dir in sorted(outputs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if not (model_dir / "chat_pairs.json").exists():
            continue
        model_id = f"outputs/{model_dir.name}"
        print(f"Processing {model_id}...")
        results[model_id] = merge_model(model_dir)
        print(f"  Conditions: {sorted(set().union(*(row['conditions'].keys() for row in results[model_id])))}")
    return results


def export_merged_json(results: Dict[str, List[Dict]], output_path: Path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nExported merged data to {output_path}")


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent
    print("Joining evaluator condition outputs by ground truth...\n")
    results = process_all_models(root_dir)
    output_path = Path(__file__).parent.parent / "data" / "merged_predictions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_merged_json(results, output_path)
    print("\nDone!")
