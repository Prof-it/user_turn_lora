#!/usr/bin/env python3
"""Freeze the existing human-eval sample IDs for additional raters."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXISTING_RATINGS = ROOT / "outputs" / "Qwen-Qwen2.5-3B-Instruct" / "human_eval_ratings.csv"
OUTPUT_PATH = ROOT / "evaluator" / "data" / "frozen_human_eval_sample_ids.json"


def main() -> None:
    ratings = pd.read_csv(EXISTING_RATINGS)
    sample_ids = ratings["sample_id"].dropna().astype(str).drop_duplicates().tolist()
    OUTPUT_PATH.write_text(json.dumps({"sample_ids": sample_ids}, indent=2) + "\n")
    print(f"Saved {len(sample_ids)} sample IDs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
