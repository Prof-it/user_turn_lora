#!/usr/bin/env python3
"""
Aggregate multiple quick-eval CSV exports and compute inter-rater agreement.
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


CATEGORY_PATTERN = re.compile(r"^(?P<condition>.+)__(?P<category>relevance|coherence|naturalness)$")


def quadratic_weighted_kappa(a: List[int], b: List[int], min_rating: int = 1, max_rating: int = 5) -> float:
    """Compute quadratic weighted Cohen's kappa for two raters."""
    if len(a) != len(b) or not a:
        return float("nan")

    categories = list(range(min_rating, max_rating + 1))
    index = {rating: idx for idx, rating in enumerate(categories)}
    n = len(categories)

    observed = [[0.0 for _ in categories] for _ in categories]
    for x, y in zip(a, b):
        observed[index[x]][index[y]] += 1.0

    row_marginals = [sum(row) for row in observed]
    col_marginals = [sum(observed[row][col] for row in range(n)) for col in range(n)]
    total = float(len(a))

    expected = [
        [(row_marginals[i] * col_marginals[j]) / total for j in range(n)]
        for i in range(n)
    ]
    weights = [
        [((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)]
        for i in range(n)
    ]

    observed_weighted = sum(weights[i][j] * observed[i][j] for i in range(n) for j in range(n)) / total
    expected_weighted = sum(weights[i][j] * expected[i][j] for i in range(n) for j in range(n)) / total
    if expected_weighted == 0:
        return float("nan")
    return 1.0 - (observed_weighted / expected_weighted)


def load_rater_exports(paths: List[Path]) -> Tuple[pd.DataFrame, List[str], List[Tuple[str, str]]]:
    """Load exported CSVs and detect scored condition/category columns."""
    frames = []
    raters = []
    detected_fields = set()

    for index, path in enumerate(paths):
        df = pd.read_csv(path)
        rater_id = path.stem or f"rater_{index + 1}"
        df["rater_id"] = rater_id
        frames.append(df)
        raters.append(rater_id)
        for column in df.columns:
            match = CATEGORY_PATTERN.match(column)
            if match:
                detected_fields.add((match.group("condition"), match.group("category")))

    merged = pd.concat(frames, ignore_index=True)
    return merged, raters, sorted(detected_fields)


def summarize_scores(df: pd.DataFrame, fields: List[Tuple[str, str]]) -> List[Dict]:
    """Compute per-condition mean/std summaries across all raters."""
    rows = []
    for condition, category in fields:
        column = f"{condition}__{category}"
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        rows.append(
            {
                "condition": condition,
                "category": category,
                "mean": float(values.mean()) if not values.empty else None,
                "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                "count": int(values.count()),
            }
        )
    return rows


def compute_pairwise_agreement(df: pd.DataFrame, raters: List[str], fields: List[Tuple[str, str]]) -> List[Dict]:
    """Compute pairwise quadratic weighted kappa for each condition/category."""
    rows = []
    for left, right in combinations(raters, 2):
        left_df = df[df["rater_id"] == left].set_index("sample_id")
        right_df = df[df["rater_id"] == right].set_index("sample_id")
        shared_ids = left_df.index.intersection(right_df.index)
        if shared_ids.empty:
            continue

        for condition, category in fields:
            column = f"{condition}__{category}"
            left_scores = pd.to_numeric(left_df.loc[shared_ids, column], errors="coerce")
            right_scores = pd.to_numeric(right_df.loc[shared_ids, column], errors="coerce")
            paired = pd.DataFrame({"left": left_scores, "right": right_scores}).dropna()
            if paired.empty:
                continue
            kappa = quadratic_weighted_kappa(
                paired["left"].astype(int).tolist(),
                paired["right"].astype(int).tolist(),
            )
            rows.append(
                {
                    "rater_a": left,
                    "rater_b": right,
                    "condition": condition,
                    "category": category,
                    "quadratic_weighted_kappa": kappa,
                    "n_shared_samples": int(len(paired)),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Analyze multiple exported quick-eval CSV files")
    parser.add_argument("csv_paths", nargs="+", type=Path, help="Paths to per-rater CSV exports")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluator/data/multi_rater"))
    args = parser.parse_args()

    merged, raters, fields = load_rater_exports(args.csv_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    score_summary = summarize_scores(merged, fields)
    agreement_summary = compute_pairwise_agreement(merged, raters, fields)

    merged.to_csv(args.output_dir / "merged_ratings.csv", index=False)
    pd.DataFrame(score_summary).to_csv(args.output_dir / "score_summary.csv", index=False)
    pd.DataFrame(agreement_summary).to_csv(args.output_dir / "pairwise_kappa.csv", index=False)

    summary = {
        "raters": raters,
        "num_raters": len(raters),
        "num_rows": int(len(merged)),
        "fields": [{"condition": condition, "category": category} for condition, category in fields],
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved merged outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
