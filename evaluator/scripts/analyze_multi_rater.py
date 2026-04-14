#!/usr/bin/env python3
"""
Aggregate quick-eval CSV exports and compute sample-level human-eval statistics.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from scipy.stats import wilcoxon


CATEGORIES = ("relevance", "coherence", "naturalness")
PRIMARY_CONDITIONS = ("base", "fine_tuned")
OPTIONAL_CONDITIONS = ("prompt_zero_shot", "prompt_few_shot", "gpt_zero_shot", "gpt_few_shot")
LEGACY_PREFIXES = {
    "baseline": "base",
    "finetuned": "fine_tuned",
    "zeroshot": "gpt_zero_shot",
    "fewshot": "gpt_few_shot",
}
DATASET_LABELS = {
    "GEM/schema_guided_dialog": "SGD",
    "allenai/WildChat-1M": "WildChat",
}


def rating_columns(conditions: Sequence[str], include_avg: bool = True) -> List[str]:
    columns = [f"{condition}__{category}" for condition in conditions for category in CATEGORIES]
    if include_avg:
        columns.extend(f"{condition}__avg" for condition in conditions)
    return columns


def detect_conditions(df: pd.DataFrame) -> List[str]:
    candidates = (*PRIMARY_CONDITIONS, *OPTIONAL_CONDITIONS)
    return [
        condition
        for condition in candidates
        if all(f"{condition}__{category}" in df.columns for category in CATEGORIES)
    ]


def normalize_export(path: Path, index: int) -> pd.DataFrame:
    """Load one rater export and normalize legacy column names without mixing prompt sources."""
    df = pd.read_csv(path)
    missing_base = {"sample_id", "dataset"} - set(df.columns)
    if missing_base:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_base)}")

    rename: Dict[str, str] = {}
    for legacy, normalized in LEGACY_PREFIXES.items():
        for category in (*CATEGORIES, "avg"):
            old = f"{legacy}_{category}"
            if old in df.columns:
                rename[old] = f"{normalized}__{category}"
    df = df.rename(columns=rename)

    conditions = detect_conditions(df)
    missing_primary = [condition for condition in PRIMARY_CONDITIONS if condition not in conditions]
    if missing_primary:
        raise ValueError(f"{path} is missing primary conditions: {missing_primary}")

    score_columns = rating_columns(conditions, include_avg=False)
    for column in score_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[score_columns].isna().any().any():
        raise ValueError(f"{path} contains non-numeric or missing score values")

    for condition in conditions:
        columns = [f"{condition}__{category}" for category in CATEGORIES]
        df[f"{condition}__avg"] = df[columns].mean(axis=1)

    keep = ["sample_id", "dataset", *rating_columns(conditions)]
    for optional in ("winner", "timestamp"):
        if optional in df.columns:
            keep.append(optional)
    normalized = df[keep].copy()
    normalized["rater_id"] = path.stem or f"rater_{index + 1}"
    return normalized


def load_rater_exports(paths: List[Path]) -> Tuple[pd.DataFrame, List[str]]:
    frames = [normalize_export(path, index) for index, path in enumerate(paths)]
    merged = pd.concat(frames, ignore_index=True)
    raters = [path.stem or f"rater_{index + 1}" for index, path in enumerate(paths)]
    return merged, raters


def quadratic_weighted_kappa(a: List[int], b: List[int], min_rating: int = 1, max_rating: int = 5) -> float:
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
    expected = [[(row_marginals[i] * col_marginals[j]) / total for j in range(n)] for i in range(n)]
    weights = [[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)]
    observed_weighted = sum(weights[i][j] * observed[i][j] for i in range(n) for j in range(n)) / total
    expected_weighted = sum(weights[i][j] * expected[i][j] for i in range(n) for j in range(n)) / total
    return float("nan") if expected_weighted == 0 else 1.0 - (observed_weighted / expected_weighted)


def krippendorff_alpha_ordinal(items: Iterable[List[int]]) -> float:
    pooled: List[int] = []
    observed_disagreement = 0.0
    for ratings in items:
        valid = [int(rating) for rating in ratings if pd.notna(rating)]
        if len(valid) < 2:
            continue
        pooled.extend(valid)
        ordered_distance = sum((left - right) ** 2 for left in valid for right in valid)
        observed_disagreement += ordered_distance / (len(valid) - 1)

    total = len(pooled)
    if total < 2:
        return float("nan")
    do = observed_disagreement / total
    de = sum((left - right) ** 2 for left in pooled for right in pooled) / (total * (total - 1))
    return float("nan") if de == 0 else 1.0 - (do / de)


def sample_level_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["sample_id", "dataset"], as_index=False)[rating_columns(PRIMARY_CONDITIONS)].mean()


def summarize_conditions(sample_df: pd.DataFrame) -> List[Dict]:
    rows = []
    for condition in PRIMARY_CONDITIONS:
        row: Dict[str, object] = {"condition": condition, "n_samples": int(len(sample_df))}
        for category in (*CATEGORIES, "avg"):
            values = sample_df[f"{condition}__{category}"]
            row[f"{category}_mean"] = float(values.mean())
            row[f"{category}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return rows


def safe_wilcoxon(left: pd.Series, right: pd.Series) -> float:
    diff = left - right
    if diff.abs().sum() == 0:
        return 1.0
    return float(wilcoxon(left, right, zero_method="wilcox").pvalue)


def summarize_domains(sample_df: pd.DataFrame) -> List[Dict]:
    rows = []
    domains = [("All", sample_df)]
    domains.extend((DATASET_LABELS.get(name, name), part) for name, part in sample_df.groupby("dataset"))
    for label, part in domains:
        base = part["base__avg"]
        fine_tuned = part["fine_tuned__avg"]
        rows.append(
            {
                "domain": label,
                "n_samples": int(len(part)),
                "base_mean": float(base.mean()),
                "base_std": float(base.std(ddof=1)),
                "fine_tuned_mean": float(fine_tuned.mean()),
                "fine_tuned_std": float(fine_tuned.std(ddof=1)),
                "delta": float((fine_tuned - base).mean()),
                "wilcoxon_p": safe_wilcoxon(fine_tuned, base),
            }
        )
    return rows


def summarize_pairwise_tests(sample_df: pd.DataFrame) -> List[Dict]:
    left = sample_df["fine_tuned__avg"]
    right = sample_df["base__avg"]
    return [
        {
            "left": "fine_tuned",
            "right": "base",
            "left_mean": float(left.mean()),
            "right_mean": float(right.mean()),
            "delta": float((left - right).mean()),
            "wilcoxon_p": safe_wilcoxon(left, right),
            "n_samples": int(len(sample_df)),
        }
    ]


def compute_pairwise_agreement(df: pd.DataFrame, raters: List[str]) -> List[Dict]:
    rows = []
    for left, right in combinations(raters, 2):
        left_df = df[df["rater_id"] == left].set_index("sample_id")
        right_df = df[df["rater_id"] == right].set_index("sample_id")
        shared_ids = left_df.index.intersection(right_df.index)
        for condition in PRIMARY_CONDITIONS:
            for category in CATEGORIES:
                column = f"{condition}__{category}"
                paired = pd.DataFrame(
                    {"left": left_df.loc[shared_ids, column], "right": right_df.loc[shared_ids, column]}
                ).dropna()
                rows.append(
                    {
                        "rater_a": left,
                        "rater_b": right,
                        "condition": condition,
                        "category": category,
                        "quadratic_weighted_kappa": quadratic_weighted_kappa(
                            paired["left"].astype(int).tolist(),
                            paired["right"].astype(int).tolist(),
                        ),
                        "n_shared_samples": int(len(paired)),
                    }
                )
    return rows


def compute_krippendorff(df: pd.DataFrame) -> List[Dict]:
    rows = []
    for condition in PRIMARY_CONDITIONS:
        for category in CATEGORIES:
            column = f"{condition}__{category}"
            pivot = df.pivot_table(index="sample_id", columns="rater_id", values=column, aggfunc="first")
            alpha = krippendorff_alpha_ordinal([row.dropna().astype(int).tolist() for _, row in pivot.iterrows()])
            rows.append({"condition": condition, "category": category, "krippendorff_alpha": alpha})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze multiple exported quick-eval CSV files")
    parser.add_argument("csv_paths", nargs="+", type=Path, help="Paths to per-rater CSV exports")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluator/data/multi_rater"))
    args = parser.parse_args()

    merged, raters = load_rater_exports(args.csv_paths)
    sample_df = sample_level_frame(merged)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    condition_summary = summarize_conditions(sample_df)
    domain_summary = summarize_domains(sample_df)
    paired_tests = summarize_pairwise_tests(sample_df)
    pairwise_kappa = compute_pairwise_agreement(merged, raters)
    krippendorff = compute_krippendorff(merged)

    merged.to_csv(args.output_dir / "merged_ratings.csv", index=False)
    sample_df.to_csv(args.output_dir / "sample_level_ratings.csv", index=False)
    pd.DataFrame(condition_summary).to_csv(args.output_dir / "condition_summary.csv", index=False)
    pd.DataFrame(domain_summary).to_csv(args.output_dir / "domain_summary.csv", index=False)
    pd.DataFrame(paired_tests).to_csv(args.output_dir / "paired_tests.csv", index=False)
    pd.DataFrame(pairwise_kappa).to_csv(args.output_dir / "pairwise_kappa.csv", index=False)
    pd.DataFrame(krippendorff).to_csv(args.output_dir / "krippendorff_alpha.csv", index=False)

    summary = {
        "raters": raters,
        "num_raters": len(raters),
        "num_samples": int(len(sample_df)),
        "num_rater_rows": int(len(merged)),
        "primary_conditions": list(PRIMARY_CONDITIONS),
        "note": "Prompt columns are preserved in merged_ratings.csv but excluded from the multi-rater aggregate because rater exports used different prompt-baseline sources.",
        "mean_pairwise_quadratic_weighted_kappa": float(
            pd.DataFrame(pairwise_kappa)["quadratic_weighted_kappa"].mean()
        ),
        "mean_krippendorff_alpha": float(pd.DataFrame(krippendorff)["krippendorff_alpha"].mean()),
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved multi-rater outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
