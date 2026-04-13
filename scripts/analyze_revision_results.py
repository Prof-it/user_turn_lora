#!/usr/bin/env python3
"""
Aggregate revision experiments into paper-ready tables and plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from revision_analysis_helpers import (
    apply_padded_ylim,
    padded_limits,
    plot_prompt_fairness,
    plot_rollout_by_step,
    prompt_rows,
    resolve_rollout_file,
    sweep_rows,
)


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "paper" / "figures"

MODEL_SPECS = [
    ("LiquidAI-LFM2.5-1.2B-Instruct", "LiquidAI 1.2B"),
    ("Qwen-Qwen2.5-3B-Instruct", "Qwen 3B"),
    ("meta-llama-Llama-3.2-3B-Instruct", "Llama 3B"),
    ("allenai-OLMo-3-7B-Instruct", "OLMo 7B"),
]


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _single_turn_rows() -> pd.DataFrame:
    rows = []
    for model_dir, label in MODEL_SPECS:
        base = pd.read_csv(OUTPUTS / model_dir / "eval_bleurt_bertscore_summary.csv").iloc[0]
        ft = pd.read_csv(OUTPUTS / model_dir / "eval_ft_bleurt_bertscore_summary.csv").iloc[0]
        rows.append(
            {
                "model_dir": model_dir,
                "model": label,
                "single_turn_base_bertscore": float(base["bertscore_f1_macro"]),
                "single_turn_base_bleurt": float(base["bleurt_macro"]),
                "single_turn_base_ppl": float(base["ppl_content_macro"]),
                "single_turn_ft_bertscore": float(ft["bertscore_f1_macro"]),
                "single_turn_ft_bleurt": float(ft["bleurt_macro"]),
                "single_turn_ft_ppl": float(ft["ppl_content_macro"]),
            }
        )
    return pd.DataFrame(rows)


def _load_rollout_summary(model_dir: str, filename: str, *, prefer_best_sweep: bool = False) -> dict | None:
    path = resolve_rollout_file(OUTPUTS, model_dir, filename, prefer_best_sweep=prefer_best_sweep)
    if path is None:
        return None
    payload = _read_json(path)
    return payload["summary"] if "summary" in payload else payload


def _rollout_rows() -> pd.DataFrame:
    rows = []
    for model_dir, label in MODEL_SPECS:
        ft = _load_rollout_summary(
            model_dir,
            "rollout_reference_assisted_finetuned_sgd_summary.json",
            prefer_best_sweep=True,
        )
        base = _load_rollout_summary(model_dir, "rollout_reference_assisted_base_sgd_summary.json")
        free = None
        if model_dir == "LiquidAI-LFM2.5-1.2B-Instruct":
            free = _load_rollout_summary(
                model_dir,
                "rollout_free_assistant_finetuned_sgd_LiquidAI-LFM2.5-1.2B-Instruct_summary.json",
            )

        row = {"model_dir": model_dir, "model": label}
        if ft:
            row.update(
                {
                    "ref_ft_bertscore": ft["user_bertscore_f1_macro"],
                    "ref_ft_bleurt": ft["user_bleurt_macro"],
                    "ref_ft_degradation": ft["user_bertscore_degradation"],
                    "ref_ft_collapse": ft["collapse_rate"],
                }
            )
        if base:
            row.update(
                {
                    "ref_base_bertscore": base["user_bertscore_f1_macro"],
                    "ref_base_bleurt": base["user_bleurt_macro"],
                    "ref_base_degradation": base["user_bertscore_degradation"],
                    "ref_base_collapse": base["collapse_rate"],
                }
            )
        if free:
            row.update(
                {
                    "free_ft_user_bertscore": free["user_bertscore_f1_macro"],
                    "free_ft_user_bleurt": free["user_bleurt_macro"],
                    "free_ft_assistant_bertscore": free["assistant_bertscore_f1_macro"],
                    "free_ft_assistant_bleurt": free["assistant_bleurt_macro"],
                    "free_ft_collapse": free["collapse_rate"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_revision_table() -> pd.DataFrame:
    single_turn = _single_turn_rows()
    rollout = _rollout_rows()
    prompt = prompt_rows(OUTPUTS, MODEL_SPECS)
    sweep = sweep_rows(OUTPUTS, MODEL_SPECS)
    merged = single_turn.merge(rollout, on=["model_dir", "model"], how="left")
    merged = merged.merge(prompt, on=["model_dir", "model"], how="left")
    merged = merged.merge(sweep, on=["model_dir", "model"], how="left")
    merged["single_turn_ft_minus_base_bleurt"] = merged["single_turn_ft_bleurt"] - merged["single_turn_base_bleurt"]
    if "ref_base_bleurt" in merged:
        merged["rollout_ft_minus_base_bleurt"] = merged["ref_ft_bleurt"] - merged["ref_base_bleurt"]
        merged["rollout_ft_minus_base_bertscore"] = merged["ref_ft_bertscore"] - merged["ref_base_bertscore"]
    if "prompt_few_shot_bleurt" in merged:
        merged["prompt_few_shot_minus_zero_shot_bleurt"] = (
            merged["prompt_few_shot_bleurt"] - merged["prompt_zero_shot_bleurt"]
        )
    return merged


def _save_table(df: pd.DataFrame) -> None:
    out_path = OUTPUTS / "revision_summary_table.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    df[
        [
            "model",
            "single_turn_base_bertscore",
            "single_turn_base_bleurt",
            "single_turn_base_ppl",
            "single_turn_ft_bertscore",
            "single_turn_ft_bleurt",
            "single_turn_ft_ppl",
        ]
    ].to_csv(OUTPUTS / "revision_single_turn_table.csv", index=False)
    rollout_cols = [c for c in [
        "model",
        "ref_base_bertscore",
        "ref_base_bleurt",
        "ref_base_degradation",
        "ref_base_collapse",
        "ref_ft_bertscore",
        "ref_ft_bleurt",
        "ref_ft_degradation",
        "ref_ft_collapse",
    ] if c in df.columns]
    df[rollout_cols].to_csv(OUTPUTS / "revision_rollout_table.csv", index=False)
    print(f"Saved {OUTPUTS / 'revision_single_turn_table.csv'}")
    print(f"Saved {OUTPUTS / 'revision_rollout_table.csv'}")


def _plot_rollout_quality(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["ref_ft_bertscore"]).copy()
    x = range(len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(x, plot_df["ref_ft_bertscore"], color=["#3B82F6", "#10B981", "#F59E0B", "#EF4444"][: len(plot_df)])
    axes[0].set_xticks(list(x), plot_df["model"], rotation=20, ha="right")
    axes[0].set_title("Reference-Assisted Rollout BERTScore")
    axes[0].set_ylabel("Macro BERTScore F1")
    apply_padded_ylim(axes[0], plot_df["ref_ft_bertscore"], min_span=0.02)

    axes[1].bar(x, plot_df["ref_ft_bleurt"], color=["#3B82F6", "#10B981", "#F59E0B", "#EF4444"][: len(plot_df)])
    axes[1].set_xticks(list(x), plot_df["model"], rotation=20, ha="right")
    axes[1].set_title("Reference-Assisted Rollout BLEURT")
    axes[1].set_ylabel("Macro BLEURT")
    apply_padded_ylim(axes[1], plot_df["ref_ft_bleurt"], min_span=0.02)

    fig.tight_layout()
    out_path = FIGURES / "revision_rollout_quality.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_rollout_stability(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["ref_ft_degradation"]).copy()
    x = range(len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(x, plot_df["ref_ft_degradation"], color="#6366F1")
    axes[0].set_xticks(list(x), plot_df["model"], rotation=20, ha="right")
    axes[0].set_title("Rollout Degradation")
    axes[0].set_ylabel("Step-0 minus Final-Step BERTScore")
    apply_padded_ylim(axes[0], plot_df["ref_ft_degradation"], min_span=0.01)

    collapse_values = plot_df["ref_ft_collapse"].fillna(0.0)
    axes[1].bar(x, collapse_values, color="#FCA5A5", edgecolor="#DC2626", linewidth=1.2)
    axes[1].scatter(x, collapse_values, color="#DC2626", s=70, zorder=3)
    for xpos, value in zip(x, collapse_values):
        axes[1].annotate(f"{value:.2f}", (xpos, value), xytext=(0, 6), textcoords="offset points", ha="center", color="#991B1B", fontsize=9)
    axes[1].set_xticks(list(x), plot_df["model"], rotation=20, ha="right")
    axes[1].set_title("Rollout Collapse Rate")
    axes[1].set_ylabel("Collapse Rate")
    axes[1].axhline(0.0, color="#7F1D1D", linewidth=1.0, alpha=0.7)
    apply_padded_ylim(axes[1], collapse_values, min_span=0.01)

    fig.tight_layout()
    out_path = FIGURES / "revision_rollout_stability.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_liquidai_base_vs_ft(df: pd.DataFrame) -> None:
    row = df.loc[df["model_dir"] == "LiquidAI-LFM2.5-1.2B-Instruct"].iloc[0]
    metrics = pd.DataFrame(
        {
            "condition": ["Base", "Fine-tuned"],
            "bertscore": [row["ref_base_bertscore"], row["ref_ft_bertscore"]],
            "bleurt": [row["ref_base_bleurt"], row["ref_ft_bleurt"]],
            "degradation": [row["ref_base_degradation"], row["ref_ft_degradation"]],
            "collapse": [row["ref_base_collapse"], row["ref_ft_collapse"]],
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors = ["#9CA3AF", "#2563EB"]
    for axis, column, title in [
        (axes[0], "bertscore", "User BERTScore"),
        (axes[1], "bleurt", "User BLEURT"),
        (axes[2], "degradation", "Degradation"),
        (axes[3], "collapse", "Collapse Rate"),
    ]:
        axis.bar(metrics["condition"], metrics[column], color=colors)
        axis.set_title(title)
        apply_padded_ylim(axis, metrics[column], min_span=0.01)
    fig.tight_layout()
    out_path = FIGURES / "revision_liquidai_base_vs_ft_rollout.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_single_vs_multi(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["ref_ft_bleurt"]).copy()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(plot_df["single_turn_ft_bleurt"], plot_df["ref_ft_bleurt"], s=80, color="#0F766E")
    for _, row in plot_df.iterrows():
        ax.annotate(row["model"], (row["single_turn_ft_bleurt"], row["ref_ft_bleurt"]), xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Single-Turn BLEURT")
    ax.set_ylabel("Reference-Assisted Rollout BLEURT")
    ax.set_title("Single-Turn vs Multi-Turn Quality")
    x_limits = padded_limits(plot_df["single_turn_ft_bleurt"], min_span=0.02)
    y_limits = padded_limits(plot_df["ref_ft_bleurt"], min_span=0.02)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path = FIGURES / "revision_single_vs_multi_bleurt.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = build_revision_table().sort_values("ref_ft_bleurt", ascending=False, na_position="last")
    _save_table(df)
    _plot_rollout_quality(df)
    _plot_rollout_stability(df)
    plot_rollout_by_step(OUTPUTS, FIGURES, MODEL_SPECS)
    plot_prompt_fairness(FIGURES, df)
    if {"ref_base_bertscore", "ref_ft_bertscore"}.issubset(df.columns):
        _plot_liquidai_base_vs_ft(df)
    _plot_single_vs_multi(df)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
