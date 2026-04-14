from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def padded_limits(values, *, pad_ratio: float = 0.12, min_span: float = 0.01) -> tuple[float, float] | None:
    numeric = [float(value) for value in values if pd.notna(value)]
    if not numeric:
        return None

    lower = min(numeric)
    upper = max(numeric)
    span = upper - lower

    if span == 0:
        center = lower
        half_span = max(min_span / 2, abs(center) * pad_ratio, 0.001)
        return center - half_span, center + half_span

    pad = max(span * pad_ratio, min_span * 0.25)
    return lower - pad, upper + pad


def apply_padded_ylim(
    ax: plt.Axes,
    values,
    *,
    pad_ratio: float = 0.12,
    min_span: float = 0.01,
) -> None:
    limits = padded_limits(values, pad_ratio=pad_ratio, min_span=min_span)
    if limits is not None:
        ax.set_ylim(*limits)


def resolve_best_sweep_dir(outputs: Path, model_dir: str) -> Path | None:
    best_config = outputs / model_dir / "targeted_sweep" / "best_config.json"
    if not best_config.exists():
        return None

    summary = pd.read_json(best_config, typ="series")
    output_dir = summary.get("output_dir")
    if not output_dir:
        return None

    candidate = Path(output_dir)
    if not candidate.is_absolute():
        candidate = outputs.parent / candidate
    return candidate if candidate.exists() else None


def resolve_rollout_file(outputs: Path, model_dir: str, filename: str, *, prefer_best_sweep: bool = False) -> Path | None:
    search_roots = []
    if prefer_best_sweep:
        best_dir = resolve_best_sweep_dir(outputs, model_dir)
        if best_dir is not None:
            search_roots.append(best_dir / "rollouts")
    search_roots.append(outputs / model_dir / "rollouts")

    for root in search_roots:
        candidate = root / filename
        if candidate.exists():
            return candidate
    return None


def prompt_rows(outputs: Path, model_specs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for model_dir, label in model_specs:
        row = {"model_dir": model_dir, "model": label}
        for mode in ("zero_shot", "few_shot"):
            path = outputs / model_dir / f"eval_prompt_{mode}_bleurt_bertscore_summary.csv"
            if not path.exists():
                continue
            summary = pd.read_csv(path).iloc[0]
            row[f"prompt_{mode}_bertscore"] = float(summary["bertscore_f1_macro"])
            row[f"prompt_{mode}_bleurt"] = float(summary["bleurt_macro"])
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_rows(outputs: Path, model_specs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for model_dir, label in model_specs:
        path = outputs / model_dir / "targeted_sweep" / "summary.csv"
        if not path.exists():
            rows.append({"model_dir": model_dir, "model": label})
            continue
        best = pd.read_csv(path).iloc[0]
        rows.append(
            {
                "model_dir": model_dir,
                "model": label,
                "sweep_best_alpha": int(best["lora_alpha"]),
                "sweep_best_lr": float(best["learning_rate"]),
                "sweep_best_bertscore": float(best["bertscore_f1_macro"]),
                "sweep_best_bleurt": float(best["bleurt_macro"]),
                "sweep_best_ppl": float(best["ppl_macro"]),
            }
        )
    return pd.DataFrame(rows)


def plot_rollout_by_step(outputs: Path, figures: Path, model_specs: list[tuple[str, str]], *, dataset: str = "all") -> None:
    series = []
    for model_dir, label in model_specs:
        ft_path = resolve_rollout_file(
            outputs,
            model_dir,
            f"rollout_reference_assisted_finetuned_{dataset}_steps.csv",
            prefer_best_sweep=True,
        )
        if ft_path is not None:
            df = pd.read_csv(ft_path).groupby("step_index", as_index=False)["user_bertscore_f1"].mean()
            df["model"] = label
            df["condition"] = "Fine-tuned"
            series.append(df)

        base_path = resolve_rollout_file(
            outputs,
            model_dir,
            f"rollout_reference_assisted_base_{dataset}_steps.csv",
        )
        if base_path is not None:
            df = pd.read_csv(base_path).groupby("step_index", as_index=False)["user_bertscore_f1"].mean()
            df["model"] = label
            df["condition"] = "Base"
            series.append(df)
    if not series:
        return

    plot_df = pd.concat(series, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, condition), group in plot_df.groupby(["model", "condition"]):
        linestyle = "--" if condition == "Base" else "-"
        alpha = 0.65 if condition == "Base" else 1.0
        ax.plot(
            group["step_index"],
            group["user_bertscore_f1"],
            marker="o",
            linestyle=linestyle,
            alpha=alpha,
            label=f"{model} ({condition})",
        )
    ax.set_xlabel("Rollout Step")
    ax.set_ylabel("Mean User BERTScore F1")
    ax.set_title("Reference-Assisted Rollout Quality by Step")
    apply_padded_ylim(ax, plot_df["user_bertscore_f1"], min_span=0.02)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = figures / "revision_rollout_by_step.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_prompt_fairness(figures: Path, df: pd.DataFrame) -> None:
    needed = {"prompt_zero_shot_bleurt", "prompt_few_shot_bleurt"}
    if not needed.issubset(df.columns):
        return
    plot_df = df.dropna(subset=sorted(needed)).copy()
    if plot_df.empty:
        return

    x = range(len(plot_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], plot_df["prompt_zero_shot_bleurt"], width=width, label="Zero-shot", color="#9CA3AF")
    ax.bar([i + width / 2 for i in x], plot_df["prompt_few_shot_bleurt"], width=width, label="Few-shot", color="#2563EB")
    ax.set_xticks(list(x), plot_df["model"], rotation=20, ha="right")
    ax.set_ylabel("BLEURT")
    ax.set_title("Open-Model Prompt Baselines")
    apply_padded_ylim(
        ax,
        pd.concat([plot_df["prompt_zero_shot_bleurt"], plot_df["prompt_few_shot_bleurt"]], ignore_index=True),
        min_span=0.02,
    )
    ax.legend()
    fig.tight_layout()
    out_path = figures / "revision_prompt_fairness.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
