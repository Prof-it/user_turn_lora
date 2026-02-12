"""Generate publication-quality figures for IEEE ICETSIS 2026 paper.

Uses tueplots for font sizing and overrides figure dimensions
to match IEEE two-column format exactly:
  - Single column: 3.5 in (88.9 mm)
  - Full width:    7.16 in (182 mm)

References:
  https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from pathlib import Path
from tueplots import bundles, figsizes

# ---------- IEEE dimensions ----------
IEEE_COL_WIDTH = 3.5  # inches
IEEE_FULL_WIDTH = 7.16  # inches
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
DPI = 300

# tueplots base (ICML is closest to IEEE); override sizes
_base = bundles.icml2022(usetex=False, family="serif")
_base["figure.figsize"] = (IEEE_COL_WIDTH, IEEE_COL_WIDTH / GOLDEN_RATIO)
_base["figure.constrained_layout.use"] = False
_base["figure.autolayout"] = False
_base["savefig.dpi"] = DPI
_base["savefig.pad_inches"] = 0.015

plt.rcParams.update(_base)

# ---------- paths ----------
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODEL_DIRS = {
    "Qwen-3B": "Qwen-Qwen2.5-3B-Instruct",
    "Llama-3B": "meta-llama-Llama-3.2-3B-Instruct",
    "OLMo-7B": "allenai-OLMo-3-7B-Instruct",
    "LiquidAI-1.2B": "LiquidAI-LFM2.5-1.2B-Instruct",
}

COLORS = {
    "Qwen-3B": "#1f77b4",
    "Llama-3B": "#ff7f0e",
    "OLMo-7B": "#2ca02c",
    "LiquidAI-1.2B": "#d62728",
}

MARKERS = {
    "Qwen-3B": "D",
    "Llama-3B": "^",
    "OLMo-7B": "o",
    "LiquidAI-1.2B": "s",
}


def load_temperature_data():
    """Load temperature sweep CSVs for all models."""
    frames = []
    for label, dirname in MODEL_DIRS.items():
        csv_path = OUT_DIR / dirname / "temperature_sweep.csv"
        df = pd.read_csv(csv_path)
        df["model"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_temperature_sweep(df):
    """Two-panel temperature sweep: BERTScore and BLEURT."""
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_FULL_WIDTH, 2.6))

    for metric_col, metric_label, ax in [
        ("bertscore_f1", "BERTScore F1", axes[0]),
        ("bleurt", "BLEURT", axes[1]),
    ]:
        for model in MODEL_DIRS:
            color = COLORS[model]
            marker = MARKERS[model]
            # baseline (dashed)
            base = df[(df["model"] == model) & (df["model_type"] == "baseline")]
            ax.plot(
                base["temperature"], base[metric_col],
                color=color, marker=marker, markersize=4,
                linestyle="--", linewidth=1.0, alpha=0.6,
            )
            # fine-tuned (solid)
            ft = df[(df["model"] == model) & (df["model_type"] == "finetuned")]
            ax.plot(
                ft["temperature"], ft[metric_col],
                color=color, marker=marker, markersize=4,
                linestyle="-", linewidth=1.2,
                label=model,
            )

        ax.set_xlabel("Temperature")
        ax.set_ylabel(metric_label)
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.3, 0.4, 0.5, 0.6, 0.7]))

    # single legend below the plots
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0, alpha=0.6))
    labels.append("Base")
    handles.append(Line2D([0], [0], color="gray", linestyle="-", linewidth=1.2))
    labels.append("Fine-tuned")

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=6,
        fontsize="x-small",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.3)

    out_path = FIG_DIR / "temperature_sweep.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(FIG_DIR / "temperature_sweep.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_cross_model_comparison():
    """Grouped bar chart: base vs fine-tuned for BERTScore, BLEURT, PPL."""
    # data from Table I in paper
    models = ["Qwen-3B", "Llama-3B", "OLMo-7B", "LiquidAI-1.2B"]
    bert_base =   [0.840, 0.828, 0.813, 0.756]
    bert_ft =     [0.846, 0.854, 0.798, 0.837]
    bleurt_base = [0.374, 0.338, 0.285, 0.323]
    bleurt_ft =   [0.384, 0.393, 0.229, 0.373]
    ppl_base =    [62.9, 100.6, 11.1, 409.6]
    ppl_ft =      [10.4, 11.2, 9.0, 9.4]

    fig, axes = plt.subplots(1, 3, figsize=(IEEE_FULL_WIDTH, 2.6))

    x = np.arange(len(models))
    w = 0.35

    for ax, base_vals, ft_vals, ylabel in [
        (axes[0], bert_base, bert_ft, "BERTScore F1"),
        (axes[1], bleurt_base, bleurt_ft, "BLEURT"),
        (axes[2], ppl_base, ppl_ft, "Perplexity"),
    ]:
        bars_base = ax.bar(x - w / 2, base_vals, w, label="Base", color="#a6cee3", edgecolor="black", linewidth=0.4)
        bars_ft = ax.bar(x + w / 2, ft_vals, w, label="Fine-tuned", color="#1f78b4", edgecolor="black", linewidth=0.4)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize="x-small")

        if ylabel == "Perplexity":
            ax.set_yscale("log")
        elif ylabel == "BERTScore F1":
            ax.set_ylim(0.72, 0.88)
        elif ylabel == "BLEURT":
            ax.set_ylim(0.18, 0.42)

    axes[2].legend(fontsize="x-small", frameon=False, loc="upper left")
    fig.subplots_adjust(bottom=0.25, wspace=0.4)

    out_path = FIG_DIR / "cross_model_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(FIG_DIR / "cross_model_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    df_temp = load_temperature_data()
    plot_temperature_sweep(df_temp)
    plot_cross_model_comparison()
    print("Done.")
