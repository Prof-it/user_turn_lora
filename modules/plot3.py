"""
Plot 3: Cross-Model Comparison

Creates bar charts comparing all models side-by-side.
Base results in light orange, fine-tuned in dark orange.
Different models distinguished by bar patterns (solid, hatched, dotted, etc.)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .helpers import discover_model_directories, get_model_name


# Color scheme
COLORS = {
    "base": "#FFB366",      # Lighter orange
    "fine_tuned": "#E67300"  # Darker orange
}

# Hatch patterns for different models
HATCH_PATTERNS = [
    "",      # Solid (no hatch)
    "//",    # Diagonal lines
    "\\\\",  # Back diagonal
    "xx",    # Cross-hatch
    "..",    # Dots
    "++",    # Plus signs
    "oo",    # Circles
    "**",    # Stars
]


def load_model_metrics(model_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load summary metrics for a single model."""
    base_path = model_dir / "eval_bleurt_bertscore_summary.csv"
    ft_path = model_dir / "eval_ft_bleurt_bertscore_summary.csv"
    
    base_df = pd.read_csv(base_path)
    ft_df = pd.read_csv(ft_path)
    
    return {
        "BERTScore-F1": {
            "base": base_df["bertscore_f1_macro"].iloc[0],
            "fine_tuned": ft_df["bertscore_f1_macro"].iloc[0],
        },
        "BLEURT": {
            "base": base_df["bleurt_macro"].iloc[0],
            "fine_tuned": ft_df["bleurt_macro"].iloc[0],
        },
        "Perplexity": {
            "base": base_df["ppl_content_macro"].iloc[0],
            "fine_tuned": ft_df["ppl_content_macro"].iloc[0],
        }
    }


def load_all_models_metrics(root_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load metrics for all discovered models.
    
    Returns:
        Dict structure: {model_name: {metric_name: {base, fine_tuned}}}
    """
    model_dirs = discover_model_directories(root_dir)
    all_metrics = {}
    
    for model_dir in model_dirs:
        model_name = get_model_name(model_dir)
        # Shorten model names for display
        short_name = model_name.replace("-Instruct", "").replace("-userturn-qlora", "")
        all_metrics[short_name] = load_model_metrics(model_dir)
    
    return all_metrics


def create_cross_model_comparison_plot(
    root_dir: Path,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
    dpi: int = 150
) -> plt.Figure:
    """
    Create cross-model comparison plot with all models side-by-side.
    
    Args:
        root_dir: Root directory containing model subdirectories
        output_path: Path to save the figure (optional)
        figsize: Figure size tuple
        dpi: DPI for saved figure
    
    Returns:
        matplotlib Figure object
    """
    all_metrics = load_all_models_metrics(root_dir)
    model_names = list(all_metrics.keys())
    n_models = len(model_names)
    
    if n_models == 0:
        raise ValueError("No models found to compare")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    
    metric_configs = [
        ("BERTScore-F1", "BERTScore-F1", True),
        ("BLEURT", "BLEURT", True),
        ("Perplexity", "Perplexity", False)
    ]
    
    # Bar width and positions
    bar_width = 0.35
    group_width = bar_width * 2 + 0.1  # Width for one model (base + ft + gap)
    
    for ax, (metric_key, ylabel, higher_is_better) in zip(axes, metric_configs):
        x_positions = []
        
        for i, model_name in enumerate(model_names):
            metrics = all_metrics[model_name][metric_key]
            base_val = metrics["base"]
            ft_val = metrics["fine_tuned"]
            
            # Position for this model's bars
            base_x = i * (group_width + 0.3)
            ft_x = base_x + bar_width + 0.05
            
            hatch = HATCH_PATTERNS[i % len(HATCH_PATTERNS)]
            
            # Base bar (light orange)
            ax.bar(
                base_x, base_val, bar_width,
                color=COLORS["base"],
                edgecolor="black",
                linewidth=0.8,
                hatch=hatch,
                label=f"{model_name} Base" if i == 0 else None
            )
            
            # Fine-tuned bar (dark orange)
            ax.bar(
                ft_x, ft_val, bar_width,
                color=COLORS["fine_tuned"],
                edgecolor="black",
                linewidth=0.8,
                hatch=hatch,
                label=f"{model_name} Fine-tuned" if i == 0 else None
            )
            
            x_positions.append((base_x + ft_x) / 2)
        
        # Styling
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        
        # Set y-axis limits
        all_vals = [all_metrics[m][metric_key][k] for m in model_names for k in ["base", "fine_tuned"]]
        if metric_key == "BERTScore-F1":
            y_min = min(all_vals) * 1.1 if min(all_vals) < 0 else 0
            y_max = max(all_vals) * 1.1 if max(all_vals) > 0 else 0.5
            ax.set_ylim(bottom=y_min, top=y_max)
        elif metric_key == "BLEURT":
            # BLEURT typically ranges 0-1, don't force start at 0
            y_min = min(all_vals) - 0.05
            y_max = max(all_vals) + 0.05
            ax.set_ylim(bottom=y_min, top=y_max)
        else:
            ax.set_ylim(bottom=0)
        
        # Format y-axis
        if metric_key in ["BERTScore-F1", "BLEURT"]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # Create custom legend
    legend_elements = []
    for i, model_name in enumerate(model_names):
        hatch = HATCH_PATTERNS[i % len(HATCH_PATTERNS)]
        # Add a patch for each model
        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor="white", edgecolor="black", hatch=hatch, label=model_name))
    
    # Add base/fine-tuned color legend
    from matplotlib.patches import Patch
    legend_elements.append(Patch(facecolor=COLORS["base"], edgecolor="black", label="Base"))
    legend_elements.append(Patch(facecolor=COLORS["fine_tuned"], edgecolor="black", label="Fine-tuned"))
    
    fig.legend(handles=legend_elements, loc="upper center", ncol=len(model_names) + 2, 
               bbox_to_anchor=(0.5, 1.0))
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved cross-model comparison plot to: {output_path}")
    
    return fig


if __name__ == "__main__":
    import sys
    root_dir = Path(__file__).parent.parent
    
    output_path = root_dir / "cross_model_comparison.png"
    fig = create_cross_model_comparison_plot(root_dir, output_path)
    plt.show()
