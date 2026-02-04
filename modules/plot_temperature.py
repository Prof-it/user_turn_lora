"""
Plot 4: Temperature Sweep Visualization

Creates line plots showing metric performance across temperatures
for baseline vs fine-tuned models.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional

from .helpers import discover_model_directories, get_model_name, get_figsize


# Color scheme
COLORS = {
    "baseline": "#FFB366",      # Lighter orange
    "finetuned": "#E67300"      # Darker orange
}

# Line styles for different models
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]


def load_temperature_sweep_data(root_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load temperature sweep CSVs for all models."""
    model_dirs = discover_model_directories(root_dir)
    data = {}
    
    for model_dir in model_dirs:
        sweep_path = model_dir / "temperature_sweep.csv"
        if sweep_path.exists():
            model_name = get_model_name(model_dir)
            # Shorten model names
            short_name = (model_name
                .replace("-Instruct", "")
                .replace("LiquidAI-LFM2.5-1.2B", "LiquidAI")
                .replace("Qwen-Qwen2.5-3B", "Qwen2.5")
                .replace("allenai-OLMo-3-7B", "OLMo")
                .replace("meta-llama-Llama-3.2-3B", "Llama-3.2"))
            data[short_name] = pd.read_csv(sweep_path)
    
    return data


def create_temperature_sweep_plot(
    root_dir: Path,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Create temperature sweep line plot showing all models.
    
    Two panels: BLEURT and BERTScore across temperatures.
    Each panel shows baseline (dashed) and fine-tuned (solid) for all models.
    """
    data = load_temperature_sweep_data(root_dir)
    
    if not data:
        raise ValueError("No temperature sweep data found")
    
    if figsize is None:
        figsize = get_figsize(column="full", nrows=1, ncols=2)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    
    model_names = list(data.keys())
    temperatures = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    metrics = [
        ("bleurt", "BLEURT", axes[0]),
        ("bertscore_f1", "BERTScore-F1", axes[1])
    ]
    
    for metric_col, metric_label, ax in metrics:
        for i, model_name in enumerate(model_names):
            df = data[model_name]
            
            baseline = df[df["model_type"] == "baseline"].sort_values("temperature")
            finetuned = df[df["model_type"] == "finetuned"].sort_values("temperature")
            
            # Baseline (dashed, lighter)
            ax.plot(
                baseline["temperature"], 
                baseline[metric_col],
                linestyle="--",
                marker=MARKERS[i % len(MARKERS)],
                markersize=4,
                color=COLORS["baseline"],
                alpha=0.7,
                label=f"{model_name} (base)" if metric_col == "bleurt" else None
            )
            
            # Fine-tuned (solid, darker)
            ax.plot(
                finetuned["temperature"], 
                finetuned[metric_col],
                linestyle="-",
                marker=MARKERS[i % len(MARKERS)],
                markersize=4,
                color=COLORS["finetuned"],
                alpha=0.7 + 0.1 * i,
                linewidth=1.5,
                label=f"{model_name} (FT)" if metric_col == "bleurt" else None
            )
        
        ax.set_xlabel("Temperature")
        ax.set_ylabel(metric_label)
        ax.set_xticks(temperatures)
        ax.grid(True, alpha=0.3)
    
    # Legend only on first plot
    axes[0].legend(loc="lower left", fontsize=6, ncol=2)
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved temperature sweep plot to: {output_path}")
    
    return fig


def generate_latex_table(root_dir: Path) -> str:
    """Generate LaTeX table for temperature sweep results."""
    data = load_temperature_sweep_data(root_dir)
    
    if not data:
        return "% No data found"
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Temperature Sweep Results: BLEURT scores across temperatures for baseline and fine-tuned models. Best scores per model are \textbf{bolded}.}",
        r"\label{tab:temperature_sweep}",
        r"\small",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Type & T=0.3 & T=0.4 & T=0.5 & T=0.6 & T=0.7 \\",
        r"\midrule",
    ]
    
    for model_name, df in data.items():
        baseline = df[df["model_type"] == "baseline"].sort_values("temperature")
        finetuned = df[df["model_type"] == "finetuned"].sort_values("temperature")
        
        # Find best values
        best_base = baseline["bleurt"].max()
        best_ft = finetuned["bleurt"].max()
        
        # Baseline row
        base_vals = []
        for temp in [0.3, 0.4, 0.5, 0.6, 0.7]:
            val = baseline[baseline["temperature"] == temp]["bleurt"].values[0]
            if val == best_base:
                base_vals.append(f"\\textbf{{{val:.3f}}}")
            else:
                base_vals.append(f"{val:.3f}")
        
        lines.append(f"{model_name} & Base & {' & '.join(base_vals)} \\\\")
        
        # Fine-tuned row
        ft_vals = []
        for temp in [0.3, 0.4, 0.5, 0.6, 0.7]:
            val = finetuned[finetuned["temperature"] == temp]["bleurt"].values[0]
            if val == best_ft:
                ft_vals.append(f"\\textbf{{{val:.3f}}}")
            else:
                ft_vals.append(f"{val:.3f}")
        
        lines.append(f" & FT & {' & '.join(ft_vals)} \\\\")
        lines.append(r"\addlinespace")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    from .helpers import setup_plot_style
    setup_plot_style()
    
    root_dir = Path(__file__).parent.parent / "outputs"
    
    # Generate plot
    output_path = root_dir / "temperature_sweep_comparison.png"
    fig = create_temperature_sweep_plot(root_dir, output_path)
    
    # Generate LaTeX table
    latex = generate_latex_table(root_dir)
    print("\nLaTeX Table:")
    print(latex)
    
    plt.close(fig)
