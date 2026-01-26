"""
Plot 2: Domain-Specific Performance Analysis

Creates bar charts showing relative change (%) by domain (Open-domain vs Task-oriented)
for each metric. Joins CSV data with JSON metadata to get domain information.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Color scheme for domains
COLORS = {
    "Open-domain": "#E67300",    # Darker orange (WildChat)
    "Task-oriented": "#FFB366"   # Lighter orange (Schema-Guided Dialog)
}

# Dataset to domain mapping
DATASET_TO_DOMAIN = {
    "allenai/WildChat-1M": "Open-domain",
    "GEM/schema_guided_dialog": "Task-oriented"
}


def load_chat_pairs_with_domain(model_dir: Path) -> pd.DataFrame:
    """Load chat_pairs.json and extract domain info for each example."""
    json_path = model_dir / "chat_pairs.json"
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    records = []
    for i, item in enumerate(data):
        dataset = item.get("meta", {}).get("dataset", "unknown")
        domain = DATASET_TO_DOMAIN.get(dataset, "Unknown")
        records.append({
            "index": i,
            "dataset": dataset,
            "domain": domain,
            "target_user": item.get("target_user", ""),
            "pred_user": item.get("pred_user", ""),
            "pred_user_ft": item.get("pred_user_ft", "")
        })
    
    return pd.DataFrame(records)


def load_and_merge_data(model_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSV data with domain info.
    
    Returns:
        Tuple of (base_df_with_domain, ft_df_with_domain)
    """
    # Load per-example CSVs
    base_path = model_dir / "eval_bleurt_bertscore_per_example.csv"
    ft_path = model_dir / "eval_ft_bleurt_bertscore_per_example.csv"
    
    base_df = pd.read_csv(base_path)
    ft_df = pd.read_csv(ft_path)
    
    # Check if domain column exists directly in CSV (new format)
    if "domain" in base_df.columns and "domain" in ft_df.columns:
        return base_df, ft_df
    
    # Fallback: Load domain info from JSON (old format)
    domain_df = load_chat_pairs_with_domain(model_dir)
    
    # Add index for merging
    base_df["index"] = base_df.index
    ft_df["index"] = ft_df.index
    
    # Merge with domain info
    base_merged = base_df.merge(domain_df[["index", "domain", "dataset"]], on="index", how="left")
    ft_merged = ft_df.merge(domain_df[["index", "domain", "dataset"]], on="index", how="left")
    
    return base_merged, ft_merged


def calculate_domain_metrics(model_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate metrics grouped by domain.
    
    Returns:
        Dict structure: {metric_name: {domain: {base_mean, ft_mean, relative_change}}}
    """
    base_df, ft_df = load_and_merge_data(model_dir)
    
    metrics = {}
    metric_cols = {
        "BERTScore-F1": "bertscore_f1",
        "BLEURT": "bleurt", 
        "Perplexity": "ppl_content"
    }
    
    domains = ["Open-domain", "Task-oriented"]
    
    for metric_name, col_name in metric_cols.items():
        metrics[metric_name] = {}
        
        for domain in domains:
            base_domain = base_df[base_df["domain"] == domain]
            ft_domain = ft_df[ft_df["domain"] == domain]
            
            if len(base_domain) == 0 or len(ft_domain) == 0:
                continue
            
            base_mean = base_domain[col_name].mean()
            ft_mean = ft_domain[col_name].mean()
            
            # Calculate relative change
            if metric_name == "Perplexity":
                # Lower is better for perplexity
                relative_change = ((base_mean - ft_mean) / base_mean) * 100
            else:
                # Higher is better
                if base_mean != 0:
                    relative_change = ((ft_mean - base_mean) / abs(base_mean)) * 100
                else:
                    relative_change = 0
            
            metrics[metric_name][domain] = {
                "base_mean": base_mean,
                "ft_mean": ft_mean,
                "base_std": base_domain[col_name].std(),
                "ft_std": ft_domain[col_name].std(),
                "relative_change": relative_change,
                "n_samples": len(base_domain)
            }
    
    return metrics


def create_domain_comparison_plot(
    model_dir: Path,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5),
    dpi: int = 150
) -> plt.Figure:
    """
    Create domain-specific performance plot showing relative change (%) by domain.
    
    Matches the reference image style with grouped bars per metric.
    """
    metrics = calculate_domain_metrics(model_dir)
    
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    metric_names = ["BERTScore-F1", "BLEURT", "Perplexity"]
    domains = ["Open-domain", "Task-oriented"]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    # Plot bars for each domain
    for i, domain in enumerate(domains):
        values = []
        for metric in metric_names:
            if domain in metrics[metric]:
                values.append(metrics[metric][domain]["relative_change"])
            else:
                values.append(0)
        
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, 
            values, 
            width, 
            label=domain,
            color=COLORS[domain],
            edgecolor="black",
            linewidth=0.5
        )
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            va = "bottom" if height >= 0 else "top"
            y_offset = 2 if height >= 0 else -2
            
            ax.annotate(
                f"{val:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                va=va,
                fontweight="bold"
            )
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    
    # Styling
    ax.set_ylabel("Relative Change Δ (%)")
    ax.set_xlabel("Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(title="Domain", loc="upper right")
    
    # Set y-axis limits with some padding
    all_values = []
    for metric in metric_names:
        for domain in domains:
            if domain in metrics[metric]:
                all_values.append(metrics[metric][domain]["relative_change"])
    
    if all_values:
        y_min = min(all_values) * 1.2 if min(all_values) < 0 else min(all_values) * 0.8
        y_max = max(all_values) * 1.2 if max(all_values) > 0 else max(all_values) * 0.8
        ax.set_ylim(y_min - 10, y_max + 10)
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved domain comparison plot to: {output_path}")
    
    return fig


def create_domain_absolute_comparison_plot(
    model_dir: Path,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
    dpi: int = 150
) -> plt.Figure:
    """
    Create domain-specific plot showing absolute values (base vs fine-tuned) per domain.
    """
    metrics = calculate_domain_metrics(model_dir)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    
    metric_configs = [
        ("BERTScore-F1", "BERTScore-F1"),
        ("BLEURT", "BLEURT"),
        ("Perplexity", "Perplexity")
    ]
    
    domains = ["Open-domain", "Task-oriented"]
    
    for ax, (metric_key, ylabel) in zip(axes, metric_configs):
        x = np.arange(len(domains))
        width = 0.35
        
        base_vals = []
        ft_vals = []
        base_stds = []
        ft_stds = []
        
        for domain in domains:
            if domain in metrics[metric_key]:
                base_vals.append(metrics[metric_key][domain]["base_mean"])
                ft_vals.append(metrics[metric_key][domain]["ft_mean"])
                base_stds.append(metrics[metric_key][domain]["base_std"])
                ft_stds.append(metrics[metric_key][domain]["ft_std"])
            else:
                base_vals.append(0)
                ft_vals.append(0)
                base_stds.append(0)
                ft_stds.append(0)
        
        # Create grouped bars
        bars1 = ax.bar(x - width/2, base_vals, width, label="Base", 
                       color=COLORS["Task-oriented"], edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width/2, ft_vals, width, label="Fine-tuned",
                       color=COLORS["Open-domain"], edgecolor="black", linewidth=0.5)
        
        # Add error bars
        ax.errorbar(x - width/2, base_vals, yerr=base_stds, fmt="none", 
                   color="black", capsize=3, capthick=1)
        ax.errorbar(x + width/2, ft_vals, yerr=ft_stds, fmt="none",
                   color="black", capsize=3, capthick=1)
        
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(domains)
        ax.legend()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved domain absolute comparison plot to: {output_path}")
    
    return fig


def print_domain_statistics(model_dir: Path) -> None:
    """Print domain statistics for debugging/verification."""
    metrics = calculate_domain_metrics(model_dir)
    
    print("\n" + "=" * 60)
    print("Domain-Specific Performance Statistics")
    print("=" * 60)
    
    for metric_name, domain_data in metrics.items():
        print(f"\n{metric_name}:")
        print("-" * 40)
        for domain, stats in domain_data.items():
            print(f"  {domain}:")
            print(f"    Base mean: {stats['base_mean']:.4f} (±{stats['base_std']:.4f})")
            print(f"    FT mean:   {stats['ft_mean']:.4f} (±{stats['ft_std']:.4f})")
            print(f"    Δ:         {stats['relative_change']:+.2f}%")
            print(f"    N samples: {stats['n_samples']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_dir = Path(sys.argv[1])
    else:
        model_dir = Path(__file__).parent.parent / "Qwen" / "Qwen2.5-3B-Instruct"
    
    if model_dir.exists():
        print_domain_statistics(model_dir)
        
        output_path = model_dir / "domain_comparison.png"
        fig = create_domain_comparison_plot(model_dir, output_path)
        
        output_path2 = model_dir / "domain_absolute_comparison.png"
        fig2 = create_domain_absolute_comparison_plot(model_dir, output_path2)
        
        plt.show()
    else:
        print(f"Model directory not found: {model_dir}")
