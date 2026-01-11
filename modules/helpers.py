"""
Helper utilities for plotting modules.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


def setup_plot_style():
    """Configure matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def discover_model_directories(root_dir: Path) -> List[Path]:
    """
    Discover all model directories containing evaluation data.
    
    Looks for directories containing eval_bleurt_bertscore_summary.csv
    """
    model_dirs = []
    
    for path in root_dir.rglob("eval_bleurt_bertscore_summary.csv"):
        model_dir = path.parent
        # Verify it has all required files
        required_files = [
            "eval_bleurt_bertscore_summary.csv",
            "eval_ft_bleurt_bertscore_summary.csv",
            "eval_bleurt_bertscore_per_example.csv",
            "eval_ft_bleurt_bertscore_per_example.csv",
            "chat_pairs.json"
        ]
        
        if all((model_dir / f).exists() for f in required_files):
            model_dirs.append(model_dir)
    
    return model_dirs


def get_model_name(model_dir: Path) -> str:
    """Extract model name from directory path."""
    return model_dir.name


def ensure_output_dir(output_dir: Path) -> Path:
    """Ensure output directory exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
