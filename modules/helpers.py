"""
Helper utilities for plotting modules.

Uses tueplots icml2024 bundle for conference-quality figures.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import tueplots
from tueplots import bundles, figsizes


def setup_plot_style(usetex: bool = False, column: str = "half"):
    """
    Configure matplotlib style using tueplots icml2024 bundle.
    
    Args:
        usetex: Use LaTeX rendering (requires LaTeX installation)
        column: "half" for single column, "full" for full width
    """
    # Use ICML 2024 bundle - let tueplots handle all styling
    config = bundles.icml2024(usetex=usetex, column=column)
    plt.rcParams.update(config)


def get_figsize(column: str = "half", nrows: int = 1, ncols: int = 1) -> tuple:
    """
    Get figure size using tueplots icml2024 sizing.
    
    Args:
        column: "half" for single column, "full" for full width
        nrows: Number of subplot rows
        ncols: Number of subplot columns
    
    Returns:
        Tuple of (width, height) in inches
    """
    if column == "full":
        size_config = figsizes.icml2024_full(nrows=nrows, ncols=ncols)
    else:
        size_config = figsizes.icml2024_half(nrows=nrows, ncols=ncols)
    
    return size_config["figure.figsize"]


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
