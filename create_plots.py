import argparse
from pathlib import Path
import sys

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.helpers import setup_plot_style, discover_model_directories, get_model_name
from modules.plot1 import create_benchmark_comparison_plot
from modules.plot2 import create_domain_comparison_plot, print_domain_statistics
from modules.plot3 import create_cross_model_comparison_plot


def generate_all_plots(model_dir: Path, verbose: bool = True) -> None:
    """
    Generate all plots for a single model directory.
    
    Args:
        model_dir: Path to model directory containing CSV/JSON files
        verbose: Print progress information
    """
    model_name = get_model_name(model_dir)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Generating plots for: {model_name}")
        print(f"Directory: {model_dir}")
        print(f"{'='*60}")
    
    # Plot 1: Benchmark Comparison (Base vs Fine-tuned)
    if verbose:
        print("\n[1/2] Creating benchmark comparison plot...")
    
    try:
        output_path = model_dir / "benchmark_comparison.png"
        fig = create_benchmark_comparison_plot(model_dir, output_path)
        fig.clf()
        if verbose:
            print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error creating benchmark plot: {e}")
    
    # Plot 2: Domain-Specific Performance
    if verbose:
        print("\n[2/2] Creating domain comparison plot...")
    
    try:
        output_path = model_dir / "domain_comparison.png"
        fig = create_domain_comparison_plot(model_dir, output_path)
        fig.clf()
        if verbose:
            print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error creating domain plot: {e}")
    
    # Print statistics
    if verbose:
        print_domain_statistics(model_dir)
    
    if verbose:
        print(f"\nCompleted plots for {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis plots for model evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_plots.py
    python create_plots.py --model-dir Qwen/Qwen2.5-3B-Instruct
    python create_plots.py --list-models
        """
    )
    
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Specific model directory to process"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List discovered model directories and exit"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup matplotlib style
    setup_plot_style()
    
    # Determine root directory
    root_dir = Path(__file__).parent
    
    if args.list_models:
        print("Discovered model directories:")
        model_dirs = discover_model_directories(root_dir)
        for d in model_dirs:
            print(f"  - {d.relative_to(root_dir)}")
        return
    
    if args.model_dir:
        # Process specific model directory
        if args.model_dir.is_absolute():
            model_dir = args.model_dir
        else:
            model_dir = root_dir / args.model_dir
        
        if not model_dir.exists():
            print(f"Error: Model directory not found: {model_dir}")
            sys.exit(1)
        
        generate_all_plots(model_dir, verbose=not args.quiet)
    else:
        # Discover and process all model directories
        model_dirs = discover_model_directories(root_dir)
        
        if not model_dirs:
            print("No model directories found with required evaluation files.")
            print("Expected files: eval_bleurt_bertscore_summary.csv, eval_ft_bleurt_bertscore_summary.csv,")
            print("                eval_bleurt_bertscore_per_example.csv, eval_ft_bleurt_bertscore_per_example.csv,")
            print("                chat_pairs.json")
            sys.exit(1)
        
        print(f"Found {len(model_dirs)} model directory(ies) to process")
        
        for model_dir in model_dirs:
            generate_all_plots(model_dir, verbose=not args.quiet)
        
        # Generate cross-model comparison plot at root if multiple models
        if len(model_dirs) > 1:
            if not args.quiet:
                print("\n[Cross-Model] Creating cross-model comparison plot...")
            try:
                output_path = root_dir / "cross_model_comparison.png"
                fig = create_cross_model_comparison_plot(root_dir, output_path)
                fig.clf()
                if not args.quiet:
                    print(f"  ✓ Saved: {output_path}")
            except Exception as e:
                print(f"  ✗ Error creating cross-model plot: {e}")
    
    print("\n" + "="*60)
    print("All plots generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
