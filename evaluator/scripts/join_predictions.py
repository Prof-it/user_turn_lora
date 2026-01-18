#!/usr/bin/env python3
"""
Join predictions CSV with chat_pairs JSON by ground truth text.
Creates a merged dataset for human evaluation.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

def load_chat_pairs(model_dir: Path) -> pd.DataFrame:
    """Load chat_pairs.json and create DataFrame with ground truth."""
    json_path = model_dir / "chat_pairs.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    
    records = []
    for i, item in enumerate(data):
        records.append({
            "index": i,
            "ground_truth": item["target_user"],
            "conversation": item["conversation"],
            "dataset": item["meta"].get("dataset", ""),
            "num_turns": item["meta"].get("num_turns", len(item["conversation"])),
        })
    
    return pd.DataFrame(records)

def load_predictions(model_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load base and fine-tuned predictions CSVs."""
    base_path = model_dir / "eval_bleurt_bertscore_per_example.csv"
    ft_path = model_dir / "eval_ft_bleurt_bertscore_per_example.csv"
    
    base_df = pd.read_csv(base_path)
    ft_df = pd.read_csv(ft_path)
    
    return base_df, ft_df

def join_by_ground_truth(model_dir: Path) -> pd.DataFrame:
    """
    Join predictions with chat_pairs by ground truth text.
    This is more reliable than index-based matching.
    """
    # Load data
    chat_pairs_df = load_chat_pairs(model_dir)
    base_df, ft_df = load_predictions(model_dir)
    
    # Rename columns for clarity
    base_df = base_df.rename(columns={
        "ref": "ground_truth",
        "pred": "pred_base",
        "bertscore_f1": "bertscore_f1_base",
        "bleurt": "bleurt_base",
        "ppl_content": "ppl_base"
    })
    
    ft_df = ft_df.rename(columns={
        "ref": "ground_truth",
        "pred_ft": "pred_ft",
        "bertscore_f1": "bertscore_f1_ft",
        "bleurt": "bleurt_ft",
        "ppl_content": "ppl_ft"
    })
    
    # Strip whitespace from ground truth for matching
    chat_pairs_df["ground_truth_clean"] = chat_pairs_df["ground_truth"].str.strip()
    base_df["ground_truth_clean"] = base_df["ground_truth"].str.strip()
    ft_df["ground_truth_clean"] = ft_df["ground_truth"].str.strip()
    
    # Join by ground truth
    merged = chat_pairs_df.merge(
        base_df[["ground_truth_clean", "pred_base", "bertscore_f1_base", "bleurt_base", "ppl_base"]],
        on="ground_truth_clean",
        how="left"
    )
    
    merged = merged.merge(
        ft_df[["ground_truth_clean", "pred_ft", "bertscore_f1_ft", "bleurt_ft", "ppl_ft"]],
        on="ground_truth_clean",
        how="left"
    )
    
    # Drop the clean column
    merged = merged.drop(columns=["ground_truth_clean"])
    
    return merged

def process_all_models(root_dir: Path) -> Dict[str, pd.DataFrame]:
    """Process all model directories and return merged DataFrames."""
    results = {}
    
    # Find model directories
    for org_dir in root_dir.iterdir():
        if not org_dir.is_dir() or org_dir.name.startswith(".") or org_dir.name == "evaluator":
            continue
        
        for model_dir in org_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            
            # Check if required files exist
            if not (model_dir / "chat_pairs.json").exists():
                continue
            if not (model_dir / "eval_bleurt_bertscore_per_example.csv").exists():
                continue
                
            model_id = f"{org_dir.name}/{model_dir.name}"
            print(f"Processing {model_id}...")
            
            try:
                merged = join_by_ground_truth(model_dir)
                results[model_id] = merged
                
                # Print stats
                total = len(merged)
                with_base = merged["pred_base"].notna().sum()
                with_ft = merged["pred_ft"].notna().sum()
                print(f"  Total samples: {total}")
                print(f"  With base predictions: {with_base}")
                print(f"  With fine-tuned predictions: {with_ft}")
                
            except Exception as e:
                print(f"  Error: {e}")
    
    return results

def export_merged_json(results: Dict[str, pd.DataFrame], output_path: Path):
    """Export merged data as JSON for the Next.js app."""
    export_data = {}
    
    for model_id, df in results.items():
        samples = []
        for _, row in df.iterrows():
            samples.append({
                "index": int(row["index"]),
                "ground_truth": row["ground_truth"],
                "dataset": row["dataset"],
                "num_turns": int(row["num_turns"]),
                "pred_base": row["pred_base"] if pd.notna(row["pred_base"]) else None,
                "pred_ft": row["pred_ft"] if pd.notna(row["pred_ft"]) else None,
                "metrics_base": {
                    "bertscore_f1": float(row["bertscore_f1_base"]) if pd.notna(row["bertscore_f1_base"]) else None,
                    "bleurt": float(row["bleurt_base"]) if pd.notna(row["bleurt_base"]) else None,
                    "ppl": float(row["ppl_base"]) if pd.notna(row["ppl_base"]) else None,
                },
                "metrics_ft": {
                    "bertscore_f1": float(row["bertscore_f1_ft"]) if pd.notna(row["bertscore_f1_ft"]) else None,
                    "bleurt": float(row["bleurt_ft"]) if pd.notna(row["bleurt_ft"]) else None,
                    "ppl": float(row["ppl_ft"]) if pd.notna(row["ppl_ft"]) else None,
                },
            })
        export_data[model_id] = samples
    
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\nExported merged data to {output_path}")

if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent  # Go up to user_turn_lora
    
    print("Joining predictions with chat_pairs by ground truth...\n")
    results = process_all_models(root_dir)
    
    # Export for Next.js app
    output_path = Path(__file__).parent.parent / "data" / "merged_predictions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_merged_json(results, output_path)
    
    print("\nDone!")
