"""
OpenAI prompt-only baseline for user turn prediction.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .condition_eval import save_condition_outputs
from .prompting import build_prompt_messages, load_eval_data, load_few_shot_examples


load_dotenv()


@dataclass
class PromptConfig:
    """Configuration for the OpenAI prompt baseline."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.4
    max_tokens: int = 256
    mode: str = "zero-shot"
    num_few_shot_examples: int = 3
    eval_samples: int = 400


def predict_with_openai(
    client: OpenAI,
    conversation,
    config: PromptConfig,
    few_shot_examples=None,
) -> str:
    """Generate one prediction using the OpenAI Chat Completions API."""
    messages = build_prompt_messages(conversation, config.mode, few_shot_examples)
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def run_prompt_baseline(
    config: PromptConfig,
    output_dir: Path,
    model_dir: Optional[Path] = None,
) -> Dict:
    """Run the OpenAI baseline and save outputs in the shared condition format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)
    eval_data = load_eval_data(model_dir, config.eval_samples)
    few_shot_examples = None
    if config.mode == "few-shot":
        few_shot_examples = load_few_shot_examples(
            num_examples=config.num_few_shot_examples,
            model_dir=model_dir,
        )

    predictions = []
    print(f"\nGenerating {config.mode} predictions for {len(eval_data)} samples...")
    for item in tqdm(eval_data):
        try:
            pred = predict_with_openai(client, item["conversation"], config, few_shot_examples)
        except Exception as exc:
            print(f"Error generating prediction: {exc}")
            pred = ""
            time.sleep(1)
        predictions.append(pred)

    output_prefix = config.mode.replace("-", "_")
    summary = save_condition_outputs(
        output_dir=output_dir,
        output_prefix=output_prefix,
        eval_pairs=eval_data,
        predictions=predictions,
        metadata={
            "model": config.model,
            "mode": config.mode,
            "temperature": config.temperature,
            "num_few_shot_examples": config.num_few_shot_examples if config.mode == "few-shot" else 0,
        },
    )
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(
            [{**item, "pred_prompt_baseline": pred} for item, pred in zip(eval_data, predictions)],
            f,
            indent=2,
        )
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Prompt Baseline Results ({config.mode})")
    print(f"{'=' * 60}")
    print(f"Model: {config.model}")
    print(f"Samples: {summary['num_examples']}")
    print(f"BERTScore F1: {summary['bertscore_f1_macro']:.4f}")
    print(f"BLEURT: {summary['bleurt_macro']:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run OpenAI prompt baseline for user turn prediction")
    parser.add_argument("--mode", type=str, choices=["zero-shot", "few-shot"], default="zero-shot")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--samples", type=int, default=400, help="Number of evaluation samples")
    parser.add_argument("--num-examples", type=int, default=3, help="Few-shot example count")
    parser.add_argument("--temperature", type=float, default=0.4, help="Generation temperature")
    parser.add_argument("--model-dir", type=Path, default=None, help="Path to an existing model output directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for results")
    args = parser.parse_args()

    config = PromptConfig(
        model=args.model,
        mode=args.mode,
        temperature=args.temperature,
        eval_samples=args.samples,
        num_few_shot_examples=args.num_examples,
    )
    output_dir = args.output_dir or Path("outputs") / "prompt_baseline" / f"{args.model.replace('/', '-')}-{args.mode}"
    run_prompt_baseline(config, output_dir, args.model_dir)


if __name__ == "__main__":
    main()
