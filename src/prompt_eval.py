"""
Prompt-conditioned evaluation for local open-source base models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional


if TYPE_CHECKING:
    from .config import PipelineConfig


def load_pipeline_config(model_dir: Path) -> "PipelineConfig":
    """Load a saved pipeline config from an existing model output directory."""
    from .config_loader import load_saved_pipeline_config

    return load_saved_pipeline_config(model_dir)


def run_local_prompt_eval(
    model_dir: Path,
    mode: str,
    num_examples: int = 3,
    temperature: Optional[float] = None,
) -> Dict:
    """Evaluate a base HF model under a prompting condition."""
    from .condition_eval import save_condition_outputs
    from .generation import generate_from_messages
    from .model import cleanup_model, load_base_model, load_tokenizer
    from .prompting import build_prompt_messages, load_eval_data, load_few_shot_examples

    config = load_pipeline_config(model_dir)
    if temperature is not None:
        config.temperature = temperature

    eval_pairs = load_eval_data(model_dir, config.num_eval_samples)
    few_shot_examples = None
    if mode == "few-shot":
        few_shot_examples = load_few_shot_examples(
            num_examples=num_examples,
            model_dir=model_dir,
        )

    tokenizer = load_tokenizer(config)
    model = load_base_model(config, for_training=False)
    blocked_texts = [
        config.special_tokens.get("assistant_start", ""),
        config.special_tokens.get("system_start", ""),
        config.special_tokens.get("user_start", ""),
    ]

    predictions = []
    try:
        for item in eval_pairs:
            messages = build_prompt_messages(item["conversation"], mode, few_shot_examples)
            pred = generate_from_messages(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                config=config,
                blocked_texts=[text for text in blocked_texts if text],
            )
            predictions.append(pred)
    finally:
        cleanup_model(model=model)

    output_prefix = f"eval_prompt_{mode.replace('-', '_')}"
    summary = save_condition_outputs(
        output_dir=model_dir,
        output_prefix=output_prefix,
        eval_pairs=eval_pairs,
        predictions=predictions,
        metadata={
            "model_name": config.model_name,
            "condition_id": output_prefix,
            "mode": mode,
            "temperature": config.temperature,
            "num_few_shot_examples": num_examples if mode == "few-shot" else 0,
        },
    )
    print(f"{output_prefix}: BERTScore={summary['bertscore_f1_macro']:.4f}, BLEURT={summary['bleurt_macro']:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a base model with zero-shot or few-shot prompting")
    parser.add_argument("--model-dir", type=Path, required=True, help="Existing model output directory")
    parser.add_argument("--mode", choices=["zero-shot", "few-shot"], required=True)
    parser.add_argument("--num-examples", type=int, default=3, help="Few-shot example count")
    parser.add_argument("--temperature", type=float, default=None, help="Override generation temperature")
    args = parser.parse_args()
    run_local_prompt_eval(args.model_dir, args.mode, num_examples=args.num_examples, temperature=args.temperature)


if __name__ == "__main__":
    main()
