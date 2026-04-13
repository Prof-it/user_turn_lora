"""
CLI for multi-turn rollout evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .rollout import run_free_assistant_rollout, run_reference_assisted_rollout
from .trajectory_data import load_rollout_dialogues


def _serialize_dialogues(dialogue_results: List[Dict]) -> List[Dict]:
    return [result["dialogue"] for result in dialogue_results]


def _serialize_steps(dialogue_results: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for result in dialogue_results:
        rows.extend(result["steps"])
    return rows


def _slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def _output_prefix(mode: str, simulator_kind: str, dataset: str, assistant_model_name: str | None = None) -> str:
    simulator_slug = simulator_kind.replace("-", "_")
    prefix = f"rollout_{mode}_{simulator_slug}_{dataset}"
    if assistant_model_name:
        prefix = f"{prefix}_{_slugify_model_name(assistant_model_name)}"
    return prefix


def run_rollout_evaluation(
    *,
    model_dir: Path,
    simulator_kind: str,
    rollout_mode: str,
    dataset_name: str,
    num_dialogues: int,
    seed_pairs: int,
    max_rollout_steps: int,
    min_rollout_steps: int,
    assistant_model_name: str | None,
    assistant_temperature: float,
    assistant_max_new_tokens: int,
    num_few_shot_examples: int,
    user_bertscore_threshold: float | None,
    user_bleurt_threshold: float | None,
) -> Dict:
    from .rollout_metrics import compute_rollout_metrics
    from .rollout_models import load_assistant_runtime, load_user_runtime

    dialogues = load_rollout_dialogues(
        dataset_name,
        num_dialogues=num_dialogues,
        seed_pairs=seed_pairs,
        min_rollout_steps=min_rollout_steps,
        max_rollout_steps=max_rollout_steps,
    )
    if not dialogues:
        raise ValueError("No dialogues matched the rollout constraints.")

    print(f"Loaded {len(dialogues)} dialogues for rollout evaluation")
    user_runtime = load_user_runtime(
        model_dir,
        simulator_kind=simulator_kind,
        num_few_shot_examples=num_few_shot_examples,
    )
    assistant_runtime = None
    if rollout_mode == "free_assistant":
        if not assistant_model_name:
            raise ValueError("--assistant-model-name is required for free_assistant mode")
        assistant_runtime = load_assistant_runtime(
            model_name=assistant_model_name,
            temperature=assistant_temperature,
            max_new_tokens=assistant_max_new_tokens,
        )

    try:
        dialogue_results = []
        for index, dialogue in enumerate(dialogues, start=1):
            if rollout_mode == "reference_assisted":
                result = run_reference_assisted_rollout(
                    dialogue,
                    user_generate=user_runtime.generate,
                    seed_pairs=seed_pairs,
                    max_rollout_steps=max_rollout_steps,
                )
            else:
                result = run_free_assistant_rollout(
                    dialogue,
                    user_generate=user_runtime.generate,
                    assistant_generate=assistant_runtime.generate,
                    seed_pairs=seed_pairs,
                    max_rollout_steps=max_rollout_steps,
                )
            dialogue_results.append(result)
            if index % 5 == 0 or index == len(dialogues):
                print(f"  {index}/{len(dialogues)} dialogues completed")

        dialogue_summaries = _serialize_dialogues(dialogue_results)
        rollout_rows = _serialize_steps(dialogue_results)
        metrics = compute_rollout_metrics(
            rollout_rows,
            dialogue_summaries,
            user_bertscore_threshold=user_bertscore_threshold,
            user_bleurt_threshold=user_bleurt_threshold,
        )
        return {
            "metadata": {
                "rollout_mode": rollout_mode,
                "dataset": dataset_name,
                "num_dialogues": len(dialogues),
                "seed_pairs": seed_pairs,
                "max_rollout_steps": max_rollout_steps,
                "simulator": user_runtime.metadata,
                "assistant": assistant_runtime.metadata if assistant_runtime else None,
            },
            "dialogues": dialogue_summaries,
            "steps": metrics["step_rows"],
            "summary": metrics["summary"],
        }
    finally:
        user_runtime.cleanup()
        if assistant_runtime is not None:
            assistant_runtime.cleanup()


def save_rollout_results(model_dir: Path, results: Dict) -> Path:
    prefix = _output_prefix(
        results["metadata"]["rollout_mode"],
        results["metadata"]["simulator"]["simulator_kind"],
        results["metadata"]["dataset"],
        results["metadata"]["assistant"]["assistant_model_name"] if results["metadata"]["assistant"] else None,
    )
    output_dir = model_dir / "rollouts"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"{prefix}_summary.json", "w") as f:
        json.dump({"metadata": results["metadata"], "summary": results["summary"]}, f, indent=2)
    with open(output_dir / f"{prefix}_dialogues.json", "w") as f:
        json.dump(results["dialogues"], f, indent=2)
    with open(output_dir / f"{prefix}_steps.json", "w") as f:
        json.dump(results["steps"], f, indent=2)
    pd.DataFrame(results["steps"]).to_csv(output_dir / f"{prefix}_steps.csv", index=False)
    return output_dir / f"{prefix}_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-turn rollout evaluation")
    parser.add_argument("--model-dir", type=Path, required=True, help="Existing model output directory")
    parser.add_argument(
        "--simulator-kind",
        choices=["finetuned", "base", "prompt_zero_shot", "prompt_few_shot"],
        default="finetuned",
    )
    parser.add_argument("--rollout-mode", choices=["reference_assisted", "free_assistant"], default="reference_assisted")
    parser.add_argument("--dataset", choices=["wildchat", "sgd", "all"], default="all")
    parser.add_argument("--num-dialogues", type=int, default=20)
    parser.add_argument("--seed-pairs", type=int, default=1)
    parser.add_argument("--max-rollout-steps", type=int, default=3)
    parser.add_argument("--min-rollout-steps", type=int, default=2)
    parser.add_argument("--assistant-model-name", type=str, default=None)
    parser.add_argument("--assistant-temperature", type=float, default=0.2)
    parser.add_argument("--assistant-max-new-tokens", type=int, default=128)
    parser.add_argument("--num-few-shot-examples", type=int, default=3)
    parser.add_argument("--user-bertscore-threshold", type=float, default=None)
    parser.add_argument("--user-bleurt-threshold", type=float, default=None)
    args = parser.parse_args()

    results = run_rollout_evaluation(
        model_dir=args.model_dir,
        simulator_kind=args.simulator_kind,
        rollout_mode=args.rollout_mode,
        dataset_name=args.dataset,
        num_dialogues=args.num_dialogues,
        seed_pairs=args.seed_pairs,
        max_rollout_steps=args.max_rollout_steps,
        min_rollout_steps=args.min_rollout_steps,
        assistant_model_name=args.assistant_model_name,
        assistant_temperature=args.assistant_temperature,
        assistant_max_new_tokens=args.assistant_max_new_tokens,
        num_few_shot_examples=args.num_few_shot_examples,
        user_bertscore_threshold=args.user_bertscore_threshold,
        user_bleurt_threshold=args.user_bleurt_threshold,
    )
    summary_path = save_rollout_results(args.model_dir, results)
    print(json.dumps(results["summary"], indent=2))
    print(f"Saved rollout outputs to {summary_path}")


if __name__ == "__main__":
    main()
