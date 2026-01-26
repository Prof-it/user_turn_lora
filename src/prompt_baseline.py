"""
Prompt-only baseline for user turn prediction.

Uses OpenAI API (GPT-4o) to predict user turns without any fine-tuning.
The conversation context is formatted as a string with role prefixes,
ending with "User:" to prompt the model to complete the user's turn.

Usage:
    python -m src.prompt_baseline --mode zero-shot --samples 400
    python -m src.prompt_baseline --mode few-shot --num-examples 3
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from openai import OpenAI


@dataclass
class PromptConfig:
    """Configuration for prompt-only baseline."""
    model: str = "gpt-4o-mini"  # ~8B params, comparable to 3-7B open source models
    temperature: float = 0.4
    max_tokens: int = 256
    mode: str = "zero-shot"  # "zero-shot" or "few-shot"
    num_few_shot_examples: int = 3
    eval_samples: int = 400


SYSTEM_PROMPT_ZERO_SHOT = """You are simulating a human user in a conversation with an AI assistant.

Given a conversation history between a user and an assistant, predict what the user would say next.

Rules:
- Respond ONLY with what the user would say - no explanations, no meta-commentary
- Match the user's speaking style from the conversation history
- Be natural and conversational
- Do NOT act like an assistant - you ARE the user"""

SYSTEM_PROMPT_FEW_SHOT = """You are simulating a human user in a conversation with an AI assistant.

Given a conversation history between a user and an assistant, predict what the user would say next.

Rules:
- Respond ONLY with what the user would say - no explanations, no meta-commentary
- Match the user's speaking style from the conversation history
- Be natural and conversational
- Do NOT act like an assistant - you ARE the user

Here are some examples of conversation contexts and what the user said next:

{examples}

Now predict the next user turn for the following conversation:"""


def format_conversation_context(conversation: List[Dict]) -> str:
    """
    Format conversation history as a string with role prefixes.
    
    Args:
        conversation: List of {"role": "user"|"assistant", "content": "..."}
    
    Returns:
        Formatted string like:
        User: Hello
        Assistant: Hi there!
        User: How are you?
        Assistant: I'm doing well!
        User:
    """
    lines = []
    for msg in conversation:
        role = msg["role"].capitalize()
        content = msg["content"].strip()
        lines.append(f"{role}: {content}")
    
    # End with "User:" to prompt completion
    lines.append("User:")
    
    return "\n".join(lines)


def format_few_shot_example(example: Dict) -> str:
    """Format a single few-shot example."""
    context_str = format_conversation_context(example["conversation"])
    # Remove the trailing "User:" since we'll show the actual response
    context_str = context_str.rsplit("\nUser:", 1)[0]
    target = example["target_user"].strip()
    
    return f"""---
Conversation:
{context_str}

User's next message: {target}
---"""


def load_eval_data(model_dir: Optional[Path] = None, num_samples: int = 400) -> List[Dict]:
    """
    Load evaluation data from existing chat_pairs.json or generate fresh.
    """
    # Try to load from an existing model output directory
    if model_dir and (model_dir / "chat_pairs.json").exists():
        with open(model_dir / "chat_pairs.json") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples from {model_dir / 'chat_pairs.json'}")
        return data[:num_samples]
    
    # Otherwise load fresh from datasets
    from .config import PipelineConfig
    from .data import load_data, build_eval_examples
    
    config = PipelineConfig()
    config.num_eval_samples = num_samples
    _, eval_pairs = load_data(config)
    eval_pairs = build_eval_examples(eval_pairs)
    
    return eval_pairs[:num_samples]


def load_few_shot_examples(num_examples: int = 3, seed: int = 42) -> List[Dict]:
    """
    Load few-shot examples from training data.
    Uses a fixed seed for reproducibility.
    """
    from .config import PipelineConfig
    from .data import load_data, build_train_examples
    
    config = PipelineConfig()
    config.num_train_samples = 100  # Load a small subset
    train_pairs, _ = load_data(config)
    train_pairs = build_train_examples(train_pairs)
    
    # Use fixed seed for reproducibility
    import random
    rng = random.Random(seed)
    examples = rng.sample(train_pairs, min(num_examples, len(train_pairs)))
    
    return examples


def predict_with_openai(
    client: OpenAI,
    conversation: List[Dict],
    config: PromptConfig,
    few_shot_examples: Optional[List[Dict]] = None
) -> str:
    """
    Generate user turn prediction using OpenAI API.
    """
    context_str = format_conversation_context(conversation)
    
    if config.mode == "few-shot" and few_shot_examples:
        examples_str = "\n\n".join(format_few_shot_example(ex) for ex in few_shot_examples)
        system_prompt = SYSTEM_PROMPT_FEW_SHOT.format(examples=examples_str)
    else:
        system_prompt = SYSTEM_PROMPT_ZERO_SHOT
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_str}
    ]
    
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    
    return response.choices[0].message.content.strip()


def run_prompt_baseline(
    config: PromptConfig,
    output_dir: Path,
    model_dir: Optional[Path] = None
) -> Dict:
    """
    Run prompt-only baseline evaluation.
    
    Args:
        config: Prompt configuration
        output_dir: Where to save results
        model_dir: Optional path to load eval data from existing model outputs
    
    Returns:
        Dictionary with results and metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    # Load evaluation data
    eval_data = load_eval_data(model_dir, config.eval_samples)
    
    # Load few-shot examples if needed
    few_shot_examples = None
    if config.mode == "few-shot":
        few_shot_examples = load_few_shot_examples(config.num_few_shot_examples)
        print(f"Loaded {len(few_shot_examples)} few-shot examples")
        
        # Save few-shot examples for reproducibility
        with open(output_dir / "few_shot_examples.json", "w") as f:
            json.dump(few_shot_examples, f, indent=2)
    
    # Generate predictions
    predictions = []
    references = []
    
    print(f"\nGenerating {config.mode} predictions for {len(eval_data)} samples...")
    
    for item in tqdm(eval_data):
        conversation = item["conversation"]
        target = item["target_user"]
        
        try:
            pred = predict_with_openai(client, conversation, config, few_shot_examples)
        except Exception as e:
            print(f"Error generating prediction: {e}")
            pred = ""
            time.sleep(1)  # Rate limit backoff
        
        predictions.append(pred)
        references.append(target)
        
        # Add prediction to item
        item["pred_prompt_baseline"] = pred
    
    # Save predictions
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"Saved predictions to {predictions_path}")
    
    # Compute metrics (lazy import to avoid slow startup)
    print("\nComputing metrics...")
    from .evaluate import compute_bertscore, compute_bleurt
    
    bertscore_results = compute_bertscore(references, predictions)
    bleurt_results = compute_bleurt(references, predictions)
    
    metrics = {
        "model": config.model,
        "mode": config.mode,
        "temperature": config.temperature,
        "num_samples": len(eval_data),
        "num_few_shot_examples": config.num_few_shot_examples if config.mode == "few-shot" else 0,
        "bertscore_f1": bertscore_results["f1"],
        "bertscore_precision": bertscore_results["precision"],
        "bertscore_recall": bertscore_results["recall"],
        "bleurt": bleurt_results["mean"],
    }
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Prompt Baseline Results ({config.mode})")
    print(f"{'='*60}")
    print(f"Model: {config.model}")
    print(f"Samples: {len(eval_data)}")
    print(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")
    print(f"BLEURT: {metrics['bleurt']:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run prompt-only baseline for user turn prediction"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zero-shot", "few-shot"],
        default="zero-shot",
        help="Prompting mode"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=400,
        help="Number of evaluation samples"
    )
    
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of few-shot examples (only used in few-shot mode)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Generation temperature"
    )
    
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to existing model output dir to load eval data from"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    config = PromptConfig(
        model=args.model,
        mode=args.mode,
        temperature=args.temperature,
        eval_samples=args.samples,
        num_few_shot_examples=args.num_examples,
    )
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_name = args.model.replace("/", "-")
        output_dir = Path("outputs") / "prompt_baseline" / f"{model_name}-{args.mode}"
    
    run_prompt_baseline(config, output_dir, args.model_dir)


if __name__ == "__main__":
    main()
