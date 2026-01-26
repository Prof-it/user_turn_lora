#!/usr/bin/env python
"""
Simple prompt baseline script - generates predictions only.
Metrics computed separately to avoid torch import issues.
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

SYSTEM_PROMPT = """You are simulating a human user in a conversation with an AI assistant.

Given a conversation history between a user and an assistant, predict what the user would say next.

Rules:
- Respond ONLY with what the user would say - no explanations, no meta-commentary
- Match the user's speaking style from the conversation history
- Be natural and conversational
- Do NOT act like an assistant - you ARE the user"""


def format_conversation(conversation):
    """Format conversation as string ending with User:"""
    lines = []
    for msg in conversation:
        role = msg["role"].capitalize()
        content = msg["content"].strip()
        lines.append(f"{role}: {content}")
    lines.append("User:")
    return "\n".join(lines)


def load_few_shot_examples(num_examples=3, seed=42):
    """Load few-shot examples from training data."""
    import random
    train_path = Path("outputs/Qwen-Qwen2.5-3B-Instruct/chat_pairs.json")
    with open(train_path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    # Use last 100 as "training" pool (different from eval)
    train_pool = data[-100:]
    return rng.sample(train_pool, min(num_examples, len(train_pool)))


def format_few_shot_example(example):
    """Format a single few-shot example."""
    lines = []
    for msg in example["conversation"]:
        role = msg["role"].capitalize()
        content = msg["content"].strip()
        lines.append(f"{role}: {content}")
    context = "\n".join(lines)
    target = example["target_user"].strip()
    return f"---\nConversation:\n{context}\n\nUser's next message: {target}\n---"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero-shot", "few-shot"], default="zero-shot")
    parser.add_argument("--num-examples", type=int, default=3)
    args = parser.parse_args()
    
    # Config
    model = "gpt-4o-mini"
    temperature = 0.4
    max_tokens = 256
    mode = args.mode
    
    # Load eval data
    eval_path = Path("outputs/Qwen-Qwen2.5-3B-Instruct/chat_pairs.json")
    with open(eval_path) as f:
        eval_data = json.load(f)
    
    # Use first 400 for eval (same as other experiments)
    eval_data = eval_data[:400]
    print(f"Loaded {len(eval_data)} samples")
    
    # Load few-shot examples if needed
    few_shot_examples = None
    if mode == "few-shot":
        few_shot_examples = load_few_shot_examples(args.num_examples)
        print(f"Loaded {len(few_shot_examples)} few-shot examples")
    
    # Output dir
    output_dir = Path(f"outputs/prompt_baseline/gpt-4o-mini-{mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save few-shot examples for reproducibility
    if few_shot_examples:
        with open(output_dir / "few_shot_examples.json", "w") as f:
            json.dump(few_shot_examples, f, indent=2)
    
    # Initialize client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Generate predictions
    print(f"\nGenerating predictions with {model}...")
    
    # Build system prompt
    if mode == "few-shot" and few_shot_examples:
        examples_str = "\n\n".join(format_few_shot_example(ex) for ex in few_shot_examples)
        system_prompt = SYSTEM_PROMPT + f"\n\nHere are some examples:\n\n{examples_str}\n\nNow predict the next user turn:"
    else:
        system_prompt = SYSTEM_PROMPT
    
    for item in tqdm(eval_data):
        context_str = format_conversation(item["conversation"])
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_str}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            pred = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}")
            pred = ""
            time.sleep(1)
        
        item["pred_prompt_baseline"] = pred
    
    # Save predictions
    output_path = output_dir / "predictions.json"
    with open(output_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"\nSaved predictions to {output_path}")
    
    # Save config
    config = {
        "model": model,
        "mode": mode,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "num_samples": len(eval_data),
        "num_few_shot_examples": len(few_shot_examples) if few_shot_examples else 0
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Done! Run metrics separately with:")
    print(f"  python scripts/compute_prompt_metrics.py {output_dir}")


if __name__ == "__main__":
    main()
