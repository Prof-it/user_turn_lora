"""
Shared prompt construction utilities for user-simulation baselines.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional


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
    """Format conversation history as role-prefixed plain text."""
    lines = []
    for msg in conversation:
        role = msg["role"].capitalize()
        content = msg["content"].strip()
        lines.append(f"{role}: {content}")
    lines.append("User:")
    return "\n".join(lines)


def format_few_shot_example(example: Dict) -> str:
    """Format one few-shot example in a stable, readable layout."""
    context_str = format_conversation_context(example["conversation"])
    context_str = context_str.rsplit("\nUser:", 1)[0]
    target = example["target_user"].strip()
    return f"""---
Conversation:
{context_str}

User's next message: {target}
---"""


def build_prompt_messages(
    conversation: List[Dict],
    mode: str,
    few_shot_examples: Optional[List[Dict]] = None,
) -> List[Dict[str, str]]:
    """Build a chat message list for zero-shot or few-shot prompting."""
    context_str = format_conversation_context(conversation)
    if mode == "few-shot" and few_shot_examples:
        examples_str = "\n\n".join(format_few_shot_example(ex) for ex in few_shot_examples)
        system_prompt = SYSTEM_PROMPT_FEW_SHOT.format(examples=examples_str)
    else:
        system_prompt = SYSTEM_PROMPT_ZERO_SHOT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_str},
    ]


def load_eval_data(model_dir: Optional[Path] = None, num_samples: int = 400) -> List[Dict]:
    """Load eval examples from an existing model directory or rebuild them."""
    if model_dir and (model_dir / "chat_pairs.json").exists():
        with open(model_dir / "chat_pairs.json") as f:
            data = json.load(f)
        return data[:num_samples]

    from .config import PipelineConfig
    from .data import build_eval_examples, load_data

    config = PipelineConfig()
    config.num_eval_samples = num_samples
    _, eval_pairs = load_data(config)
    return build_eval_examples(eval_pairs)[:num_samples]


def load_few_shot_examples(
    num_examples: int = 3,
    seed: int = 42,
    model_dir: Optional[Path] = None,
) -> List[Dict]:
    """Sample few-shot examples from saved training pairs or rebuild them."""
    if model_dir and (model_dir / "training_pairs.json").exists():
        with open(model_dir / "training_pairs.json") as f:
            train_pairs = json.load(f)
    else:
        from .config import PipelineConfig
        from .data import load_data

        config = PipelineConfig()
        config.num_train_samples = 100
        train_pairs, _ = load_data(config)

    rng = random.Random(seed)
    return rng.sample(train_pairs, min(num_examples, len(train_pairs)))
